#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Tuple
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import statistics
import warnings
import difflib
import tempfile

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr

from data.bioparse.parser.pdb_to_complex import pdb_to_complex as pdb_to_complex_raw
from data.bioparse import VOCAB, Block, const
from data.bioparse.utils import recur_index
from evaluation import diversity
from evaluation.rmsd import compute_rmsd
from evaluation.dockq import dockq
from utils.random_seed import setup_seed
from evaluation.seq import align_sequences
# from evaluation.openmm_relaxer import ForceFieldMinimizer
# from evaluation.energy import pyrosetta_interface_energy
from evaluation.clash import eval_pdb_clash
from evaluation.dihedrals import jsd_angle_profile, dihedral_distribution
from utils.logger import print_log


def pdb_to_complex(pdb_path, selected_chains):
# Optimization: Call raw function directly to avoid slow temp file IO on cluster
    # Assuming remove_het=True handles the HETATM filtering
    return pdb_to_complex_raw(pdb_path, selected_chains, remove_het=True)

# Global variable for worker process
_pyrosetta_fn = None

def init_worker():
    global _pyrosetta_fn
    try:
        from evaluation.energy import pyrosetta_interface_energy
        _pyrosetta_fn = pyrosetta_interface_energy
    except ImportError:
        print_log("Failed to import pyrosetta. Ensure it is installed.", level='ERROR')
        _pyrosetta_fn = None


def _get_ref_pdb(_id, root_dir):
    return os.path.join(root_dir, 'references', f'{_id}_ref.pdb')


def _get_gen_pdb(_id, number, root_dir, use_rosetta=False):
    suffix = '_rosetta' if use_rosetta else ''
    return os.path.join(root_dir, 'candidates', _id, f'{number}{suffix}.pdb')


def _mean(vals):
    return sum(vals) / len(vals)


def robust_dG(pdb, rec_chains, lig_chains, gen_block_idx=None, n=3, relax=True, relax_save_pdb=None):
    if _pyrosetta_fn is None:
        print_log("PyRosetta function not initialized in worker.", level='ERROR')
        return 0.0

    energies = []
    
    # helper for relax options
    relax_opt = {}
    if gen_block_idx is not None:
        def _to_rosetta_res_id(idx):
            res_id = [idx[0], idx[1][0]]
            if idx[1][1] != '': res_id.append(idx[1][1])
            return res_id
        relax_opt = {
            'flexible_residue_first': _to_rosetta_res_id(gen_block_idx[0]),
            'flexible_residue_last': _to_rosetta_res_id(gen_block_idx[-1])
        }

    min_dG = None

    for _ in range(n):
        try:
            dG = _pyrosetta_fn(
                pdb_path=os.path.abspath(pdb),
                receptor_chains=rec_chains,
                ligand_chains=lig_chains,
                relax=relax,
                relax_opt=relax_opt,
                relax_save_pdb=os.path.abspath(relax_save_pdb) if relax_save_pdb else None
            )
        except Exception as e:
            print_log(f'PyRosetta energy calc failed: {e}', level='WARN')
            dG = 0.0
            
        energies.append(dG)

        # save pose with the best energy
        if relax_save_pdb is not None:
            if (min_dG is None or energies[-1] <= min_dG) and os.path.exists(relax_save_pdb): # update best pose
                with open(relax_save_pdb, 'r') as fin: min_pose_text = fin.read()
                min_dG = energies[-1]
            elif min_dG is not None and energies[-1] > min_dG: # recover original pose
                with open(relax_save_pdb, 'w') as fout: fout.write(min_pose_text)

    return _mean(energies)


def extract_paired_coords(gen_blocks: List[Block], ref_blocks: List[Block], ca_only: bool):
    # Full atom RMSD
    gen_x, ref_x = [], []
    for gen_block, ref_block in zip(gen_blocks, ref_blocks):
        allow_atoms = { atom.name for atom in ref_block }
        if ca_only: allow_atoms = { 'CA': 1 } if 'CA' in allow_atoms else {}
        elif gen_block.name != ref_block.name: # only backbone atoms
            allow_atoms = { atom: 1 for atom in const.backbone_atoms if atom in allow_atoms }
    
        gen_atoms, ref_atoms = {}, {}
        for atom in gen_block: gen_atoms[atom.name] = atom.get_coord()
        for atom in ref_block: ref_atoms[atom.name] = atom.get_coord()
        for ref_atom in allow_atoms:
            if ref_atom in gen_atoms:
                ref_x.append(ref_atoms[ref_atom])
                gen_x.append(gen_atoms[ref_atom])
    return np.array(gen_x), np.array(ref_x)


@dataclass
class Task:
    root_dir: str
    id: str
    number: int
    target_chains_ids: List[str]
    ligand_chains_ids: List[str]
    gen_block_idx: List[tuple]
    pmetric: float
    is_antibody: bool
    calc_dg: bool = False
    ref_metrics: dict = None
    metrics: dict = None
    seq: str = None
    ca_coords: List[tuple] = None


def run_ref_metrics(task: Task):
    if task.is_antibody:
        ref_pdb = _get_ref_pdb(os.path.dirname(task.id), task.root_dir)
    else:
        ref_pdb = _get_ref_pdb(task.id, task.root_dir)
    
    task.ref_metrics = {}
    # set reference dG
    # set reference dG
    if task.calc_dg:
        print_log(f"[{task.id}] Start ref dG", level='INFO')
        ref_dG_relax = robust_dG(
            ref_pdb, task.target_chains_ids,
            task.ligand_chains_ids,
            task.gen_block_idx if task.is_antibody else None,
            n=3,
            relax_save_pdb=ref_pdb.rstrip('.pdb') + '_rosetta.pdb')
        task.ref_metrics['ref_dG'] = round(ref_dG_relax, 2)
        print_log(f"[{task.id}] End ref dG", level='INFO')
    else:
        task.ref_metrics['ref_dG'] = 0.0


    # ref clash
    clash_inner, clash_outer = eval_pdb_clash(ref_pdb, task.target_chains_ids, task.ligand_chains_ids)
    task.ref_metrics['ref_Clash_inner'] = round(clash_inner, 4)
    task.ref_metrics['ref_Clash_outer'] = round(clash_outer, 4)

    return task


def run_basic_metrics(task: Task):
    task.metrics = {}

    # get complex
    gen_pdb = _get_gen_pdb(task.id, task.number, task.root_dir)
    try:
        print_log(f'[DEBUG] {task.id}_{task.number}: Start pdb_to_complex {gen_pdb}', level='INFO')
        gen_cplx = pdb_to_complex(gen_pdb, task.target_chains_ids + task.ligand_chains_ids)
        print_log(f'[DEBUG] {task.id}_{task.number}: End pdb_to_complex', level='INFO')
    except Exception as e:
        print_log(f'{gen_pdb} read in complex error! {task}', level='ERROR')
        task.metrics = {}  # Use regular dict instead of defaultdict(lambda) to avoid pickle error
        return task
    if task.is_antibody:
        ref_pdb = _get_ref_pdb(os.path.dirname(task.id), task.root_dir)
    else:
        ref_pdb = _get_ref_pdb(task.id, task.root_dir)
    ref_cplx = pdb_to_complex(ref_pdb, task.target_chains_ids + task.ligand_chains_ids)

    # get generated blocks
    gen_block_idx = [(idx[0], tuple(idx[1])) for idx in task.gen_block_idx]
    try:
        gen_blocks = [recur_index(gen_cplx, block_id) for block_id in gen_block_idx]
    except Exception:
        print_log(f'{gen_pdb} generation part missing', level='ERROR')
    ref_blocks = [recur_index(ref_cplx, block_id) for block_id in gen_block_idx]
    
    # get sequence
    gen_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in gen_blocks])
    ref_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in ref_blocks]) 

    # set sequence
    task.seq = gen_seq

    # aar
    assert len(gen_seq) == len(ref_seq)
    print_log(f'[DEBUG] {task.id}_{task.number}: Start align_sequences', level='INFO')
    _, seq_id = align_sequences(gen_seq, ref_seq)
    print_log(f'[DEBUG] {task.id}_{task.number}: End align_sequences', level='INFO')
    task.metrics['AAR'] = round(seq_id, 4)

    # CA coordinates
    gen_ca_x, ref_ca_x = extract_paired_coords(gen_blocks, ref_blocks, ca_only=True)

    # set CA coordinates
    task.ca_coords = gen_ca_x

    # CA RMSD
    c_rmsd = compute_rmsd(gen_ca_x, ref_ca_x, need_align=False)
    task.metrics['C_RMSD(CA)'] = round(c_rmsd, 2)
    l_rmsd = compute_rmsd(gen_ca_x, ref_ca_x, need_align=True)
    task.metrics['L_RMSD(CA)'] = round(l_rmsd, 2)
    
    # DockQ
    try:
        dockq_score = round(dockq(gen_pdb, ref_pdb, task.target_chains_ids, task.ligand_chains_ids), 3)
    except (ValueError, KeyError):
        dockq_score = 0
    task.metrics['DockQ'] = dockq_score

    # clash
    try:
        clash_inner, clash_outer = eval_pdb_clash(gen_pdb, task.target_chains_ids, task.ligand_chains_ids)
    except Exception:
        clash_inner, clash_outer = float('nan'), float('nan')
    task.metrics['Clash_inner'] = round(clash_inner, 4)
    task.metrics['Clash_outer'] = round(clash_outer, 4)

    return task


def run_dG(task: Task):
    print_log(f"[{task.id}_{task.number}] Start run dG", level='INFO')
    gen_pdb = _get_gen_pdb(task.id, task.number, task.root_dir)

    # Rosetta dG relax
    dG = robust_dG(
        gen_pdb, task.target_chains_ids, task.ligand_chains_ids,
        task.gen_block_idx if task.is_antibody else None,
        n=3,
        relax_save_pdb=gen_pdb.rstrip('.pdb') + '_rosetta.pdb')
    dG = round(dG, 2)
    task.metrics['dG'] = dG
    task.metrics['ddG'] = dG - task.ref_metrics['ref_dG']
    task.metrics['ddG <= 0 (IMP)'] = dG <= task.ref_metrics['ref_dG']
    print_log(f"[{task.id}_{task.number}] End run dG", level='INFO')

    return task


def aggregate_metrics(tasks: List[Task]):
    pmets = [task.pmetric for task in tasks]
    indexes = list(range(len(tasks)))

    aggr_results = {}

    # aggregation
    def nan_filter(values):
        return [val for val in values if not np.isnan(val)]

    # Find first task with valid metrics (skip failed tasks with empty dict)
    first_valid = next((t for t in tasks if t.metrics), None)
    if first_valid is None:
        return aggr_results  # All tasks failed
    
    for name in first_valid.metrics:
        # Use .get() to handle tasks with missing metrics (e.g., failed tasks)
        vals = [task.metrics.get(name, float('nan')) for task in tasks]
        vals_not_none = nan_filter(vals)
        if len(vals_not_none) < 3: corr = 0 # at least 3 not being nan
        else: corr = spearmanr(vals, pmets, nan_policy='omit').statistic
        if np.isnan(corr): corr = 0
        corr = round(corr, 3)
        if len(vals_not_none) == 0: vals_not_none = [float('nan')] # all failed
        aggr_res = {
            'max': max(vals_not_none),
            'min': min(vals_not_none),
            'mean': sum(vals_not_none) / len(vals_not_none),
            'random': vals_not_none[0],
            'max*': vals[(max if corr > 0 else min)(indexes, key=lambda i: pmets[i])],
            'min*': vals[(min if corr > 0 else max)(indexes, key=lambda i: pmets[i])],
            'pmet_corr': corr,
            'individual': vals,
            'individual_pmet': pmets
        }
        aggr_results[name] = aggr_res

    if len(tasks) > 1:
        seqs = [task.seq for task in tasks if (task.seq is not None and task.ca_coords is not None)]
        ca_coords = [task.ca_coords for task in tasks if (task.seq is not None and task.ca_coords is not None)]
        if len(seqs) > 1:
            print_log(f'[DEBUG] Start diversity calc for {len(seqs)} sequences', level='INFO')
            seq_div, struct_div, co_div, _ = diversity.diversity(seqs, np.array(ca_coords))
            print_log(f'[DEBUG] End diversity calc', level='INFO')
        else:
            seq_div, struct_div, co_div = 0, 0, 0
        aggr_results['Sequence Diversity'] = round(seq_div, 3)
        aggr_results['Struct Diversity'] = round(struct_div, 3)
        aggr_results['Codesign Diversity'] = round(co_div, 3)
    
    return aggr_results


def run_ref_pipeline(input: Tuple[dict, bool, bool]):
    item, is_antibody, calc_dg = input
    ref_task = Task(
        root_dir = item['root_dir'],
        id = item['id'],
        number = item['n'],
        target_chains_ids = item['target_chains_ids'],
        ligand_chains_ids = item['ligand_chains_ids'],
        gen_block_idx = item['gen_block_idx'],
        pmetric = round(item['pmetric'], 4),
        is_antibody = is_antibody,
        calc_dg = calc_dg
    )

    ref_funcs = [
        run_ref_metrics,
    ]
    for fn in ref_funcs:
        ref_task = fn(ref_task)
    
    return ref_task.id, ref_task.ref_metrics


def run_candidate_pipeline(input: Tuple[dict, bool, bool, dict]):
    item, is_antibody, calc_dg, ref_metrics = input
    task = Task(
        root_dir = item['root_dir'],
        id = item['id'],
        number = item['n'],
        target_chains_ids = item['target_chains_ids'],
        ligand_chains_ids = item['ligand_chains_ids'],
        gen_block_idx = item['gen_block_idx'],
        pmetric = round(item['pmetric'], 4),
        is_antibody = is_antibody,
        calc_dg = calc_dg,
        ref_metrics = ref_metrics
    )

    funcs = [
        run_basic_metrics,
    ]
    if calc_dg:
        funcs.append(run_dG)

    for fn in funcs:
        task = fn(task)
    
    return task



def main(args):
    root_dir = os.path.dirname(args.results)
    # load results
    with open(args.results, 'r') as fin:
        lines = fin.read().strip().split('\n')
    id2items = {}
    for line in lines:
        item = json.loads(line)
        item['root_dir'] = root_dir
        _id = item['id']
        if _id not in id2items: id2items[_id] = []
        id2items[_id].append(item)
    ids = list(id2items.keys())
    
    # Filter specific IDs requested to be skipped
    ids = [i for i in ids if i not in ['3wbn', '6pj8', '6hgt']]
    
    eval_results_path = os.path.join(os.path.dirname(args.results), f'eval_report_{args.log_suffix}.jsonl')

    # Check if eval results already exist and are complete
    skip_processing = False
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            existing_lines = f.read().strip().split('\n')
            if len(existing_lines) == len(ids):
                print_log(f"Found complete eval results ({len(existing_lines)} IDs). Skipping processing...", level='INFO')
                skip_processing = True
                # Load existing results
                ref_metrics_list, metrics_list = [], []
                for line in existing_lines:
                    record = json.loads(line)
                    ref_metrics_list.append({k: record[k] for k in record if k.startswith('ref_')})
                    metrics_list.append({k: record[k] for k in record if not k.startswith('ref_') and k != 'id'})
            else:
                print_log(f"Eval results incomplete ({len(existing_lines)}/{len(ids)} IDs). Re-processing...", level='INFO')

    if not skip_processing:
        # Run parallel processing with multiprocessing.Pool
        from multiprocessing import Pool
        
        # Open file for writing
        with open(eval_results_path, 'w') as fout:
            # 1. Initialize Pool globally (if parallel)
            pool = None
            if args.num_workers > 1:
                pool = Pool(processes=args.num_workers, initializer=init_worker)

            try:
                # Stage 1: Calculate Reference Metrics (Pre-calc for all IDs)
                print_log("Stage 1: Calculating Reference Metrics...", level='INFO')
                ref_inputs = [(id2items[_id][0], args.antibody, args.calc_dg) for _id in ids]
                id_to_ref_metrics = {}
                
                if pool:
                    for res in tqdm(pool.imap_unordered(run_ref_pipeline, ref_inputs), total=len(ref_inputs)):
                        _id, ref_metrics = res
                        id_to_ref_metrics[_id] = ref_metrics
                else:
                    if args.calc_dg: init_worker()
                    for inp in tqdm(ref_inputs):
                        _id, ref_metrics = run_ref_pipeline(inp)
                        id_to_ref_metrics[_id] = ref_metrics

                # Stage 2: Calculate Candidate Metrics (Async Non-Blocking)
                print_log("Stage 2: Calculating Candidate Metrics (Async Non-Blocking)...", level='INFO')
                
                # Prepare ALL inputs
                all_cand_inputs = []
                id_counts = {} # To know how many items to expect per ID
                for _id in ids:
                    ref = id_to_ref_metrics[_id]
                    items = id2items[_id]
                    id_counts[_id] = len(items)
                    for item in items:
                        all_cand_inputs.append((item, args.antibody, args.calc_dg, ref))
                
                print_log(f"Total candidate tasks: {len(all_cand_inputs)}", level='INFO')

                # Buffers for async collection
                id_buffers = defaultdict(list)
                finished_ids_count = 0
                
                ref_metrics_list, metrics_list = [], [] # For final summary

                # Result Iterator
                if pool:
                    # imap_unordered ensures we get results as SOON as they finish, maximizing CPU utility.
                    # The pool will keep pre-fetching and running tasks while we (main thread) are processing the current result.
                    result_iter = pool.imap_unordered(run_candidate_pipeline, all_cand_inputs, chunksize=1)
                else:
                    result_iter = (run_candidate_pipeline(inp) for inp in all_cand_inputs)

                # Process Loop
                pbar = tqdm(total=len(all_cand_inputs))
                for task in result_iter:
                    pbar.update(1)
                    
                    # Add to buffer
                    _id = task.id
                    id_buffers[_id].append(task)
                    
                    # Check if this ID is complete
                    if len(id_buffers[_id]) == id_counts[_id]:
                        # Trigger Aggregation & Write
                        # Note: Workers are still running in background processing other IDs!
                        # The time spent here does NOT waste CPU cycles of the workers.
                        
                        tasks = id_buffers[_id]
                        ref = tasks[0].ref_metrics # They all share same ref metrics
                        
                        # Aggregate
                        aggr_res = aggregate_metrics(tasks)
                        
                        # Store
                        ref_metrics_list.append(ref)
                        metrics_list.append(aggr_res)
                        
                        # Write
                        record = { 'id': _id }
                        record.update(ref)
                        record.update(aggr_res)

                        fout.write(json.dumps(record) + '\n')
                        fout.flush()
                        
                        print_log(f"Finished ID {_id} ({len(tasks)} items). Saved.", level='INFO')
                        
                        # Clear buffer to free memory
                        del id_buffers[_id]
                        finished_ids_count += 1
                
                pbar.close()
                print_log(f"All {finished_ids_count} IDs completed.", level='INFO')

            finally:
                if pool:
                    pool.close()
                    pool.join()


    
    log_file = open(os.path.join(os.path.dirname(args.results), f'eval_final_{args.log_suffix}.log'), 'w')
    def print_and_log(s, **kwargs):
        print(s, **kwargs)
        log_file.write(s + '\n')

    if not ref_metrics_list:
        print_and_log("No results generated.")
        log_file.close()
        return

    # calculate reference
    for name in ref_metrics_list[0]:
        vals = [m[name] for m in ref_metrics_list]
        print_and_log(f'reference {name}: mean {sum(vals) / len(vals)}, median {statistics.median(vals)}')

    # individual level results
    print_and_log('Point-wise evaluation results:')
    # Filter out empty metrics (from IDs where all tasks failed)
    metrics_list = [m for m in metrics_list if m]
    if not metrics_list:
         print_and_log("No metrics available.")
    else:
        for name in metrics_list[0]:
            # Use .get() to handle records missing this metric (e.g., from partial failures)
            vals = [item.get(name) for item in metrics_list]
            vals = [v for v in vals if v is not None]
            if not vals:
                print_and_log(f'{name}: No valid data')
                continue
            if isinstance(vals[0], dict):
                if (('RMSD' in name) or ('dG' in name) or ('ddG' in name)) and '<=' not in name:
                    aggr = 'min'
                elif 'Clash' in name: # clash ratio. Average performance should be measured
                    aggr = 'mean'
                else:
                    aggr = 'max'
                all_vals = [val[aggr] for val in vals]
                not_nan_idx = [i for i in range(len(all_vals)) if not np.isnan(all_vals[i])]
                aggr_vals = [val for val in all_vals if not np.isnan(val)]
                if '>=' in name or '<=' in name:  # percentage
                    if len(aggr_vals) > 0:
                        print_and_log(f'{name}: {sum(aggr_vals) / len(aggr_vals)}')
                    else:
                        print_and_log(f'{name}: NaN')
                else:
                    if len(aggr_vals) > 0:
                        if 'RMSD' in name:  # use median as conventions
                            print_and_log(f'{name} (median): {statistics.median(aggr_vals)}')
                        else:
                            print_and_log(f'{name} (mean): {sum(aggr_vals) / len(aggr_vals)}')
                    else:
                         print_and_log(f'{name}: NaN')

                    if len(not_nan_idx) == 0:
                        print_log('all values are nan!', level='WARN')
                        lowest_i, highest_i = 0, 0
                    else:
                        lowest_i = min(not_nan_idx, key=lambda i: all_vals[i])
                        highest_i = max(not_nan_idx, key=lambda i: all_vals[i])
                    print_and_log(f'\tlowest: {all_vals[lowest_i]}, id: {ids[lowest_i]}', end='')
                    print_and_log(f'\thighest: {all_vals[highest_i]}, id: {ids[highest_i]}')
                    corrs = [val['pmet_corr'] for val in vals if val['pmet_corr'] != 0]
                    if len(corrs) == 0: corrs = [0]
                    print_and_log(f'\tcorrelation with flow matching likelihood: {sum(corrs) / len(corrs)}')
                
            else:
                print_and_log(f'{name} (mean): {sum(vals) / len(vals)}')
                lowest_i = min([i for i in range(len(vals))], key=lambda i: vals[i])
                highest_i = max([i for i in range(len(vals))], key=lambda i: vals[i])
                print_and_log(f'\tlowest: {vals[lowest_i]}, id: {ids[lowest_i]}')
                print_and_log(f'\thighest: {vals[highest_i]}, id: {ids[highest_i]}')
    
    # get distribution of dihedral angles
    print_and_log('\n')
    pdbs, selected_residues = [], []
    for _id in id2items:
        for item in id2items[_id]:
            pdb = _get_gen_pdb(item['id'], item['n'], root_dir)
            pdbs.append(pdb)
            selected_residues.append(item['gen_block_idx'])
    print_and_log('JSD of dihedral angles:')
    dist = dihedral_distribution(pdbs, all_selected_residues=selected_residues, num_cpus=args.num_workers)
    profile = jsd_angle_profile(dist, 'antibody' if args.antibody else 'peptide')
    for key in profile:
        if key not in ['backbone_overall', 'sidechain_overall']: print_and_log(f'{key}: {profile[key]}')
    print_and_log('\n')
    print_and_log(f'backbone_overall (JSD_bb): {profile["backbone_overall"]}')
    print_and_log(f'sidechain_overall (JSD_sc): {profile["sidechain_overall"]}')
    print_and_log('\n')

    log_file.close()


def parse():
    parser = argparse.ArgumentParser(description='calculate metrics')
    parser.add_argument('--results', type=str, required=True, help='Path to test set')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use')
    parser.add_argument('--antibody', action='store_true', help='Special reference id and only relax CDR regions')
    parser.add_argument('--calc_dg', action='store_true', help='Whether to calculate dG (slow)')
    parser.add_argument('--log_suffix', type=str, default='', help='Suffix of the log file (eval_final.log)')

    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(0)
    main(parse())
