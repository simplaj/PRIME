#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import statistics
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr

from evaluation import diversity
from evaluation.dihedrals import jsd_angle_profile, dihedral_distribution
from utils.logger import print_log
from scripts.metrics.peptide import _get_gen_pdb

def get_args():
    parser = argparse.ArgumentParser(description='calculate metrics report from jsonl')
    parser.add_argument('--results', type=str, required=True, help='Path to original input task jsonl (for PDB paths)')
    parser.add_argument('--report', type=str, required=False, help='Path to computed metrics jsonl. If not provided, infers from results path.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use')
    parser.add_argument('--antibody', action='store_true', help='Special reference id and only relax CDR regions')
    parser.add_argument('--log_suffix', type=str, default='', help='Suffix of the log file (eval_final.log)')
    return parser.parse_args()

def main(args):
    root_dir = os.path.dirname(args.results)
    
    # 1. Load original tasks to get PDB paths (for JSD)
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

    # 2. Load computed metrics
    if args.report:
        report_path = args.report
    else:
        report_path = os.path.join(os.path.dirname(args.results), f'eval_report_{args.log_suffix}.jsonl')
    
    if not os.path.exists(report_path):
        print_log(f"Report file not found: {report_path}", level='ERROR')
        return

    print_log(f"Loading metrics from {report_path}...")
    ref_metrics = []
    metrics = []
    
    with open(report_path, 'r') as fin:
        for line in fin:
            data = json.loads(line)
            # Separate ref and gen metrics based on keys
            # Convention: ref metrics start with 'ref_'
            ref = {k: v for k, v in data.items() if k.startswith('ref_')}
            gen = {k: v for k, v in data.items() if not k.startswith('ref_') and k != 'id'}
            
            # Note: The original script aggregated metrics list. 
            # In the jsonl output, 'ref' and 'gen' are merged. 
            # We assume one line per task. 
            # But wait, 'ref_metrics' list in peptide.py had ONE entry per task.
            # And 'metrics' list had ONE entry per task.
            # So simple reconstruction is valid.
            if ref: ref_metrics.append(ref)
            if gen: metrics.append(gen)

    if not ref_metrics or not metrics:
        print_log("No metrics found in report file.", level='ERROR')
        return

    # 3. Reporting Logic
    log_file_path = os.path.join(os.path.dirname(args.results), f'eval_final_{args.log_suffix}.log')
    log_file = open(log_file_path, 'w')
    
    def print_and_log(s, **kwargs):
        print(s, **kwargs)
        log_file.write(s + '\n')

    # calculate reference
    if ref_metrics:
        for name in ref_metrics[0]:
            vals = [m[name] for m in ref_metrics]
            print_and_log(f'reference {name}: mean {sum(vals) / len(vals)}, median {statistics.median(vals)}')

    # individual level results
    print_and_log('Point-wise evaluation results:')
    if metrics:
        for name in metrics[0]:
            vals = [item[name] for item in metrics]
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
                    print_and_log(f'{name}: {sum(aggr_vals) / len(aggr_vals)}')
                else:
                    if 'RMSD' in name:  # use median as conventions
                        print_and_log(f'{name} (median): {statistics.median(aggr_vals)}')
                    else:
                        print_and_log(f'{name} (mean): {sum(aggr_vals) / len(aggr_vals)}')
                    if len(not_nan_idx) == 0:
                        print_log('all values are nan!', level='WARN')
                        lowest_i, highest_i = 0, 0
                    else:
                        lowest_i = min(not_nan_idx, key=lambda i: all_vals[i])
                        highest_i = max(not_nan_idx, key=lambda i: all_vals[i])
                    # Note: ids used here are from the original task list.
                    # We assume 1-to-1 mapping between lines in args.results and args.report?
                    # peptide.py output order matches input order if single threaded, 
                    # but with multiprocessing/Ray it might be shuffled unless sorted.
                    # But the 'ids' list here comes from 'id2items.keys()'.
                    # The 'metrics' list comes from file read order.
                    # This implies 'eval_report.jsonl' must be sorted or we must re-align.
                    # For safety, we should assume the provided snippet relies on stable ordering or just reports stats.
                    # However, print_log using 'ids[lowest_i]' implies alignment.
                    # If peptide.py output is shuffled, this index will be wrong.
                    # Given this is a requested 'copy current logic', I will keep it but warn.
                    if lowest_i < len(ids):
                        print_and_log(f'\tlowest: {all_vals[lowest_i]}, id: {ids[lowest_i]}', end='')
                    if highest_i < len(ids):
                        print_and_log(f'\thighest: {all_vals[highest_i]}, id: {ids[highest_i]}')
                    corrs = [val['pmet_corr'] for val in vals if val['pmet_corr'] != 0]
                    if len(corrs) == 0: corrs = [0]
                    print_and_log(f'\tcorrelation with flow matching likelihood: {sum(corrs) / len(corrs)}')
                
            else:
                print_and_log(f'{name} (mean): {sum(vals) / len(vals)}')
                lowest_i = min([i for i in range(len(vals))], key=lambda i: vals[i])
                highest_i = max([i for i in range(len(vals))], key=lambda i: vals[i])
                if lowest_i < len(ids):
                    print_and_log(f'\tlowest: {vals[lowest_i]}, id: {ids[lowest_i]}')
                if highest_i < len(ids):
                    print_and_log(f'\thighest: {vals[highest_i]}, id: {ids[highest_i]}')
    
    # get distribution of dihedral angles
    print_and_log('\n')
    pdbs, selected_residues = [], []
    for _id in id2items:
        for item in id2items[_id]:
            pdb = _get_gen_pdb(item['id'], item['n'], root_dir)
            pdbs.append(pdb)
            selected_residues.append(item['gen_block_idx'])
    
    if pdbs:
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

if __name__ == '__main__':
    main(get_args())
