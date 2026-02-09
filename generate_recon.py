#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
import json
import torch
import yaml
from tqdm import tqdm
from copy import deepcopy

from rdkit import Chem

from data import create_dataset
from data.bioparse import VOCAB
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data.base import Summary
from models.LDM.data_utils import OverwriteTask
from utils.random_seed import setup_seed


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    elif hasattr(data, 'to'):
        return data.to(device)
    return data


def overwrite(cplx, summary, S, X, A, ll, bonds, intra_bonds, out_path):
    """与 generate.py 保持一致的保存逻辑"""
    task = OverwriteTask(
        cplx=cplx,
        select_indexes=summary.select_indexes,
        generate_mask=summary.generate_mask,
        target_chain_ids=summary.target_chain_ids,
        ligand_chain_ids=summary.ligand_chain_ids,
        S=S, X=X, A=A, ll=ll,
        inter_bonds=bonds,
        intra_bonds=intra_bonds,
        out_path=out_path
    )
    cplx, gen_mol, overwrite_indexes = task.get_overwritten_results(check_validity=False)
    if cplx is None or gen_mol is None:
        return None
    return {
        'id': summary.id,
        'pmetric': task.get_total_likelihood(),
        'smiles': Chem.MolToSmiles(gen_mol),
        'gen_seq': task.get_generated_seq(),
        'target_chains_ids': summary.target_chain_ids,
        'ligand_chains_ids': summary.ligand_chain_ids,
        'gen_block_idx': overwrite_indexes,
        'gen_pdb': os.path.abspath(out_path),
        'ref_pdb': os.path.abspath(summary.ref_pdb),
    }


def format_id(summary):
    if '|' in summary.id:
        summary.id = summary.id.split('|')[0].strip('.pdb')


def main(args):
    setup_seed(args.seed)
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    
    # 加载模型
    model = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    autoencoder = getattr(model, 'autoencoder', model)
    autoencoder.to(device).eval()
    
    # 加载数据集
    config = yaml.safe_load(open(args.config, 'r'))
    _, _, test_set = create_dataset(config['dataset'])
    print(f"Test set: {len(test_set)} samples")
    
    # 创建保存目录 (与 generate.py 一致)
    save_dir = args.save_dir
    ref_save_dir = os.path.join(save_dir, 'references')
    cand_save_dir = os.path.join(save_dir, 'candidates')
    os.makedirs(ref_save_dir, exist_ok=True)
    os.makedirs(cand_save_dir, exist_ok=True)
    
    # 测试
    all_logs = []
    max_samples = args.max_samples or len(test_set)
    
    for idx in tqdm(range(min(max_samples, len(test_set))), desc="RAE Reconstruction"):
        batch = to_device(test_set.collate_fn([test_set[idx]]), device)
        
        with torch.no_grad():
            # 纯 RAE: encode → decode (无 diffusion)
            batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = autoencoder.generate(
                **batch,
                n_iter=autoencoder.default_num_steps,
                fixseq=args.fixseq,
            )
        
        # 保存结果 (与 generate.py 一致)
        for S, X, A, ll, bonds, intra_bonds in zip(batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds):
            cplx = deepcopy(test_set.get_raw_data(idx))
            summary: Summary = deepcopy(test_set.get_summary(idx))
            format_id(summary)
            
            # 保存 reference
            summary.ref_pdb = os.path.join(ref_save_dir, summary.ref_pdb)
            os.makedirs(os.path.dirname(summary.ref_pdb), exist_ok=True)
            complex_to_pdb(cplx, summary.ref_pdb, summary.target_chain_ids + summary.ligand_chain_ids)
            
            # 保存 candidate
            os.makedirs(os.path.join(cand_save_dir, summary.id), exist_ok=True)
            save_path = os.path.join(cand_save_dir, summary.id, '0.pdb')
            
            log = overwrite(cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path)
            if log:
                all_logs.append(log)
    
    # 保存 results.jsonl (与 generate.py 一致，每行一个 JSON)
    results_path = os.path.join(save_dir, 'results.jsonl')
    with open(results_path, 'w') as f:
        for log in all_logs:
            log['n'] = 0
            log['struct_only'] = args.fixseq
            f.write(json.dumps(log) + '\n')
    
    # 保存 index.json (供 peptide_o_ray.py 使用)
    index_path = os.path.join(save_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump(all_logs, f, indent=2)

    
    print(f"\n保存完成: {len(all_logs)} 个样本")
    print(f"  - References: {ref_save_dir}")
    print(f"  - Candidates: {cand_save_dir}")
    print(f"  - Index: {index_path}")
    print(f"\n运行评估: python -m scripts.metrics.peptide_o_ray --results {save_dir}/results.jsonl --calc_dg --antibody --num_workers 64")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAE Reconstruction Test')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True, help='保存结果目录')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fixseq', action='store_true', help='固定序列，仅重构结构')
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    main(args)
