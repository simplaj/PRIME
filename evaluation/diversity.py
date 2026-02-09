#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats.contingency import association

from .seq import align_sequences
from .rmsd import compute_rmsd
from utils.logger import print_log

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import multiprocessing as mp


def _align_pair(args):
    """Worker 函数，计算单对序列的距离"""
    i, j, seq1, seq2 = args
    _, sim = align_sequences(seq1, seq2, symmetric=True)
    return i, j, 1 - sim


def seq_diversity(seqs: List[str], th: float=0.6, n_workers: int=None) -> float:
    '''
        th: sequence distance (1 - sequence identity)
        n_workers: 并行进程数，默认为 CPU 核心数
        
    使用对称性 + 多进程并行加速 (约 4-8x)
    '''
    N = len(seqs)
    if N <= 1:
        return 1.0, np.array([1])
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # 只计算上三角 (i < j)，利用对称性
    pairs = [(i, j, seqs[i], seqs[j]) for i, j in combinations(range(N), 2)]
    
    dists = np.zeros((N, N))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(_align_pair, pairs, chunksize=max(1, len(pairs) // (n_workers * 4)))
        
        for i, j, dist in results:
            dists[i][j] = dist
            dists[j][i] = dist  # 对称填充
    
    Z = linkage(squareform(dists), 'single')
    cluster = fcluster(Z, t=th, criterion='distance')
    return len(np.unique(cluster)) / len(seqs), cluster


# def seq_diversity(seqs: List[str], th: float=0.6) -> float:
#     '''
#         th: sequence distance (1 - sequence identity)
#     '''
#     dists = []
#     for i, seq1 in enumerate(seqs):
#         dists.append([])
#         for j, seq2 in enumerate(seqs):
#             _, sim = align_sequences(seq1, seq2, symmetric=True)
#             dists[i].append(1 - sim)
#     dists = np.array(dists)
#     Z = linkage(squareform(dists), 'single')
#     cluster = fcluster(Z, t=th, criterion='distance')
#     return len(np.unique(cluster)) / len(seqs), cluster


def struct_diversity(structs: np.ndarray, th: float=2.0) -> float:
    '''
    structs: N*L*3, alpha carbon coordinates
    th: threshold for clustering (distance < th)
    '''
    ca_dists = np.sum((structs[:, None] - structs[None, :]) ** 2, axis=-1) # [N, N, L]
    rmsd = np.sqrt(np.mean(ca_dists, axis=-1))
    Z = linkage(squareform(rmsd), 'single') # since the distances might not be euclidean distances (e.g. rmsd)
    cluster = fcluster(Z, t=th, criterion='distance')
    return len(np.unique(cluster)) / structs.shape[0], cluster


def diversity(seqs: List[str], structs: np.ndarray):
    print_log(f'start seq div')
    seq_div, seq_clu = seq_diversity(seqs)
    print_log(f'end seq div')
    if structs is None:
        return seq_div, None, seq_div, None
    print_log('struct div')
    struct_div, struct_clu = struct_diversity(structs)
    print_log('end struct div')

    print_log('start clu')
    co_div = np.sqrt(seq_div * struct_div)

    n_seq_clu, n_struct_clu = np.max(seq_clu), np.max(struct_clu) # clusters start from 1
    if n_seq_clu == 1 or n_struct_clu == 1:
        consistency = 1.0 if n_seq_clu == n_struct_clu else 0.0
    else:
        table = [[0 for _ in range(n_struct_clu)] for _ in range(n_seq_clu)]
        for seq_c, struct_c in zip(seq_clu, struct_clu):
            table[seq_c - 1][struct_c - 1] += 1
        consistency = association(np.array(table), method='cramer')
    print_log('end cluster')

    return seq_div, struct_div, co_div, consistency


if __name__ == '__main__':
    N, L = 100, 10
    a = np.random.randn(N, L, 3)
    print(struct_diversity(a))
    from data.bioparse.const import aas
    aas = [tup[0] for tup in aas]
    seqs = np.random.randint(0, len(aas), (N, L))
    seqs = [''.join([aas[i] for i in idx]) for idx in seqs]
    print(seq_diversity(seqs, 0.4))