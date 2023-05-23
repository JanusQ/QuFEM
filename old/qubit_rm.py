
import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from tqdm import tqdm

from jax import numpy as jnp
import jax
import time
from functools import lru_cache
from line_profiler import LineProfiler

qnum = 24
stats01s = sio.loadmat(f'{qnum}qubit_data/stats_0.mat')['stats01s']
measure_fids = sio.loadmat(
    f'{qnum}qubit_data/measure_fids.mat')['measure_fids']


'''开始的部分'''
meas_mats, meas_mats_inv = [], []
for qubit in range(qnum):
    meas_mat = np.array([[
        measure_fids[qubit][0], 1-measure_fids[qubit][1]],
        [1-measure_fids[qubit][0], measure_fids[qubit][1]]
    ])
    meas_mats.append(meas_mat)
    meas_mats_inv.append(np.linalg.inv(meas_mat))

i = 5

dim0 = stats01s.shape
qnum = dim0[2]
stats_num = dim0[0]
stats_counts = defaultdict(int)

for j in range(stats_num):
    stat_str = ''.join([str(ii) for ii in stats01s[j, i, :]])
    stats_counts[stat_str] += 1

stats_probs = {
    basis: count / stats_num
    for basis, count in stats_counts.items()
}


# def sw_rm(qnum, stats_counts, meas_mats_inv, threshold=1e-10):
#     all_local_vecs = np.zeros(shape=(qnum, 2, 2))
#     for qubit in range(qnum):
#         for local_basis in (0, 1):
#             local_vec = np.zeros(2)
#             local_vec[local_basis] = 1
#             local_vec = meas_mats_inv[qubit] @ local_vec
#             all_local_vecs[qubit][local_basis] = local_vec
#     all_local_vecs = jnp.array(all_local_vecs)
    
#     @jax.jit
#     def transform(basis, all_local_vecs): 
#         now_basis_values = [
#             [local_basis, local_value*count]
#             for local_basis, local_value in enumerate(all_local_vecs[0][basis[0]])
#         ]

#         for qubit in range(1, qnum):
#             local_vec = all_local_vecs[qubit][basis[qubit]]
#             next_basis_values = []
            
#             for local_basis in (0, 1):
#                 for now_basis, value in now_basis_values:
#                     next_value = value * local_vec[local_basis]

#                     # if next_value < threshold and next_value > -threshold:
#                     #     continue

#                     next_basis_values.append(
#                         [now_basis << 1 | local_basis, next_value])

#             now_basis_values = next_basis_values

#         return now_basis_values

#     rm_prob = defaultdict(float)
#     for basis, count in tqdm(stats_counts.items()):
#         basis = np.array([int(c) for c in basis])
#         now_basis_values = transform(basis, all_local_vecs)
#         for now_basis, now_value in now_basis_values:
#             rm_prob[now_basis] += now_value

#     sum_prob = sum(rm_prob.values())
#     rm_prob = {
#         basis: value / sum_prob
#         for basis, value in rm_prob.items()
#     }
#     return rm_prob

def sw_rm(qnum, stats_counts, meas_mats_inv, threshold=1e-10):
    all_local_vecs = np.zeros(shape=(qnum, 2, 2))

    for qubit in range(qnum):
        for local_basis in (0, 1):
            local_vec = np.zeros(2)
            local_vec[local_basis] = 1
            local_vec = meas_mats_inv[qubit] @ local_vec
            all_local_vecs[qubit][local_basis] = local_vec

    rm_prob = defaultdict(float)
    for basis, count in tqdm(stats_counts.items()):
        basis = [int(c) for c in basis]

        now_basis_values = [
            [local_basis, local_value*count]
            for local_basis, local_value in enumerate(all_local_vecs[0][basis[0]])
        ]

        for qubit in range(1, qnum):
            next_basis_values = []
            for local_basis, local_value in enumerate(all_local_vecs[qubit][basis[qubit]]):
                for now_basis, value in now_basis_values:
                    next_value = value * local_value

                    if next_value < threshold and next_value > -threshold:
                        continue

                    next_basis_values.append(
                        [now_basis << 1 | local_basis, next_value])

            now_basis_values = next_basis_values

        for basis, value in now_basis_values:
            rm_prob[basis] += value

    sum_prob = sum(rm_prob.values())
    rm_prob = {
        basis: value / sum_prob
        for basis, value in rm_prob.items()
    }
    return rm_prob

# rm_prob = sw_rm(qnum, stats_counts, meas_mats_inv, threshold=stats_num * 1e-8)

lp = LineProfiler()
lp_wrap = lp(sw_rm)
lp_wrap(qnum, stats_counts, meas_mats_inv, threshold = stats_num * 1e-5)
lp.print_stats()


# 77  41841702 6693525000.0    160.0     14.0                  for basis, value in now_basis_values:
# 78  41841702 5773210000.0    138.0     12.1                      next_value = value * local_value
# 79
# 80  37948702 6391452000.0    168.4     13.4                      if next_value < threshold and next_value > -threshold:
# 81   3893000  337651000.0     86.7      0.7                          continue
# 82
# 83  37948702 5692320000.0    150.0     11.9                      basis = basis << 1 | local_basis
# 84  37948702 16384346000.0    431.7     34.3                      next_basis_values.append([basis, next_value])
# 85
# 86      5805  522952000.0  90086.5      1.1              now_basis_values = next_basis_values
# 87
# 88  17028625 2160551000.0    126.9      4.5          for basis, value in now_basis_values:
# 89  17028625 3266397000.0    191.8      6.8              rm_prob[basis] += value
# 90
