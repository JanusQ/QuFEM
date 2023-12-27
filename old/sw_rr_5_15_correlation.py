# import
from collections import defaultdict
import itertools
import scipy.io as sio
import random
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import numpy as np
import scipy.linalg as la
from qiskit.result import LocalReadoutMitigator
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import jax
from qiskit.result import Counts
from jax import numpy as jnp
from jax import vmap
from qiskit_aer import AerSimulator
# from jax import random
from qiskit_aer.noise import NoiseModel, ReadoutError
import random
import math
import time

def ghz(n_qubits):
    cir = QuantumCircuit(n_qubits)
    cir.h(0)
    for i in range(n_qubits - 1):
        cir.cx(i, i + 1)
    cir.measure_all()
    return cir


n_qubits = 3
measure_fids = np.ones(shape=(n_qubits, 2)) * 0.99
noise_model = NoiseModel()

for i in range(n_qubits):
    re = ReadoutError([[measure_fids[i][0], 1 - measure_fids[i][0]],
                      [1 - measure_fids[i][1], measure_fids[i][1]]])
    noise_model.add_readout_error(re, qubits=[i])
simulator = AerSimulator(noise_model=noise_model)

# 拿到错误的模拟值
n_samples = 3000
before_rm_counts = simulator.run(
    ghz(n_qubits), shots=n_samples).result().get_counts()
before_rm_prob = {k: v / n_samples for k, v in before_rm_counts.items()}
plot_histogram(before_rm_prob, title = f'before_rm_{n_qubits}', filename=f'before_rm_{n_qubits}')

'''生成矫正的矩阵, meas_mats是读取噪声的form_ulation, meas_mats_inv是矫正矩阵'''
meas_mats, meas_mats_inv = [], []
for qubit in range(n_qubits):
    meas_mat = np.array([[
        measure_fids[qubit][0], 1-measure_fids[qubit][1]],
        [1-measure_fids[qubit][0], measure_fids[qubit][1]]
    ])
    meas_mats.append(meas_mat)
    meas_mats_inv.append(np.linalg.inv(meas_mat))


if n_qubits < 10:
    '''qiskit的矫正'''
    mit = LocalReadoutMitigator(meas_mats, list(range(n_qubits)))
    qiskit_rm_prob = mit.quasi_probabilities(before_rm_counts)
    qiskit_rm_prob = {
        basis: value
        for basis, value in qiskit_rm_prob.items()
        if value > 1e-3
    }
    plot_histogram(qiskit_rm_prob, title = f'qiskit_rm_{n_qubits}', filename=f'qiskit_rm_{n_qubits}')

    '''用纯数学的方法试一下'''
    before_rm_prob_vec = np.zeros(2**n_qubits)
    for basis, prob in before_rm_prob.items():
        before_rm_prob_vec[int(basis, base=2)] = prob

    tensor_meas_mat_inv = np.linalg.inv(meas_mats[0])
    for qubit in range(1, n_qubits):
        tensor_meas_mat_inv = np.kron(
            tensor_meas_mat_inv, np.linalg.inv(meas_mats[qubit]))

    rm_prob = {}
    rm_prob_vec = tensor_meas_mat_inv @ before_rm_prob_vec
    for basis, prob in enumerate(rm_prob_vec):
        rm_prob[bin(basis).replace('0b', '')] = prob
    rm_prob = {
        basis: value
        for basis, value in rm_prob.items()
        if value > 1e-3
    }
    plot_histogram(rm_prob, title = f'math_rm_{n_qubits}', filename=f'math_rm_{n_qubits}')

'''基于纯数学的形式改成了稀疏计算'''


def list2int(arrs):
    init = arrs[0]
    for i in range(1, len(arrs)):
        init = init << 1 | arrs[i]
    return init

# 看下有没有更快的方法
# def kron_basis(*arrs: list):
#     arrs = reversed(arrs)
#     grid = np.meshgrid(*arrs)
#     return list2int([e.ravel() for e in grid ])

def kron_basis(arr1, arr2, offest):
    grid = np.meshgrid(arr2, arr1)
    return grid[1].ravel() << offest | grid[0].ravel()
    # return grid[1].ravel() << offest | grid[0].ravel()

# def reduce(arr, initial_value, func):
#     '''func(now_value, elm, )'''
#     init = func
#     return 

def sw_rm_corr(stats_counts: dict, group2meas_mats_inv: dict, threshold=None):
    '''每个group里面不能有重复的qubit'''
    n_qubits = len(list(stats_counts.keys())[0])
    
    if threshold is None:
        sum_count = sum(stats_counts.values())
        threshold = sum_count * 1e-5

    rm_prob = defaultdict(float)

    pqubit2lqubits = defaultdict(list)  # 现在一个会对应张成后的多个了
    n_lqubits = 0
    for group in group2meas_mats_inv:
        for i, qubit in enumerate(group):
            pqubit2lqubits[qubit].append(n_lqubits + i) 
        n_lqubits += len(group)
    
    total_group_size = 0
    for group in group2meas_mats_inv:
        total_group_size += len(group)
    
    
    lbasis2pbasis = {}
    for basis in itertools.product([0, 1], repeat = n_qubits):
        pbasis = list2int(basis)
        
        lbasis = np.zeros(n_lqubits, dtype=np.int64)
        for pqubit, lqubits in pqubit2lqubits.items():
            lbasis[lqubits] = basis[pqubit]
        lbasis = list2int(lbasis)
        
        lbasis2pbasis[lbasis] = pbasis

    
    
    for basis, count in stats_counts.items():
        basis = [int(c) for c in basis]
        
        now_basis = None  #basis_1q[basis[0]]
        now_values = None
        
        finished_group_size = 0
        for group, meas_mats_inv in group2meas_mats_inv.items():
            group_size = len(group)
            finished_group_size += group_size
            group_basis = [basis[qubit] for qubit in group]
            
            group_mitigated_vec = meas_mats_inv[:,list2int(group_basis)]
            group_basis = np.arange(2**group_size)
            
            # TODO: 这里就可以logical qubit 和physical qubit不对应的剃掉了
            
            if now_basis is None:
                next_basis = group_basis
                next_values = group_mitigated_vec * count
            else:
                next_basis = kron_basis(now_basis, group_basis, group_size)
                next_values = np.kron(now_values, group_mitigated_vec)

                filter_basis = np.array(list(lbasis2pbasis.keys()), dtype=np.int64) >> (total_group_size - finished_group_size)
                filter = []
                for basis in next_basis:
                    filter.append(np.any(filter_basis == basis))
                next_basis = next_basis[filter]
                next_values = next_values[filter]
            
            
            # filter = np.logical_or(next_values > threshold, next_values < -threshold)
            # now_basis = next_basis[filter]
            # now_values = next_values[filter]

            now_basis = next_basis
            now_values = next_values

        for basis, value in zip(now_basis, now_values):
            if basis in lbasis2pbasis:
                rm_prob[lbasis2pbasis[basis]] += value
            # rm_prob += value

    sum_prob = sum(rm_prob.values())
    rm_prob = {
        basis: value / sum_prob
        for basis, value in rm_prob.items()
    }
    return rm_prob

# TODO: 多次两两校准最后逼近，线性的复杂度

group2meas_mats_inv = {
    # (0, 1, 2): np.kron(np.kron(meas_mats_inv[0], meas_mats_inv[1]),meas_mats_inv[2]),
    (0, 1): np.kron(meas_mats_inv[0], meas_mats_inv[1]),
    (0, 2): np.kron(meas_mats_inv[0], meas_mats_inv[2]),
    # (1, 2): np.kron(meas_mats_inv[1], meas_mats_inv[2]),
    # (0,): meas_mats_inv[0],
    # (1,): meas_mats_inv[1],
    # (2,): meas_mats_inv[2],
}

start_time = time.time()
rm_prob = sw_rm_corr(before_rm_counts, group2meas_mats_inv, threshold=n_samples * 1e-5)

print('sw_rr cost', time.time() - start_time, 's')
rm_prob = {
    basis: value
    for basis, value in rm_prob.items()
    if value > 1e-3
}
# rm_prob = {
#     '1' * n_qubits: rm_prob['1' * n_qubits],
#     '0' * n_qubits: rm_prob['0' * n_qubits]
# }
plot_histogram(rm_prob, title = f'sw_rm_{n_qubits}', filename=f'sw_rm_{n_qubits}')
