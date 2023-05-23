# import
from collections import defaultdict
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
from typing import List, Optional, Tuple, Dict

from utils import to_bitstring

'''
    基的张成
    如：   
'''
def kron_basis(arr1, arr2):
    grid = np.meshgrid(arr2, arr1)
    return grid[1].ravel() << 1 | grid[0].ravel()

class LocalMitigator():
    def __init__(self, n_qubits, M_per_qubits = None):
        self.n_qubits = n_qubits
        self.M_per_qubits = M_per_qubits
        
    def characterize_M(self, protocol_results: dict):
        '''
            protocol_results: {
                '理论的字符串': '实际的字符串分布'
            }
        '''
        n_qubits = self.n_qubits
        

        measure_success = np.zeros(shape=(n_qubits, 2))
        n_measures = np.zeros(shape=(n_qubits, 2))

        for real_bitstring, status_count in protocol_results.items():
            
            '''TODO: 实际上是或多或少可以用上的'''
            if '2' in real_bitstring: continue
            
            real_bitstring = [int(bit) for bit in real_bitstring]
            for measure_bitstring, count in status_count.items():
                for qubit, value in enumerate(measure_bitstring):
                    real_value = real_bitstring[qubit]
                    value = int(value)
                    if value == real_value:
                        measure_success[qubit][real_value] += count
                    n_measures[qubit][real_value] += count
        
        measure_fids = measure_success/n_measures
        
        self.M_per_qubits = np.array([
            np.array([
                # [measure_fids[qubit][0],    1-measure_fids[qubit][0]],
                # [1-measure_fids[qubit][1],  measure_fids[qubit][1]]
                [measure_fids[qubit][0],    1-measure_fids[qubit][1]],
                [1-measure_fids[qubit][0],  measure_fids[qubit][1]]
            ])
            for qubit in range(n_qubits)
        ])
        
        self.M_per_qubits_inverse = np.array([
            np.linalg.inv(self.M_per_qubits[qubit])
            for qubit in range(n_qubits)
        ])
            
        return self.M_per_qubits
        
        
    def mitigate(self, stats_counts: dict, threshold=None):
        meas_mats_inv = self.M_per_qubits_inverse
        
        n_qubits = len(list(stats_counts.keys())[0])
        
        if threshold is None:
            sum_count = sum(stats_counts.values())
            threshold = sum_count * 1e-15

        basis_1q = np.array([[1, 0], [0, 1]])  # 向量表示
        all_local_vecs = np.zeros(shape=(n_qubits, 2, 2))
        for qubit in range(n_qubits):
            all_local_vecs[qubit][0] = meas_mats_inv[qubit] @  basis_1q[0]
            all_local_vecs[qubit][1] = meas_mats_inv[qubit] @  basis_1q[1]

        rm_prob = defaultdict(float)
        basis_1q = np.array([0, 1]) # 2进制表示
        # for basis, count in tqdm(stats_counts.items()):
        for basis, count in stats_counts.items():
            basis = [int(c) for c in basis]

            now_basis = basis_1q  #basis_1q[basis[0]]
            now_values = all_local_vecs[0][basis[0]] * count

            for qubit in range(1, n_qubits):
                next_basis = kron_basis(now_basis, basis_1q) #basis_1q[basis[qubit]])
                next_values = np.kron(now_values, all_local_vecs[qubit][basis[qubit]])
                
                # filter = np.logical_or(next_values > threshold, next_values < -threshold)
                # now_basis = next_basis[filter]
                # now_values = next_values[filter]
                now_basis = next_basis
                now_values = next_values

            for basis, value in zip(now_basis, now_values):
                rm_prob[basis] += value

        rm_prob = {
            basis: value
            for basis, value in rm_prob.items()
            if value > 0
        }
        
        sum_prob = sum(rm_prob.values())
        return {
            to_bitstring(basis, self.n_qubits): value / sum_prob
            for basis, value in rm_prob.items()
        }

    # start_time = time.time()
    # rm_prob = sw_rm(before_rm_counts, meas_mats_inv, threshold=n_samples * 1e-5)

    # print('sw_rr cost', time.time() - start_time, 's')
    # rm_prob = {
    #     basis: value
    #     for basis, value in rm_prob.items()
    #     if value > 1e-3
    # }
    # # rm_prob = {
    # #     '1' * n_qubits: rm_prob['1' * n_qubits],
    # #     '0' * n_qubits: rm_prob['0' * n_qubits]
    # # }
    # plot_histogram(rm_prob, title = f'sw_rm_{n_qubits}', filename=f'sw_rm_{n_qubits}')
