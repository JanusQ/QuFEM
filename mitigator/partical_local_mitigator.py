'''两两交替的'''

from collections import defaultdict

import qiskit.circuit.random
import scipy.io as sio
import random
import itertools
from qiskit.circuit.library import XGate, IGate
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import numpy as np
import scipy.linalg as la
from qiskit.quantum_info import hellinger_fidelity
from qiskit.result import LocalReadoutMitigator
from qiskit.visualization import plot_histogram
from qiskit import Aer, QuantumCircuit, execute
import matplotlib.pyplot as plt
import jax
from qiskit.result import Counts
from jax import numpy as jnp
from jax import vmap
from qiskit_aer import AerSimulator
# from jax import random
from qiskit_aer.noise import NoiseModel, ReadoutError, QuantumError, pauli_error, depolarizing_error
import random
import math
import time
from typing import List, Optional, Tuple
from utils import to_bitstring

# 改成支持只测量一部分的

def kron_basis(arr1, arr2, offest):
    grid = np.meshgrid(arr2, arr1)
    return grid[1].ravel() << offest | grid[0].ravel()

        
def list2int(arrs):
    init = arrs[0]
    for i in range(1, len(arrs)):
        init = init << 1 | arrs[i]
    return init

class ParticalLocalMitigator():
    def __init__(self, n_qubits, group2M = None):
        self.n_qubits = n_qubits
        self.group2M = group2M
        
        if group2M is not None:
            self.group2invM = {
                group: np.linalg.inv(M)
                for group, M in group2M.items()
            }
            
            self.groups = []
            self.inner_qubit_map = [] 
            for group in group2M:
                self.inner_qubit_map += list(group)  # 假设已经sort过了
                self.groups.append(group)
                
            self.inner_qubit_map_inv = [0] * self.n_qubits
            for real_pos, old_pos in enumerate(self.inner_qubit_map):
                self.inner_qubit_map_inv[old_pos] = real_pos
            
    def random_group(self, group_size): 
        qubits = list(range(self.n_qubits))
        
        groups = []
        while len(qubits) != 0:
            now_group = []
            for _ in range(group_size):
                now_group.append(random.choice(qubits))
                qubits = [
                    qubit
                    for qubit in qubits
                    if qubit != now_group[len(now_group)-1]
                ]
                if len(qubits) == 0:
                    break
            now_group.sort()
            groups.append(now_group)
            
        return groups
        
    def characterize_M(self, protocol_results, groups: List[List[int]]):
        '''假设groups里面的比特不重叠'''
        self.group2M = {}
        self.group2invM = {}
        
        self.inner_qubit_map = []  # 存了mitigate里面对应的格式
        self.groups = []

        for group in groups:
            group_size = len(group)
            group.sort()
            
            self.inner_qubit_map += group
            self.groups.append(tuple(group))
            
            M = np.zeros(shape=(2**group_size, 2**group_size))
            
            for real_bitstring, status_count in protocol_results.items():
                if '2' in real_bitstring: continue
                
                real_bitstring = [
                    real_bitstring[qubit]
                    for qubit in group
                ]
                real_bitstring = int(''.join(real_bitstring), base=2)

                for measure_bitstring, count in status_count.items():                    
                    measure_bitstring = [
                        measure_bitstring[qubit]
                        for qubit in group
                    ]
                    measure_bitstring = int(''.join(measure_bitstring), base=2)
                    M[measure_bitstring][real_bitstring] += count
            
            for column_index in range(2**group_size):
                M[:,column_index] /= np.sum(M[:,column_index])
            
            self.group2M[tuple(group)] = M
            self.group2invM[tuple(group)] = np.linalg.inv(M)
        
        
        # 返回原来的位置
        # [2, 0, 1] -> [1, 2, 0]
        self.inner_qubit_map_inv = [0] * self.n_qubits
        for real_pos, old_pos in enumerate(self.inner_qubit_map):
            self.inner_qubit_map_inv[old_pos] = real_pos
            
        
    
    @staticmethod
    def permute(stats_counts, qubit_order: list):
        permuted_stat_counts = {}
        for bitstring, count in stats_counts.items():
            new_bitstring = ['0'] * len(bitstring)
            for now_pos, old_pos in enumerate(qubit_order):
                new_bitstring[now_pos] = bitstring[old_pos]
            permuted_stat_counts[''.join(new_bitstring)] = count
        return permuted_stat_counts
    
            
    # , group2invM = None
    def mitigate(self, stats_counts: dict, threshold: float = None, circuit: QuantumCircuit = None, mask_bitstring = None):
        # if circuit is not None:
        #     measured_qubits = tuple([
        #         instruction.qubits[0].index
        #         for instruction in circuit
        #         if instruction.operation.name == 'measure'
        #     ])
        # elif mask_bitstring is not None:
        #     measured_qubits = tuple([
        #         qubit
        #         for qubit in range(n_qubits)
        #         if mask_bitstring[qubit] != '2'
        #     ])
        # else:
        #     measured_qubits = tuple(range(n_qubits))
        # if group_basis[0] == 2:  # 对应的就是没有测量的
        #     assert len(group_basis) == 1
        #     group_mitigated_vec = np.array([1])
        #     group_basis = np.arange(2**group_size)
        # else:
        
        '''假设group之间没有重叠的qubit'''
        n_qubits = self.n_qubits
        
        stats_counts = self.permute(stats_counts, self.inner_qubit_map)
        
        # stats_counts = dict(stats_counts)
        # if group2invM is None:
        group2invM = self.group2invM
        groups = self.groups
        
        if threshold is None:
            sum_count = sum(stats_counts.values())
            threshold = sum_count * 1e-15

        rm_prob = defaultdict(float)
        for basis, count in stats_counts.items():
            basis = [int(c) for c in basis]

            now_basis = None  #basis_1q[basis[0]]
            now_values = None
            
            pointer = 0
            for group in groups:
                invM = group2invM[group]
                group_size = len(group)
                
                # group_basis = [basis[qubit] for qubit in group]
                group_basis = basis[pointer: pointer + group_size]

                pointer += group_size
                

                group_mitigated_vec = invM[:,list2int(group_basis)]
                group_basis = np.arange(2**group_size)
                    
                if now_basis is None:
                    next_basis = group_basis
                    next_values = group_mitigated_vec * count
                else:
                    next_basis = kron_basis(now_basis, group_basis, group_size)
                    next_values = np.kron(now_values, group_mitigated_vec)
                
                # TODO: 还没有在这份代码里面测试过
                # filter = np.logical_or(next_values > threshold, next_values < -threshold)
                # now_basis = next_basis[filter]
                # now_values = next_values[filter]

                now_basis = next_basis
                now_values = next_values

            for basis, value in zip(now_basis, now_values):
                rm_prob[basis] += value  # 这里的basis是按照group的顺序的
        
        rm_prob = {
            basis: value
            for basis, value in rm_prob.items()
            if value > 0
        }
        sum_prob = sum(rm_prob.values())
        rm_prob = {
            basis: value / sum_prob
            for basis, value in rm_prob.items()
        }
        
        rm_prob = {
            to_bitstring(bitstring, self.n_qubits): prob
            for bitstring, prob in rm_prob.items()
        }
        rm_prob = self.permute(rm_prob, self.inner_qubit_map_inv)

        
        
        # 实际的qubit->按照group之后的顺序
        # qubit_map = defaultdict(list)  # 现在一个会对应张成后的多个了
        # pointer = 0
        # for group in groups:
        #     for i, qubit in enumerate(group):
        #         qubit_map[qubit] = pointer + i 
        #     pointer += len(group)
            
        # inv_qubit_map = {
        #     l_qubit: qubit
        #     for qubit, l_qubit in qubit_map.items()
        # }
        
        # new_rm_prob = {}
        # for basis, value in rm_prob.items():
        #     lbasis = 0
        #     for qubit in range(n_qubits):
        #         lbasis |= (basis & 1) << inv_qubit_map[qubit]
        #         basis = basis >> 1

        #     new_rm_prob[lbasis] = value
        # rm_prob = new_rm_prob

        return rm_prob
        

