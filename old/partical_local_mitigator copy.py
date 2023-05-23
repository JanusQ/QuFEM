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
            groups.append(now_group)
            
        return groups
        
    def characterize_M(self, protocol_results, groups: List[List[int]]):
        '''假设groups里面的比特不重叠'''
        self.group2M = {}
        self.group2invM = {}
        for group in groups:
            group_size = len(group)
            group.sort()
            
            M = np.zeros(shape=(2**group_size, 2**group_size))
            
            for real_bitstring, status_count in protocol_results.items():
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
            
    def mitigate(self, stats_counts: dict, threshold: float = None):
        '''假设group之间没有重叠的qubit'''
        n_qubits = self.n_qubits
        
        # stats_counts = dict(stats_counts)
        group2invM = self.group2invM
        
        
        if threshold is None:
            sum_count = sum(stats_counts.values())
            threshold = sum_count * 1e-15

        rm_prob = defaultdict(float)
        for basis, count in stats_counts.items():
            basis = [int(c) for c in basis]
            
            now_basis = None  #basis_1q[basis[0]]
            now_values = None
            
            for group, invM in group2invM.items():
                group_size = len(group)
                group_basis = [basis[qubit] for qubit in group]
                
                group_mitigated_vec = invM[:,list2int(group_basis)]
                group_basis = np.arange(2**group_size)
                
                if now_basis is None:
                    next_basis = group_basis
                    next_values = group_mitigated_vec * count
                else:
                    next_basis = kron_basis(now_basis, group_basis, group_size)
                    next_values = np.kron(now_values, group_mitigated_vec)
                
                # filter = np.logical_or(next_values > threshold, next_values < -threshold)
                # now_basis = next_basis[filter]
                # now_values = next_values[filter]

                now_basis = next_basis
                now_values = next_values

            for basis, value in zip(now_basis, now_values):
                rm_prob[basis] += value  # 这里的basis是按照group的顺序的
        
        # 实际的qubit->按照group之后的顺序  
        qubit_map = defaultdict(list)  # 现在一个会对应张成后的多个了
        pointer = 0
        for group in group2invM:
            for i, qubit in enumerate(group):
                qubit_map[qubit] = pointer + i 
            pointer += len(group)
            
        inv_qubit_map = {
            l_qubit: qubit
            for qubit, l_qubit in qubit_map.items()
        }
        
        new_rm_prob = {}
        for basis, value in rm_prob.items():
            lbasis = 0
            for qubit in range(n_qubits):
                lbasis |= (basis & 1) << inv_qubit_map[qubit]
                basis = basis >> 1

            new_rm_prob[lbasis] = value
        rm_prob = new_rm_prob

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
        return rm_prob
        

