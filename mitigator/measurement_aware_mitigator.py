from collections import defaultdict

import qiskit.circuit.random
import scipy.io as sio
import random

from qiskit.circuit.library import XGate, IGate
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import numpy as np
import scipy.linalg as la
from qiskit.quantum_info import hellinger_fidelity
from qiskit.result import LocalReadoutMitigator


from qiskit import Aer, QuantumCircuit, execute
import matplotlib.pyplot as plt
import jax
from qiskit.result import Counts
from jax import numpy as jnp
from jax import vmap
# from jax import random
import random
import math
import time
from mitigator.partical_local_mitigator import ParticalLocalMitigator, kron_basis, list2int
from utils import all_bitstrings, to_bitstring
from correlation_analysis import PdBasedProtocolResults, correlation_based_partation, construct_bayesian_network
# from mitigator.multi_stage_mitigator import MultiStageMitigator
from functools import lru_cache
from typing import List, Optional, Tuple
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# '''基于MultiStageMitigator改来的，考虑了测量的影响'''
'''基于ParticalLocalMitigator改来的，考虑了测量的影响'''

class MeasurementAwareMitigator(ParticalLocalMitigator):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

        # , groups: List = None, bayesian_network_model: BayesianNetwork = None, bayesian_infer_model :VariableElimination = None
        # self.bayesian_network_model = bayesian_network_model 
        # self.bayesian_infer_model = bayesian_infer_model

        # if groups is not None:
        #     self.assign_groups(groups)

    # TODO: 放到ParticalLocalMitigator里面
    @staticmethod
    def assign_groups(groups, n_qubits):
        '''假设groups里面的比特不重叠'''
        qubit_map = []  # 存了mitigate里面对应的格式
        qubit_map_inv = [0] * n_qubits

        for group in groups:
            group.sort()
            qubit_map += group
            groups.append(tuple(group))

        # 返回原来的位置
        # [2, 0, 1] -> [1, 2, 0]
        for real_pos, old_pos in enumerate(qubit_map):
            qubit_map_inv[old_pos] = real_pos
        
        return qubit_map, qubit_map_inv
            
    def characterize_M(self, protocol_results, groups: List[List[int]]):
        protocol_results = PdBasedProtocolResults(protocol_results, self.n_qubits)
        self.bayesian_network_model, self.bayesian_infer_model = construct_bayesian_network(protocol_results)
        self.groups = groups
        
        # self.assign_groups(groups)
        
    
    @lru_cache
    @staticmethod
    def unmeasure_index(group_size):
        return enumerate([index for index, bitstring in enumerate(all_bitstrings(group_size, base = 3)) if '2' not in bitstring])
    
    @lru_cache
    def get_partical_local_mitigator(self, measured_qubits) -> ParticalLocalMitigator:
        # bayesian_network_model: BayesianNetwork  = self.bayesian_network_model
        bayesian_infer_model: VariableElimination =  self.bayesian_infer_model
        
        groups = [
            tuple([qubit for qubit in group if qubit in measured_qubits])
            for group in self.groups
        ]

        # qubit_map = measured_qubits
        # qubit_map_inv = [0] * len(measured_qubits)
        # for real_pos, old_pos in enumerate(qubit_map):
        #     qubit_map_inv[old_pos] = real_pos
            
        group2M = {}
        # group2M_inv = {}
        for group in groups:
            group_size = len(group)
            M = np.zeros(shape = (2**group_size, 2**group_size))
            
            for bitstring in all_bitstrings(group_size):
                posterior_p = bayesian_infer_model.query([f'{qubit}_read' for qubit in group], evidence={
                    f'{qubit}_set': int(bit)
                    for qubit, bit in enumerate(bitstring)
                })
                posterior_v = posterior_p.values.reshape(3**group_size) #变成了M中列的数据
                
                # 剃掉包含2的
                posterior_v = posterior_v[self.unmeasure_index(group_size)]
                
                assert abs(sum(posterior_v) - 1) < 1e-2, sum(posterior_v)  # 后面可以删了
                
                M[:,int(bitstring, base=2)] =  posterior_v
                # print(posterior_v)
            
            group = []
            group2M[group] = M
            # group2M_inv[group] = np.linalg.inv(M)
        
        
        
        return ParticalLocalMitigator(self.n_qubits, group2M)
            
    
    def mitigate(self, stats_counts: dict, circuit: QuantumCircuit, threshold: float = None):
        '''假设group之间没有重叠的qubit'''
        n_qubits = self.n_qubits

        measured_qubits = tuple([
            instruction.qubits[0].index
            for instruction in circuit
            if instruction.operation.name == 'measure'
        ])
        
        plm = self.get_partical_local_mitigator(measured_qubits)
        
        return plm.mitigate(stats_counts)
        
        
        
        
        
        
        