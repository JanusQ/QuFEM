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
from utils import all_bitstrings, to_bitstring, downsample
from correlation_analysis import PdBasedProtocolResults, correlation_based_partation, construct_bayesian_network
# from mitigator.multi_stage_mitigator import MultiStageMitigator
from functools import lru_cache
from typing import List, Optional, Tuple
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# '''基于MultiStageMitigator改来的，考虑了测量的影响'''
'''基于ParticalLocalMitigator改来的，考虑了测量的影响'''

class BayesianMitigator(ParticalLocalMitigator):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
    
    def characterize_M(self, protocol_results, groups: List[List[int]]):
        protocol_results = PdBasedProtocolResults(protocol_results, self.n_qubits)
        self.bayesian_network_model, self.bayesian_infer_model = construct_bayesian_network(protocol_results, self.n_qubits, groups)
        self.groups = [
            sorted(group)
            for group in groups
        ]

    @lru_cache
    def unmeasureed_index(self, group_size):
        return np.array([index for index, bitstring in enumerate(all_bitstrings(group_size, base = 3)) if '2' not in bitstring])
    
    # 这里的映射很乱
    @lru_cache
    def get_partical_local_mitigator(self, measured_qubits: List) -> ParticalLocalMitigator:
        n_qubits = self.n_qubits
        n_measured_qubits = len(measured_qubits)

        bayesian_infer_model: VariableElimination =  self.bayesian_infer_model

        group2M = {}
        for group in self.groups:
            group_measured_qubits = [qubit for qubit in group if qubit in measured_qubits]
            n_group_measured_qubits = len(group_measured_qubits)
            if n_group_measured_qubits == 0:
                continue
            M = np.zeros(shape = (2**n_group_measured_qubits, 2**n_group_measured_qubits))
            
            for bitstring in all_bitstrings(n_group_measured_qubits):
                posterior_p = bayesian_infer_model.query([f'{qubit}_read' for qubit in group_measured_qubits],   # if qubit in measured_qubits
                    evidence={
                        f'{qubit}_set': int(bitstring[group_measured_qubits.index(qubit)]) if qubit in measured_qubits else 2
                        for qubit in group
                    }
                )
                posterior_v = posterior_p.values.reshape(3**n_group_measured_qubits) # 变成了M中列的数据
                posterior_v = posterior_v[self.unmeasureed_index(n_group_measured_qubits)] # 剃掉包含2的
                
                assert abs(sum(posterior_v) - 1) < 1e-2, sum(posterior_v)  # 后面可以删了
                
                M[:,int(bitstring, base=2)] =  posterior_v

            remap_group = tuple([measured_qubits.index(qubit) for qubit in group_measured_qubits])
            group2M[remap_group] = M

            # print('Bayesian:')
            # print(np.round(M, 3))
        
        return ParticalLocalMitigator(n_measured_qubits, group2M)
            
    def mitigate(self, stats_counts: dict, circuit: QuantumCircuit = None, threshold: float = None, mask_bitstring: str = None):
        '''假设group之间没有重叠的qubit'''
        n_qubits = self.n_qubits
        
        if circuit is not None:
            measured_qubits =[
                instruction.qubits[0].index
                for instruction in circuit
                if instruction.operation.name == 'measure'
            ]
            measured_qubits.sort()
            measured_qubits = tuple(measured_qubits)
        elif mask_bitstring is not None:
            measured_qubits = tuple([
                qubit
                for qubit in range(n_qubits)
                if mask_bitstring[qubit] != '2'
            ])
        else:
            measured_qubits = tuple(range(n_qubits))
        # measured qubits 必须得按照降序排列

        plm = self.get_partical_local_mitigator(measured_qubits)
        
        stats_counts = downsample(stats_counts, measured_qubits)  # 剃掉不测量的比特

        mitigated_stats_counts = plm.mitigate(stats_counts, threshold = threshold)
        
        extend_status_counts = {}
        for bitstring, count in mitigated_stats_counts.items():
            extend_bitstring = ['2'] * self.n_qubits
            for pos, qubit in enumerate(measured_qubits):
                extend_bitstring[qubit] = bitstring[pos]
            extend_status_counts[''.join(extend_bitstring)] = count

        return extend_status_counts