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
from ray_func import batch, wait
from utils import all_bitstrings, to_bitstring, downsample_status_count
from correlation_analysis_new import PdBasedProtocolResults, correlation_based_partation, construct_bayesian_network
# from mitigator.multi_stage_mitigator import MultiStageMitigator
from functools import lru_cache
from typing import List, Optional, Tuple
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import pickle
import ray

# '''基于MultiStageMitigator改来的,考虑了测量的影响'''
'''基于ParticalLocalMitigator改来的,考虑了测量的影响'''

class BayesianMitigator(ParticalLocalMitigator):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
    
    def characterize_M(self, protocol_results, groups: List[List[int]], multi_process: bool = False):
        self.bayesian_network_model, self.bayesian_infer_model = construct_bayesian_network(protocol_results, self.n_qubits, groups, multi_process = multi_process)
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
         
    def get_M(self, measured_qubits: list = None):                     #得到对应比特数的full  M
        n_qubits = self.n_qubits

        measured_qubits = tuple(measured_qubits)
        bayesian_infer_model: VariableElimination =  self.bayesian_infer_model

        group=[i for i in range(n_qubits)]
        
        group_measured_qubits = [qubit for qubit in group if qubit in measured_qubits]
        n_group_measured_qubits = len(group_measured_qubits)
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


            # print('Bayesian:')
            # print(np.round(M, 3))
        
        return M 
    
    def get_partial_M(self,measured_qubits,measured_bitstring):                     #从136bit数据，得到部分指定bit的M,M3
        measured_qubits = tuple(measured_qubits)
        bayesian_infer_model: VariableElimination =  self.bayesian_infer_model

        group=measured_qubits
        
        group_measured_qubits = [qubit for qubit in group if qubit in measured_qubits]
        n_group_measured_qubits = len(group_measured_qubits)
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

        P = np.zeros(shape = (len(measured_bitstring), len(measured_bitstring)))
        for i in range(len(measured_bitstring)):
            for j in range(len(measured_bitstring)):
                P[i,j]=M[int(measured_bitstring[i],base=2),int(measured_bitstring[j],base=2)]


        for c in range(len(measured_bitstring)):
            for i in range(len(measured_bitstring)):
                P[i][c]=P[i][c]/sum(M[:][c])
            # print('Bayesian:')
            # print(np.round(M, 3))

        return P
    
    def partial_M(self, measured_qubits):                     #从136bit数据，得到部分指定bit的M
        measured_qubits = tuple(measured_qubits)
        bayesian_infer_model: VariableElimination =  self.bayesian_infer_model

        n_measured_qubits = len(measured_qubits)
        M = np.zeros(shape = (2**n_measured_qubits, 2**n_measured_qubits))
            
        for bitstring in all_bitstrings(n_measured_qubits):
            posterior_p = bayesian_infer_model.query([f'{qubit}_read' for qubit in measured_qubits],   # if qubit in measured_qubits
                evidence={
                    f'{qubit}_set': int(bitstring[measured_qubits.index(qubit)]) if qubit in measured_qubits else 2
                    for qubit in measured_qubits
                }
            )
            posterior_v = posterior_p.values.reshape(3**n_measured_qubits) # 变成了M中列的数据
            posterior_v = posterior_v[self.unmeasureed_index(n_measured_qubits)] # 剃掉包含2的
                
            assert abs(sum(posterior_v) - 1) < 1e-2, sum(posterior_v)  # 后面可以删了
                
            M[:,int(bitstring, base=2)] =  posterior_v

        for c in range(n_measured_qubits):
            for i in range(n_measured_qubits):
                M[i][c]=M[i][c]/sum(M[:][c])

        
        return M

            
    def batch_mitigate(self, stats_counts_list: list, mask_bitstring_list: list, threshold: float = None,):
        futures = []
        
        stats_counts_list = list(stats_counts_list)
        mask_bitstring_list = list(mask_bitstring_list)
        
        assert len(stats_counts_list) == len(mask_bitstring_list)
        
        @ray.remote
        def _mitigate(self, stats_counts_list, mask_bitstring_list, threshold):
            self = wait(self)
            return [
                self.mitigate(stats_count, threshold = threshold, mask_bitstring = mask_bitstring)
                for stats_count, mask_bitstring in zip(stats_counts_list, mask_bitstring_list)
            ]
        
        if len(stats_counts_list) < 120:
            batch_size = len(stats_counts_list)
        else:
            batch_size = len(stats_counts_list) // 120
        
        self_token = ray.put(self)
        for start in range(0, len(stats_counts_list), batch_size):
            futures.append(_mitigate.remote(self_token, stats_counts_list[start: start+batch_size], mask_bitstring_list[start: start+batch_size], threshold))
        
        all_results = []
        # print('start batch_mitigate')
        for sub_results in wait(futures, show_progress=True):
            all_results += sub_results

        return all_results


    def mitigate(self, stats_counts: dict, circuit: QuantumCircuit = None, threshold: float = None, mask_bitstring: str = None, 
                 measured_qubits: list = None):
        '''假设group之间没有重叠的qubit'''
        n_qubits = self.n_qubits
        
        if measured_qubits is not None:
            measured_qubits = tuple(measured_qubits)
        elif circuit is not None:
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
        stats_counts = downsample_status_count(stats_counts, measured_qubits)  # 剃掉不测量的比特

        mitigated_stats_counts = plm.mitigate(stats_counts, threshold = threshold)

        extend_status_counts = {}
        for bitstring, count in mitigated_stats_counts.items():
            extend_bitstring = ['2'] * self.n_qubits
            for pos, qubit in enumerate(measured_qubits):
                extend_bitstring[qubit] = bitstring[pos]
            extend_status_counts[''.join(extend_bitstring)] = count

        return extend_status_counts
        # return mitigated_stats_counts

    

    def add_error(self, stats_counts: dict, measured_qubits: list, threshold: float = None):
        '''输入没有噪声的，反过来预测有噪声的情况'''
        n_qubits = self.n_qubits
        
        plm: ParticalLocalMitigator = self.get_partical_local_mitigator(tuple(measured_qubits))
        stats_counts = downsample_status_count(stats_counts, measured_qubits)  # 剃掉不测量的比特
        mitigated_stats_counts = plm.add_error(stats_counts, threshold = threshold)
        extend_status_counts = {}
        for bitstring, count in mitigated_stats_counts.items():
            extend_bitstring = ['2'] * self.n_qubits
            for pos, qubit in enumerate(measured_qubits):
                extend_bitstring[qubit] = bitstring[pos]
            extend_status_counts[''.join(extend_bitstring)] = count

        return extend_status_counts