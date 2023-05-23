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
from qiskit_aer import AerSimulator
# from jax import random
from qiskit_aer.noise import NoiseModel, ReadoutError, QuantumError, pauli_error, depolarizing_error
import random
import math
import time
from mitigator.partical_local_mitigator import ParticalLocalMitigator
from utils import to_bitstring
from correlation_analysis import correlation_based_partation

class MultiStageMitigator():
    def __init__(self, n_qubits, n_stages = 2):
        self.n_qubits = n_qubits
        self.n_stages = n_stages
    
    def eval_plm(self, plm: ParticalLocalMitigator, protocol_results):
        mitigated_protocol_results = {
            real_bitstring: {
                measured_bitstring: value * 1000  # 防止精度丢失
                for measured_bitstring, value in plm.mitigate(status_count).items()  # 可能要乘以一个大的数字防止精度降低
            }
            for real_bitstring, status_count in protocol_results.items()
        }
  
        n_success = 0
        n_total = 0
        for real_bitstring, status_count in mitigated_protocol_results.items():
            n_total += sum(status_count.values())
            n_success += status_count[real_bitstring]
            
        # print(n_success/n_total)
        return n_success/n_total, mitigated_protocol_results
    
    def characterize_M(self, protocol_results, group_size = 2, partation_method = 'max-cut'):
        n_qubits = self.n_qubits
        n_stages = self.n_stages
        
        self.plms = []
        self.plm_scores = []
        
        # def prob(status_count, bitstring):
        #     return status_count[bitstring]/sum(status_count.values())
        
        # TODO: 可以整一个树搜索
        
        protocol_results = {
            bitstring: status_count
            for bitstring, status_count in protocol_results.items()
            if '2' not in bitstring
        }
        
        for stage in range(n_stages):
            # print('Stage:', stage)
            
            # random selection
            best_plm = None
            best_mitgated_protocol_results = None
            max_score = 0
            for _ in range(3):  
                '''目前看来好的划分是会对校准结果产生影响的'''
                plm = ParticalLocalMitigator(n_qubits)
                groups = plm.random_group(group_size) #TODO: 用类似集成学习方式划分group
                plm.characterize_M(protocol_results, groups)

                score, mitgated_protocol_results = self.eval_plm(plm, protocol_results)
                if score > max_score:
                    best_plm = plm
                    max_score = score
                    best_mitgated_protocol_results = mitgated_protocol_results
                    
                # print('score:', score)
            
            plm = best_plm
            protocol_results = best_mitgated_protocol_results
            score = max_score
            # print('max_score:', max_score)
            
            '''一般都比较好，但是和最优相比还是差了'''
            # # max-cut
            # plm = ParticalLocalMitigator(n_qubits)
            # groups = correlation_based_partation(protocol_results, group_size, n_qubits)
            # # group = plm.random_group(group_size) #TODO: 用类似集成学习方式划分group
            # # group = [[q,] for q in range(n_qubits)]
            # plm.characterize_M(protocol_results, groups)

            # score, protocol_results = self.eval_plm(plm, protocol_results)
            # print('correlation_based_partation score', score)
            
            # # 几种方法选最好的
            # if score < max_score:
            #     protocol_results = best_mitgated_protocol_results
            #     plm = best_plm
            #     score = max_score
            
            self.plms.append(plm)
            self.plm_scores.append(score)
            
        best_plm_index = np.argmax(self.plm_scores)
        self.plms = self.plms[:best_plm_index+1]
        self.plm_scores = self.plm_scores[:best_plm_index+1]
        
        return self.plm_scores[-1]
            
        
    def mitigate(self, stats_counts: dict, threshold: float = None):
        for plm in self.plms:
            stats_counts = plm.mitigate(stats_counts, threshold)

            if plm != self.plms[-1]:
                stats_counts = {
                    bitstring: count * 1000  # 防止精度丢失
                    for bitstring, count in stats_counts.items()
                }
            
        return stats_counts
        