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
from tqdm import tqdm

class MultiStageMitigator():
    def __init__(self, n_qubits, n_stages = 2):
        self.n_qubits = n_qubits
        self.n_stages = n_stages
    
    def eval_plm(self, plm: ParticalLocalMitigator, protocol_results):
        mitigated_protocol_results = {
            real_bitstring: {
                measured_bitstring: value * 1000  # 防止精度丢失
                for measured_bitstring, value in plm.mitigate(status_count, mask_bitstring = real_bitstring, threshold = 1e-6).items()  # 可能要乘以一个大的数字防止精度降低
            }
            for real_bitstring, status_count in tqdm(protocol_results.items())
        }
  
        n_success = 0
        n_total = 0
        for real_bitstring, status_count in mitigated_protocol_results.items():
            n_total += sum(status_count.values())
            if real_bitstring in status_count: #改成hamming distance的
                n_success += status_count[real_bitstring]
            
        # print(n_success/n_total)
        return n_success/n_total, mitigated_protocol_results
    
    def characterize_M(self, protocol_results, group_size = 2, partation_method = 'max-cut', BasisMitigator = ParticalLocalMitigator):
        n_qubits = self.n_qubits
        n_stages = self.n_stages
        
        self.plms = []
        self.plm_scores = []
        
        # def prob(status_count, bitstring):
        #     return status_count[bitstring]/sum(status_count.values())
        
        # TODO: 可以整一个树搜索
        if BasisMitigator == ParticalLocalMitigator:
            # ParticalLocalMitigator没法计算包含非测量的
            protocol_results = {
                bitstring: status_count
                for bitstring, status_count in protocol_results.items()
                if '2' not in bitstring
            }
            
        for stage in range(n_stages):
            print('Stage:', stage)
            
            # random selection
            # best_plm = None
            # best_mitgated_protocol_results = None
            # max_score = 0
            # for _ in range(3):  
            #     '''目前看来好的划分是会对校准结果产生影响的'''
            #     plm: ParticalLocalMitigator = BasisMitigator(n_qubits)
            #     groups = plm.random_group(group_size) #TODO: 用类似集成学习方式划分group  # [[0, 1], [2]]#
            #     plm.characterize_M(protocol_results, groups)

            #     score, mitgated_protocol_results = self.eval_plm(plm, protocol_results)
            #     if score > max_score:
            #         best_plm = plm
            #         max_score = score
            #         best_mitgated_protocol_results = mitgated_protocol_results
                    
            #     print('score:', score)
            
            # plm = best_plm
            # protocol_results = best_mitgated_protocol_results
            # score = max_score
            # print('max_score:', max_score)
            
            '''一般都比较好，但是和最优相比可能还是差了'''
            # # max-cut
            plm: ParticalLocalMitigator = BasisMitigator(n_qubits)
            groups = correlation_based_partation(protocol_results, group_size, n_qubits)
            plm.characterize_M(protocol_results, groups)

            score, protocol_results = self.eval_plm(plm, protocol_results)
            print('correlation_based_partation score', score)
            
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
            
        
    def mitigate(self, stats_counts: dict, threshold: float = None, measured_qubits = None):
        for plm in self.plms:
            stats_counts = plm.mitigate(stats_counts, threshold =  threshold, measured_qubits = measured_qubits)

            if plm != self.plms[-1]:
                stats_counts = {
                    bitstring: count * 1000  # 防止精度丢失
                    for bitstring, count in stats_counts.items()
                }
            
        return stats_counts
    
    
    def add_error(self, stats_counts: dict, measured_qubits, threshold: float = None):
        '''输入没有噪声的，反过来预测有噪声的情况'''
        inv_plms = list(self.plms)
        inv_plms.reverse()
        for plm in inv_plms:
            stats_counts = plm.add_error(stats_counts, measured_qubits = measured_qubits, threshold =  threshold)

            if plm != self.plms[-1]:
                stats_counts = {
                    bitstring: count * 1000  # 防止精度丢失
                    for bitstring, count in stats_counts.items()
                }
            
        return stats_counts