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
import random
import math
import time
from mitigator.partical_local_mitigator import ParticalLocalMitigator
from utils import to_bitstring
from correlation_analysis_new import correlation_based_partation
from tqdm import tqdm


def hamming_distance(string1, string2):
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter


class MultiStageMitigator():
    def __init__(self, n_qubits, n_stages = 2):
        self.n_qubits = n_qubits
        self.n_stages = n_stages
    
    def eval_plm_hamming(self, plm: ParticalLocalMitigator, protocol_results: dict, multi_process = False, threshold = 1e-5):
        
        if not multi_process:
            mitigated_protocol_results = {
                real_bitstring: {
                    measured_bitstring: value * 1000  # 防止精度丢失
                    for measured_bitstring, value in plm.mitigate(status_count, mask_bitstring = real_bitstring, threshold = threshold).items()  # 可能要乘以一个大的数字防止精度降低
                }
                for real_bitstring, status_count in protocol_results.items()
            }
        else:
            stats_counts_list = list(protocol_results.values())
            mask_bitstring_list = list(protocol_results.keys())
            mitigated_protocol_results = plm.batch_mitigate(stats_counts_list, mask_bitstring_list, threshold = threshold)
            mitigated_protocol_results = {
                real_bitstring: {
                    measured_bitstring: value * 1000  # 防止精度丢失
                    for measured_bitstring, value in status_count.items()  # 可能要乘以一个大的数字防止精度降低
                }
                for real_bitstring, status_count in zip(mask_bitstring_list, mitigated_protocol_results)
            }

        total_dist = 0
        n_total = 0
        
        for real_bitstring, status_count in mitigated_protocol_results.items():
            n_total += sum(status_count.values())
            for measured_bitstring, count in status_count.items():
                total_dist += hamming_distance(measured_bitstring, real_bitstring) * count

        return total_dist/n_total, mitigated_protocol_results
    
    def characterize_M(self, protocol_results,group_size = 2, partation_method = ['random', 'max-cut'], BasisMitigator = ParticalLocalMitigator, threshold= 1e-5, multi_process= True):
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
            
            
            candidate_plms = []
            candidate_scores = []
            candidate_mitgated_protocol_results = []
            # random selection
            if 'random' in partation_method:
                # for _ in range(3):  
                for _ in range(2):  
                    '''目前看来好的划分是会对校准结果产生影响的'''
                    plm: ParticalLocalMitigator = BasisMitigator(n_qubits)
                    groups = plm.random_group(group_size) #TODO: 用类似集成学习方式划分group  # [[0, 1], [2]]#
                    # groups = [[0, 3], [1, 6], [2, 7], [4, 9], [5, 8]]
                    plm.characterize_M(protocol_results, groups, multi_process = multi_process)

                    score, mitgated_protocol_results = self.eval_plm_hamming(plm, protocol_results, threshold= threshold, multi_process= multi_process)
                    
                    candidate_plms.append(plm)
                    candidate_scores.append(score)
                    candidate_mitgated_protocol_results.append(mitgated_protocol_results)
                    
            '''一般都比较好，但是和最优相比可能还是差了'''
            if 'max-cut' in partation_method:
                # # max-cut
                plm: ParticalLocalMitigator = BasisMitigator(n_qubits)
                groups = correlation_based_partation(protocol_results, group_size, n_qubits)
                plm.characterize_M(protocol_results, groups, multi_process = multi_process)
                score, mitgated_protocol_results = self.eval_plm_hamming(plm, protocol_results,threshold= threshold, multi_process= multi_process)
                # print('correlation_based_partation score', score)

                candidate_plms.append(plm)
                candidate_scores.append(score)
                candidate_mitgated_protocol_results.append(mitgated_protocol_results)
            
            # # 几种方法选最好的
            best_plm_index = np.argmin(candidate_scores)

            self.plms.append(candidate_plms[best_plm_index])
            self.plm_scores.append(candidate_scores[best_plm_index])
            protocol_results = candidate_mitgated_protocol_results[best_plm_index]
            
        best_plm_index = np.argmin(self.plm_scores)
        self.plms = self.plms[:best_plm_index+1]
        self.plm_scores = self.plm_scores[:best_plm_index+1]
        # print(groups)
        
        return self.plm_scores[-1]
        # return groups  
        
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
        