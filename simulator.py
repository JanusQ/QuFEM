'''只加了读取噪声的simulator'''

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
from utils import all_bitstrings, wait, random_group
from typing import List, Optional, Tuple

class Simulator():
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.simulator = AerSimulator()

        self._all_bitstings = None
        return

    def execute(self, circuit: List[QuantumCircuit], n_samples: int = 10000) -> List[dict]:
        # if not isinstance(circuit, (list, set, tuple)):
        #     circuit = [circuit]
        
        result = self.simulator.run(circuit, shots=n_samples).result().get_counts()

        if not isinstance(result, (list, set, tuple)):
            result = [result]
        
        return [
            {
            bitstring[::-1]: count  # qiskit是N-0, 我们统一成0-N
            for bitstring, count in stats_count.items()
            }
            for stats_count in result
        ]

    @property
    def all_bitstrings(self):
        return all_bitstrings(self.n_qubits)

class LocalSimulator(Simulator):
    '''每个比特一个M'''

    def __init__(self, n_qubits, M_per_qubit):
        super().__init__(n_qubits)
        self.M_per_qubit = M_per_qubit

        noise_model = NoiseModel()
        for i in range(n_qubits):
            re = ReadoutError(M_per_qubit[i])
            noise_model.add_readout_error(re, qubits=[i])
        self.simulator = AerSimulator(noise_model=noise_model)

    @staticmethod
    def gen_random_M(n_qubits):
        measure_fids = np.random.random(size=(n_qubits, 2))
        measure_fids = np.abs(measure_fids) / 10 + .9

        meas_mats = []
        for qubit in range(n_qubits):
            # probabilities[0] = [P("0"|"0"), P("1"|"0")]
            # probabilities[1] = [P("0"|"1"), P("1"|"1")]
            '''和读取矫正的格式好像不一样'''
            meas_mat = np.array([
                [measure_fids[qubit][0],    1-measure_fids[qubit][0]],
                [1-measure_fids[qubit][1],  measure_fids[qubit][1]]
            ])
            # meas_mat = np.array([
            #     [measure_fids[qubit][0],    1-measure_fids[qubit][1]],
            #     [1-measure_fids[qubit][0],  measure_fids[qubit][1]]
            # ])
            meas_mats.append(meas_mat)
            # meas_mats_inv.append(np.linalg.inv(meas_mat))

        return np.array(meas_mats)


class NonLocalSimulator(Simulator):
    '''所有比特一个M'''

    def __init__(self, n_qubits, M):
        super().__init__(n_qubits)
        self.M = M
        return

    @staticmethod
    def gen_random_M(n_qubits, min_fidelity = 0.90):
        assert n_qubits > 1 # 不然 error_others.max() - error_others.min()会编程nan
        
        M = np.zeros(shape=(2**n_qubits, 2**n_qubits))
        for basis in range(2**n_qubits):
            correct = np.random.random(size=1) * (1 - min_fidelity) + min_fidelity  # [1, 10]
            error_others = np.random.random(size=2**n_qubits-1)
            # if error_others.size == 1:
            #     error_others = np.array([(1-correct)])
            # else:
            error_others = (error_others - error_others.min()) / \
                (error_others.max() - error_others.min()) * (1-correct)
            column = np.concatenate(
                [error_others[:basis], correct, error_others[basis:]]).reshape(-1)
            M[:, basis] = column/sum(column)
        return M

    def execute(self, circuit: List[QuantumCircuit], n_samples: int = 10000):
        M = self.M
        
        noise_free_results = Simulator.execute(self, circuit, n_samples)

        error_results = []
        
        for noise_free_result in noise_free_results:
            error_result = defaultdict(int)
            for bitstring, count in noise_free_result.items():
                transfer_prob = M[:, int(bitstring, base=2)]
                error_bitstrings = random.choices(
                    self.all_bitstrings, weights=transfer_prob, k=count)
                for error_bitstring in error_bitstrings:
                    error_result[error_bitstring] += 1
            error_results.append(dict(error_result))

        # if not isinstance(circuit, list):
        #     return error_results[0]
        # else:
        return error_results


# class PartialLocalSimulator(Simulator):
#     '''几个比特一个M'''
#     pass

class MeasurementAwareNonLocalSimulator(Simulator):
    def __init__(self, n_qubits, error_model):
        super().__init__(n_qubits)
        
        local_M, sub_groups, sub_group_Ms = error_model
        
        self.local_simulator = LocalSimulator(n_qubits, local_M)
        self.sub_groups = sub_groups
        self.sub_group_Ms = sub_group_Ms
        
        
        
    @staticmethod
    def gen_random_M(n_qubits):
        # 对于一个比特来说有 读0，读1，不读
        # 读可能会对别的比特的读取错误产生影响，不读不会影响
        
        # 先生成一个大的矩阵
        local_M = LocalSimulator.gen_random_M(n_qubits)
        
        sub_groups = random_group(range(n_qubits), 2)  # 生成一些两两的 [[0, 1], [2, 3]] # 
        sub_groups = [sub_group for sub_group in sub_groups if len(sub_group) > 1]
        sub_group_Ms = [
            NonLocalSimulator.gen_random_M(len(sub_group), min_fidelity = 0.95)
            for sub_group in sub_groups
        ]

        return local_M, sub_groups, sub_group_Ms
    
    def execute(self, circuits: List[QuantumCircuit], n_samples: int = 10000):
        # 读取那些比特进行了读取
        local_error_results = self.local_simulator.execute(circuits, n_samples)
        sub_groups, sub_group_Ms = self.sub_groups, self.sub_group_Ms
        # 生成对应的读取矩阵
        
        if not isinstance(circuits, (list, set, tuple)):
            circuits = [circuits]
        
        new_error_results = []
        for circuit, result in zip(circuits, local_error_results):
            measured_qubits = [
                instruction.qubits[0].index
                for instruction in circuit
                if instruction.operation.name == 'measure'
            ]
            # print(circuit)
            
            for sub_group, sub_group_M in zip(sub_groups, sub_group_Ms):
                if any([qubit not in measured_qubits for qubit in sub_group]): continue
                new_error_result = defaultdict(int)
                for bitstring, count in result.items():
                    bitstring = list(bitstring)
                    
                    # 这里写的复杂了
                    transfer_prob = sub_group_M[:,int(''.join([bitstring[measured_qubits.index(qubit)] for qubit in sub_group]), base=2)]
                    error_bitstrings = random.choices(all_bitstrings(len(sub_group)), weights=transfer_prob, k=count)
                    
                    for error_bitstring in error_bitstrings:
                        for index, qubit in enumerate(sub_group):
                            bitstring[measured_qubits.index(qubit)] = error_bitstring[sub_group.index(qubit)]
                        new_error_result[''.join(bitstring)] += 1
                result = new_error_result

            # 把没有测量的比特也标记为2放进去
            new_result = {}
            for bitstring, count in result.items():
                new_bitstring = ''
                for qubit in range(self.n_qubits):
                    if qubit in measured_qubits:
                        new_bitstring = new_bitstring + bitstring[measured_qubits.index(qubit)]
                    else:
                        new_bitstring = new_bitstring + '2'
                new_result[new_bitstring] = count
            result = new_result

            new_error_results.append(dict(result))
            
        return new_error_results
    