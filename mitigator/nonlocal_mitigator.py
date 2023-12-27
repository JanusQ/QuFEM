# import
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
from utils import to_bitstring

class NonLocalMitigator():
    def __init__(self, n_qubits, M = None):
        self.n_qubits = n_qubits
        self.M = M
        if M is not None:
            # self.invM = np.linalg.inv(self.M)
            self.invM = self.M
        
    def characterize_M(self, protocol_results):
    #def characterize_M(self):
        n_qubits = self.n_qubits

        # self.M = np.random.rand(2**n_qubits, 2**n_qubits)

        self.M = np.zeros(shape=(2**n_qubits, 2**n_qubits))

        for real_bitstring, status_count in protocol_results.items():
            
            if '2' in real_bitstring: continue
            
            real_bitstring = int(real_bitstring, base=2)
            total_count = sum(status_count.values())
            for measure_bitstring, count in status_count.items():
                measure_bitstring = int(measure_bitstring, base=2)
                self.M[measure_bitstring][real_bitstring] += count / total_count
        
        self.invM = np.linalg.inv(self.M)
        # self.invM = self.M
        return self.M
        
    def mitigate(self, stats_counts: dict):
        '''用纯数学的方法'''
        # total_count = sum(stats_counts.values())
        error_count_vec = np.zeros(2 ** self.n_qubits)
        for basis, count in stats_counts.items():
            error_count_vec[int(basis, base=2)] = count
            
        rm_prob = {}
        rm_count_vec = self.invM @ error_count_vec
        rm_count_vec[rm_count_vec < 0] = 0
        rm_count_vec /= sum(rm_count_vec)
        for basis, prob in enumerate(rm_count_vec):
            rm_prob[to_bitstring(basis, self.n_qubits)] = prob

        return rm_prob
    
    def add_error(self, stats_counts: dict):
        error_count_vec = np.zeros(2 ** self.n_qubits)
        for basis, count in stats_counts.items():
            error_count_vec[int(basis, base=2)] = count
            
        rm_prob = {}
        rm_count_vec = self.M @ error_count_vec
        rm_count_vec[rm_count_vec < 0] = 0
        rm_count_vec /= sum(rm_count_vec)
        for basis, prob in enumerate(rm_count_vec):
            rm_prob[to_bitstring(basis, self.n_qubits)] = prob

        return rm_prob
     