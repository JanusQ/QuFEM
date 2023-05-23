# import
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

def ghz(n_qubits):
    cir = QuantumCircuit(n_qubits)
    cir.h(0)
    for i in range(n_qubits - 1):
        cir.cx(i, i + 1)
    cir.measure_all()
    return cir


def inverse(arr):
    return jnp.where(arr == 0, 1, 0)


def flip(arr, flag):
    '''flag为1的时候arr对应的元素会翻转'''
    return jnp.abs(arr - flag)


def rr(qc_sample, measure_fids, n_rm_sample=100):
    n_samples, n_qubits = qc_sample.shape
    # key = random.PRNGKey(314)
    
    def _rr_sample_construct_prob(measure_fid, sample):
        transfer_p = 1 - measure_fid[inverse(sample)] # 是另外一个态转移来的概率
        transfer = jax.random.bernoulli(jax.random.PRNGKey(random.randint(0, 100)), transfer_p, shape=(n_rm_sample,))
        return jnp.where(transfer == True, 1, 0)

    def _rr_sample_mit(sample, transfer):
        return flip(sample, transfer)

    def rr_sample(sample, measure_fids):
        transfers = jnp.array([
            _rr_sample_construct_prob(qubit_fid, qubit_value)
            for qubit_fid, qubit_value in zip(measure_fids, sample)
        ])
        # transfers = vmap(_rr_sample_construct_prob, in_axes=(0, 0))(measure_fids, sample)
        
        # if np.any(transfers == 1):
        #     print('flip')
            
        # rm_samples = [
        #     _rr_sample_mit(sample, transfers[index])
        #     for index in range(transfers.shape[0])
        # ]
        transfers = transfers.T
        rm_samples = vmap(_rr_sample_mit, in_axes=(None, 0))(sample, transfers)
        
        # for i, _ in enumerate(rm_samples):
        #     if np.sum(_ - sample) != 0:
        #         print(_, sample, transfers[i])
        
        return rm_samples

    
    # all_rm_samples = [
    #     rr_sample(elm, measure_fids)
    #     for elm in qc_sample
    # ]
    all_rm_samples = vmap(rr_sample, in_axes=(0, None))(qc_sample, measure_fids)
    
    all_rm_samples = np.array(all_rm_samples).astype(np.str_)

    counts = defaultdict(int)
    for rm_samples in all_rm_samples:
        for rm_sample in rm_samples:
            counts[''.join(rm_sample)] += 1

    total_count = n_samples * n_rm_sample
    counts = {
        sample: count / total_count
        for sample, count in counts.items()
    }
    return counts


n_qubits = 4
measure_fids = jnp.ones(shape=(n_qubits, 2)) * 0.99
noise_model = NoiseModel()

for i in range(n_qubits):
    re = ReadoutError([[measure_fids[i][0], 1 - measure_fids[i][0]],
                      [1 - measure_fids[i][1], measure_fids[i][1]]])
    noise_model.add_readout_error(re, qubits=[i])
simulator = AerSimulator(noise_model=noise_model)

# 拿到错误的模拟值
n_samples = 3000
before_rr_counts = simulator.run(
    ghz(n_qubits), shots=n_samples).result().get_counts()

meas_mats, meas_mats_inv = [], []
for qubit in range(n_qubits):
    meas_mat = np.array([[
        measure_fids[qubit][0], 1-measure_fids[qubit][1]],
        [1-measure_fids[qubit][0], measure_fids[qubit][1]]
    ])
    meas_mats.append(meas_mat)
    meas_mats_inv.append(np.linalg.inv(meas_mat))


mit = LocalReadoutMitigator(meas_mats, list(range(n_qubits)))
qiskit_rr_prob = mit.quasi_probabilities(before_rr_counts)
plot_histogram(qiskit_rr_prob, filename='qiskit_rr')
      
def get_k(s_counts, shots):
    qc_sample = np.zeros((shots, n_qubits))
    c_sample = 0
    for k, v in s_counts.items():
        qc_sample[c_sample: c_sample + v] = np.array([int(i) for i in k])
        c_sample = c_sample + v
    return qc_sample

rr_prob = rr(get_k(before_rr_counts, n_samples), measure_fids, n_rm_sample=1000)

before_rr_prob = {k: v / n_samples for k, v in before_rr_counts.items()}

plot_histogram(rr_prob, filename='rr')
plot_histogram(before_rr_prob, filename='before_rr')

# 用纯数学的方法试一下
before_rr_prob_vec = np.zeros(2**n_qubits)
for basis, prob in before_rr_prob.items():
    before_rr_prob_vec[int(basis, base=2)] = prob

# tensor_meas_mat = meas_mats[0]
# for qubit in range(1, n_qubits):
#     tensor_meas_mat = np.kron(tensor_meas_mat, meas_mats[qubit])
    
tensor_meas_mat_inv = np.linalg.inv(meas_mats[0])
for qubit in range(1, n_qubits):
    tensor_meas_mat_inv = np.kron(tensor_meas_mat_inv, np.linalg.inv(meas_mats[qubit]))

rr_prob = {}
rr_prob_vec = tensor_meas_mat_inv @ before_rr_prob_vec
for basis, prob in enumerate(rr_prob_vec):
    rr_prob[bin(basis).replace('0b', '')] = prob
plot_histogram(rr_prob, filename='math_rr')

def transform(matrix, qubit, basis, value):
    local_vec = np.zeros(2)
    local_vec[basis[qubit]] = 1
    local_vec = matrix @ local_vec
    
    # new_basis_value = []
    # for local_basis, local_v in enumerate(local_vec):
    #     extend_basis = np.array(basis)
    #     extend_basis[qubit] = local_basis
    #     new_basis_value.append([local_basis, local_v * value])
        
    return local_vec #new_basis_value

def merge(basis_value1, basis_value2):
    return

rr_prob = defaultdict(float)

for basis, prob in before_rr_prob.items():
    print(basis, prob)
    basis = [int(c) for c in basis]
    local_vecs = []
    for qubit in range(n_qubits):
        local_vecs.append(transform(meas_mats_inv[qubit], qubit, basis, prob))
    
    now_basis_values = [
        [[local_basis], local_value*prob]
        for local_basis, local_value in enumerate(local_vecs[0])
    ]
    
    for qubit in range(1, n_qubits):
        next_basis_values = []
        for local_basis, local_value in enumerate(local_vecs[qubit]):
            for basis, value in now_basis_values:
                basis = list(basis)
                next_value = value * local_value
                
                # if np.abs(next_value) < 1e-10:
                #     continue
                
                basis.append(local_basis)  # 可以改成进制的
                next_basis_values.append([basis, next_value])
                
        now_basis_values = next_basis_values
    
    for basis, value in now_basis_values:
        basis = np.array(basis)
        rr_prob[''.join(basis.astype(np.str_))] += value
                
plot_histogram(rr_prob, filename='new_math_rr')