# import
from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm
import qiskit.circuit.random
import scipy.io as sio
import random
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import numpy as np
import scipy.linalg as la
from qiskit.quantum_info import hellinger_fidelity
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


def ghz(n_qubits):
    cir = QuantumCircuit(n_qubits)
    cir.h(0)
    for i in range(n_qubits - 1):
        cir.cx(i, i + 1)
    cir.measure_all()
    return cir


def compare(n_qubits, depth, error_value):
    measure_fids = np.random.random(size=(n_qubits, 2))
    measure_fids = np.abs(measure_fids) / 10 + .9
    noise_model = NoiseModel()

    for i in range(n_qubits):
        re = ReadoutError([[measure_fids[i][0], 1 - measure_fids[i][0]],
                           [1 - measure_fids[i][1], measure_fids[i][1]]])
        noise_model.add_readout_error(re, qubits=[i])
    simulator = AerSimulator(noise_model=noise_model)
    noise_free_simulator = AerSimulator()
    # 拿到错误的模拟值
    n_samples = 3000

    # before_rm_counts = simulator.run(
    #     ghz(n_qubits), shots=n_samples).result().get_counts()
    while (True):
        qc = qiskit.circuit.random.random_circuit(num_qubits=n_qubits, depth=depth, max_operands= 2, measure= True)
        try:
            before_rm_counts = simulator.run(qc, shots=n_samples).result().get_counts()
            break
        except Exception:
            continue

    ideal_count  = noise_free_simulator.run(qc, shots=n_samples).result().get_counts()
    ideal_rm_prob = {k: v / n_samples for k, v in ideal_count.items()}


    before_rm_prob = {k: v / n_samples for k, v in before_rm_counts.items()}
    #plot_histogram(before_rm_prob, title=f'before_rm_{n_qubits}_{depth}_{error_value}',
                   # filename=f'{n_qubits}_{depth}_{error_value}before_rm.png')


    print('before',hellinger_fidelity(ideal_rm_prob, before_rm_prob))


    '''生成矫正的矩阵, meas_mats是读取噪声的form_ulation, meas_mats_inv是矫正矩阵'''
    meas_mats, meas_mats_inv = [], []
    for qubit in range(n_qubits):
        meas_mat = np.array([[
            measure_fids[qubit][0], 1 - measure_fids[qubit][1]],
            [1 - measure_fids[qubit][0], measure_fids[qubit][1]]
        ])
        meas_mats.append(meas_mat)
        meas_mats_inv.append(np.linalg.inv(meas_mat))

    '''qiskit的矫正'''
    if n_qubits < 10:
        mit = LocalReadoutMitigator(meas_mats, list(range(n_qubits)))
        qiskit_rm_prob = mit.quasi_probabilities(before_rm_counts)
        qiskit_rm_prob = {
            basis: value
            for basis, value in qiskit_rm_prob.items()
            if value > 1e-3
        }
        #plot_histogram(qiskit_rm_prob, title=f'qiskit_rm_{n_qubits}_{depth}_{error_value}',
                       # filename=f'{n_qubits}_{depth}_{error_value}qiskit_rm.png')

        def get_k(s_counts, shots):
            qc_sample = np.zeros((shots, n_qubits))
            c_sample = 0
            for k, v in s_counts.items():
                qc_sample[c_sample: c_sample + v] = np.array([int(i) for i in k])
                c_sample = c_sample + v
            return qc_sample

        '''用纯数学的方法试一下'''
        before_rm_prob_vec = np.zeros(2 ** n_qubits)
        for basis, prob in before_rm_prob.items():
            before_rm_prob_vec[int(basis, base=2)] = prob

        tensor_meas_mat_inv = np.linalg.inv(meas_mats[0])
        for qubit in range(1, n_qubits):
            tensor_meas_mat_inv = np.kron(
                tensor_meas_mat_inv, np.linalg.inv(meas_mats[qubit]))

        math_rm_prob = {}
        rm_prob_vec = tensor_meas_mat_inv @ before_rm_prob_vec
        for basis, prob in enumerate(rm_prob_vec):
            math_rm_prob[bin(basis).replace('0b', '')] = prob
        rm_prob = {
            basis: value
            for basis, value in math_rm_prob.items()
            if value > 1e-3
        }
        #plot_histogram(math_rm_prob, title=f'math_rm_{n_qubits}_{depth}_{error_value}',
                       # filename=f'{n_qubits}_{depth}_{error_value}math_rm.png')

    '''基于纯数学的形式改成了稀疏计算'''


    def sw_rm(qnum, stats_probs, meas_mats_inv, threshold = 1e-10): 
        @lru_cache
        def transform_(qubit, local_basis):
            local_vec = np.zeros(2)
            local_vec[local_basis] = 1
            local_vec = meas_mats_inv[qubit] @ local_vec
            return local_vec  # new_basis_value

        rm_prob = defaultdict(float)
        threshold = 1e-10
        for basis, prob in tqdm(stats_probs.items()):
            basis = [int(c) for c in basis]
            local_vecs = []
            for qubit in range(qnum):
                local_vecs.append(transform_(qubit, basis[qubit]))

            now_basis_values = [
                [local_basis, local_value*prob]
                for local_basis, local_value in enumerate(local_vecs[0])
            ]

            for qubit in range(1, qnum):
                next_basis_values = []
                for local_basis, local_value in enumerate(local_vecs[qubit]):
                    for basis, value in now_basis_values:
                        next_value = value * local_value

                        if next_value < threshold:  # 1e-2 ** n_qubits:
                            continue
                        
                        basis = basis << 2 | local_basis
                        next_basis_values.append([basis, next_value])

                now_basis_values = next_basis_values

            for basis, value in now_basis_values:
                rm_prob[basis] += value

        sum_prob = sum(rm_prob.values())
        rm_prob = {
            basis: value / sum_prob
            for basis, value in rm_prob.items()
        }
        return rm_prob

    start_time = time.time()
    rm_prob = sw_rm(n_qubits, before_rm_prob, meas_mats_inv)
    print('sw_rr cost', time.time() - start_time, 's')
    rm_prob = {
        basis: value
        for basis, value in rm_prob.items()
        if value > 1e-3
    }
    # rm_prob = {
    #     '1' * n_qubits: rm_prob['1' * n_qubits],
    #     '0' * n_qubits: rm_prob['0' * n_qubits]
    # }
    #plot_histogram(rm_prob, title=f'sw_rm_{n_qubits}_{depth}_{error_value}',
                   # filename=f'{n_qubits}_{depth}_{error_value}sw_rm.png')
    # math_sw = hellinger_fidelity(rm_prob, math_rm_prob)
    # qiskit_sw = hellinger_fidelity(rm_prob, qiskit_rm_prob)
    # qiskit_math = hellinger_fidelity(math_rm_prob, qiskit_rm_prob)
    # print('math_sw:', math_sw)
    # print('qiskit_sw:', qiskit_sw)
    # print('qiskit_math:', qiskit_math)
    print('sw', hellinger_fidelity(ideal_rm_prob, rm_prob))
    return (hellinger_fidelity(ideal_rm_prob, before_rm_prob),hellinger_fidelity(ideal_rm_prob, rm_prob))
    # return math_sw, qiskit_sw, qiskit_math

depth = 10
n_qubits = 4
error_value = 0.99
befores, afters = [],[]
for n_qubits in range(4, 10, 1):
    error_value =95
    depth = 5
    for i in range(10):
        before, after = compare(n_qubits, depth, error_value/100)
        befores.append(before)
        afters.append(after)
x = [i for i in range(4, 10, 1)]
import matplotlib.pyplot as  plt

befores = np.array(befores).reshape((6,10)).mean(axis = 1)
afters = np.array(afters).reshape((6,10)).mean(axis = 1)
fig, ax= plt.subplots(figsize = (8, 6))
ax.plot(x, befores, label = 'before')
ax.plot(x, afters, label = 'after')
ax.legend()
fig.savefig('n_qubits.svg')
# befores, afters = [],[]
# for error_value in range(90, 100, 1):
#     n_qubits = 5
#     depth = 5
#     for i in range(10):
#         before, after = compare(n_qubits, depth, error_value/100)
#         befores.append(before)
#         afters.append(after)
# x = [i for i in range(90, 100, 1)]
# import matplotlib.pyplot as  plt
#
# befores = np.array(befores).reshape((-1,10)).mean(axis = 1)
# afters = np.array(afters).reshape((-1,10)).mean(axis = 1)
# fig, ax= plt.subplots(figsize = (8, 6))
# ax.plot(x, befores, label = 'before')
# ax.plot(x, afters, label = 'after')
# ax.legend()
# fig.savefig('error_value.svg')
