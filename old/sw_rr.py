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
import time

def ghz(n_qubits):
    cir = QuantumCircuit(n_qubits)
    cir.h(0)
    for i in range(n_qubits - 1):
        cir.cx(i, i + 1)
    cir.measure_all()
    return cir


n_qubits = 3
measure_fids = jnp.ones(shape=(n_qubits, 2)) * 0.99
noise_model = NoiseModel()

for i in range(n_qubits):
    re = ReadoutError([[measure_fids[i][0], 1 - measure_fids[i][0]],
                      [1 - measure_fids[i][1], measure_fids[i][1]]])
    noise_model.add_readout_error(re, qubits=[i])
simulator = AerSimulator(noise_model=noise_model)

# 拿到错误的模拟值
n_samples = 3000
before_rm_counts = simulator.run(
    ghz(n_qubits), shots=n_samples).result().get_counts()
before_rm_prob = {k: v / n_samples for k, v in before_rm_counts.items()}
plot_histogram(before_rm_prob, title = f'before_rm_{n_qubits}', filename=f'before_rm_{n_qubits}')

'''生成矫正的矩阵, meas_mats是读取噪声的form_ulation, meas_mats_inv是矫正矩阵'''
meas_mats, meas_mats_inv = [], []
for qubit in range(n_qubits):
    meas_mat = np.array([[
        measure_fids[qubit][0], 1-measure_fids[qubit][1]],
        [1-measure_fids[qubit][0], measure_fids[qubit][1]]
    ])
    meas_mats.append(meas_mat)
    meas_mats_inv.append(np.linalg.inv(meas_mat))


if n_qubits < 10:
    '''qiskit的矫正'''
    mit = LocalReadoutMitigator(meas_mats, list(range(n_qubits)))
    qiskit_rm_prob = mit.quasi_probabilities(before_rm_counts)
    qiskit_rm_prob = {
        basis: value
        for basis, value in qiskit_rm_prob.items()
        if value > 1e-3
    }
    plot_histogram(qiskit_rm_prob, title = f'qiskit_rm_{n_qubits}', filename=f'qiskit_rm_{n_qubits}')

    '''用纯数学的方法试一下'''
    before_rm_prob_vec = np.zeros(2**n_qubits)
    for basis, prob in before_rm_prob.items():
        before_rm_prob_vec[int(basis, base=2)] = prob

    tensor_meas_mat_inv = np.linalg.inv(meas_mats[0])
    for qubit in range(1, n_qubits):
        tensor_meas_mat_inv = np.kron(
            tensor_meas_mat_inv, np.linalg.inv(meas_mats[qubit]))

    rm_prob = {}
    rm_prob_vec = tensor_meas_mat_inv @ before_rm_prob_vec
    for basis, prob in enumerate(rm_prob_vec):
        rm_prob[bin(basis).replace('0b', '')] = prob
    rm_prob = {
        basis: value
        for basis, value in rm_prob.items()
        if value > 1e-3
    }
    plot_histogram(rm_prob, title = f'math_rm_{n_qubits}', filename=f'math_rm_{n_qubits}')

'''基于纯数学的形式改成了稀疏计算'''
def sw_rr(before_rm_prob, meas_mats_inv):
    def transform_(matrix, local_basis):
        local_vec = np.zeros(2)
        local_vec[local_basis] = 1
        local_vec = matrix @ local_vec
        return local_vec  # new_basis_value

    rm_prob = defaultdict(float)

    for basis, prob in before_rm_prob.items():
        # print(basis, prob)
        basis = [int(c) for c in basis]
        local_vecs = []
        for qubit in range(n_qubits):
            local_vecs.append(transform_(meas_mats_inv[qubit], basis[qubit]))

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

                    if np.abs(next_value) < 1e-10:  # 1e-2 ** n_qubits:
                        continue

                    basis.append(local_basis)  # 可以改成二进制的
                    next_basis_values.append([basis, next_value])

            now_basis_values = next_basis_values

        for basis, value in now_basis_values:
            basis = np.array(basis)
            rm_prob[''.join(basis.astype(np.str_))] += value

        new_rm_prob = defaultdict(float)
        for basis, value in rm_prob.items():
            if np.abs(value) < 1e-10:  # 1e-2 ** n_qubits:
                continue
            new_rm_prob[basis] = value
        rm_prob = new_rm_prob

    sum_prob = sum(rm_prob.values())
    rm_prob = {
        basis: value / sum_prob
        for basis, value in rm_prob.items()
    }

    return rm_prob

start_time = time.time()
rm_prob = sw_rr(before_rm_prob, meas_mats_inv)
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
plot_histogram(rm_prob, title = f'sw_rm_{n_qubits}', filename=f'sw_rm_{n_qubits}')
