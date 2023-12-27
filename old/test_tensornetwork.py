import tensornetwork

import inspect
import numpy as np
import tensorcircuit as tc

from qiskit.result import LocalReadoutMitigator
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
# from jax import random
from qiskit_aer.noise import NoiseModel, ReadoutError
import pennylane as qml
import tensornetwork as tn
from collections import defaultdict

def ghz(n_qubits):
    cir = QuantumCircuit(n_qubits)
    cir.h(0)
    for i in range(n_qubits - 1):
        cir.cx(i, i + 1)
    cir.measure_all()
    return cir


n_qubits = 4
measure_fids = np.ones(shape=(n_qubits, 2)) * 0.99


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

# qiskit的校准
meas_mats = [
    np.array([[
        measure_fids[qubit][0], 1-measure_fids[qubit][1]],
        [1-measure_fids[qubit][0], measure_fids[qubit][1]]
    ])
    for qubit in range(n_qubits)
]

mit = LocalReadoutMitigator(meas_mats, list(range(n_qubits)))
qiskit_rr_probs = mit.quasi_probabilities(before_rr_counts)
plot_histogram(qiskit_rr_probs, filename='qiskit_rr')

before_rr_probs = {k: v / n_samples for k, v in before_rr_counts.items()}
before_rr_probs_vec = np.zeros(2**n_qubits)
for k, v in before_rr_probs.items():
    before_rr_probs_vec[int(k, base=2)] = v


K = tc.set_backend("tensorflow")
n_rr_sample = n_samples * 1000
tensor = tc.Circuit(n_qubits, inputs=np.sqrt(before_rr_probs_vec),)
for qubit in range(n_qubits):
    tensor.any(qubit, unitary=np.linalg.inv(meas_mats[qubit]))

r = tensor.sample(
    batch=n_rr_sample,
    allow_state=True,
    random_generator=tc.backend.get_random_state(42),
    format_="sample_bin"
)
r = np.array(r).astype(np.str_)
rr_count = defaultdict(int)
for rr_sample in r:
    rr_count[''.join(rr_sample)] += 1

rr_probs = {
    sample: count / n_rr_sample
    for sample, count in rr_count.items()
}

plot_histogram(rr_probs, filename='rr')
plot_histogram(before_rr_probs, filename='before_rr')
