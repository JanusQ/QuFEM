from benchmark import ghz
from simulator import LocalSimulator, NonLocalSimulator, Simulator, MeasurementAwareNonLocalSimulator as MALSimulator
from qiskit.visualization import plot_histogram
from config import fig_path, join
import os

from mitigator.protocol_circuits import EnumeratedProtocol, MeasuremtAwareEnumeratedProtocol
from mitigator.local_mitigator import LocalMitigator
from mitigator.nonlocal_mitigator import NonLocalMitigator
from mitigator.partical_local_mitigator import ParticalLocalMitigator
from mitigator.multi_stage_mitigator import MultiStageMitigator
from mitigator.measurement_aware_mitigator import BayesianMitigator

import numpy as np
from utils import all_bitstrings
from qiskit.quantum_info.analysis import hellinger_fidelity
from correlation_analysis import correlation_based_partation, construct_bayesian_network
import matplotlib.pyplot as plt

'''开始考虑读取的影响了'''

n_qubits = 3
M = np.array([[[0.90347497, 0.09652503],
        [0.02001257, 0.97998743]],

       [[0.91547926, 0.08452074],
        [0.08481369, 0.91518631]],

       [[0.90903764, 0.09096236],
        [0.01686524, 0.98313476]]]), [[0, 2]], [np.array([[0.98212119, 0.        , 0.01418251, 0.0019242 ],
       [0.00575889, 0.9731274 , 0.        , 0.02580488],
       [0.        , 0.01589443, 0.96468217, 0.        ],
       [0.01211993, 0.01097817, 0.02113532, 0.97227092]])]
# simulator = MALSimulator(n_qubits, MALSimulator.gen_random_M(n_qubits))
simulator = MALSimulator(n_qubits, M)
noise_free_simulator = Simulator(n_qubits)  # 没有噪音

bitstrings, protocol_circuits = MeasuremtAwareEnumeratedProtocol(n_qubits).gen_circuits()
protocol_results = {
    bitstring: status_count
    for bitstring, status_count in zip(bitstrings, simulator.execute(protocol_circuits, n_samples = 1000))
}

# construct_bayesian_network(protocol_results, 2, n_qubits)

# correlation_based_partation(protocol_results, n_qubits)

# print(ls.M_per_qubit)
# print(lm.M_per_qubits)

# for bitstring, status_count in protocol_results.items():
#     fig, ax = plt.subplots()
#     plot_histogram(status_count, filename=f'temp/fig/measure_aware_protocols/{bitstring}.png', title = bitstring, ax = None)
#     plt.close()
# exit()

n_samples = 10000
threshold = 1e-5
circuit = ghz(n_qubits)
result = simulator.execute(circuit, n_samples)[0]  # 返回默认都是一个数组提高效率
# plot_histogram(error_result, filename = join(fig_path, 'ls_ghz'))
noise_free_result = noise_free_simulator.execute(circuit)[0]

# mam = BayesianMitigator(n_qubits)
# groups = mam.random_group(2)
# mam.characterize_M(protocol_results, groups)
# result_mam = mam.mitigate(result, circuit, threshold = threshold * n_samples)


# lm = LocalMitigator(n_qubits)
# lm.characterize_M(protocol_results)
# result_l = lm.mitigate(result, threshold = threshold * n_samples)

# # lm = NonLocalMitigator(n_qubits, M =  np.kron(M[0], M[1]))
# nlm = NonLocalMitigator(n_qubits)
# nlm.characterize_M(protocol_results)
# result_nl = nlm.mitigate(result)

# fidelity_plms = []
# for group_size in range(1, n_qubits+1):
#     plm = ParticalLocalMitigator(n_qubits)
#     group = plm.random_group(group_size)  # n_qubits的时候是个nlm一样的
#     # group = [[0], [1], [2]]
#     plm.characterize_M(protocol_results, group)
#     # print(plm.group2M)
#     rm_result_partical_local = plm.mitigate(result, threshold = threshold * n_samples)
#     # print(rm_result_partical_local)
#     fidelity_plms.append(hellinger_fidelity(rm_result_partical_local, noise_free_result))

# fidelity_mlms = []
# score_mlms = []
# for i in range(10):
#     mlm = MultiStageMitigator(n_qubits, n_stages = 5)
#     score = mlm.characterize_M(protocol_results, group_size = 3)
#     result_m = mlm.mitigate(result, threshold = threshold * n_samples)
    
#     fidelity_mlms.append(hellinger_fidelity(result_m, noise_free_result))
#     score_mlms.append(score)


fidelity_mamlms = []
score_mlms = []
for i in range(10):
    mamlm = MultiStageMitigator(n_qubits, n_stages = 5)
    score = mamlm.characterize_M(protocol_results, group_size = 2, BasisMitigator=BayesianMitigator)
    result_mam = mamlm.mitigate(result, threshold = threshold * n_samples)
    
    fidelity_mamlms.append(hellinger_fidelity(result_mam, noise_free_result))
    score_mlms.append(score)


# print(
#     hellinger_fidelity(result_l, noise_free_result), 
#     hellinger_fidelity(result_nl, noise_free_result),
#     fidelity_plms,
#     fidelity_mlms,
#     hellinger_fidelity(result, noise_free_result)
# )
# error_result = nls.execute(circuit)
# plot_histogram(error_result, filename = join(fig_path, 'nls_ghz'))
print('end')

'''先假设当前的是对的吧'''
'''mitigator里面都换成位运算的'''