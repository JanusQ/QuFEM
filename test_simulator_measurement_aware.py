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

import numpy as np
from utils import all_bitstrings
from qiskit.quantum_info.analysis import hellinger_fidelity
from correlation_analysis import correlation_based_partation, construct_bayesian_network
import matplotlib.pyplot as plt

'''开始考虑读取的影响了'''

n_qubits = 2

simulator = MALSimulator(n_qubits, MALSimulator.gen_random_M(n_qubits))
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
circuits = ghz(n_qubits)
result = simulator.execute(circuits, n_samples)[0]  # 返回默认都是一个数组提高效率
# plot_histogram(error_result, filename = join(fig_path, 'ls_ghz'))


lm = LocalMitigator(n_qubits)
lm.characterize_M(protocol_results)
result_l = lm.mitigate(result, threshold = threshold * n_samples)

# lm = NonLocalMitigator(n_qubits, M =  np.kron(M[0], M[1]))
nlm = NonLocalMitigator(n_qubits)
nlm.characterize_M(protocol_results)
result_nl = nlm.mitigate(result)

noise_free_result = noise_free_simulator.execute(circuits)[0]
# total_count = sum(noise_free_result.values())
# noise_free_result = {
#     int(bitstring, base=2): count / total_count
#     for bitstring, count in noise_free_result.items()
# }

# '''应该还是有问题的'''
fidelity_plms = []
for group_size in range(1, n_qubits+1):
    plm = ParticalLocalMitigator(n_qubits)
    group = plm.random_group(group_size)  # n_qubits的时候是个nlm一样的
    # group = [[0], [1], [2]]
    plm.characterize_M(protocol_results, group)
    # print(plm.group2M)
    rm_result_partical_local = plm.mitigate(result, threshold = threshold * n_samples)
    # print(rm_result_partical_local)
    fidelity_plms.append(hellinger_fidelity(rm_result_partical_local, noise_free_result))

fidelity_mlms = []
score_mlms = []
for i in range(10):
    mlm = MultiStageMitigator(n_qubits, n_stages = 5)
    score = mlm.characterize_M(protocol_results, group_size = 3)
    result_m = mlm.mitigate(result, threshold = threshold * n_samples)
    
    fidelity_mlms.append(hellinger_fidelity(result_m, noise_free_result))
    score_mlms.append(score)
# print(result_m)

# error_result = {
#     int(bitstring, base=2): count / total_count
#     for bitstring, count in error_result.items()
# }

print(
    hellinger_fidelity(result_l, noise_free_result), 
    hellinger_fidelity(result_nl, noise_free_result),
    fidelity_plms,
    fidelity_mlms,
    hellinger_fidelity(result, noise_free_result)
)
# error_result = nls.execute(circuit)
# plot_histogram(error_result, filename = join(fig_path, 'nls_ghz'))
print('end')
'''先假设当前的是对的吧'''


'''TODO: 检验下NonLocalSimulator和LocalSimulator在Local的情况结果是不是一样的, 好像差不多'''


'''mitigator里面都换成位运算的'''