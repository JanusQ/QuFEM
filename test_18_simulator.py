from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import pickle
from benchmark import ghz
from correlation_analysis import PdBasedProtocolResults, construct_bayesian_network
from mitigator.measurement_aware_mitigator import BayesianMitigator
from mitigator.multi_stage_mitigator import MultiStageMitigator
from mitigator.partical_local_mitigator import ParticalLocalMitigator
from simulator import LocalSimulator, MeasurementAwareNonLocalSimulator, NonLocalSimulator, Simulator
from qiskit.visualization import plot_histogram
from config import fig_path, join
import os
import numpy as np
from mitigator.protocol_circuits import EnumeratedProtocol, MeasuremtAwareEnumeratedProtocol
from mitigator.local_mitigator import LocalMitigator
from mitigator.nonlocal_mitigator import NonLocalMitigator
from qiskit.quantum_info import hellinger_fidelity
from qiskit.visualization import plot_histogram
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import downsample

# 要让其他的mitigator也支持部分测量的
def eval_plm(plm: ParticalLocalMitigator, protocol_results):
    mitigated_protocol_results = {
        real_bitstring: {
            measured_bitstring: value * 1000  # 防止精度丢失
            for measured_bitstring, value in plm.mitigate(status_count, mask_bitstring = real_bitstring, threshold = 1e-8).items()  # 可能要乘以一个大的数字防止精度降低
        }
        for real_bitstring, status_count in tqdm(protocol_results.items())
    }

    n_success = 0
    n_total = 0
    for real_bitstring, status_count in mitigated_protocol_results.items():
        n_total += sum(status_count.values())
        if real_bitstring in status_count:
            n_success += status_count[real_bitstring]
        
    print(n_success/n_total)
    return n_success/n_total, mitigated_protocol_results
    

n_qubits = 11
with open(f'protocal_result_{n_qubits}bit.pkl','rb') as f:
    all_protocol_results = pickle.load(f)
    
n_qubits = 5
bitstrings, protocol_circuits = MeasuremtAwareEnumeratedProtocol(n_qubits).gen_circuits()
simulator = MeasurementAwareNonLocalSimulator(n_qubits, MeasurementAwareNonLocalSimulator.gen_random_M(n_qubits))
all_protocol_results = {
    bitstring: status_count
    for bitstring, status_count in zip(bitstrings, simulator.execute(protocol_circuits, n_samples = 1000))
}

mamlm = MultiStageMitigator(n_qubits, n_stages = 4)
score = mamlm.characterize_M(all_protocol_results, group_size = 2, BasisMitigator=BayesianMitigator)
# error_result =  mamlm.add_error({
#     '0'*n_qubits: 1
# }, measured_qubits = list(range(n_qubits)))
# plot_histogram(error_result)
# fig, ax = plt.subplots()
# plot_histogram(stats_count, title = real_bitstring, ax = ax)
# fig.savefig(os.path.join('temp/fig/ibmq_manila_protocols', real_bitstring + '.svg'))
# plt.close()
# plt.show()
for real_bitstring, status_count in all_protocol_results.items():
    measured_qubits = [qubit for qubit, bit in enumerate(real_bitstring) if bit != '2']
    mitigated_result = mamlm.mitigate(status_count, measured_qubits = measured_qubits, threshold = 1e-8)
    error_result =  mamlm.add_error(mitigated_result, measured_qubits = measured_qubits)
    print(status_count[real_bitstring]/sum(status_count.values()), mitigated_result[real_bitstring], hellinger_fidelity(error_result, mitigated_result))
    

mitigator = BayesianMitigator(n_qubits)
groups = mitigator.random_group(group_size = 2)
mitigator.characterize_M(all_protocol_results ,groups)
score, mitigated_protocol_results = eval_plm(mitigator, all_protocol_results)



print('finish')