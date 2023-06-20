from collections import defaultdict
import copy
import pickle
from benchmark import ghz
from correlation_analysis import PdBasedProtocolResults, construct_bayesian_network
from mitigator.measurement_aware_mitigator import MeasurementAwareMitigator
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

def eval_plm(plm: ParticalLocalMitigator, protocol_results):
        mitigated_protocol_results = {
            real_bitstring: {
                measured_bitstring: value * 1000  # 防止精度丢失
                for measured_bitstring, value in plm.mitigate(status_count, mask_bitstring = real_bitstring).items()  # 可能要乘以一个大的数字防止精度降低
            }
            for real_bitstring, status_count in protocol_results.items()
        }
  
        n_success = 0
        n_total = 0
        for real_bitstring, status_count in mitigated_protocol_results.items():
            n_total += sum(status_count.values())
            n_success += status_count[real_bitstring]
            
        # print(n_success/n_total)
        return n_success/n_total, mitigated_protocol_results
    
for i in range(2, 6): #range(2, 12):
    print('\n\n\n\n')
    simulator = MeasurementAwareNonLocalSimulator(i, MeasurementAwareNonLocalSimulator.gen_random_M(i))
    NonLocalSimulator
    bitstings, protocol_circuits = MeasuremtAwareEnumeratedProtocol(i).gen_circuits()
    protocol_results = {
        bitsting: status_count
        for bitsting, status_count in zip(bitstings, simulator.execute(protocol_circuits, n_samples = 1000,))
    }
    
    # bitstring_dataset, protocol_results_dataset, cor, uncertainty = get_min_bitstring(n_qubits = i)
    
    print(len(bitstings), 3**i)
    
    mitigator_2 = MeasurementAwareMitigator(i)
    mitigator_10 = MeasurementAwareMitigator(i)
    # groups_2 = mitigator_2.random_group(group_size = 2)
    
    _qubit = [q for q in range(i)]
    for j in simulator.sub_groups:
        for k in j:
            _qubit.remove(k)
            
    groups_2 = simulator.sub_groups + [_qubit]
    groups_10 = mitigator_10.random_group(group_size = 10)
    mitigator_2.characterize_M(protocol_results, groups_2)
    mitigator_10.characterize_M(protocol_results, groups_10)
    
    circuit = ghz(i)
    result = simulator.execute(circuit, n_samples = 1000)[0]
    
    fixed_status_counts_2 = mitigator_2.mitigate(result, threshold = 0.1 )
    fixed_status_counts_10 = mitigator_10.mitigate(result, threshold = 0.1 )
    # print(result)
    # print(fixed_status_counts)
    # print(result['0'*i], fixed_status_counts['0'*i], result['1'*i], fixed_status_counts['1'*i])
    print(len(bitstings), 3**i)
    
    bitstings, protocol_circuits = EnumeratedProtocol(i).gen_circuits()
    protocol_results = {
        bitsting: status_count
        for bitsting, status_count in zip(bitstings, simulator.execute(protocol_circuits, n_samples = 1000))
    }
    lm = LocalMitigator(i)
    lm.characterize_M(protocol_results)
    result_l = lm.mitigate(result, threshold = 1e-5 * 1000)
    # print(result_l['0'*i], result_l['1'*i])
    
    nlm = NonLocalMitigator(i)
    nlm.characterize_M(protocol_results)
    result_nl = nlm.mitigate(result)
    
    groud_truth = {'0'*i: 0.5, '1'*i: 0.5}
    print('minimize_2:', hellinger_fidelity(groud_truth, fixed_status_counts_2))
    print('minimize_10:', hellinger_fidelity(groud_truth, fixed_status_counts_10))
    print('local:', hellinger_fidelity(groud_truth, result_l))
    print('non-local:', hellinger_fidelity(groud_truth, result_nl))
print('finish')