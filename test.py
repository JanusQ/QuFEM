from collections import defaultdict
import copy
import pickle
from benchmark import ghz
from correlation_analysis import PdBasedProtocolResults, construct_bayesian_network
from simulator import LocalSimulator, MeasurementAwareNonLocalSimulator, NonLocalSimulator, Simulator
from qiskit.visualization import plot_histogram
from config import fig_path, join
import os
import numpy as np

from qiskit.quantum_info import hellinger_fidelity

from mitigator.measurement_aware_mitigator import BayesianMitigator
from mitigator.partical_local_mitigator import ParticalLocalMitigator
from mitigator.protocol_circuits import EnumeratedProtocol, MeasuremtAwareEnumeratedProtocol
from mitigator.local_mitigator import LocalMitigator
from mitigator.nonlocal_mitigator import NonLocalMitigator
from utils import eval_plm

for n_qubits in range(2, 6): #range(2, 12):
    print('n_qubits =', n_qubits)
    
    simulator_path = f'./temp/simulator_{n_qubits}.pkl'
    # simulator = MeasurementAwareNonLocalSimulator(n_qubits, MeasurementAwareNonLocalSimulator.gen_random_M(n_qubits))
    # with open(simulator_path, 'wb') as file:
    #     pickle.dump(simulator, file)
        
    with open(simulator_path, 'rb') as file:
        simulator: MeasurementAwareNonLocalSimulator = pickle.load(file)
    # NonLocalSimulator
    
    protocol_path = f'./temp/protocol_{n_qubits}.pkl'
    
    # bitstings, protocol_circuits = MeasuremtAwareEnumeratedProtocol(n_qubits).gen_circuits()
    # protocol_results = {
    #     bitsting: status_count
    #     for bitsting, status_count in zip(bitstings, simulator.execute(protocol_circuits, n_samples = 10000,))
    # }
    # with open(protocol_path, 'wb') as file:
    #     pickle.dump(protocol_results, file)
        
    with open(protocol_path, 'rb') as file:
        protocol_results: dict = pickle.load(file)
        
    lm = LocalMitigator(n_qubits)
    lm.characterize_M(protocol_results)

    nlm = NonLocalMitigator(n_qubits)
    nlm.characterize_M(protocol_results)
    
    # mitigator_2 = BayesianMitigator(n_qubits)
    mitigator_10 = BayesianMitigator(n_qubits)
    # groups_2 = mitigator_2.random_group(group_size = 2)
    
    # _qubit = [q for q in range(n_qubits)]
    # for j in simulator.sub_groups:
    #     for k in j:
    #         _qubit.remove(k)
            
    # groups_2 = simulator.sub_groups + [_qubit]
    # mitigator_2.characterize_M(protocol_results, groups_2)
    
    # groups_10 = mitigator_10.random_group(group_size = 10)
    groups_10 = [list(range(n_qubits))]
    mitigator_10.characterize_M(protocol_results, groups_10)
    
    circuit = ghz(n_qubits)
    result = simulator.execute(circuit, n_samples = 1000)[0]
    
    # bitstings, protocol_circuits = EnumeratedProtocol(n_qubits).gen_circuits()
    # protocol_results = {
    #     bitsting: status_count
    #     for bitsting, status_count in zip(bitstings, simulator.execute(protocol_circuits, n_samples = 10000))
    # }

    
    # fixed_status_counts_2 = mitigator_2.mitigate(result, threshold = 0.1 )
    fixed_status_counts_10 = mitigator_10.mitigate(result, threshold = 0.1 )
    result_l = lm.mitigate(result, threshold = 1e-5 * 1000)
    result_nl = nlm.mitigate(result)
    
    # groud_truth = {'0'*n_qubits: 0.5, '1'*n_qubits: 0.5}
    # print('ghz fidelity:')
    # print('minimize_2:', hellinger_fidelity(groud_truth, fixed_status_counts_2))
    # print('minimize_10:', hellinger_fidelity(groud_truth, fixed_status_counts_10))
    # print('local:', hellinger_fidelity(groud_truth, result_l))
    # print('non-local:', hellinger_fidelity(groud_truth, result_nl))
    
    print('eval_plm:')
    # print('minimize_2:', eval_plm(mitigator_2, protocol_results))
    print('minimize_10:', eval_plm(mitigator_10, protocol_results))
    print('local:', eval_plm(lm, protocol_results))
    print('non-local:', eval_plm(nlm, protocol_results))
    print()
    
    
print('finish')