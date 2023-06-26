from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import pickle
from benchmark import ghz
from correlation_analysis import PdBasedProtocolResults, construct_bayesian_network
from mitigator.measurement_aware_mitigator import BayesianMitigator
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




def get_iter_protocol_results(cnt, bitstring_dataset):
    iter_res = {}
    i = 0
    for k, v in all_res.items():
        if k not in bitstring_dataset:
            i+=1
            iter_res[k] = v
            bitstring_dataset.append(k)
        if i == cnt:
            return iter_res
        
def task(n_qubits, real_bitstring, result, n_samples):
    local_tmp = np.zeros((n_qubits, n_qubits, 2, 3))
    local_uncertainty = np.zeros((n_qubits, n_qubits, 2, 3))
    for measure_bitstring, count in result.items():
        origin_bitstring = copy.deepcopy(real_bitstring)
        if '2' in origin_bitstring:
            origin_bitstring = origin_bitstring.replace('2','0')
            measure_bitstring = measure_bitstring.replace('2','0')
        if int(origin_bitstring, base=2) ^ int(measure_bitstring, base=2) != 0:
            error_qubits = bin(int(origin_bitstring, base=2) ^ int(measure_bitstring, base=2)).replace('0b','').zfill(n_qubits)#[::-1]
            for qubit_idx, is_error in enumerate(error_qubits):
                if is_error == '0':
                    continue
                
                for i in range(n_qubits):
                    if qubit_idx == i:
                        continue
                    local_tmp[qubit_idx][i][int(origin_bitstring[qubit_idx])][int(real_bitstring[i])] += count / n_samples
                    local_uncertainty[qubit_idx][i][int(origin_bitstring[qubit_idx])][int(real_bitstring[i])] += 1
    return local_tmp, local_uncertainty
    
def get_min_bitstring(n_qubits, n_samples = 1000, C = 0.005, hyper = 4, groups = None):
    # n_qubits = 8
    # n_samples = 1000
    # C = 0.005
    # simulator = MeasurementAwareNonLocalSimulator(n_qubits, MeasurementAwareNonLocalSimulator.gen_random_M(n_qubits))

    tmp = np.zeros((n_qubits, n_qubits, 2, 3))
    uncertainty = np.zeros((n_qubits, n_qubits, 2, 3))

    bitstring_dataset = []
    protocol_results_dataset = {}

    cnt = hyper * n_qubits
    
    iter = 0
    iter_score = []
    while True:
        iter += 1
        # bitstrings, protocol_circuits = MeasuremtAwareEnumeratedProtocol(n_qubits).gen_random_circuits(cnt, bitstring_dataset)
        # if len(bitstrings) == 0:
        #     break
        # protocol_results = {
        #     bitsting: status_count
        #     for bitsting, status_count in zip(bitstrings, simulator.execute(protocol_circuits, n_samples = n_samples))
        # }
        
        protocol_results = get_iter_protocol_results(cnt, bitstring_dataset)
        
        
        protocol_results_dataset = {**protocol_results, **protocol_results_dataset}
        
        executor = ThreadPoolExecutor()
        
        futures = []
        for real_bitstring, result in protocol_results.items():
            futures.append(executor.submit(task, n_qubits, real_bitstring, result, n_samples))
            
        for future in as_completed(futures):
            local_tmp, local_uncertainty = future.result()
            tmp += local_tmp
            uncertainty += local_uncertainty


        mitigator = BayesianMitigator(n_qubits)
        if groups is None:
            groups = mitigator.random_group(group_size = 2)
        mitigator.characterize_M(protocol_results_dataset ,groups)
        score, mitigated_protocol_results = eval_plm(mitigator, protocol_results_dataset)
        iter_score.append(score)
        
        cor = tmp / uncertainty
        if np.all((cor / np.sqrt(uncertainty))[uncertainty > 0] < C):
            break

        # M = np.nanmean(cor, axis = 3).mean(axis = 2)
        # array([[       nan, 0.02244027, 0.02157613, 0.02101489],
        #        [0.02065234,        nan, 0.01878396, 0.0199301 ],
        #        [0.01614292, 0.01453051,        nan, 0.01543383],
        #        [0.01275245, 0.01504462, 0.01389959,        nan]])
        # read_out_error = np.nanmean(M, axis = 1)

    
    return bitstring_dataset, protocol_results_dataset, cor, uncertainty, iter_score


def eval_plm(plm: ParticalLocalMitigator, protocol_results):
    mitigated_protocol_results = {
        real_bitstring: {
            measured_bitstring: value * 1000  # 防止精度丢失
            for measured_bitstring, value in plm.mitigate(status_count, mask_bitstring = real_bitstring, threshold = 1e-3).items()  # 可能要乘以一个大的数字防止精度降低
        }
        for real_bitstring, status_count in protocol_results.items()
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
    all_res = pickle.load(f)

bitstring_dataset, protocol_results_dataset, cor, uncertainty, iter_score = get_min_bitstring(n_qubits, C = 1e-8)
print('iter_count:', len(iter_score))
print('score:', iter_score)
print('dataset_size:', len(bitstring_dataset))
print('finish')