from collections import defaultdict
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
        n_success += status_count[real_bitstring]
        
    # print(n_success/n_total)
    return n_success/n_total, mitigated_protocol_results
    
for i in range(18, 19): #range(2, 12):
    
    with open('protocal_result_fix.pkl','rb') as f:
        protocal_result = pickle.load(f)
    
    mitigator = BayesianMitigator(i)
    groups = mitigator.random_group(group_size = 2)
    mitigator.characterize_M(protocal_result, groups)
    
    score, mitigated_protocol_results = eval_plm(mitigator, protocal_result)
 
            
print('finish')