from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from correlation_analysis import calculate_correlation
from mitigator.measurement_aware_mitigator import BayesianMitigator
from mitigator.partical_local_mitigator import ParticalLocalMitigator
from utils import load
from mitigator.protocol_circuits import MeasuremtAwareEnumeratedProtocol
class Dataloader():
    def __init__(self, simulator,):
        self.n_qubits = simulator.n_qubits
        self.simulator = simulator
    
    def get_iter_protocol_results(self, all_res, cnt, bitstring_dataset):
        iter_res = {}
        i = 0
        for k, v in all_res.items():
            if k not in bitstring_dataset:
                i+=1
                iter_res[k] = v
                bitstring_dataset.append(k)
            if i == cnt:
                return iter_res
        
    def eval_plm(self, plm: ParticalLocalMitigator, protocol_results):
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
            
        # print(n_success/n_total)
        return n_success/n_total, mitigated_protocol_results
            
    
    def get_data(self,  n_samples = 1000, threshold = 1e-6, hyper = 4, groups = None, eval = False, machine_data = None):
        n_qubits = self.n_qubits
        simulator = self.simulator
    
        tmp = np.zeros((n_qubits, n_qubits, 2, 3))
        uncertainty = np.zeros((n_qubits, n_qubits, 2, 3))

        bitstring_dataset = []
        protocol_results_dataset = {}

        cnt = hyper * n_qubits
        
        iter = 0
        stop = 0
        iter_score = []
        while stop < 2:
            
            if machine_data is not None:
                protocol_results = self.get_iter_protocol_results(machine_data, cnt, bitstring_dataset)
            else:
                bitstrings, protocol_circuits = MeasuremtAwareEnumeratedProtocol(n_qubits).gen_random_circuits(cnt, bitstring_dataset)
                if len(bitstrings) == 0:
                    break
                protocol_results = {
                    bitsting: status_count
                    for bitsting, status_count in zip(bitstrings, simulator.execute(protocol_circuits, n_samples = n_samples))
                }
            
            
            
            protocol_results_dataset = {**protocol_results, **protocol_results_dataset}
            
            executor = ProcessPoolExecutor()
            
            futures = []
            for real_bitstring, result in protocol_results.items():
                futures.append(executor.submit(calculate_correlation, n_qubits, real_bitstring, result, n_samples))
                
            for future in as_completed(futures):
                local_tmp, local_uncertainty = future.result()
                tmp += local_tmp
                uncertainty += local_uncertainty

            
            cor = tmp / uncertainty
            if np.all((cor / np.sqrt(uncertainty))[uncertainty > 0] < threshold):
                break


            if eval:
                mitigator = BayesianMitigator(n_qubits)
                try:
                    if groups is None:
                        groups = mitigator.random_group(group_size = 2)
                    mitigator.characterize_M(protocol_results_dataset ,groups)
                    
                except Exception as e:
                    print(e)
                    groups = None
                    continue
                    
                score, mitigated_protocol_results = self.eval_plm(mitigator, protocol_results_dataset)
                if len(iter_score) != 0 and (score - iter_score[-1]) < 1e-3:
                    stop += 1 
                print(f'qubits: {n_qubits}, iter: {iter}, score: {score}, datasize: {len(bitstring_dataset)}')
                iter_score.append(score)
                iter += 1
                
        return bitstring_dataset, protocol_results_dataset, cor, uncertainty, iter_score
