from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from mitigator.measurement_aware_mitigator import BayesianMitigator
from mitigator.partical_local_mitigator import ParticalLocalMitigator
from ray_func import wait
from utils import load
from mitigator.protocol_circuits import MeasuremtAwareEnumeratedProtocol
from mitigator.multi_stage_mitigator import MultiStageMitigator
import ray

def hamming_distance(string1, string2):
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter



class Dataloader():
    def __init__(self, simulator = None):
        self.n_qubits = simulator.n_qubits
        self.simulator = simulator


    def get_iter_protocol_results(self, all_res, cnt, bitstring_dataset, filter = None ):
        iter_res = {}
        i = 0
        for k, v in all_res.items():
            if k not in bitstring_dataset:
                if filter is not None:
                    if int(k[filter[0]]) != filter[2] or int(k[filter[1]]) != filter[3]:
                        continue
                i+=1
                iter_res[k] = v
                bitstring_dataset.append(k)
            if i == cnt:
                return iter_res
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
            
    #threshold = 1e-6
    def get_data(self,  n_samples = 1, threshold = 1e-3, hyper = 4, groups = None, eval = False, machine_data = None):
        n_qubits = self.n_qubits
        simulator = self.simulator
    
        tmp = np.zeros((n_qubits, n_qubits, 2, 3))
        uncertainty = np.zeros((n_qubits, n_qubits, 2, 3))

        bitstring_dataset = []
        protocol_results_dataset = {}

        cnt = hyper * n_qubits
        
        qubit_errors = np.zeros(shape=n_qubits)
        qubit_count = np.zeros(shape=n_qubits)
        states_error = np.zeros((n_qubits, n_qubits, 2, 3))
        states_count = np.zeros((n_qubits, n_qubits, 2, 3))
        states_datasize = np.zeros((n_qubits, n_qubits, 2, 3))

        filter = None
        while True:
            
            if machine_data is not None:
                protocol_results = self.get_iter_protocol_results(machine_data, cnt, bitstring_dataset, filter = filter)

                while len(protocol_results.values()) == 0:
                    if np.nanmax(eq6) == 0:
                        print(f'当 threshold = ', threshold, '被薅空了')
                        return bitstring_dataset, protocol_results_dataset, eq6, uncertainty
                    
                    eq6[filter] = 0
                    filter = np.nanargmax(eq6)
                    filter = np.unravel_index(filter, eq6.shape)
                    print('new filter: ', filter)
                    protocol_results = self.get_iter_protocol_results(machine_data, cnt, bitstring_dataset, filter = filter)

                    
            else:
                bitstrings, protocol_circuits = MeasuremtAwareEnumeratedProtocol(n_qubits).gen_random_circuits(cnt, bitstring_dataset, filter = filter)
                if len(bitstrings) == 0:
                    break
                protocol_results = {
                    bitsting: status_count
                    for bitsting, status_count in zip(bitstrings, simulator.execute(protocol_circuits, n_samples = n_samples))
                }
            
            # if protocol_results is None:
            #     print(protocol_results_dataset)

            protocol_results_dataset = {**protocol_results, **protocol_results_dataset}
            
            
            
            for real_bitstring, status_count in protocol_results.items():
                for measured_bitstring, count in status_count.items():
                    for qubit in range(n_qubits):
                        if real_bitstring[qubit] != measured_bitstring[qubit]:
                            qubit_errors[qubit] += count
                        qubit_count[qubit] += count
                        
            iter_qubit_errors = qubit_errors / qubit_count
            
            # [
            #     (real, [mea, count])
            # ]

            
            for real_bitstring, status_count in protocol_results.items():
                for measure_bitstring, count in status_count.items():
                    for qubit1 in range(n_qubits):
                        if real_bitstring[qubit1] == '2':
                            continue
                        
                        is_error = real_bitstring[qubit1] != measure_bitstring[qubit1]
                        for qubit2 in range(n_qubits):
                            if is_error:
                                states_error[qubit1][qubit2][int(real_bitstring[qubit1])][int(real_bitstring[qubit2])] += count
                            states_count[qubit1][qubit2][int(real_bitstring[qubit1])][int(real_bitstring[qubit2])] += count

                            states_datasize[qubit1][qubit2][int(real_bitstring[qubit1])][int(real_bitstring[qubit2])] += 1
                        
            iter_states_error = states_error / states_count

            for qubit in range(n_qubits):
                iter_states_error[qubit] -= iter_qubit_errors[qubit]
                
            eq6 = np.abs(iter_states_error)/states_datasize
                
            
            if np.nanmax(eq6) < threshold:
                break
            
            unsatisfy_states = np.nanargmax(eq6)
            unsatisfy_states = np.unravel_index(unsatisfy_states, eq6.shape)

            print(np.nanmax(eq6), threshold, unsatisfy_states) 
            filter = unsatisfy_states
        
        return bitstring_dataset, protocol_results_dataset, eq6, uncertainty