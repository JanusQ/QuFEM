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


    def get_iter_protocol_results(self, machine_data, cnt, filter = None ):
        
        if filter is None:
            iter_res = machine_data[:cnt]
            machine_data =  machine_data[cnt:]
        else:
            iter_res, new_machine_data = [], []
            for ele in machine_data:
                real_bitstring, status_count = ele 
                if real_bitstring[filter[0]] != filter[2] or real_bitstring[filter[1]] != filter[3]:
                    new_machine_data.append(ele)
                else:
                    iter_res.append(ele)
            new_machine_data += iter_res[cnt:]
            iter_res = iter_res[:cnt]
            machine_data = new_machine_data
        return iter_res, machine_data
    

        
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
    def get_data(self,  n_samples = 1, threshold = 1e-3, hyper = 1, groups = None, eval = False, machine_data = None):
        n_qubits = self.n_qubits
        simulator = self.simulator
    
        bitstring_dataset =  []
        protocol_results_dataset = []

        cnt = hyper * n_qubits
        
        qubit_errors = np.zeros(shape=n_qubits)
        qubit_count = 0
        states_error = np.zeros((n_qubits, n_qubits, 2, 3))
        states_count = np.zeros((n_qubits, n_qubits, 2, 3))
        states_datasize = np.zeros((n_qubits, n_qubits, 2, 3))

        filter = None
        
        step = 1
        kth_max = 0
        while True:
            
            if machine_data is not None:
                protocol_results, machine_data = self.get_iter_protocol_results(machine_data, 1000 if step == 0 else cnt, filter = filter)

                while len(protocol_results) == 0:
                    if len(machine_data) == 0:
                        print(f'当 threshold = ', threshold, '被薅空了')
                        return protocol_results_dataset
                    kth_max += 1

                    nan_count = np.sum(np.isnan(eq6))
                    filter = np.argsort(eq6,  axis = None )[-1-nan_count-kth_max]
                    filter = np.unravel_index(filter, eq6.shape)
                    print('new filter: ', filter)
                    protocol_results, machine_data = self.get_iter_protocol_results(machine_data, cnt, filter = filter)

                    
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

            protocol_results_dataset += protocol_results
            
            # for real_bitstring, status_count in protocol_results:
            for ele in protocol_results:
                real_bitstring, status_count  = ele
                meas_np, cnt_np = status_count
                qubit_count += np.sum(cnt_np)  # 固定值
                

                for q0 in range(n_qubits):
                    if real_bitstring[q0] == 2:
                         continue
                    for q1 in range(n_qubits):
                        states_count[q0][q1][real_bitstring[q0]][real_bitstring[q1]] += qubit_count
                        states_datasize[q0][q1][real_bitstring[q0]][real_bitstring[q1]] += 1

                    error_index = meas_np[:,q0] != real_bitstring[q0]
                    # error_meas_np = meas_np[error_index]
                    error_cnt_np = cnt_np[error_index]
                    
                    total_error_cnt_np = np.sum(error_cnt_np)
                    for q1 in range(n_qubits):
                        states_error[q0][q1][real_bitstring[q0]][real_bitstring[q1]] += total_error_cnt_np
                    
                
                        
            
            iter_qubit_errors = qubit_errors / qubit_count               
            iter_states_error = states_error / states_count

            for qubit in range(n_qubits):
                iter_states_error[qubit] -= iter_qubit_errors[qubit]
                
            eq6 = np.abs(iter_states_error)/states_datasize
                
            if np.nanmax(eq6) < threshold:
                break
            
            nan_count = np.sum(np.isnan(eq6))
            filter = np.argsort(eq6, axis = None )[-1-nan_count-kth_max]
            filter = np.unravel_index(filter, eq6.shape)
            
            # print(np.nanmax(eq6), threshold, filter, len(protocol_results_dataset), len(machine_data))
        
        return  protocol_results_dataset