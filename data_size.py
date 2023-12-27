from dataloader import Dataloader
from simulator import Simulator
import pickle
from utils import to_dic,result_sim,downsample_protocol_result
from mitigator.multi_stage_mitigator import MultiStageMitigator
from mitigator.measurement_aware_mitigator import BayesianMitigator
from qiskit.quantum_info.analysis import hellinger_fidelity
from np_to_dic import np_to_dic
from circuit.dataset_loader import gen_algorithms
from sim import Simulator as sim
import random

def random_subset(original_dict, subset_size):
    # 从原始字典的键中随机选取指定数量的键
    random_keys = random.sample(list(original_dict.keys()), min(subset_size, len(original_dict)))

    # 创建新字典，包含选取的键值对
    new_dict = {key: original_dict[key] for key in random_keys}

    return new_dict


qubits = 36
simulator = Simulator(qubits)

protocol_result = pickle.load(file=open('dataset/36_np.pickle','rb'))

dataloader = Dataloader(simulator)
simulator = Simulator(qubits)
protocol_result_golden = pickle.load(file=open('dataset/36_data.pickle','rb'))

# for th in [1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7,9.9e-8]:
# for th in [1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,6e-7,7e-7,8e-7]:

#     protocol_results_dataset = dataloader.get_data(eval = True, machine_data = protocol_result, threshold = th)
#     print("data size:",len(protocol_results_dataset))
#     protocol_result_sec = np_to_dic(protocol_results_dataset)
#     protocol_result_random = random_subset(protocol_result_golden, len(protocol_results_dataset))


n_qubit = 10
als = gen_algorithms(n_qubit, None, False)
for alg in als:
    circuits = alg['qiskit_circuit']
    circuits.measure_all()
    print(alg['id'])
    simulator = Simulator(n_qubit)
    error_result = simulator.execute(circuits)[0]
    sim_ideal = sim(n_qubit) 
    ideal_result = sim_ideal.execute(circuits)[0]
    # print("门噪声:",hellinger_fidelity(error_result,ideal_result))

    multiStageMitigator1 = MultiStageMitigator(n_qubit, n_stages = 1)
    multiStageMitigator2 = MultiStageMitigator(n_qubit, n_stages = 1)
    multiStageMitigator3 = MultiStageMitigator(n_qubit, n_stages = 1)
    measured_bit = [i for i in range(n_qubit)]

    protocol_result_golden = downsample_protocol_result(protocol_result_golden,measured_bit)

    group1 = multiStageMitigator1.characterize_M(protocol_result_golden, group_size = 2, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process= True)  
    error_result = multiStageMitigator1.add_error(error_result,measured_bit,0)
    error_result1 = error_result.copy()
    error_result2 = error_result.copy()

    # print("读取噪声:",hellinger_fidelity(error_result,ideal_result))
    mitigate_result1 = multiStageMitigator1.mitigate(error_result, threshold = 0, measured_qubits = measured_bit)
    print("golden:",hellinger_fidelity(mitigate_result1,ideal_result))

    for th in [4.14e-7,4.10e-7,4.06e-7]:
        protocol_results_dataset = dataloader.get_data(eval = True, machine_data = protocol_result, threshold = th)
        print("data size:",len(protocol_results_dataset))
        protocol_result_sec = np_to_dic(protocol_results_dataset)
        protocol_result_random = random_subset(protocol_result_golden, len(protocol_results_dataset))

        protocol_result_sec = downsample_protocol_result(protocol_result_sec,measured_bit)



        group2 = multiStageMitigator2.characterize_M(protocol_result_sec, group_size = 2, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process= True)  
        mitigate_result2 = multiStageMitigator2.mitigate(error_result1, threshold = 0, measured_qubits = measured_bit)
        print("sec:",hellinger_fidelity(mitigate_result2,ideal_result))

        group3 = multiStageMitigator3.characterize_M(protocol_result_random, group_size = 2, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process= True)  
        mitigate_result3 = multiStageMitigator3.mitigate(error_result2, threshold = 0, measured_qubits = measured_bit)
        print("random:",hellinger_fidelity(mitigate_result3,ideal_result))







# protocol_result2 = pickle.load(file=open('read_test/readout_error_mitigation-test/dataset/36_data.pickle','rb'))
# multiStageMitigator1 = MultiStageMitigator(n_qubits, n_stages = 1)
# score1 = multiStageMitigator1.characterize_M(protocol_result2, group_size = 1, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process= True)
# error_result = multiStageMitigator1.add_error(noise_free_result,measured_bit,1e-6)
# error_result = {bit:count*samples for bit,count in error_result.items()}
# dataloader = Dataloader(simulator)
# for i in range(1):     
#     threshold = 1e-5#1e-6 +5e-6*i
#     protocol_results_dataset = dataloader.get_data(eval = True, threshold = threshold, machine_data = protocol_result)
#     protocol_results_dataset = to_dic(protocol_results_dataset)
#     print(len(protocol_results_dataset))
#     multiStageMitigator1 = MultiStageMitigator(n_qubits, n_stages = 1)
#     score2 = multiStageMitigator1.characterize_M(protocol_results_dataset, group_size = 2, partation_method = 'max-cut', BasisMitigator = BayesianMitigator, multi_process= True)
#     mitigation_result1 = multiStageMitigator1.mitigate(error_result, threshold = samples*1e-5, measured_qubits = measured_bit)
#     print(hellinger_fidelity(mitigation_result1,noise_free_result))