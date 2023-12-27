from mitigator.measurement_aware_mitigator import BayesianMitigator
from mitigator.multi_stage_mitigator import MultiStageMitigator
import pickle
from qiskit.quantum_info.analysis import hellinger_fidelity
from utils import downsample_protocol_result
import numpy as np
from simulator import Simulator
from sim import Simulator as sim
from circuit.dataset_loader import gen_algorithms


protocol_results = pickle.load(file=open('dataset/data_79qubits.pickle','rb'))

for n_qubit in [10]:

    als = gen_algorithms(n_qubit, None, False)
    for alg in als:
        circuits = alg['qiskit_circuit']
        circuits.measure_all()
        print(alg['id'])
        simulator = Simulator(n_qubit)
        error_result = simulator.execute(circuits)[0]
        sim_ideal = sim(n_qubit) 
        ideal_result = sim_ideal.execute(circuits)[0]
        print("门噪声:",hellinger_fidelity(error_result,ideal_result))

        multiStageMitigator1 = MultiStageMitigator(n_qubit, n_stages = 1)
        measured_bit = [i for i in range(n_qubit)]
        protocol_results = downsample_protocol_result(protocol_results,measured_bit)
        group1 = multiStageMitigator1.characterize_M(protocol_results, group_size = 10, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process= True)  
        error_result = multiStageMitigator1.add_error(error_result,measured_bit,0)
        print("读取噪声:",hellinger_fidelity(error_result,ideal_result))

        group1 = multiStageMitigator1.characterize_M(protocol_results, group_size = 2, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process= True)
        mitigate_result = multiStageMitigator1.add_error(error_result,measured_bit,0)
        print("random:",hellinger_fidelity(mitigate_result,ideal_result))

        group2 = multiStageMitigator1.characterize_M(protocol_results, group_size = 2, partation_method = 'max-cut', BasisMitigator = BayesianMitigator, multi_process= True)
        mitigate_result = multiStageMitigator1.add_error(error_result,measured_bit,0)
        print("max_cut:",hellinger_fidelity(mitigate_result,ideal_result))
