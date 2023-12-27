from mitigator.measurement_aware_mitigator import BayesianMitigator
from mitigator.multi_stage_mitigator import MultiStageMitigator
import pickle
from m3_D import get_space,sol
from qiskit.quantum_info.analysis import hellinger_fidelity
from utils import downsample_protocol_result
import time
import numpy as np
from utils import result_sim
from simulator import Simulator
from benchmark import ghz
from qbeep import qbeep
from sim import Simulator as sim
from circuit.dataset_loader import gen_algorithms
from mitigator.partical_local_mitigator import ParticalLocalMitigator


protocol_results_131 = pickle.load(file=open('/home/zhanghanyu/read_test/readout_error_mitigation-test/dataset/new_data_18.pickle','rb'))

for n_qubit in [10]:
    print("qubit :",n_qubit)

    als = gen_algorithms(n_qubit, None, False)
    for alg in als:
        circuits = alg['qiskit_circuit']
        circuits.measure_all()
        print(alg['id'])
        simulator = Simulator(n_qubit)
        result = simulator.execute(circuits)[0]

        # sim_ideal = sim(n_qubit) 
        # ideal_result = sim_ideal.execute(circuits)[0]

        # multiStageMitigator1 = MultiStageMitigator(n_qubit, n_stages = 2)
        # measured_bit = [i for i in range(n_qubit)]
        # protocol_results = downsample_protocol_result(protocol_results_131,measured_bit)
        # groups = multiStageMitigator1.characterize_M(protocol_results, group_size = 2, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process=  True)  
            
        # result = multiStageMitigator1.add_error(result,measured_bit,0)
        result = {bit:count*10000 for bit,count in result.items()}
        # print("没有校准 :",hellinger_fidelity(result,ideal_result))
        # copied_dict = result.copy()

        # multi_result1 = multiStageMitigator1.mitigate(copied_dict, threshold = 5000 * 1e-5, measured_qubits = measured_bit)
        # print("FEFA :",hellinger_fidelity(multi_result1,ideal_result))


        # multiStageMitigator2 = ParticalLocalMitigator(n_qubit)
        # gp = multiStageMitigator2.random_group(group_size = 4)
        # multiStageMitigator2.characterize_M(protocol_results, gp)
        # multi_result =multiStageMitigator2.mitigate(result)
        # print("JigSaw :",hellinger_fidelity(multi_result,ideal_result))

        # multi_result = {bit:count*10000 for bit,count in multi_result.items()}
        result_qbeep_fefa = qbeep(circuits,result)
        # print("FEFA + QBEEP :",hellinger_fidelity(result_qbeep_fefa,ideal_result))

