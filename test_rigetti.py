import pickle
from utils import downsample_protocol_result,to_dic
import random
from circuit.dataset_loader import gen_algorithms
from mitigator.multi_stage_mitigator import MultiStageMitigator
from mitigator.measurement_aware_mitigator import BayesianMitigator
from qiskit.quantum_info.analysis import hellinger_fidelity
from benchmark import ghz
from simulator import Simulator
from sim import Simulator as sim

protocol_results_131 = pickle.load(file=open('dataset/down_data.pickle','rb'))


sub_result_11 = []
for _ in range(12):
    numbers = [i for i in range(131)]
    n_qubits = 11
    measured_bit = random.sample(numbers, n_qubits)
    measured_bit.sort()
    protocol_results = downsample_protocol_result(protocol_results_131,measured_bit)
    multiStageMitigator1 = MultiStageMitigator(n_qubits, n_stages = 1)
    measured_bit = [i for i in range(n_qubits)]
    groups = multiStageMitigator1.characterize_M(protocol_results, group_size = 1, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process=  True)   #当小bit  threshold 设置大一点
    samples = 10000

    result_fid = {}
    als = gen_algorithms(n_qubits, None, False)
    for alg in als:
        circuits = alg['qiskit_circuit']
        circuits.measure_all()
        simulator = Simulator(n_qubits)  # 没有噪音
        sim_ideal = sim(n_qubits)  

        error_result = simulator.execute(circuits)[0]

        ideal_result = sim_ideal.execute(circuits)[0]


        print("门噪声  :",hellinger_fidelity(error_result,ideal_result))

        
        error_result = multiStageMitigator1.add_error(error_result,measured_bit,1e-6)
        error_result = {bit:count*samples for bit,count in error_result.items()}

        print("读取噪声 :",hellinger_fidelity(error_result,ideal_result))
        error_fidelity = hellinger_fidelity(error_result,ideal_result)

        multiStageMitigator2 = MultiStageMitigator(n_qubits, n_stages = 2)
        score2 = multiStageMitigator2.characterize_M(protocol_results, group_size = 2, partation_method = 'max-cut',BasisMitigator = BayesianMitigator,multi_process= True)
        multi_result = multiStageMitigator2.mitigate(error_result, threshold = samples * 1e-5, measured_qubits = measured_bit)
        multi_fidelity = hellinger_fidelity(multi_result,ideal_result)


        bayesianMitigator = BayesianMitigator(n_qubits)
        groups = bayesianMitigator.random_group(group_size = 1)
        bayesianMitigator.characterize_M(protocol_results,groups)
        N = bayesianMitigator.get_M(measured_bit)
        from mitigator.nonlocal_mitigator import NonLocalMitigator
        nonlocalmitigator = NonLocalMitigator(n_qubits,N)
        non_mitigation_result = nonlocalmitigator.mitigate(error_result)
        nonlocal_fidelity = hellinger_fidelity(non_mitigation_result,ideal_result)



        from mitigator.local_mitigator import LocalMitigator

        localMitigator = LocalMitigator(n_qubits)
        M = localMitigator.characterize_M(protocol_results)
        local_result = localMitigator.mitigate(error_result,threshold=0)
        local_fidelity = hellinger_fidelity(local_result,ideal_result)

        temp_result = [multi_fidelity/error_fidelity,nonlocal_fidelity/error_fidelity,local_fidelity/error_fidelity]
        sub_result_11.append(temp_result)


pickle.dump(sub_result_11, file=open('dataset/sub_result_11','wb'))



        
        



    

