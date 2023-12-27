from mitigator.measurement_aware_mitigator import BayesianMitigator
from mitigator.multi_stage_mitigator import MultiStageMitigator
import pickle
from qiskit.quantum_info.analysis import hellinger_fidelity
from utils import downsample_protocol_result,normalize_dict_values
from simulator import Simulator
from qbeep import qbeep
from sim import Simulator as sim
from circuit.dataset_loader import gen_algorithms
import mthree
from IBU import ibu
from qiskit.providers.fake_provider import FakeWashington




protocol_results = pickle.load(file=open('dataset/36_data.pickle','rb'))
sample = 10000


for n_qubit in [10]:
    print("qubit :",n_qubit)

    als = gen_algorithms(n_qubit, None, False)
    for alg in als:
        circuits = alg['qiskit_circuit']
        circuits.measure_all()
        print(alg['id'])
        simulator = Simulator(n_qubit)
        error_result = simulator.execute(circuits)[0]

        sim_ideal = sim(n_qubit) 
        ideal_result = sim_ideal.execute(circuits)[0]

        multiStageMitigator1 = MultiStageMitigator(n_qubit, n_stages = 2)
        measured_bit = [i for i in range(n_qubit)]
        protocol_results = downsample_protocol_result(protocol_results,measured_bit)
        groups = multiStageMitigator1.characterize_M(protocol_results, group_size = 2, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process=  True)  
            
        error_result = multiStageMitigator1.add_error(error_result,measured_bit,0)
        error_result = {bit:count*sample for bit,count in error_result.items()}
        print("没有校准 :",hellinger_fidelity(error_result,ideal_result))
        error_result_copy = error_result.copy()

        multi_result = multiStageMitigator1.mitigate(error_result, threshold = sample * 1e-5, measured_qubits = measured_bit)
        qbeep_result = qbeep(circuits,error_result)


        backend = FakeWashington()
        mit = mthree.M3Mitigation(backend)
        mit.cals_from_system(measured_bit)
        mthree_result = mit.apply_correction(error_result_copy, measured_bit, distance=3)
        mthree_result = normalize_dict_values(mthree_result)
        ibu_result = ibu(circuits,error_result_copy,n_qubit,ideal_result)

        print("QuFEM :",hellinger_fidelity(multi_result,ideal_result))
        print("M3 :",hellinger_fidelity(mthree_result,ideal_result))
        print("IBU :",max(hellinger_fidelity(ibu_result,ideal_result)))
        print("Q-BEEP :",hellinger_fidelity(qbeep_result,ideal_result))

