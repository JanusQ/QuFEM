import pickle
from utils import downsample_protocol_result
from benchmark import ghz
from simulator import Simulator
from sim import Simulator as sim
import mthree
from IBU import ibu
from qiskit.providers.fake_provider import FakeWashington
from mitigator.multi_stage_mitigator import MultiStageMitigator
from mitigator.measurement_aware_mitigator import BayesianMitigator

def L1(n_qbits,result,samples):
    x = 0
    for i in result.keys():
        if i == '0'*n_qbits or i == '1'*n_qbits:
            x+=abs((result[i]-0.5*samples))
        else:
            x+=abs(result[i])
    return (1*samples-0.5*x)/samples

def normalize_dict_values(input_dict):
    # 将字典中所有负数的值置为0
    for key, value in input_dict.items():
        if value < 0:
            input_dict[key] = 0

    # 归一化
    total = sum(input_dict.values())
    normalized_dict = {key: value / total for key, value in input_dict.items()}

    return normalized_dict

protocol_results = pickle.load(file=open('dataset/data_131qubits.pickle','rb'))

for n_qubits in [16]:
    print("qubits:",n_qubits)
    measured_bit = [i for i in range(n_qubits)]
    samples = 10000

    circuits = ghz(n_qubits)
    sim_noise = Simulator(n_qubits,0.035,0.048)  
    sim_ideal = sim(n_qubits)  

    error_result = sim_noise.execute(circuits)[0]
    ideal_result = sim_ideal.execute(circuits)[0]

    print("Raw fidelity  :",L1(n_qubits,error_result,samples))


    ibu_result = ibu(circuits,error_result,n_qubits,ideal_result)                 # max_iters,tol 需要更改
    print("Fidelity after ibu mitagation :",max(L1(n_qubits,ibu_result,1)))
    backend = FakeWashington()
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(measured_bit)
    mthree_result = mit.apply_correction(error_result, measured_bit, distance=3)
    mthree_result = normalize_dict_values(mthree_result)
    
    # print("Fidelity after mthree mitagation :",mthree_result.expval())
    print("Fidelity after mthree mitagation :",L1(n_qubits,mthree_result,1))


    multiStageMitigator = MultiStageMitigator(n_qubits, n_stages = 2)
    protocol_results = downsample_protocol_result(protocol_results,measured_bit)
    groups = multiStageMitigator.characterize_M(protocol_results, group_size = 2, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process=  True)  
    QuFEM_result = multiStageMitigator.mitigate(error_result, threshold = samples * 1e-5, measured_qubits = measured_bit)
    print("###################")