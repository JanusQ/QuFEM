from mitigator.measurement_aware_mitigator import BayesianMitigator
from mitigator.multi_stage_mitigator import MultiStageMitigator
import pickle
from qiskit.quantum_info.analysis import hellinger_fidelity
from utils import downsample_protocol_result
from simulator import Simulator
from sim import Simulator as sim
from benchmark import ghz

protocol_results = pickle.load(file=open('dataset/36_data.pickle','rb'))
sample = 10000
n_qubit = 8


circuits = ghz(n_qubit)
simulator = Simulator(n_qubit)
error_result = simulator.execute(circuits)[0]

sim_ideal = sim(n_qubit) 
ideal_result = sim_ideal.execute(circuits)[0]

multiStageMitigator1 = MultiStageMitigator(n_qubit, n_stages = 1)
measured_bit = [i for i in range(n_qubit)]
protocol_results = downsample_protocol_result(protocol_results,measured_bit)
groups = multiStageMitigator1.characterize_M(protocol_results, group_size = 2, partation_method = 'random', BasisMitigator = BayesianMitigator, multi_process=  True)  
error_result = multiStageMitigator1.add_error(error_result,measured_bit,0)
error_result = {bit:count*sample for bit,count in error_result.items()}
print("没有校准 :",hellinger_fidelity(error_result,ideal_result))

for iteration in [1,2,3,4]:
    multiStageMitigator = MultiStageMitigator(n_qubit, n_stages = iteration)
    for gp in [1,2,3,4,5,6]:
        group = multiStageMitigator.characterize_M(protocol_results, group_size = gp, partation_method = 'max-cut', BasisMitigator = BayesianMitigator, multi_process=  True)  
        multi_result = multiStageMitigator.mitigate(error_result, threshold = sample * 1e-5, measured_qubits = measured_bit)
        print("Iteration : ",iteration,"  group size : ",gp,"  fidelity : ",hellinger_fidelity(multi_result,ideal_result))

