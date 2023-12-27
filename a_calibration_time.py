import pickle
from utils import downsample_protocol_result
from benchmark import ghz
from simulator import Simulator
from sim import Simulator as sim
from IBU import ibu
from m3_D import *
from mitigator.partical_local_mitigator_copy import ParticalLocalMitigator


protocol_results = pickle.load(file=open('dataset/data_131qubits.pickle','rb'))
for n_qubits in [18]:
    print("qubits:",n_qubits)
    measured_bit = [i for i in range(n_qubits)]

    circuits = ghz(n_qubits)
    sim_noise = Simulator(n_qubits)  
    sim_ideal = sim(n_qubits)
    error_result = sim_noise.execute(circuits)[0]
    ideal_result = sim_ideal.execute(circuits)[0]

    Mitigator = ParticalLocalMitigator(n_qubits)
    group = Mitigator.random_group(group_size = 2)
    Mitigator.characterize_M(groups=group)

    error_result = {bit:count*10000 for bit,count in error_result.items()}
    t1 = time.time()
    our = Mitigator.mitigate(stats_counts = error_result, threshold =1e-5*10000)
    print("QuFEM calibration time",(time.time()-t1)*2)


    group1 = Mitigator.random_group(group_size = 1)
    Mitigator.characterize_M(groups=group1)

    t2 = time.time()
    our = Mitigator.mitigate(stats_counts = error_result, threshold =1e-5*10000)
    print("CTMP calibration time",(time.time()-t1)*2)



    ibu_result = ibu(circuits,error_result,n_qubits,ideal_result)

    space = get_space(3,error_result,n_qubits)
    m3_result = sol(error_result,space)