import numpy as np
from tqdm import tqdm
import numpy as np
from itertools import combinations
import time
def hamming_distance(str1,str2):
    str1_arry = [i for i in str1]
    str2_arry = [i for i in str2]
    ham_distance=0
    for i in range(len(str1_arry)):
        if str1_arry[i]!=str2_arry[i]:
            ham_distance=ham_distance+1
    return ham_distance


def bitstring_verse(bitstring,verse):
    bitstring = list(bitstring)
    for i in verse:
        tmp = str(int(bitstring[i])^1)
        bitstring[i] = tmp
    bitstring = ''.join(bitstring)

    return bitstring

def get_space(D,status_counter,n):   
    M_space = set()
    itmlist = [i for i in range(n)]
    comb=[]
    for d in range(1,D):
        comb = comb+list(combinations(itmlist,d))

    for measured_bitstring in tqdm(status_counter):
        bitstring_space=[]
        for index in comb:
            new_bitstring = bitstring_verse(measured_bitstring,index)
            bitstring_space.append(new_bitstring)
        M_space = M_space.union(bitstring_space)
    M_space=list(M_space)
    M_space.sort()
    return M_space

def sol(status_counter,M_space):
    M = np.random.rand(len(M_space), len(M_space))
    V = {
        bitstring: status_counter[bitstring] if bitstring in status_counter else 0
        for bitstring in M_space
    }

    Y=[]
    for key, value in V.items():
        Y.append(value)

    Y=np.array(Y)

    t1 = time.time()
    r = np.linalg.solve(M, Y) 
    print("m3 calibration time",time.time()-t1)

    count=0
    result_mitigation = V.copy()
    for key, value in V.items():
        if r[count]<0:
            result_mitigation[key]=0
        else:
            result_mitigation[key]=int(r[count])
        count=count+1

    x = sum(result_mitigation.values())
    for key in result_mitigation:
        result_mitigation[key]=result_mitigation[key]/x

    return result_mitigation

'''
from qiskit.quantum_info.analysis import hellinger_fidelity

import time
import pickle
from mitigator.measurement_aware_mitigator import BayesianMitigator
from dataloader import Dataloader
from simulator import MeasurementAwareNonLocalSimulator
from simulator import  Simulator

n_qubits=18
n_qubits_measured=8
protocol_results = pickle.load(file=open('/home/tsw/workspace/readout_error_mitigation-test/dataset/protocal_result_18bit.pkl','rb'))

simulator = MeasurementAwareNonLocalSimulator(n_qubits, MeasurementAwareNonLocalSimulator.gen_random_M(n_qubits))
n_samples = 5000
circuits = ghz(n_qubits_measured)
result = simulator.execute(circuits, n_samples)[0]


noise_free_simulator = Simulator(n_qubits_measured)
noise_free_result = noise_free_simulator.execute(circuits)[0]




loader = Dataloader(simulator)
bitstring_dataset, protocol_results_dataset, cor, uncertainty, iter_score = loader.get_data(eval= True)


mitigator = BayesianMitigator(n_qubits)
groups = mitigator.random_group(group_size = 2)
mitigator.characterize_M(protocol_results,groups)

space=get_y(2,result)
print(len(space))
print(2**n_qubits)
space=tuple(space)
M=mitigator.get_M_measured(space)
t1=time.time()
result_mitigation=sol(M,result,space)
t2=time.time()
print(t2-t1)

print(hellinger_fidelity(result,noise_free_result))
print(hellinger_fidelity(result_mitigation,noise_free_result))


space=get_y(2,result)
print(len(space))
space=tuple(space)
M=mitigator.get_M_measured(space)

result_mitigation=sol(M,result,space)
'''