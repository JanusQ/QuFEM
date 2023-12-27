#136 bit   memory and speed 10,18,27,65,136             local 不过滤
#18 bit:    阈值和准确度     数据量和准确度    hamming distance
#5 bit: 考虑读取和不考虑读取   multistage 和 真实 matrix 对比 local 真实

from mitigator.measurement_aware_mitigator import BayesianMitigator
from simulator import MeasurementAwareNonLocalSimulator
from benchmark import ghz
import pickle
from m3_D import get_space,sol
from simulator import  Simulator
from qiskit.quantum_info.analysis import hellinger_fidelity
from utils import downsample_protocol_result
import time
import random
import numpy as np

def result_r(num,n,samples):         #num:非零值数目,n:比特数,samples:采样数
    bitstring=[]
    count = np.random.random(num)
    count /= count.sum()
    count = list(count*samples)
    List = random.sample(range(0,10000),num)          
    for i in range(num):
        binary_num = str(bin(List[i])[2:])
        if len(binary_num)!=n:
            binary_num='0'*(n-len(binary_num))+binary_num
        bitstring.append(binary_num)
    result = { bitstring : count 
              for bitstring,count in zip(bitstring,count)
              }
    return result

n_qubits_measured = 131
simulator = MeasurementAwareNonLocalSimulator(n_qubits_measured, MeasurementAwareNonLocalSimulator.gen_random_M(n_qubits_measured))
n_samples = 10000
circuits = ghz(n_qubits_measured)
result = simulator.execute(circuits, n_samples)[0]      #没有mitigate的结果


# with open('down_data.pickle', 'wb') as f:
#     pickle.dump(down_protocol_result, f)

down_protocol_result = pickle.load(file=open('./dataset/down_data.pickle','rb'))
bayesianMitigator = BayesianMitigator(131)
groups = bayesianMitigator.random_group(group_size = 2)
bayesianMitigator.characterize_M(down_protocol_result,groups)
# n_samples = 10000

for n_qubits_measured in [8]:
    measured_bit=[i for i in range(n_qubits_measured)]     #测量的bit
    b1=time.time()
    beyesian_result = bayesianMitigator.mitigate(result,threshold=n_samples*1e-5,measured_qubits = measured_bit)     
    b2=time.time()
    print(n_qubits_measured,"beyesian mitigation:",b2-b1)

# import numpy as np
# space=get_space(2,result,n_qubits_measured)
# print(len(space))
# print(2**n_qubits_measured)
# space=tuple(space)

# P = np.random.rand(len(space),len(space))
# #P = bayesianMitigator.get_partial_M(measured_bit,space)
# t1=time.time()
# m3_mitigation=sol(P,result,space)            #D=2,采样次数为10000
# t2=time.time()
# print(t2-t1)

# print("no mitigate的fidelity:",hellinger_fidelity(result,noise_free_result))
# print("m3 mitigate的fidelity:",hellinger_fidelity(m3_mitigation,noise_free_result))

#                  8               10 bit         18 bit          27 bit          32 bit          34 bit              42 bit
# m3 空间大小      256               843           11566           51134           87325           126532              195631
# 全部空间         256              1024           262144        134217728       4294967296      17179869184        4398046511104
# speed      0.003705501556     0.0122115612    2.788094711     86.65966701      390.97406     1169.0272867679    3986.89165019989
 




#10bit 0.1270437240600586
#18bit 1.3838436603546143
#27bit 5.774710416793823
#65bit 108.64721751213074
#90bit 523.0092809200287


#非零值为100
#10bit 0.07091212272644043
#18bit 0.21317219734191895
#27bit 0.7642440795898438
#36bit 4.059460639953613
#48bit 12.89157223701477
#65bit 73.05849552154541
#90bit 277.39750480651855



