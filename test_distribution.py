from utils import result_sim
from mitigator.partical_local_mitigator_copy import ParticalLocalMitigator
import time 
import matplotlib.pyplot as plt
from benchmark import ghz
from simulator import Simulator
samples = 10000
num_nonzero = 200
from circuit.dataset_loader import gen_algorithms
# 每个时间都存一下

# p[0,1]-bitstring-时间的影响
# 

# 200,300,400,500,600,700,
for n_qubit in [400]:#104,112,120,128,136
    result_ideal = result_sim(num_nonzero, n_qubit, samples,"laplace")     #laplace,norm,uniform
    # result_idea2 = result_sim(num_nonzero, n_qubit, samples,"norm")
    # result_idea3 = result_sim(num_nonzero, n_qubit, samples,"uniform")

        # circuits = ghz(n_qubit)
        # circuits = alg['qiskit_circuit']
        # circuits.measure_all()
        # simulator = Simulator(n_qubit)  # 没有噪音
    
        # error_result = simulator.execute(circuits)[0]
    multiStageMitigator = ParticalLocalMitigator(n_qubit)
    group1 = multiStageMitigator.random_group(group_size=1)
    multiStageMitigator.characterize_M(group1)
    t1 = time.time()
    result_1 = multiStageMitigator.mitigate(result_ideal,threshold = samples*1e-5, max_basis_size=n_qubit*3)
    t2 = time.time()
    result_1 = {bit : count*samples for bit,count in result_1.items() }#if '-' not in bit}#if '-' not in bit}
    t3 = time.time()
    result_1 = multiStageMitigator.mitigate(result_1,threshold = samples*1e-5, max_basis_size=n_qubit*3)
    t4 = time.time()
    # result_2 = multiStageMitigator.mitigate(result_idea2,threshold = samples*1e-5)
    # t3 = time.time()
    # result_3 = multiStageMitigator.mitigate(result_idea3,threshold = samples*1e-5)
    # t4 = time.time()
    print("iter 1:",t2-t1)
    print("iter 1:",t4-t3)
    # print(t3-t2)
    # print(t4-t3)
    # result_1 = {bit:count*10000 for bit,count in result_1.items() if "-" not in bit and count>1e-5}
    # result_2 = {bit:count*10000 for bit,count in result_2.items() if "-" not in bit and count>1e-5}
    # result_3 = {bit:count*10000 for bit,count in result_3.items() if "-" not in bit and count>1e-5}
    # group2 = multiStageMitigator.random_group(group_size=1)
    # multiStageMitigator.characterize_M(group2)
    # t5 = time.time()
    # result1 = multiStageMitigator.mitigate(result_1,threshold = samples*1e-5)
    # t6 = time.time()
    # result2 = multiStageMitigator.mitigate(result_2,threshold = samples*1e-5)
    # t7 = time.time()
    # result3 = multiStageMitigator.mitigate(result_3,threshold = samples*1e-5)
    # t8 = time.time()
    # print((t4-t1+t8-t5)/3)
    print("######")





# 200,300,400,500,600,700
# laplace    35.406,616.72216
# norm       31.911,576.22974
# uniform    29.776,369.21449

# bit = [128,144,160,176,192]
# laplace = [2.50913,4.52052,6.66199,8.972611,15.174934]
# norm = [3.38547,2.54353,4.33939,12.25513,14.9603298]
# uniform = [6.26334,11.0532,15.9068,18.23341,20.802567]
# average = [(laplace[i]+norm[i]+uniform[i])/3 for i in range(5)] 
# fig, ax = plt.subplots()    
# ax.scatter(bit, laplace)
# ax.plot(bit, laplace ,markersize = 6,linewidth = 2, label='laplace',marker = '^' )

# ax.scatter(bit, norm)
# ax.plot(bit, norm ,markersize = 6,linewidth = 2, label='norm',marker = '^' )

# ax.scatter(bit, uniform)
# ax.plot(bit, uniform ,markersize = 6,linewidth = 2, label='uniform',marker = '^' )

# ax.scatter(bit, average)
# ax.plot(bit, average ,markersize = 6,linewidth = 2, label='average',marker = '^' )
# ax.set_xlabel('quibts')
# ax.set_ylabel('time')
# ax.legend() # 添加图例
# plt.title("distribution")
# plt.show()
# plt.savefig("read_test/readout_error_mitigation-test/pic/distribution.png")
# plt.savefig("read_test/readout_error_mitigation-test/pic/distribution.svg")










