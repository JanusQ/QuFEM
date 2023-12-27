from ray_func import wait
import pickle
from mitigator.measurement_aware_mitigator import BayesianMitigator
from mitigator.multi_stage_mitigator import MultiStageMitigator
from utils import downsample_protocol_result
from simulator import Simulator
from dataloader import Dataloader
import ray
from utils import all_bitstrings
import numpy as np 

# n_qubits = 136
# down_protocol_result = pickle.load(file=open('dataset/all_task_id_136_result_new.pkl','rb'))

n_qubits = 131
down_protocol_result = pickle.load(file=open('dataset/all_task_id_131_result_new.pkl','rb'))

# n_qubits = 5
# down_protocol_result = pickle.load(file=open('dataset/5bit_quito.res_new','rb'))

# down_protocol_result = downsample_protocol_result(protocol_results,[0,1,2,4,5,6,7,14])

# for real_bitstring in down_protocol_result:
#     if real_bitstring[0] == real_bitstring[3] == '0':
#         print()

# bit_u=[24,63,78,101,104]
# bit_all=[i for i in range(136)]
# bit_c=list(set(bit_all).difference(set(bit_u)))
# protocol_results = pickle.load(file=open('dataset/all_task_id_136_result.pkl','rb'))
# down_protocol_result = downsample_protocol_result(protocol_results,bit_c)
# n_qubits = len(bit_c)


simulator = Simulator(n_qubits)


thresholds = []
scores = []
data_sizes = []
mitigators = []

@ray.remote
def eval(threshold, down_protocol_result):
    down_protocol_result = wait(down_protocol_result)
    
    print('threshold: ', threshold)
    dataloader = Dataloader(simulator)
    # dataloader = Dataloader()

    protocol_results_dataset = dataloader.get_data(eval = True, machine_data = down_protocol_result, threshold = threshold)

    print(threshold, 'finish dataloader.get_data', len(protocol_results_dataset))
    return threshold, len(protocol_results_dataset)

    # mitigator = MultiStageMitigator(n_qubits, n_stages = 3)
    # score = mitigator.characterize_M(protocol_results_dataset ,BasisMitigator = BayesianMitigator, multi_process = True, partation_methods = ['max-cut'])
    # print('threshold, score:', threshold, score, len(protocol_results_dataset))
    
    # return threshold, len(protocol_results_dataset), score, mitigator

futures = []
x = []
down_protocol_result_token = ray.put(down_protocol_result)

for i in np.linspace(2e-10,2e-4,10):
    threshold = i
    futures.append(eval.remote(threshold, down_protocol_result_token))
    # futures.append(eval(threshold, down_protocol_result))
    x.append(threshold)


for threshold, data_size in wait(futures):
    # scores.append(score)
    thresholds.append(threshold)
    data_sizes.append(data_size)
    # mitigators.append(mitigator)
    # print(threshold, score)

# 18
# [0.0008, 0.016800000000000002, 0.0328, 0.0488, 0.06480000000000001, 0.08080000000000001, 0.09680000000000001, 0.11280000000000001, 0.1288, 0.1448, 0.16080000000000003, 0.1768, 0.19280000000000003]
# [0.13718625827689857, 0.37446803199581796, 0.5047403201359225, 0.5113805281094274, 0.5093348592741388, 0.5113805281094272, 0.5082730681889841, 0.5113805281094272, 0.9332909328773618, 0.9332909328773619, 0.9332909328773618, 0.9332909328773618, 0.9283844776880742]

print(thresholds)
# print(scores)
print(data_sizes)

# with open('temp_data/threshold_136_mitigators.pkl', 'wb') as file:
#     pickle.dump((scores, thresholds, data_sizes, mitigators), file)

# import matplotlib.pyplot as plt
# fig ,ax = plt.subplots(figsize=(20,18))
# ax.plot(x, data_sizes)
# fig.savefig('thres_size.png')
