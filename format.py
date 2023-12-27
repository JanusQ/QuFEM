# [
#     (real, [mea, count])
# ]
import numpy as np 
import pickle 
import tqdm
import ray
from ray_func import wait

from utils import downsample_protocol_result


# n_qubits = 136
# down_protocol_result = pickle.load(file=open('dataset/all_task_id_136_result.pkl','rb'))

# bit_u=[24,63,78,101,104]
# bit_all=[i for i in range(136)]
# bit_c=list(set(bit_all).difference(set(bit_u)))
# protocol_results = pickle.load(file=open('dataset/all_task_id_136_result.pkl','rb'))
# down_protocol_result = downsample_protocol_result(protocol_results,bit_c)
# n_qubits = len(bit_c)
protocol_results = pickle.load(file=open('dataset/data_131qubits.pickle','rb'))
# down_protocol_result = downsample_protocol_result(protocol_results,[i for i in range(8)])
# n_qubits = 8

@ray.remote
def fun(state_cnt):
    meas_list, cnt_list = [], []
    # meas_list, cnt_list = np.zeros(shape=(n_qubits,len(state_cnt)), dtype=np.int8), np.zeros(shape=(len(state_cnt)), dtype=np.int64)
    for i, (meas, cnt) in enumerate(state_cnt.items()):
        # meas_list[i] = np.array(list(meas)).astype(np.int8)
        # cnt_list[i] = cnt
        
        meas = np.array(list(meas)).astype(int)
        meas_list.append(meas)
        cnt_list.append(cnt)
    
    # return [meas_list, cnt_list]
    meas_np = np.array(meas_list)
    cnt_np = np.array(cnt_list)
    return [meas_np, cnt_np]


new_format = []

futures = []
reals = []
for real, state_cnt in tqdm.tqdm(protocol_results.items()):
    real = np.array(list(real)).astype(int)
    futures.append(fun.remote(state_cnt))
    #futures.append(fun(state_cnt))
    reals.append(real)

for real, future in tqdm.tqdm(zip(reals, futures)):
    new_format.append([real, wait(future)])

pickle.dump(new_format, file=open('dataset/np_131bit.pickle','wb'))
print('finish')