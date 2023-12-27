import copy
from functools import lru_cache
import ray
import threading
import inspect
import uuid
import concurrent.futures
from concurrent.futures._base import Future
from inspect import isgeneratorfunction
from collections import defaultdict
import numpy as np
import random


def to_bitstring(integer, n_qubits):
    measure_bitstring = bin(integer).replace('0b', '')
    measure_bitstring = (n_qubits - len(measure_bitstring)) * '0' + measure_bitstring
    return measure_bitstring

def matrix_distance_squared(A: np.array, B: np.array):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    return np.abs(1 - np.abs(np.sum(np.multiply(A, np.conj(B)))) / A.shape[0])


def _to_str(): 
    return

# str -> int, int -> str
def decimal(value, convert_type, base = 2):
    if convert_type == 'str':
        str_value = ''
        while value != 0:
            str_value = str(value % base) + str_value
            value //= base
        return str_value
    elif convert_type == 'int':
        int_value = 0
        for bit_pos, bit_vlaue in enumerate(value):
            bit_pos = len(value) - bit_pos - 1
            int_value += int(bit_vlaue) * (base ** bit_pos)
        return int_value

    raise Exception('unkown convert_type', convert_type)

@lru_cache
def all_bitstrings(n_qubits, base = 2):
    assert n_qubits < 30
    all_bitstings = []
    for value in range(base**n_qubits):
        bitstring = decimal(value, 'str', base = base)
        bitstring = '0' * (n_qubits - len(bitstring)) + bitstring
        all_bitstings.append(bitstring)
    # for qubit in range(2**n_qubits):
    #     bitstring = bin(qubit).replace('0b', '')
    #     bitstring = '0' * (n_qubits - len(bitstring)) + bitstring
    #     all_bitstings.append(bitstring)
    return tuple(all_bitstings)

# 是不是需要ray远程执行的函数
def is_ray_func(func):
    for name, f in inspect.getmembers(func, lambda f: hasattr(f, '_remote')):
        return True
    return False

def is_ray_future(obj):
    return isinstance(obj, ray._raylet.ObjectRef)

def wait(future, show_progress = False):
    # TODO: 可能会导致循环递归
    if isinstance(future, (list, set)):
        futures = future
        
        if not show_progress: 
            return [wait(future) for future in futures]
        else:
            from tqdm import tqdm
            results = []
            for future in tqdm(futures):
                results.append(wait(future) )
            return results
    elif is_ray_future(future):
        return ray.get(future)
    elif isinstance(future, Future):
        return future.result()
    elif isinstance(future, (dict, defaultdict)):
        return {
            key: wait(item)
            for key, item in future.items()
        }
    else:
        # raise Exception(future, 'is not future type')
        return future

def random_group(group, sub_group_size):
    group = list(group)
    
    sub_groups = []
    while len(group) != 0:
        now_group = []
        for _ in range(sub_group_size):
            now_group.append(random.choice(group))
            group.remove(now_group[len(now_group)-1])
            # group = [
            #     qubit
            #     for qubit in group
            #     if qubit != now_group[len(now_group)-1]
            # ]
            if len(group) == 0:
                break
        now_group.sort()
        if len(now_group) != 0:
            sub_groups.append(now_group)

    return sub_groups

def downsample_bitstring(bitstring, qubits):
    new_bitstring = ''.join([bitstring[qubit] for qubit in qubits])
    return new_bitstring
    
'''将M比特的变成只包含子集的'''
def downsample_status_count(stats_count: dict, qubits: list):
    new_stats_count = defaultdict(int)
    
    for bitstring, count in stats_count.items():
        new_bitstring = downsample_bitstring(bitstring, qubits)
        new_stats_count[new_bitstring] += count
    
    return dict(new_stats_count)

from tqdm import tqdm




def downsample_protocol_result(protocol_result: dict[dict], qubits:list):
    new_protocol_result = {}
    
    for real_bitsting, status_count in tqdm(protocol_result.items()):
        new_real_bitsring = downsample_bitstring(real_bitsting,qubits)
        if new_real_bitsring == "2"*len(qubits):
            continue
        new_status_count = downsample_status_count(status_count,qubits)
        
        if new_real_bitsring not in new_protocol_result:
            new_protocol_result[new_real_bitsring] = new_status_count
        else:
            for measure_bitstring, count in new_status_count.items():
                if measure_bitstring in new_protocol_result[new_real_bitsring]:
                    new_protocol_result[new_real_bitsring][measure_bitstring] += count
                else:
                    new_protocol_result[new_real_bitsring][measure_bitstring] = count
    
    for real_bitsting, status_count in new_protocol_result.items():
        total_count = sum(status_count.values())
        for measure_bitstring, count in status_count.items():
            status_count[measure_bitstring] = count / total_count
        
    return new_protocol_result
        

def store(data, filename):
    import os
    import pickle
    cnt = 0                     
    while os.path.exists(f'{filename}_{cnt}.pkl'):
        cnt += 1
    with open(f'{filename}_{cnt}.pkl','wb') as f:
        pickle.dump(data, f)
        
def load(filename):
    import pickle
    with open(filename, 'rb')as f:
        return pickle.load(f) 
    


def hamming_distance(string1, string2):
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter


from scipy import stats

def result_sim(num,n,samples,dis:str):         #num:非零值数目,n:比特数,samples:采样数  dis:laplace,norm,uniform
    bitstring=[]

    dist = getattr(stats, dis)(loc=0,scale=num)
    if dis == "uniform":
        x=np.arange(0,num)
    else:
        x=np.arange(-num*10,num*10,20)
    count = dist.pdf(x)
    count = count/count.sum()*samples
        
        
    for i in range(num):
        binary_num = list('0'*n)

        x = random.randint(1,n-1)
        for _ in range(x):
            y = random.randint(0,n-1)
            binary_num [y] = '1'
        binary_num = ''.join(binary_num)
        bitstring.append(binary_num)
    bitstring.sort()
    result = { bitstring : count 
              for bitstring,count in zip(bitstring,count)
              }
    return result


def to_dic(protocol_results_dataset):
    t =[]
    count = []
    for i in range(len(protocol_results_dataset)):
        x = np.array_str(protocol_results_dataset[i][0])
        x = x.replace('[', '').replace(']', '').replace('\n', '').replace(' ', '')
        t.append(x)

        t1 = []
        h = []
        for j in range(len(protocol_results_dataset[i][1][0])):
            x1 = np.array_str(protocol_results_dataset[i][1][0][j])
            x1 = x1.replace('[', '').replace(']', '').replace('\n', '').replace(' ', '')
            t1.append(x1)

            y = np.float16(protocol_results_dataset[i][1][1][j]/sum(protocol_results_dataset[i][1][1]))
            h.append(y)
        count.append({s1:s2 for s1,s2 in zip(t1,h)})

    protocol_results = {bit:stat for bit,stat in zip(t,count)}

    return protocol_results



def normalize_dict_values(input_dict):
    # 将字典中所有负数的值置为0
    for key, value in input_dict.items():
        if value < 0:
            input_dict[key] = 0

    # 归一化
    total = sum(input_dict.values())
    normalized_dict = {key: value / total for key, value in input_dict.items()}

    return normalized_dict


def np_to_dic(result):

    empty_dict = {}
    for i in tqdm(range(len(result))):
        x1 = [''.join(map(str, sublist)) for sublist in result[i][1][0].tolist()]
        x2 = result[i][1][1]
        my_dict = dict(zip(x1, x2))
        array_as_string = ''.join(result[i][0].ravel().astype(str))
        empty_dict[array_as_string] = my_dict

    return empty_dict
