from functools import lru_cache
import ray
import threading
import inspect
import uuid
import random
import concurrent.futures
from concurrent.futures._base import Future
from collections import Iterable
from inspect import isgeneratorfunction
from collections import defaultdict
import numpy as np


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

'''将M比特的变成只包含子集的'''
def downsample(stats_count: dict, qubits: list):
    new_stats_count = defaultdict(int)
    
    for bitstring, count in stats_count.items():
        new_bitstring = ''.join([bitstring[qubit] for qubit in qubits])
        new_stats_count[new_bitstring] += count
    
    return new_stats_count