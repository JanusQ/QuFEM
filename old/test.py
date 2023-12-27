# import
from collections import defaultdict
import numpy as np
import scipy.io as sio
import random
import numpy as np
import pymc as pm
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import numpy as np
import scipy.linalg as la
from qiskit.result import LocalReadoutMitigator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import jax

# data
file_input = sio.loadmat('./data/thermal.mat')
stats01s_all = file_input['stats01s']
measureFids = file_input['measureFids']

dim = stats01s_all.shape
n_samples = dim[0]
num_slices = dim[1]
n_qubits = dim[2]
# n_qubits = 24


# 需要校准的sequence
qc_sample = stats01s_all[:, 0, :]

def inverse(arr):
    return np.where(arr == 0, 1, 0)

def flip(arr, flag):
    '''flag为1的时候arr对应的元素会翻转'''
    arr = np.array(arr)
    return np.abs(arr - flag)

all_rm_samples = [] 
n_rm_sample = 10
for sample_index in range(n_samples):
    sample = qc_sample[sample_index]
    # transfer_p = np.zeros(n_qubits)
    
    transfers = []
    for qubit in range(n_qubits):
        transfer_p = 1- measureFids[qubit][inverse(sample[qubit])] # 是另外一个态转移来的概率
        transfer = np.random.binomial(1, transfer_p, n_rm_sample) # 转移回去的采样
        transfers.append(transfer)
    
    transfers = np.array(transfers)
    for rm_index in range(n_rm_sample):
        rm_sample = flip(sample, transfers[:,rm_index])
        all_rm_samples.append(rm_sample)

    # print(sample)
counts = defaultdict(int)
for rm_sample in all_rm_samples:
    counts[''.join(rm_sample.astype(np.str_))] += 1
    
total_count = n_samples * n_rm_sample

# counts = {
#     sample: count
#     for sample, count in counts.items()
#     if count > total_count / 1e4
# }
plot_histogram(counts)
plt.show()