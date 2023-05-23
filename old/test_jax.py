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
from jax import numpy as jnp
from jax import vmap

from jax import random
# 生成二项分布随机变量

p = 0.01
key = random.PRNGKey(314)
x = jax.random.bernoulli(key, p, shape=(100,))
print(x[x == True].shape, x[x == False].shape)  # True的概率为p

# data
file_input = sio.loadmat('./data/thermal.mat')
stats01s_all = file_input['stats01s']
measure_fids = file_input['measureFids']

dim = stats01s_all.shape
n_samples = dim[0]
num_slices = dim[1]
n_qubits = dim[2]

# 需要校准的sequence
qc_sample = stats01s_all[:, 0, :]

def inverse(arr):
    return jnp.where(arr == 0, 1, 0)

def flip(arr, flag):
    '''flag为1的时候arr对应的元素会翻转'''
    return jnp.abs(arr - flag)

def rr(qc_sample, measure_fids, n_rm_sample = 100):
    n_samples, n_qubits = qc_sample.shape
    key = random.PRNGKey(314)
    
    def _rr_sample_construct_prob(measure_fid, sample):
        transfer_p = 1 - measure_fid[inverse(sample)] # 是另外一个态转移来的概率
        transfer = jax.random.bernoulli(key, transfer_p, shape=(n_rm_sample,))
        return jnp.where(transfer == True, 1, 0)

    def _rr_sample_prob(sample, transfer): 
        return flip(sample, transfer)

    def rr_sample(sample, measure_fids):
        transfers = vmap(_rr_sample_construct_prob, in_axes=(0, 0))(measure_fids, sample)
        # print(transfers)
        rm_samples = vmap(_rr_sample_prob, in_axes=(None, 1))(sample, transfers)
        return rm_samples

    all_rm_samples = vmap(rr_sample, in_axes=(0, None))(qc_sample, measure_fids)
    all_rm_samples = np.array(all_rm_samples).astype(np.str_)

    counts = defaultdict(int)
    for rm_samples in all_rm_samples:
        for rm_sample in rm_samples:
            counts[''.join(rm_sample)] += 1
        
    total_count = n_samples * n_rm_sample

    # counts = {
    #     sample: count
    #     for sample, count in counts.items()
    #     if count > total_count / 1e2
    # }
    
    return counts
    
rr_counts = rr(qc_sample, measure_fids)
print('rr finish')

before_rr_counts = defaultdict(int)
for sample in qc_sample:
    before_rr_counts[''.join(sample.astype(np.str_))] += 1
before_rr_counts = {
    sample: count
    for sample, count in before_rr_counts.items()
    if count > 0.01
}
    
plot_histogram(rr_counts, filename ='rr')
plot_histogram(before_rr_counts, filename='before_rr')
# plt.show()