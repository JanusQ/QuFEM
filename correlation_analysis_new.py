'''分析错误之间的相关性'''
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.approximation.maxcut as maxcut
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mutual_info_score
import itertools
# import pulp
# import community
# community.best_partition
# nx.community.girvan_newman

# https://networkx.org/documentation/latest/reference/algorithms/community.html

import networkx as nx
import numpy as np
# from sklearn.cluster import KMeans
from collections import defaultdict

from tqdm import tqdm
from itertools import combinations

import networkx as nx
from sklearn.metrics import f1_score
import time
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import K2Score
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from typing import Dict
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from ray_func import batch, wait
from utils import all_bitstrings, downsample_protocol_result
import ray


class PdBasedProtocolResults:
    pass

# 将status_count展开成X，Y的格式:
def extend(protocol_results, related_qubits, qubit):
    # protocol_results = downsample_protocol_result(protocol_results, related_qubits)
    X = []
    Y = []
    for real_bitstring, status_count in protocol_results.items():
        set_types = [int(real_bitstring[qubit]) for qubit in related_qubits]
        for measure_bitstring, count in status_count.items():
            read_type = int(measure_bitstring[qubit])
            if read_type == 2:
                continue
            if count < 1:
                count =  int(count*5000)
            for _ in range(count):
                X.append(set_types)
                Y.append(read_type)
    return X, Y

# @ray.remote
def construct_cpd(qubit, related_qubits, protocol_results):
    # print('start', qubit)
    data = np.zeros(shape=(3, 3**len(related_qubits)))

    protocol_results = wait(protocol_results)
    
    # for real_bitstring, status_count in tqdm(protocol_results.items()):
    set_type_list = [0] * (3**len(related_qubits))
    for real_bitstring, status_count in protocol_results.items():
        set_types = [int(real_bitstring[qubit]) for qubit in related_qubits]
        colum_index = sum([set_type * (3**(len(related_qubits) - qubit - 1))
                            for qubit, set_type in enumerate(set_types)])
        
        # print(colum_index, len(set_types))
        set_type_list[colum_index] = set_types

        for measure_bitstring, count in status_count.items():
            read_type = int(measure_bitstring[qubit])
            data[read_type][colum_index] += count
            
            # if read_type == 2:
            #     assert real_bitstring[qubit] == '2'
    
    infer_model = None
    for colum_index in range(3**len(related_qubits)):
        if np.sum(data[:, colum_index]) == 0:
            
            op_types = []
            temp_colum_index = colum_index
            for related_qubit in related_qubits:
                op_types.append(temp_colum_index % 3)
                temp_colum_index  = temp_colum_index // 3
            op_types.reverse()
            
            if op_types[related_qubits.index(qubit)] == 2  : #all([op == 2 for op in op_types]):
                data[2, colum_index] = 1
            else:
                # 需要从周围估一下，懒得搞了就直接上机器学习了
                if infer_model is None:
                    X, Y = extend(protocol_results, related_qubits, qubit)
                    infer_model = GaussianNB() # max_depth=2, random_state=0, n_jobs = -1
                    infer_model.fit(X, Y)
                probs = infer_model.predict_proba([op_types])[0]
                data[0:2, colum_index] = probs
                
            # 本来不可能出现的条件概率
            # data[2, colum_index] = 1
            # raise Exception('应该是这个对应的没有数据，需要从周围估一下', set_type_list[colum_index], related_qubits)
            # data[1, colum_index] = 1
            # data[set_type_list[colum_index][related_qubits.index(qubit)], colum_index] = 1
            # print('应该是这个对应的没有数据，需要从周围估一下', op_types, related_qubits)
        else:
            data[:, colum_index] /= np.sum(data[:, colum_index])


    # 生成如下的格式的cpd
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 0_set     | 0_set(0) | 0_set(0) | ... | 0_set(2) | 0_set(2) | 0_set(2) |
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 1_set     | 1_set(0) | 1_set(1) | ... | 1_set(0) | 1_set(1) | 1_set(2) |
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 0_read(0) | 0.0      | 0.0      | ... | 942.0    | 41.0     | 0.0      |
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 0_read(1) | 0.0      | 0.0      | ... | 58.0     | 959.0    | 0.0      |
    # +-----------+----------+----------+-----+----------+----------+----------+
    # | 0_read(2) | 1.0      | 1.0      | ... | 0.0      | 0.0      | 0.0      |
    # +-----------+----------+----------+-----+----------+----------+----------+

    # P(qubit_read | qubit_set, other qubit_set)
    qubit_cpd = TabularCPD(f"{qubit}_read", 3,
                            data,
                            evidence=[f"{related_qubit}_set" for related_qubit in related_qubits],
                            evidence_card=[3, ] * len(related_qubits),
                            )
    
    # print('finish', qubit)
    # print(qubit_cpd)
    return qubit_cpd

@ray.remote
def construct_cpd_remote(qubit, related_qubits, protocol_results):
    return construct_cpd(qubit, related_qubits, protocol_results)

def construct_bayesian_network(protocol_results, n_qubits, groups, multi_process = True):
    q2group = {}
    for g in groups:
        for q in g:
            q2group[q] = g
    
    cpds = []
    network_edges = []

    futures = []

    protocol_result_token = ray.put(protocol_results)
    for qubit in range(n_qubits):
        # 还有比特的的set, 基本上不会影响到实际生成的值
        cpds.append(TabularCPD(f"{qubit}_set", 3, [[1/3]] * 3,))
        
        # 自己 + 其他pmi大的 TODO: 需要剪枝，用pmi
        # related_qubits = list(range(n_qubits))
        related_qubits = q2group[qubit]
        for related_qubit in related_qubits:
            network_edges.append((f'{related_qubit}_set',f'{qubit}_read'))
        
        if multi_process:
            futures.append(construct_cpd_remote.remote(qubit, related_qubits, protocol_result_token))
        else:
            futures.append(construct_cpd(qubit, related_qubits, protocol_result_token))

    for qubit_cpd in wait(futures, show_progress=False): #tqdm(as_completed(futures)):
        cpds.append(qubit_cpd)
        

    model = BayesianNetwork(network_edges)  #TODO: 可以加latent
    model.add_cpds(*cpds)
    infer = VariableElimination(model)
          
    return model, infer


def draw_correlation_graph(graph):
    plt.figure(figsize=(5, 5))  # 这里控制画布的大小，可以说改变整张图的布局
    # plt.subplot(111)
    pos = nx.spring_layout(graph, iterations=20)

    nx.draw(graph, pos, edge_color="grey", node_size=500)  # 画图，设置节点大小

    node_labels = nx.get_node_attributes(graph, 'qubit')  # 获取节点的desc属性
    nx.draw_networkx_labels(graph, pos, node_labels, font_size=10)
    # nx.draw_networkx_labels(graph, pos, node_labels=node_labels,font_size=20)  # 将desc属性，显示在节点上
    edge_labels = nx.get_edge_attributes(graph, 'freq_diff')  # 获取边的name属性，
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels, font_size=10)  # 将name属性，显示在边上

    # plt.savefig('./tu.pdf')
    plt.show()


# 相关性的度量 q1的值对q2出错的概率的影响
def correlation_based_partation(protocol_results, group_size, n_qubits):
    '''这里用的是概率差'''
    # qubit1 -> value -> qubit2 -> value -> error_frequency
    error_count = np.zeros(shape=(n_qubits, 3, n_qubits, 1))  # qubit1的值对qubit2的影响
    all_count = np.zeros(shape=(n_qubits, 3, n_qubits, 1))

    '''可能可以换一个更加合理的'''
    for real_bitstring, status_count in tqdm(protocol_results.items()):
        real_bitstring = [int(bit) for bit in real_bitstring]

        for measure_bitstring, count in status_count.items():
            measure_bitstring = np.array(
                [int(bit) for bit in measure_bitstring])

            error_qubits, correct_qubits = [], []
            for qubit in range(n_qubits):
                if measure_bitstring[qubit] != real_bitstring[qubit]:
                    error_qubits.append(qubit)
                else:
                    correct_qubits.append(qubit)

            # for correct_qubit in correct_qubits:
            for error_qubit in error_qubits:
                for qubit in range(n_qubits):
                    error_count[qubit][real_bitstring[qubit]][error_qubit] += count

            for qubit1 in range(n_qubits):
                for qubit2 in range(n_qubits):
                    all_count[qubit1][real_bitstring[qubit1]][qubit2] += count

            # error_count[:,measure_bitstring, correct_qubits, measure_bitstring[error_qubits]] += count
            # all_count[:,measure_bitstring,:,measure_bitstring] += count

    error_freq = error_count / all_count
    # np.where(np.abs(error_freq[:,:,:,0]-error_freq[:,:,:,1])>0.03)

    # 现在有三种情况了，0,1,2
    freq_diff = np.abs(error_freq[:, 0, :]-error_freq[:, 1, :]) + np.abs(error_freq[:, 0, :]-error_freq[:, 2, :]) + np.abs(error_freq[:, 1, :]-error_freq[:, 2, :])
    large_corr_qubit_pairs = np.where(freq_diff > 0.01)  # (q1, q2, 0)

    graph = nx.Graph()

    graph.add_nodes_from([[qubit, {'qubit': qubit}]
                         for qubit in range(n_qubits)])
    for q1, q2, _ in zip(*large_corr_qubit_pairs):
        if q1 == q2:
            continue
        graph.add_edge(q1, q2, freq_diff=np.round(freq_diff[q1][q2][0], 4))

    # max-cut 二分应该得不到最优解
    def partition(group):
        # print(group)
        small_partitions = []
        # current_cut_size, partition =
        for sub_group in maxcut.one_exchange(graph.subgraph(group), weight='freq_diff')[1]:
            if len(sub_group) == len(group):
                # print(sub_group)
                # 互相之间就是独立的
                return [
                    [qubit]
                    for qubit in sub_group
                ]
            if len(sub_group) <= group_size:
                small_partitions.append(sub_group)
            else:
                small_partitions += partition(sub_group)
        return small_partitions

    groups = [
        list(group)
        for group in partition(list(range(n_qubits)))
    ]

    # TODO: 检查下是不是小于group_size的

    return groups


# sample accuracy
# 标准差/sqrt(样本量)

# 相关性的度量 q1的值对q2出错的概率的影响
def correlation_based_partation_PMI(protocol_results, n_qubits):
    '''TODO: 这里用的是点互信息衡量变量之间的相似性'''

    # 类似
    from sklearn.metrics import mutual_info_score

    # 示例数据
    x = [1, 2, 1, 2, 1, 2]
    y = [1, 1, 2, 2, 1, 2]

    # 计算点互信息
    pmi = mutual_info_score(x, y)

    print("Point Mutual Information:", pmi)
    return



def calculate_correlation(n_qubits, real_bitstring, result, n_samples):
    local_tmp = np.zeros((n_qubits, n_qubits, 2, 3))
    local_uncertainty = np.zeros((n_qubits, n_qubits, 2, 3))
    for measure_bitstring, count in result.items():
        origin_bitstring = copy.deepcopy(real_bitstring)
        if '2' in origin_bitstring:
            origin_bitstring = origin_bitstring.replace('2','0')
            measure_bitstring = measure_bitstring.replace('2','0')
        if int(origin_bitstring, base=2) ^ int(measure_bitstring, base=2) != 0:
            error_qubits = bin(int(origin_bitstring, base=2) ^ int(measure_bitstring, base=2)).replace('0b','').zfill(n_qubits)#[::-1]
            for qubit_idx, is_error in enumerate(error_qubits):
                if is_error == '0':
                    continue
                
                for i in range(n_qubits):
                    if qubit_idx == i:
                        continue
                    local_tmp[qubit_idx][i][int(origin_bitstring[qubit_idx])][int(real_bitstring[i])] += count / n_samples
                    local_uncertainty[qubit_idx][i][int(origin_bitstring[qubit_idx])][int(real_bitstring[i])] += 1
    return local_tmp, local_uncertainty