'''分析错误之间的相关性'''
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.approximation.maxcut as maxcut
from scipy.stats import pearsonr
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


from itertools import combinations

import networkx as nx
from sklearn.metrics import f1_score

from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import K2Score
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from typing import Dict
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from utils import all_bitstrings

def reduce(array: list, func = lambda now_value, total: now_value + total, initial_total = None, ):
    start_index = 0
    if initial_total is None:
        initial_total = array[0]
        start_index = 1
    
    for index in range(start_index, len(array)):
        initial_total = func(array[index], initial_total)
    
    return initial_total

class PdBasedProtocolResults():
    def __init__(self, protocol_results: dict, n_qubits: int):
        self.protocol_results = protocol_results
        self.n_qubits = n_qubits
    
        columns = defaultdict(list)

        # TODO: 这个的构建可以放到外面去，封装成一个查询类
        for real_bitstring, status_count in protocol_results.items():
            for measure_bitstring, count in status_count.items():

                for qubit in range(n_qubits):
                    columns[f'{qubit}_set'].append(real_bitstring[qubit]) # set的值 0, 1, 2 (没有set)
                    columns[f'{qubit}_read'].append(measure_bitstring[qubit]) # 读到的值 0, 1, 2 (不读)
                    # columns[f'{qubit}_measure'].append(real_bitstring[qubit] != '2') # 是否进行读取 0, 1
                columns['count'].append(count)

        # 全是2的没有实际测量所以这里直接加了一个
        for qubit in range(n_qubits):
            columns[f'{qubit}_set'].append('2') # set的值 0, 1, 2 (没有set)
            columns[f'{qubit}_read'].append('2') # 读到的值 0, 1, 2 (不读)
        columns['count'].append(10000)

        self.df = pd.DataFrame(data=columns)
    
    # def join_prob()
    def __getitem__(self, qubit_set_values):
        '''return count'''
        # df = self.df
        # return df[reduce(qubit_set_values, func = lambda qv, prev: (df[f'{qv[0]}_set'] == qv[1]) & prev, initial_total= True)]
                
        filter_df = self.df
        for qubit, value in qubit_set_values:
            filter_df = filter_df[filter_df[f'{qubit}_set'] == str(value)]
        return filter_df

def construct_bayesian_network(protocol_results: PdBasedProtocolResults, n_qubits, groups):
    '''这里已经开始考虑是否读取了'''
    # columns = defaultdict(list)
    # for real_bitstring, status_count in protocol_results.items():
    #     for measure_bitstring, count in status_count.items():

    #         for qubit in range(n_qubits):
    #             columns[f'{qubit}_set'].append(real_bitstring[qubit]) # set的值 0, 1, 2 (没有set)
    #             columns[f'{qubit}_read'].append(measure_bitstring[qubit]) # 读到的值 0, 1, 2 (不读)
    #             # columns[f'{qubit}_measure'].append(real_bitstring[qubit] != '2') # 是否进行读取 0, 1
    #         columns['count'].append(count)

    # df = PdBasedProtocolResults(protocol_results).df
    # print(df)

    q2group = {}
    for g in groups:
        for q in g:
            q2group[q] = g
    
    cpds = []
    network_edges = []
    
    for qubit in range(n_qubits):
        # 还有比特的的set, 基本上不会影响到实际生成的值
        cpds.append(TabularCPD(f"{qubit}_set", 3, [[1/3]] * 3,))
        # cpds.append(TabularCPD(f"{qubit}_set", 3, [[0.20], [0.20], [0.60]],))
        
        # 自己 + 其他pmi大的 TODO: 需要剪枝，用pmi
        # related_qubits = list(range(n_qubits))
        related_qubits = q2group[qubit]
        for related_qubit in related_qubits:
            network_edges.append((f'{related_qubit}_set',f'{qubit}_read'))

        data = np.zeros(shape=(3, 3**len(related_qubits)))

        for set_types in itertools.product(*[(0, 1, 2)]*len(related_qubits)):
            colum_index = sum([set_type * (3**(len(related_qubits) - qubit - 1))
                              for qubit, set_type in enumerate(set_types)])
            
            # 替换上面的
            filter_df = protocol_results[[(related_qubit, set_type) for set_type, related_qubit in zip(set_types, related_qubits)]]
            
            if len(filter_df) == 0:
                print('Wanrning')
            
            # 这里面要不要改成int
            for read_type in (0, 1, 2):  # 对应1,2,3的操作
                data[read_type][colum_index] = filter_df[filter_df[f'{qubit}_read'] == str(
                    read_type)]['count'].sum()
                # print(elm)

            if np.sum(data[:, colum_index]) == 0:
                # 本来不可能出现的条件概率
                data[2, colum_index] = 1
                raise Exception('忘了这里是干啥的了，好像是算不出概率')
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
        
        
        print(qubit_cpd)
        cpds.append(qubit_cpd)

    
    model = BayesianNetwork(network_edges)  #TODO: 可以加latent
    model.add_cpds(*cpds)
    infer = VariableElimination(model)
    
    # for bitstring in all_bitstrings(n_qubits):
    #     posterior_p = infer.query([f'{qubit}_read' for qubit in range(n_qubits)], evidence={
    #         f'{qubit}_set': int(bit)
    #         for qubit, bit in enumerate(bitstring)
    #     })
    #     print(bitstring)
    #     print(posterior_p)
    #     posterior_v = posterior_p.values.reshape(3**n_qubits) #变成了M中列的数据
        
    #     # 剃掉包含2的，TODO: 需要换成更快的过滤方法，比如也用lru_cache
    #     posterior_v = np.array([posterior_v[index] for index, bitstring in enumerate(all_bitstrings(n_qubits, base = 3)) if '2' not in bitstring ])

    #     print(posterior_v)
        
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
    # error_count = np.zeros(shape=(n_qubits, 2, n_qubits, 2, 1))   # qubit1 -> value -> qubit2 -> value -> error_frequency
    # all_count = np.zeros(shape=(n_qubits, 2, n_qubits, 2, 1))
    # qubit1 -> value -> qubit2 -> value -> error_frequency
    error_count = np.zeros(shape=(n_qubits, 2, n_qubits, 1))
    all_count = np.zeros(shape=(n_qubits, 2, n_qubits, 1))

    '''可能可以换一个更加合理的'''
    for real_bitstring, status_count in protocol_results.items():
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
                    error_count[qubit][real_bitstring[qubit]
                                       ][error_qubit] += count

            for qubit1 in range(n_qubits):
                for qubit2 in range(n_qubits):
                    all_count[qubit1][real_bitstring[qubit1]][qubit2] += count

            # error_count[:,measure_bitstring, correct_qubits, measure_bitstring[error_qubits]] += count
            # all_count[:,measure_bitstring,:,measure_bitstring] += count

    error_freq = error_count / all_count
    # np.where(np.abs(error_freq[:,:,:,0]-error_freq[:,:,:,1])>0.03)
    # corr_qubit_pairs =
    freq_diff = np.abs(error_freq[:, 0, :]-error_freq[:, 1, :])
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


# def construct_bayesian_network(protocol_results: dict, group_size, n_qubits):
#     '''这里已经开始考虑是否进行读取了'''
#     columns = {}
#     for qubit in range(n_qubits):
#         columns[f'{qubit}_set'] = []  # set的值 0, 1, 2 (没有set)
#         columns[f'{qubit}_read'] = [] # 读到的值 0, 1, 2 (不读)
#         columns[f'{qubit}_measure'] = [] # 是否进行读取 0, 1

#     # 后面可以直接改成count自己算的
#     for real_bitstring, status_count in protocol_results.items():
#         # real_bitstring = [int(bit) for bit in real_bitstring]

#         for measure_bitstring, count in status_count.items():
#             # measure_bitstring = np.array([int(bit) for bit in measure_bitstring])

#             '''TODO: 要不要直接把status_count里面加上没有测量的，表示为2'''

#             for _ in range(count):
#                 for qubit in range(n_qubits):
#                     columns[f'{qubit}_set'].append(real_bitstring[qubit])
#                     columns[f'{qubit}_read'].append(measure_bitstring[qubit])
#                     columns[f'{qubit}_measure'].append(real_bitstring[qubit] != '2')

#     df = pd.DataFrame(data=columns)
#     print(df)

#     # 需要finetune
#     # max_cond_vars: int
#     #     The maximum number of conditional variables allowed to do the statistical
#     #     test with.

#     # scoring_method = K2Score(data=df)
#     # est = HillClimbSearch(data=df)
#     # estimated_model = est.estimate(
#     #     scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
#     # )
#     # estimated_model.query(
#     # variables=["bronc"], evidence={"smoke": "no"})
#     # , virtual_evidence=[lung_virt_evidence]
#     return
