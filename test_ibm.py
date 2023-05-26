import pickle
from simulator import LocalSimulator, NonLocalSimulator, Simulator
from qiskit.visualization import plot_histogram
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from utils import all_bitstrings

n_qubits = 5
with open('dataset/result.pkl', 'rb', ) as file:
    protocol_results = pickle.load(file)  # # ibmq_manila

M = np.zeros(shape=(2**n_qubits, 2**n_qubits))
# M = np.zeros(shape=(2**3, 2**3))
columns = defaultdict(list)
for circuit, stats_count  in zip(*protocol_results):
    measured_qubits = tuple([
        instruction.qubits[0].index
        for instruction in circuit
        if instruction.operation.name == 'measure'
    ])
    
    if 0 not in measured_qubits and 1 not in measured_qubits and 2 not in measured_qubits and 3 in measured_qubits and 4 in measured_qubits:
        continue

    apply_x_qubits = tuple([
        instruction.qubits[0].index
        for instruction in circuit
        if instruction.operation.name == 'x'
    ])

    real_bitstring = ['0'] * n_qubits
    for qubit in range(n_qubits):
        if qubit not in measured_qubits:
            real_bitstring[qubit] = '2'
        elif qubit in apply_x_qubits:
            real_bitstring[qubit] = '1'
    real_bitstring = ''.join(real_bitstring)
    
    new_stats_count = {}
    for bitstring, count in stats_count.items():
        bitstring = bitstring[::-1]
        new_bitstring = ['0'] * n_qubits
        for qubit in range(n_qubits):
            if qubit in measured_qubits:
                new_bitstring[qubit] = bitstring[measured_qubits.index(qubit)]
            else:
                new_bitstring[qubit] = '2'
        new_bitstring = ''.join(new_bitstring)
        new_stats_count[new_bitstring] = count
    stats_count = new_stats_count

    # fig, ax = plt.subplots()
    # plot_histogram(stats_count, title = real_bitstring, ax = ax)
    # fig.savefig(os.path.join('temp/fig/ibmq_manila_protocols', real_bitstring + '.svg'))
    # plt.close()
    
    for measure_bitstring, count in stats_count.items():
        for _ in range(count):
            for qubit in range(n_qubits):
                columns[f'{qubit}_set'].append(real_bitstring[qubit])
                columns[f'{qubit}_read'].append(measure_bitstring[qubit])
                columns[f'{qubit}_measure'].append(real_bitstring[qubit] != '2')
                columns[f'{qubit}_error'].append(real_bitstring[qubit] != measure_bitstring[qubit])
                
    if '2' not in real_bitstring:
        P = [
            stats_count[bitstring] if bitstring in stats_count else 0
            for bitstring in all_bitstrings(n_qubits)
            # for bitstring in all_bitstrings(3)
        ]
        # P = np.array(P)
        M[:,int(real_bitstring, base=2)] = P / np.sum(P)
        
from matplotlib.colors import LinearSegmentedColormap

#练习的数据：
data=pd.DataFrame(M)

colors = [[31, 150, 39],  [0, 54, 92], ]
# [31, 117, 175], 
# 0.4, 
'''xia <- shang'''
# colors.reverse()
colors = np.array(colors) / 256
# 定义颜色的位置
pos = [0,  1]
# 创建colormap对象
cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(pos, colors)))



#绘制热度图：
plot=sns.heatmap(data, cmap = cmap)
  
plt.show()

# def plot_correlation(data, feature_names, color_features = None, name = 'correlation'):
#     df = pd.DataFrame(data, columns=feature_names)
    
#     if color_features is not None:
#         df["class"] = pd.Series(color_features)
#         sns_plot = sns.pairplot(df, hue='class', palette="tab10")
#     else:
#         sns_plot = sns.pairplot(df, hue=None, palette="tab10")

#     sns_plot.savefig(f"{name}.png")
#     return

# print(protocol_results)
