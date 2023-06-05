1. 弄懂代码
2. 完成代码正确性验证
    1. 验证在各个算法上的效果
    2. 加上考虑稀疏性的
    3. 加上利用少量数据更新贝叶斯网络的
    5. 完成真机的实验
3. 实现baseline:
    1. Q-BEEP ISCA 2023 没有代码
    2. unfolding: 有代码
    3. 纯数学的： 已经实现
    4. m3 矩阵压缩: 有代码 https://qiskit.org/ecosystem/mthree/
    5. 神经网络的: 没有代码，比较简单
    6. IBM：百度的云平台里面有代码，https://github.com/baidu/QCompute/blob/64a2152b79b1d4b3319f563a2e5b4b98d41d6e4f/Extensions/QuantumErrorProcessing/tutorials/CN/tutorial-mem-ctmp.ipynb#L41![image](https://github.com/JanusQ/readout_error_mitigation/assets/20809866/4a805911-d770-4362-bd81-2d752e3528a5)
    7. scipy稀疏矩阵计算的 https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html#scipy.sparse.csc_matrix
3. 部署到C++上
4. (7/15完成) 补充细节实验
