
n_qubits = 3
measure_fids = jnp.ones(shape=(n_qubits, 2)) * 0.99
noise_model = NoiseModel()

for i in range(n_qubits):
    re = ReadoutError([[measure_fids[i][0], 1 - measure_fids[i][0]],
                      [1 - measure_fids[i][1], measure_fids[i][1]]])
    noise_model.add_readout_error(re, qubits=[i])
simulator = AerSimulator(noise_model=noise_model)

# 拿到错误的模拟值
n_samples = 3000
before_rm_counts = simulator.run(
    ghz(n_qubits), shots=n_samples).result().get_counts()
before_rm_prob = {k: v / n_samples for k, v in before_rm_counts.items()}
plot_histogram(before_rm_prob, title = f'before_rm_{n_qubits}', filename=f'before_rm_{n_qubits}')

'''生成矫正的矩阵, meas_mats是读取噪声的form_ulation, meas_mats_inv是矫正矩阵'''
meas_mats, meas_mats_inv = [], []
for qubit in range(n_qubits):
    meas_mat = np.array([[
        measure_fids[qubit][0], 1-measure_fids[qubit][1]],
        [1-measure_fids[qubit][0], measure_fids[qubit][1]]
    ])
    meas_mats.append(meas_mat)
    meas_mats_inv.append(np.linalg.inv(meas_mat))


if n_qubits < 10:
    '''qiskit的矫正'''
    mit = LocalReadoutMitigator(meas_mats, list(range(n_qubits)))
    qiskit_rm_prob = mit.quasi_probabilities(before_rm_counts)
    qiskit_rm_prob = {
        basis: value
        for basis, value in qiskit_rm_prob.items()
        if value > 1e-3
    }
    plot_histogram(qiskit_rm_prob, title = f'qiskit_rm_{n_qubits}', filename=f'qiskit_rm_{n_qubits}')

    '''用纯数学的方法试一下'''
    before_rm_prob_vec = np.zeros(2**n_qubits)
    for basis, prob in before_rm_prob.items():
        before_rm_prob_vec[int(basis, base=2)] = prob

    tensor_meas_mat_inv = np.linalg.inv(meas_mats[0])
    for qubit in range(1, n_qubits):
        tensor_meas_mat_inv = np.kron(
            tensor_meas_mat_inv, np.linalg.inv(meas_mats[qubit]))

    rm_prob = {}
    rm_prob_vec = tensor_meas_mat_inv @ before_rm_prob_vec
    for basis, prob in enumerate(rm_prob_vec):
        rm_prob[bin(basis).replace('0b', '')] = prob
    rm_prob = {
        basis: value
        for basis, value in rm_prob.items()
        if value > 1e-3
    }
    plot_histogram(rm_prob, title = f'math_rm_{n_qubits}', filename=f'math_rm_{n_qubits}')
