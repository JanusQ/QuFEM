from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, errors

# Create a quantum circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Create a noise model
noise_model = NoiseModel()

# Add T1, T2, and depolarizing errors to qubits
# ...

# Add readout errors
readout_error_probabilities = [[0.1, 0.2], [0.3, 0.4]]
readout_errors = [errors.readout_error.ReadoutError(probs=p, invert=False)
                  for p in readout_error_probabilities]
noise_model.add_all_qubit_readout_error(readout_errors)

# Simulate the noisy circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, noise_model=noise_model, shots=1024)
result = job.result()

# Get the results
counts = result.get_counts(qc)
print(counts)
