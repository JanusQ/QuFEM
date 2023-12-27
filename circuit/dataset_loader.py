import traceback
from circuit.algorithm.get_data import get_dataset_bug_detection
from circuit.formatter import layered_circuits_to_qiskit
from circuit.parser import get_circuit_duration, qiskit_to_layered_circuits
from circuit.random_circuit import random_circuit, random_circuit_cycle
from qiskit import transpile
import ray


def gen_algorithms(n_qubits, coupling_map, mirror):
    return get_dataset_bug_detection(n_qubits, n_qubits+1, coupling_map, mirror)
