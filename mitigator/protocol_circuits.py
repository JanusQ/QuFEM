'''生成用于测量M的电路'''
from utils import all_bitstrings, decimal
from qiskit import QuantumCircuit
from typing import List, Optional, Tuple
import random
class EnumeratedProtocol():
    '''生成的是2到2**n_qubits的暴搜'''
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def gen_circuits(self) -> List[QuantumCircuit]:
        circuits = []
        for bitstring in all_bitstrings(self.n_qubits):
            circuit = QuantumCircuit(self.n_qubits)
            for qubit, bit in enumerate(bitstring):
                if bit == '1':
                    circuit.x(qubit)
            circuit.measure_all()
            circuits.append(circuit)        
    
        return all_bitstrings(self.n_qubits), circuits

class MeasuremtAwareEnumeratedProtocol():
    '''生成的是2到2**n_qubits的暴搜'''
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def gen_circuits(self) -> List[QuantumCircuit]:
        circuits = []
        
        for bitstring in all_bitstrings(self.n_qubits, base = 3):
            if bitstring == '2' * self.n_qubits: continue  # 没有测量
            
            circuit = QuantumCircuit(self.n_qubits)
            for qubit, bit  in enumerate(bitstring):
                if bit == '1':
                    circuit.x(qubit)
                    
            measured_qubits = [
                qubit
                for qubit, measurement in enumerate(bitstring)
                if '2' != measurement
            ]
            new_creg = circuit._create_creg(len(measured_qubits), "meas")
            circuit.add_register(new_creg)
            circuit.barrier()
            circuit.measure(measured_qubits, new_creg)
            
            # print('\n')
            # print(bitstring)
            # print(circuit)
            
            circuits.append(circuit)        
        

        return all_bitstrings(self.n_qubits, base = 3), circuits
    def gen_random_circuits(self, cnt, bitstring_dataset):
            n_qubits = self.n_qubits
                
            bitstrings, circuits = [], []
        
            i = 0
            while i < cnt:
                if len(bitstring_dataset) == 3**n_qubits - 1:
                    break
                
                max = 3**self.n_qubits
                value = random.randint(0, max - 1)
                bitstring = decimal(value, 'str', base = 3)
                bitstring = '0' * (n_qubits - len(bitstring)) + bitstring
                
                if bitstring == '2' * self.n_qubits or bitstring in bitstring_dataset:
                    continue  # 没有测量
                i += 1
                bitstrings.append(bitstring)
                bitstring_dataset.append(bitstring)
                
                circuit = QuantumCircuit(self.n_qubits)
                for qubit, bit  in enumerate(bitstring):
                    if bit == '1':
                        circuit.x(qubit)
                        
                measured_qubits = [
                    qubit
                    for qubit, measurement in enumerate(bitstring)
                    if '2' != measurement
                ]
                new_creg = circuit._create_creg(len(measured_qubits), "meas")
                circuit.add_register(new_creg)
                circuit.barrier()
                circuit.measure(measured_qubits, new_creg)
                
                # print('\n')
                # print(bitstring)
                # print(circuit)
                
                circuits.append(circuit)        
            

            return bitstrings, circuits