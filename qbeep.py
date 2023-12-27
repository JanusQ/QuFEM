import qiskit
from scipy.stats import entropy
import numpy
import pandas as pd
import copy
import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from IPython import display
from itertools import permutations
import time
import math
from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import OrderedDict
import glob
import pickle
import random

def generate_binary_strings(bit_count):
    """
    Generate all possible binary strings of a given length.

    Args:
        bit_count (int): The length of the binary strings to be generated.

    Returns:
        list: A list of binary strings.
    """
    binary_strings = []

    def genbin(n, bs=''):
        """
        Recursive helper function to generate binary strings.

        Args:
            n (int): The remaining length of the binary string to be generated.
            bs (str): The current binary string generated so far.
        """
        if len(bs) == n:
            # The binary string has reached the desired length
            binary_strings.append(bs)
        else:
            # Recursive calls to append '0' or '1' and continue generating the binary string
            genbin(n, bs + '0')
            genbin(n, bs + '1')

    # Start the generation process with an empty binary string
    genbin(bit_count)

    # Return the list of generated binary strings
    return binary_strings

<<<<<<< HEAD
=======

>>>>>>> a188bb899043160906c28568ca175c5a1528ced8
def gen_ham_component(n, k):
    """
    Generate a binary string component with 'n' '1' characters and 'k' total length.

    Args:
        n (int): The number of '1' characters in the binary string.
        k (int): The total length of the binary string.

    Returns:
        str: The generated binary string component.

    Raises:
        AssertionError: If 'k' is not greater than 'n'.
    """
    # Ensure that 'k' is greater than 'n' using an assertion
    assert k > n

    # Generate the binary string component
    s1 = ''.join(['0' for _ in range(k - n)])  # Construct string of '0' characters
    s2 = ''.join(['1' for _ in range(n)])  # Construct string of '1' characters
    s = s1 + s2  # Concatenate the strings

    # Return the generated binary string component
    return s

<<<<<<< HEAD
=======

>>>>>>> a188bb899043160906c28568ca175c5a1528ced8
def xor_convert(s1, s2):
    """
    Perform element-wise XOR operation on two binary strings.

    Args:
        s1 (str): The first binary string.
        s2 (str): The second binary string.

    Returns:
        str: The result of the XOR operation as a binary string.
    """
    # Perform the XOR operation element-wise using a list comprehension
    # If the XOR result is 1, append '1' to the result string; otherwise, append '0'
    result = ''.join(['1' if int(a) ^ int(b) == 1 else '0' for a, b in zip(s1, s2)])

    # Return the result of the XOR operation
    return result

<<<<<<< HEAD
=======

>>>>>>> a188bb899043160906c28568ca175c5a1528ced8
def get_perms(n, k):
    """
    Get all binary strings of length 'k' with 'n' number of '1' characters.

    Args:
        n (int): The number of '1' characters in the binary strings.
        k (int): The length of the binary strings.

    Returns:
        list: A list of binary strings that satisfy the conditions.
    """
    bs = np.array(generate_binary_strings(k))  # Generate all binary strings of length 'k'
    nbs = []  # List to store the filtered binary strings

    for row in bs:
        if row.count('1') == n:
            # If the count of '1' characters in the binary string is equal to 'n', append it to the list
            nbs.append(row)

    return nbs  # Return the list of filtered binary strings

<<<<<<< HEAD
=======

>>>>>>> a188bb899043160906c28568ca175c5a1528ced8
def hamming(s1, s2):
    """
    Calculate the Hamming distance between two binary strings.

    Args:
        s1 (str): The first binary string.
        s2 (str): The second binary string.

    Returns:
        int: The Hamming distance between the two strings.

    Raises:
        AssertionError: If the lengths of s1 and s2 are not equal.
    """
    assert len(s1) == len(s2), "The lengths of the binary strings must be equal."

    # Calculate the Hamming distance by counting the positions where the characters are different
    distance = sum(c1 != c2 for c1, c2 in zip(s1, s2))

    return distance

<<<<<<< HEAD
=======

>>>>>>> a188bb899043160906c28568ca175c5a1528ced8
def p_failure(dq, sq, measurements, n_failure, t1, sqg_error, dqg_error, readout_err):
    """
    Calculate the probability of failure in a quantum system according to Q-BEEP

    Args:
        dq (int): Number of dq errors.
        sq (int): Number of sq errors.
        measurements (int): Number of measurements.
        n_failure (int): Number of failures.
        t1 (float): T1 relaxation time.
        sqg_error (float): SQ gate error rate.
        dqg_error (float): DQ gate error rate.
        readout_err (float): Readout error rate.

    Returns:
        float: The probability of failure.
    """
    p_fail = (1 - (((1 - readout_err) ** measurements) * ((1 - dqg_error) ** dq) * ((1 - sqg_error) ** sq) * (
        np.exp(-(1 * (sq + dq) / t1))))) ** n_failure

    return p_fail

<<<<<<< HEAD
=======

>>>>>>> a188bb899043160906c28568ca175c5a1528ced8
def rand_key(p):
    """
    Generate a random binary string of a desired length.

    Args:
        p (int): The desired length of the binary string.

    Returns:
        str: The generated random binary string.
    """
    key1 = ""  # Variable to store the generated string

    for i in range(p):
        # Use the `random.randint` function to generate a random integer (0 or 1)
        # Convert the result into a string
        temp = str(random.randint(0, 1))

        # Concatenate the random 0 or 1 to the final result
        key1 += temp

    return key1  # Return the generated random binary string

<<<<<<< HEAD
=======

>>>>>>> a188bb899043160906c28568ca175c5a1528ced8
# We need a circuit with n qubits, plus one auxiliary qubit
# Also need n classical bits to write the output to

def gen_bv_circ(n, s):
    """
    Generate a quantum circuit for the Bernstein-Vazirani algorithm.

    Args:
        n (int): The number of qubits in the circuit.
        s (str): The binary string representing the hidden binary string.

    Returns:
        qiskit.QuantumCircuit: The generated Bernstein-Vazirani circuit.
    """
    bv_circuit = qiskit.QuantumCircuit(n + 1, n)  # Create a quantum circuit with n+1 qubits and n classical bits

    # Put auxiliary qubit in state |->
    bv_circuit.h(n)
    bv_circuit.z(n)

    # Apply Hadamard gates before querying the oracle
    for i in range(n):
        bv_circuit.h(i)

    # Apply barrier
    bv_circuit.barrier()

    # Apply the inner-product oracle
    s = s[::-1]  # Reverse s to fit qiskit's qubit ordering
    for q in range(n):
        if s[q] == '0':
            bv_circuit.i(q)
        else:
            bv_circuit.cx(q, n)

    # Apply barrier
    bv_circuit.barrier()

    # Apply Hadamard gates after querying the oracle
    for i in range(n):
        bv_circuit.h(i)

    # Measurement
    for i in range(n):
        bv_circuit.measure(i, i)

    return bv_circuit  # Return the generated Bernstein-Vazirani circuit


def hamming_distance(chaine1, chaine2):
    """
    Calculate the Hamming distance between two strings.

    Args:
        chaine1 (str): The first string.
        chaine2 (str): The second string.

    Returns:
        int: The Hamming distance between the two strings.
    """
    # Calculate the Hamming distance by counting the positions where the characters are different
    distance = sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

    return distance


def pad_zero(num, tlen):
    """
    Pad zeros to the left of a binary number to achieve a target length.

    Args:
        num (str): The binary number.
        tlen (int): The target length of the binary number.

    Returns:
        str: The padded binary number.
    """
    num = num.lstrip('0b')  # Remove the '0b' prefix from the binary number
    p_num = ''  # Variable to store the padded binary number

    for i in range(tlen - len(num)):
        p_num += '0'  # Append '0' to the padded binary number for the remaining length

    return p_num + num  # Concatenate the padded binary number with the original binary number

<<<<<<< HEAD
=======

>>>>>>> a188bb899043160906c28568ca175c5a1528ced8
def gen_bins(num):
    """
    Generate binary strings representing a sequence of bins.

    Args:
        num (int): The number of bits in each binary string.

    Returns:
        list: A list of binary strings representing the sequence of bins.
    """
    bins = []  # List to store the generated binary strings

    # Generate the maximum binary string representation
    max_bin = '1'.join(['' for _ in range(num + 1)])

    # Iterate from 0 to the maximum binary value
    for i in range(int(eval('0b' + max_bin)) + 1):
        # Convert the decimal value to a binary string representation and pad zeros to match the specified length
        bins.append(pad_zero(bin(i), num))

    return bins


def hellinger_explicit(p, q):
    """
    Calculate the Hellinger distance between two discrete distributions.

    Args:
        p (list): The first discrete distribution.
        q (list): The second discrete distribution.

    Returns:
        float: The Hellinger distance between the two distributions.
    """
    list_of_squares = []  # List to store the squares of the differences

    for p_i, q_i in zip(p, q):
        # Calculate the square of the difference of the ith distribution elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

        # Append the square to the list
        list_of_squares.append(s)

    # Calculate the sum of squares
    sosq = sum(list_of_squares)

    # Calculate the Hellinger distance by dividing the sum of squares by the square root of 2
    return sosq / math.sqrt(2)


def complete_results(r_dict: dict):
    """
    Complete the results dictionary by adding all possible binary keys that are not featured.

    Args:
        r_dict (dict): The results dictionary.

    Returns:
        dict: The completed results dictionary.
    """
    # Determine the number of bits based on the length of the keys in the dictionary
    n_bits = len(list(r_dict.keys())[0])

    # Generate a binary string with all bits set to '1'
    b_str = ''.join(['1' for _ in range(n_bits)])

    # Iterate from 0 to the maximum binary value
    for i in range(int(b_str, 2) + 1):
        # Convert the decimal value to a binary string representation and pad zeros if necessary
        key_b = bin(i).lstrip('0b').zfill(n_bits)

        # If the binary key is not present in the results dictionary, add it with a value of 0
        if key_b not in r_dict:
            r_dict[key_b] = 0

    return r_dict

<<<<<<< HEAD
import numpy as np

=======

import numpy as np


>>>>>>> a188bb899043160906c28568ca175c5a1528ced8
def fid_class(dist_a, dist_b):
    """
    Calculate the fidelity between two probability distributions.

    Args:
        dist_a (list): The first probability distribution.
        dist_b (list): The second probability distribution.

    Returns:
        float: The fidelity between the two distributions.
    """
    fid = 0

    for a, b in zip(dist_a, dist_b):
        # Calculate the square root of the product of corresponding elements in the distributions
        fid += np.sqrt(a * b)

    return fid

def get_p_vals(backend, tc):
    """
    Retrieve properties and error rates from a backend and QEC for analysis.

    Args:
        backend: The backend to retrieve properties from.
        tc: The qubits indices that participate in the program

    Returns:
        tuple: A tuple containing t1, t2, sqg_error, dqg_error, and readout_err.
    """
    valid_indxs = set()

    # Extract valid indices from QEC data
    for dta in tc.data:
        dta = dta[1][0].index
        valid_indxs.add(dta)

    aq = list(valid_indxs)
    properties = backend.properties().to_dict()
    t1 = []
    t2 = []
    readout_err = []

    # Retrieve T1, T2, and readout error rates for valid indices
    for qb in valid_indxs:
        t1.append(properties['qubits'][qb][0]['value'])
        t2.append(properties['qubits'][qb][1]['value'])
        readout_err.append(properties['qubits'][qb][4]['value'])

    t1 = np.mean(t1)
    t2 = np.mean(t2)
    readout_err = np.mean(readout_err)
    sqg = ['sx', 'x', 'rz']
    dqg = ['cx']
    sqg_error = []
    dqg_error = []

    # Retrieve gate error rates for valid indices and gate types
    for gate in properties['gates']:
        if len(gate['qubits']) == 1:
            if gate['qubits'][0] not in aq:
                continue
        else:
            if gate['qubits'][0] in aq and gate['qubits'][1] in aq:
                pass
            else:
                continue
        if gate['gate'] in sqg:
            if gate['parameters'][0]['name'] == "gate_error":
                sqg_error.append(gate['parameters'][0]['value'])
        elif gate['gate'] in dqg:
            if gate['parameters'][0]['name'] == "gate_error":
                val = gate['parameters'][0]['value']
                if val < 0.04:
                    dqg_error.append(gate['parameters'][0]['value'])

    dqg_error = np.mean(dqg_error)
    sqg_error = np.mean(sqg_error)

    # Return the extracted properties and error rates
    return t1, t2, sqg_error, dqg_error, readout_err


def poiss_params(dq, sq, measurements, t1, t2, sqg_error, dqg_error, readout_err, time):
    """
    Calculate the error count for a Poisson noise model based on various parameters and error rates.

    Args:
        dq (int): Number of dq errors.
        sq (int): Number of sq errors.
        measurements (int): Number of measurements.
        t1 (float): T1 relaxation time.
        t2 (float): T2 relaxation time.
        sqg_error (float): SQ gate error rate.
        dqg_error (float): DQ gate error rate.
        readout_err (float): Readout error rate.
        time (float): Time duration.

    Returns:
        float: The calculated error count.
    """
    ec = 0

    # Calculate the error count based on the provided parameters and error rates
    ec += dqg_error * dq + sq * sqg_error + readout_err * measurements
    ec += measurements * np.exp(-t1 * 10 ** -6 / (time * 10 ** -9))
    ec += measurements * np.exp(-t2 * 10 ** -6 / (time * 10 ** -9))

    return ec


def poisson(lmda, k):
    """
    Calculate the probability mass function (PMF) of a Poisson distribution.

    Args:
        lmda (float): The average rate of occurrence.
        k (int): The number of events.

    Returns:
        float: The probability of obtaining 'k' events in the Poisson distribution.
    """
    # Calculate the PMF using the formula: lambda^k * e^(-lambda) / k!
    pmf = (lmda ** k) * np.exp(-lmda) / factorial(k)

    return pmf


def to_p_dist(results):
    """
    Convert a dictionary of results into a probability distribution.

    Args:
        results (dict): The dictionary of results.

    Returns:
        list: The probability distribution.
    """
    x = copy.deepcopy(results)  # Create a deep copy of the results dictionary
    tot = sum(x.values())  # Calculate the total sum of values in the dictionary

    # Normalize the values in the dictionary by dividing each value by the total sum
    for key in x:
        x[key] = x[key] / tot

    a = list(x.keys())  # Retrieve the keys (binary strings) from the dictionary
    bins = sorted(a, key=lambda z: int(z, 2))  # Sort the binary strings in ascending order based on their integer value

    res = []  # List to store the probability values

    # Retrieve the probability values corresponding to the sorted binary strings
    for bn in bins:
        res.append(x[bn])

    return res  # Return the probability distribution


def remove_idle_qwires(circ):
    """
    Remove idle quantum wires (qubits) from a given circuit.

    Args:
        circ (QuantumCircuit): The input circuit.

    Returns:
        QuantumCircuit: The modified circuit with idle wires removed.
    """
    dag = circuit_to_dag(circ)  # Convert the circuit to a directed acyclic graph (DAG)

    idle_wires = list(dag.idle_wires())  # Get the list of idle wires from the DAG
    for w in idle_wires:
        dag._remove_idle_wire(w)  # Remove the idle wire from the DAG
        dag.qubits.remove(w)  # Remove the idle wire from the qubits attribute of the DAG

    dag.qregs = OrderedDict()  # Set the qregs attribute of the DAG to an empty OrderedDict

    return dag_to_circuit(dag)  # Convert the modified DAG back to a circuit and return it






def qbeep(circuit,result):
    bit_string_neighbors = {}

# Iterate over the range of bit string lengths
    for i in range(5, 16):
        # Iterate over the range of indices for each bit string length
        for j in range(i - 1):
            # Check if the length key exists in the outer dictionary
            if str(i) not in bit_string_neighbors:
                bit_string_neighbors[str(i)] = {}

            # Check if the index key exists in the inner dictionary
            if str(i) not in bit_string_neighbors[str(i)]:
                # Generate and store the neighbors of the bit string with the given length and index
                s = get_perms(j, i)
                bit_string_neighbors[str(i)][str(j)] = s
            else:
                # Retrieve the stored neighbors from the inner dictionary
                s = bit_string_neighbors[str(i)][str(j)]
    fidelities = []
    # Circuits is a list of your circuits, result_bank is the corresponding
    circuits = [circuit] #circuits = [gen_bv_circ(6,'010101')]  # Your list of circuits here
    result_bank = [result]   #result_bank = [{'010101':100,'010111':25}]  # The corresponding list of results in the respective data structure
    for circuit, results in zip(circuits, result_bank):
        #t1, t2, sqg_error, dqg_error, readout_err = get_p_vals(backend, circuit)
        # For evaluation, we include a simulation to understand the correct distribution.
        sim = qiskit.Aer.get_backend('aer_simulator')
        res = qiskit.execute(circuit, sim, shots=100)
        ideal_res = res.result().get_counts()
        corr_soln = list(ideal_res.keys())[0]
        print(f"Correct Solution: {corr_soln}")
        ideal_res = complete_results(ideal_res)
        ideal_res = to_p_dist(ideal_res)
        try:
            dq = circuit.count_ops()['cx']
        except:
            dq = 0
        ops = circuit.count_ops()
        if 'cx' in ops.keys():
            sq = sum(ops.values()) - ops['barrier'] - ops['cx'] - ops['measure']
        else:
            sq = sum(ops.values()) - ops['barrier'] - ops['measure']
        measurements = ops['measure']
        #time = qiskit.schedule(circuit, backend).stop_time
        #pparam = poiss_params(dq, sq, measurements, t1, t2, sqg_error, dqg_error, readout_err, time)
        pparam=0.1
        # if results[corr_soln] == 0:
        #     # QBEEP is not applicable to
        #     print(f"This solution observed no correct bs.")
        #     continue
        results = complete_results(results)
        p_prior = results[corr_soln]
        nb = len(list(results.keys())[0])
        peaked = False
        begin = 1
        for i in range(10):
            if i > nb:
                cut_off = i - 1
                break
            # Arbitrary for debugging
            if poisson(pparam, i) > 0.2:
                peaked = True
            if poisson(pparam, i) < 0.05 and peaked:
                cut_off = i
                break
        nn = {}
        fidelities.append([])
        p_dist_corr = to_p_dist(results.copy())
        fidelities[-1].append(fid_class(p_dist_corr, ideal_res))
        # This is the most computationally costly section of our algorithm.
        # Current versions populate 2^n bins. This is not necessary, as you only need
        # to populate up to k bins, where k is the number of observations you made on your circuit.
        # We did not do this optimization in the code. But the change can be made for the same results.
        for key in results.keys():
            if results[key] == 0:
                nn[str(key)] = {}
                continue
            for i in np.arange(1, cut_off):
                s = bit_string_neighbors[str(nb)][str(i)]
                if key not in nn:
                    nn[str(key)] = {}
                nn[str(key)][str(i)] = []
                for ham in s:
                    z = xor_convert(ham, key)
                    nn[str(key)][str(i)].append(z)
        deltas = {}
        # For this, we are generating the dictionary of P(A|B) = P(B|A)*P(A)/P(B)
        # In each iteration, P(A) is each result key, and P(B) is a H Dist away bitstring
        # i.e. the Deltas objects keys represent the probability of each result belonging to Key
        # given that we observed Bstr. This is computed as P(Bstr Observed Model)*P(Class)/P(Bstr Observed)
        mitigate_result = []
        # t1 = time.time()
        for UPDATE in range(20):
            lr = 1 / (UPDATE + 1)
            inital_state = copy.deepcopy(results)
            for key in results.keys():
                deltas[key] = {}
                if results[key] != 0:
                    for h_dist in nn[key]:
                        hdist = poisson(pparam, int(h_dist))
                        # Iterate over each bitstring nn[key] distance away (HAMMING)
                        for bstr in nn[key][h_dist]:
                            # If initial_state[bstr] != 0
                            if results[key] != 0:
                                deltas[key][bstr] = hdist * results[bstr] / results[key]
                            else:
                                deltas[key][bstr] = 0
                    tot = sum(deltas[key].values())
                    if tot > 1:
                        for bstr in deltas[key].keys():
                            deltas[key][bstr] = deltas[key][bstr] / tot
            updates = copy.deepcopy(deltas)
            # In this algorithm, we iterate over each key in the updates dictionary.
            # Over each iteration, we take the update key, and the bistrstring corersponding
            # I.e. We say that the
            for change_key in updates.keys():
                for bstr in updates[change_key].keys():
                    updates[change_key][bstr] = int(lr * updates[change_key][bstr] * results[change_key])
                tchange = sum(updates[change_key].values())
                cap = results[change_key]
                if tchange > cap:
                    for bstr in updates[change_key].keys():
                        updates[change_key][bstr] = int(updates[change_key][bstr] * cap / tchange)
            change = {}
            for key in updates:
                change[key] = {}
                change[key]['out'] = sum(updates[key].values())
                change[key]['in'] = 0
                for out_key in updates.keys():
                    if out_key == key:
                        continue
                    elif key in updates[out_key]:
                        change[key]['in'] += updates[out_key][key]
            for key in change:
                results[key] = results[key] - change[key]['out'] + change[key]['in']
            mitigate_result.append(results)
        # t2 = time.time()
        # print((t2-t1)*1000)

    return mitigate_result[19]
        

