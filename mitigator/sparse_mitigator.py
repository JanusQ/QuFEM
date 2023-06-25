import random
from utils import to_bitstring
import numpy as np
from scipy.sparse import csc_matrix
from scipy import linalg
import scipy.sparse.linalg as sp_linalg
from scipy.sparse import csc_matrix, isspmatrix, coo_matrix

class SparseMitigator():
    def __init__(self, n_qubits, M = None):
        self.n_qubits = n_qubits
        self.M = M
        if M is not None and isspmatrix(M):
            self.invM = sp_linalg.inv(self.M)
        
    def characterize_M(self, protocol_results, filter = True, threshold = 1e-2):
        n_qubits = self.n_qubits
        
        row, col, data = [], [], []
        
        for real_bitstring, status_count in protocol_results.items():
            
            if '2' in real_bitstring: continue
            
            real_bitstring = int(real_bitstring, base=2)
            total_count = sum(status_count.values())
            for measure_bitstring, count in status_count.items():
                measure_bitstring = int(measure_bitstring, base=2)
                if filter and count / total_count > threshold:
                    row.append(measure_bitstring)
                    col.append(real_bitstring)
                    data.append(count / total_count)
        
        self.M = coo_matrix((data, (row, col)), shape = (2**n_qubits, 2**n_qubits))

        self.M = self.M.tocsc()
        
        # col_sums = self.M.sum(axis = 0)
        
        # for idx, col_sum in enumerate(col_sums):
        #     self.M[idx] /=  col_sum
        
        
        import time
        start = time.time()
        self.invM = sp_linalg.inv(self.M)
        print('inverse time',time.time() - start)
        return self.M
        
    def mitigate(self, stats_counts: dict):
        
        # total_count = sum(stats_counts.values())
        error_count_vec = np.zeros(2 ** self.n_qubits)
        for basis, count in stats_counts.items():
            error_count_vec[int(basis, base=2)] = count
            
        rm_prob = {}
        import time
        start = time.time()
        rm_count_vec = self.invM.dot(error_count_vec)
        print('dot time',time.time() - start)
        rm_count_vec[rm_count_vec < 0] = 0
        rm_count_vec /= sum(rm_count_vec)
        for basis, prob in enumerate(rm_count_vec):
            rm_prob[to_bitstring(basis, self.n_qubits)] = prob

        return rm_prob
     