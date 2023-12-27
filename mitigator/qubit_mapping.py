from utils import downsample_status_count

    
def permute(stats_counts, qubit_order: list):
    permuted_stat_counts = {}
    for bitstring, count in stats_counts.items():
        new_bitstring = ['0'] * len(bitstring)
        for now_pos, old_pos in enumerate(qubit_order):
            new_bitstring[now_pos] = bitstring[old_pos]
        permuted_stat_counts[''.join(new_bitstring)] = count
    return permuted_stat_counts

# N个比特 -> M个比特，N >= M 然后可以进行down sample
class DownSampleMapping():
    # physical qubit -> local qubit
    def __init__(self, included_qubits):
        self._l2p = included_qubits
        
        self._p2l = {
            p_qubit: l_qubit
            for l_qubit, p_qubit in enumerate(included_qubits)
        }
        # for real_pos, old_pos in enumerate(qubit_map):
        #     qubit_map_inv[old_pos] = real_pos
        return
    
    def __getitem__(self, physical_qubit):
        return self.p2l(physical_qubit)
    
    def p2l(self, physical_qubit):
        return self._p2l[physical_qubit]
    
    def l2p(self, logical_qubit):
        return self._l2p[logical_qubit]
    
    def stats_count_p2l(self, stats_count):
        
        return
    
    def stats_count_l2p(self, stats_count):
        return


# TODO: 实现
# N个比特 -> N个比特，比特的映射关系变了

