from dataloader import Dataloader
from simulator import MeasurementAwareNonLocalSimulator
from utils import load


iter_scores = [] 
for n_qubits in range(11,12):
# n_qubits = 20
    simulator = MeasurementAwareNonLocalSimulator(n_qubits, MeasurementAwareNonLocalSimulator.gen_random_M(n_qubits))
    loader = Dataloader(simulator)
    bitstring_dataset, protocol_results_dataset, cor, uncertainty, iter_score = loader.get_data(eval= True)
    iter_scores.append(iter_score)
print(iter_scores)


# all_res = load('dataset/protocal_result_11bit.pkl')

# simulator = MeasurementAwareNonLocalSimulator(11, MeasurementAwareNonLocalSimulator.gen_random_M(11))
# loader = Dataloader(simulator)
# bitstring_dataset, protocol_results_dataset, cor, uncertainty, iter_score = loader.get_data(eval= True, machine_data= all_res)
# print('finish')