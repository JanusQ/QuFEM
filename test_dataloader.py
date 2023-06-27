from dataloader import Dataloader
from simulator import MeasurementAwareNonLocalSimulator

iter_scores = [] 
for n_qubits in range(5,10):
# n_qubits = 20
    simulator = MeasurementAwareNonLocalSimulator(n_qubits, MeasurementAwareNonLocalSimulator.gen_random_M(n_qubits))
    loader = Dataloader(simulator)
    bitstring_dataset, protocol_results_dataset, cor, uncertainty, iter_score = loader.get_data(eval= True)
    iter_scores.append(iter_score)
    
print(iter_scores)
print('finish')