import pickle
from utils import *
from benchmark import ghz
from simulator import Simulator
from m3_D import *
from dataloader import Dataloader

protocol_results = pickle.load(file=open('dataset/np_131bit.pickle','rb'))

for n_qubits in [7,18,27,36,49,79,131]:

    circuits = ghz(n_qubits)
    sim_noise = Simulator(n_qubits,0.025,0.03)  
    error_result = sim_noise.execute(circuits)[0]

    measured_bit = [i for i in range(n_qubits)]

    dataloader = Dataloader(sim_noise)
    protocol_results_dataset = dataloader.get_data(eval = True, machine_data = protocol_results, threshold = 2.5e-5)
    protocol_results_dataset = np_to_dic(protocol_results_dataset)
    protocol_results_dataset = downsample_protocol_result(protocol_results_dataset,measured_bit)

    print('QuFEM,The number of circuits used for readout characterization', len(protocol_results_dataset))


    m3_space = get_space(3,error_result,n_qubits)
    print('M3,The number of circuits used for readout characterization', len(m3_space))
