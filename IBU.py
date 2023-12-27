from PyIBU.utils.qc_utils import *
from PyIBU.src.IBU import IBU
from PyIBU.utils.data_utils import *
import time

def ibu(circuit,error_result,n_qubits,ideal_result):

    circuit.measure_active()
    from qiskit.providers.fake_provider import FakeWashington
    backend = FakeWashington()

    ac_qubits = get_active_qubits_from_ghz_circuit(circuit)
    matrices = [get_response_matrix(backend, q) for q in ac_qubits]


    params = {
        "exp_name": "ghz",
        "method": "reduced",  # options: "full", "reduced"
        "library": "jax",  # options: "tensorflow" (for "full" only) or "jax"
        "num_qubits": n_qubits,
        "max_iters": 100000,
        "tol": 1e-5,
        "use_log": False,  # options: True or False
        "verbose": True,
        "init": "unif",  # options: "unif" or "unif_obs" or "obs"
        "smoothing": 1e-8,
        "ham_dist": 3
    }
    if params["library"] == 'tensorflow':
        params.update({
            "max_iters": tf.constant(params["max_iters"]),
            "eager_run": True
        })
        tf.config.run_functions_eagerly(params["eager_run"])

    ibu = IBU(matrices, params)
    ibu.set_obs(dict(error_result))
    ibu.initialize_guess()
    t1 = time.time()
    t_sol, max_iters, tracker = ibu.train(params["max_iters"], tol=params["tol"], soln=ideal_result)
    mitigate_result = ibu.guess_as_dict()
    print("ibu calibration time",time.time()-t1)
    # mitigate_result = vec_to_dict(t_sol)

    
    return mitigate_result

