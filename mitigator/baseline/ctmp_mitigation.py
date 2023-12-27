from qcompute_qep.measurement.correction import CTMPCorrector
# import qcompute_g

ctmp = CTMPCorrector(qcomputer, calibrator='ctmp', qubits=qubits, k=3)
ctmp = CTMPCorrector(qc=qc, qubits=qubits)
ctmp = CTMPCorrector(calibrator='ctmp')
ctmp = CTMPCorrector(calibrator='ctmp', cal_data=cal_data, qubits=qubits)