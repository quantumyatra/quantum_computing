#!/usr/bin/env python3

import qiskit
from qiskit import Aer, execute

def  get_statevector(qc, counts=False):
    backend = Aer.get_backend('statevector_simulator')
    res  = execute(qc, backend=backend).result()
    final_state = res.get_statevector()
    counts = res.get_counts()
    if counts:
        return (final_state, counts)
    return final_state
