#!/usr/bin/env python3

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute, BasicAer

import numpy as np
import matplotlib.pyplot as plt

def get_state_vector(qc):
    backend = BasicAer.get_backend('statevector_simulator')
    res = execute(qc, backend).result()
    return res.get_statevector(qc, decimals=3)
    
def create_bell_pair(qc, a, b):
    qc.h(a)
    qc.cx(a,b)
