import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile, assemble, execute, IBMQ
import math
import random
import numpy as np
from scipy.optimize import minimize
import SPSA as spsa
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.components.optimizers import SPSA
import pandas as pd
from qiskit.providers.ibmq.managed import IBMQJobManager
from datetime import datetime
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn, CircuitOp, CircuitStateFn, ListOp


#SOLVE THE Ax=b PROBLEM USING THE BRAVO-PRIETO et al. (2020) VQLS Method


###########################################################################
#FUNCTIONS
###########################################################################

#THE SELECTION OF POSSIBLE ANSATZ
def Prieto_Mod(circ, qubits, parameters):
    #PRIETO et al. (2020)
    #circ = QuantumCircuit(nqubit)
    
    for iz in range(0,len(qubits)):
        circ.ry(parameters[iz],qubits[iz])
    
    idx = nqubit-1
    for ly in range(0,nlayer):
        for oe in range(0,2):
            for iz in range(0,int(nqubit/2)-oe):
                circ.cx(2*iz+oe,2*iz+1+oe)
                
            for iz in range(0+oe,nqubit-oe):
                idx += 1
                circ.ry(parameters[idx],qubits[iz])

    return circ
               
def apply_fixed_ansatz(circ,ansatz,qubits,parameters):
    if (ansatz=='Prieto_Mod'):
        circ = Prieto_Mod(circ,qubits,parameters)    
    return circ

def number_param(ansatz,nqubit,nlayer):
    if (ansatz=='Prieto_Mod'):
        nparam = nqubit + nlayer*(2*nqubit-2)
    return nparam


#b VECTOR PREPARATION
def b_state(op_circ, qubits):
    for ia in qubits:
        op_circ.h(ia)
    return op_circ

    
#EXPECTATION VALUE FOR DENOMINATOR TERM
#LABELLED AS BETA IN PRIETO
def beta_ij(gate_type, parameters, qubits):   
    #A_l
    op_circ = QuantumCircuit(nqubit)
    for ie in range (0, nqubit):
        if (gate_type[0][ie] == 1):
            op_circ.x(qubits[ie])  
        elif (gate_type[0][ie] == 2):
            op_circ.y(qubits[ie])                
        elif (gate_type[0][ie] == 3):
            op_circ.z(qubits[ie]) 
     
    #A_l'_dagger
    ####FLAG: NOTE THAT SINCE I AM USING PAULI MATRICES 
    ####      IT FOLLOWS THAT A=A_dagger
    for ie in range (0, nqubit):
        if (gate_type[1][ie] == 1):
            op_circ.x(qubits[ie])  
        elif (gate_type[1][ie] == 2):
            op_circ.y(qubits[ie])                
        elif (gate_type[1][ie] == 3):
            op_circ.z(qubits[ie]) 

    return op_circ

#EXPECTATION VALUE FOR DENOMINATOR TERM
#LABELLED AS DELTA IN PRIETO
def delta_ijk(gate_type, parameters, qubits, k):
    #A_l
    op_circ = QuantumCircuit(nqubit)
    for ie in range (0, nqubit):
        if (gate_type[0][ie] == 1):
            op_circ.x(qubits[ie])  
        elif (gate_type[0][ie] == 2):
            op_circ.y(qubits[ie])                
        elif (gate_type[0][ie] == 3):
            op_circ.z(qubits[ie]) 
    op_circ.barrier(qubits)
    
    #U_dagger 
    ####FLAG: U_dagger=U SINCE U=HHH
    op_circ = b_state(op_circ,qubits)
    op_circ.barrier(qubits)
    
    #Z
    op_circ.z(k)
    op_circ.barrier(qubits)
    
    #U
    op_circ = b_state(op_circ,qubits)
    op_circ.barrier(qubits)
    
    #A_l'_dagger
    ####FLAG: NOTE THAT SINCE I AM USING PAULI MATRICES 
    ####      IT FOLLOWS THAT A=A_dagger
    for ie in range (0, nqubit):
        if (gate_type[1][ie] == 1):
            op_circ.x(qubits[ie])  
        elif (gate_type[1][ie] == 2):
            op_circ.y(qubits[ie])                
        elif (gate_type[1][ie] == 3):
            op_circ.z(qubits[ie]) 
    op_circ.barrier(qubits)
    
    return op_circ

#CALCULATE THE LOCAL COST FUNCTION IN PRIETO
def calculate_cost_function(parameters):   
    qubits = [x for x in range(0,nqubit)]
    
    ops      = []
    multiply = []   
    overall_sum_1 = 0 
    overall_sum_2 = 0
    
    #FIRST DEFINE |psi>
    psi = QuantumCircuit(nqubit)
    psi = apply_fixed_ansatz(psi, ansatz, qubits, parameters)
    psi = CircuitStateFn(psi)
    
    for i in range(0,len(gate_set)):
        for j in range(0, len(gate_set)):        
            #COEFFICIENTS
            multiply.append(coefficient_set[i]*coefficient_set[j])
            
            #EXPECTATION VALUE
            circ  = beta_ij([gate_set[i], gate_set[j]], parameters, qubits)
            ops.append(CircuitOp(circ))

    for i in range(0,len(gate_set)):
        for j in range(0,len(gate_set)):
            for k in range(0,nqubit):
                #COEFFICIENTS
                multiply.append(coefficient_set[i]*coefficient_set[j])
                
                #EXPECTATION VALUE
                circ = delta_ijk([gate_set[i], gate_set[j]], parameters, qubits, k)
                ops.append(CircuitOp(circ)) 
                
    startTime = datetime.now()
    #RUN CIRCUITS
    if (simulator=='IBM_sim'):
        backend  = provider.get_backend('ibmq_qasm_simulator')
    else:
        backend = Aer.get_backend('qasm_simulator')
    q_instance = QuantumInstance(backend, shots=nshots, optimization_level=0, \
                                 skip_qobj_validation=False)                                     
    measurable_expression = StateFn(ListOp(ops), is_measurement=True).compose(psi)
    expectation = PauliExpectation(group_paulis=False).convert(measurable_expression)
    sampler = CircuitSampler(q_instance).convert(expectation).eval()
    print('Transpilation + Pauli Conversion Time ', datetime.now() - startTime) 

    m = -1
    for i in range (0,len(gate_set)**2):
        m += 1
        #SUM THE 1ST PART OF THE COST FUNCTION
        overall_sum_1 += multiply[m]*sampler[m]                
    
    for i in range (0,len(gate_set)**2*nqubit):
        m += 1
        #SUM THE 2ND PART OF THE COST FUNCTION
        overall_sum_2 += multiply[m]*sampler[m]          
    
    #ABSOLUTE VALUES
    overall_sum_1 = np.abs(overall_sum_1)
    overall_sum_2 = np.abs(overall_sum_2)
    
    cost_func =  0.5* ( 1. - 1./nqubit * overall_sum_2/overall_sum_1 )
    print(cost_func)
    return cost_func

#MAKE THE A MATRIX FROM INPUT PARAMETERS
def which_pauli(pauli):
    sig_1 = np.array([[1,0],[0,1]])
    sig_x = np.array([[0,1],[1,0]])
    sig_y = np.array([[0,-1j],[1j,0]])
    sig_z = np.array([[1,0],[0,-1]])
    if (pauli == 0):
        ai = sig_1
    elif (pauli == 1):
        ai = sig_x
    elif (pauli == 2):
        ai = sig_y                
    elif (pauli == 3):
        ai = sig_z
    return ai

def makeA(gate_set,coefficient_set):
    nterm = len(gate_set[:,0])
    npaul = len(gate_set[0,:])
    A     = 0
    for i in range(0,nterm):
        ai = which_pauli(gate_set[i,npaul-1])
        for j in range(npaul-2,-1,-1):
            ai = np.kron(ai,which_pauli(gate_set[i,j]))            
        A = A + coefficient_set[i]*ai
    return A


###########################################################################
#MAIN
###########################################################################

#SETTINGS 
optimizer = 'SPSA'
ansatz = 'Prieto_Mod'
nlayer = 2
nqubit = 4
nitr = 100
nshots = 8192  #MAX SHOTS
tol = 10**(-3)
magchange = 2. #INITIAL STEP SIZE IN DIRECTION OF SPSA GRADIENT
nexp   = 1     #NUMBER OF EXPERIMENTS
simulator = 'not_IBM_sim' #'IBM_sim'
limitbounds = True
outdir = './'

if (nqubit==4):
    coefficient_set = [1.5, 0.25]
    gate_set = [[0,0,0,1], [1,3,0,0]]
elif (nqubit==6):
    coefficient_set = [1.5, 0.25]
    gate_set = [[0,0,0,0,0,1], [1,3,0,0,0,0]]
elif (nqubit==8):
    coefficient_set = [1.5, 0.25]
    gate_set = [[0,0,0,0,0,0,0,1], [1,3,0,0,0,0,0,0]]
else:
    print('Must adjust the coefficient_set and gate_set')
    exit()



#SET 'A' AND 'b'
A = makeA(np.array(gate_set),np.array(coefficient_set))
b = np.zeros(2**nqubit) + 1./np.sqrt(2**nqubit)

metric  = np.zeros((nexp))
for ex in range(0,nexp):
    startTime = datetime.now()

    #ANSATZ PARAMETERS
    nparam = number_param(ansatz,nqubit,nlayer)
    random_numbers = np.array([float(random.randint(0,6200))/1000 for i in range(0, nparam)])

    #OPTIMIZE
    th_min = np.zeros(nparam)
    th_max = np.zeros(nparam) + 2.*np.pi
    out,norm_err,cost2,traj  = spsa.SPSA(calculate_cost_function,random_numbers,tol,nitr,\
                                         th_min,th_max,False,limitbounds,magchange)

    #GET THE FINAL SOLUTION
    circ = QuantumCircuit(nqubit, 1)
    apply_fixed_ansatz(circ, ansatz, [x for x in range(0,nqubit)], out)
    backend = Aer.get_backend('statevector_simulator')
    t_circ = transpile(circ, backend)
    qobj = assemble(t_circ)
    job = backend.run(qobj)
    result = job.result()
    o = result.get_statevector(circ, decimals=10)
    o = np.abs(o)
    metric[ex] = (b.dot(A.dot(o)/(np.linalg.norm(A.dot(o)))))**2


print(metric)





    