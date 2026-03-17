import numpy as np
from scipy.integrate import solve_ivp
from FloquetExponentsVariationals import ComputeFloquetExponents
from InitPO_2 import InitCondPOHomogeneous_2


def RoutineFloquet(vapsConn,Nvariables,Npop,params,size):
        
    #Inputs model: 
    #Nvariables: (Integer) Number of variables of the neural mass model
    #Npop: (Integer) Number of populations considered in the network
    #W: (NpopsxNpop) Normalized Structural Connectivity matrix
    #Parameters: (1x15 Float) Array containing the parameters of the model
        #tau_e,tau_i, tau_se, tau_si: Time constants of the model
        #nu_e,nu_i: Baseline constant current for excitatory,inhibitory neurons
        #Delta_e,Delta_i: Mean neuron noise intensity over excitatory,inhibitory
        #Jpq: For p,q in {e,i} Synaptic strength between E-I populations
        #Iext_e,Iext_i: External currents
        #eps: Coupling strength between network populations

    #Initialize empty NvariablesxNpop matrices to store Nvariables-FloquetExp and Nvariables-FloquetMult of each dimension alpha 
    matrixFloquetExp = np.zeros((Nvariables,size),dtype=np.complex128)
    matrixFloquetMult = np.zeros((Nvariables,size),dtype=np.complex128)

    #Compute initial condition for periodic orbit and period with these params
    status, initCond, T = InitCondPOHomogeneous_2(Nvariables,params)


    for idx in range(len(vapsConn)):
        #print('------ Iteration: '+str(idx)+' ------')
        #Set eigenvalue for current iteration
        eig = vapsConn[idx]

        #Compute Nvariables Floquet exponents and multipliers corresponding to current eigenvalue alpha
        FloquetExp,FloquetMult,veps = ComputeFloquetExponents(Nvariables,eig,initCond,T,params)

        #Save Floquet exponents and multipliers in corresponding position of the matrices

        matrixFloquetExp[:,idx] = FloquetExp
        matrixFloquetMult[:,idx] = FloquetMult

    return matrixFloquetExp,matrixFloquetMult
