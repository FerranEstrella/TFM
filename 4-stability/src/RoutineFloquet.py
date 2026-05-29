import numpy as np
from scipy.integrate import solve_ivp
from FloquetExponentsVariationals import ComputeFloquetExponents
from InitPO import InitCondPOHomogeneous


def RoutineFloquet(vapsConn,params, findPO=True, initCond=0, T=0):
        
    Nvariables=6
    Npop=len(vapsConn)

    #Initialize empty NvariablesxNpop matrices to store Nvariables-FloquetExp and Nvariables-FloquetMult of each dimension alpha 
    matrixFloquetExp = np.zeros((Nvariables,Npop),dtype=np.complex128)
    matrixFloquetMult = np.zeros((Nvariables,Npop),dtype=np.complex128)

    #Compute initial condition for periodic orbit and period with these params
    if findPO:
        status, initCond, T = InitCondPOHomogeneous(params)

    for idx in range(Npop):
        #print('------ Iteration: '+str(idx)+' ------')
        #Set eigenvalue for current iteration
        eig = vapsConn[idx]

        #Compute Nvariables Floquet exponents and multipliers corresponding to current eigenvalue alpha
        FloquetExp,FloquetMult,veps = ComputeFloquetExponents(initCond,params,eig,T)

        #Save Floquet exponents and multipliers in corresponding position of the matrices

        matrixFloquetExp[:,idx] = FloquetExp
        matrixFloquetMult[:,idx] = FloquetMult

    return matrixFloquetExp,matrixFloquetMult
