import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from NetworkSystem import JacobianNetworkSystemEigenmode
from InitEP import InitCondEPHomogeneous

def RoutineVaps(vapsConn,params, findEP=True, initCond=0):
        
    #Initialize empty NvariablesxNpop matrix to store Nvariables-Vaps of each dimension alpha 
    Npop=len(vapsConn)
    Nvariables=6
    matrixVaps = np.zeros((Nvariables,Npop),dtype=np.complex128)

    if findEP:
        initCond = InitCondEPHomogeneous(params)
    
    for idx in range(Npop):
        #print('------ Iteration: '+str(idx)+' ------')
        #Set eigenvalue for current iteration
        eig = vapsConn[idx]

        #Compute Jacobian matrix for each eigenvalue
        jacobian = JacobianNetworkSystemEigenmode(initCond,params,eig)

        #Compute eigenvalues and eigenvectors for each alpha
        vapsAlpha,vepsAlpha = np.linalg.eig(jacobian)

        #Save current 6 eigenvalues
        matrixVaps[:,idx] = vapsAlpha[:]

    return matrixVaps
