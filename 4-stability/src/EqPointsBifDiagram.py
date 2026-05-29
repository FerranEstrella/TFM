# 0a. Load required Python libraries
import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from pathlib import Path

import matplotlib.pyplot as plt

from InitEP import InitCondEPHomogeneous
from RoutineEqPoints import RoutineVaps


def EqPointsBifDiagram(W_file="normalized_matrix_4cluster.npy"):
    
    # 0b. Load structural connectivity matrix
    root = Path(__file__).resolve().parent.parent
    W = np.load( root / "data" / W_file)
    Npop=W.shape[0]

    # 0c. Load Floquet Data obtained from FloquetBifDiagram.py
    saved_data = np.load(root / "scripts" / f'FloquetBifDiagram_Npop={Npop}.npz')
    dataVector_Iext_e = saved_data['vector_Iext_e']
    dataVector_Iext_e.shape
    dataVector_eps = saved_data['vector_eps']
    dataVector_eps.shape
    dataStatus = saved_data['dataStatus']
    dataStatus.shape

    # Compute eigenvalues and eigenvectors of connectivity matrix W
    vapsConn,vepsConn = np.linalg.eig(W) 

    # 1a. Set parameters of the model
    Nvariables = 6
    params = dict(tau_e = 8,
                tau_i = 8,
                tau_se=1,
                tau_si=5,
                nu_e = -5,
                nu_i = -5,
                Delta_e = 1,
                Delta_i = 1,
                Jee = 5,
                Jei = 13,
                Jii = 5,
                Jie = 13,
                Iext_i=0,
                Iext_e = 0,
                eps = 0)

    # Initialize empty (long_Iext_e,long_eps,Npop) array structure to store characteristic exponents
    dataVapsReal = np.zeros((dataStatus.shape[0],dataStatus.shape[1],Npop))
    dataVapsImaginary = np.zeros((dataStatus.shape[0],dataStatus.shape[1],Npop))

    # Initialize empty (long_Iext_e,long_eps) array structure to store IC of EP
    ICs = np.zeros((dataStatus.shape[0], dataStatus.shape[1],Nvariables))
    # Store number of positive characteristic exponents among population modes
    numPositive = np.zeros((dataStatus.shape[0], dataStatus.shape[1]))


    # For loop over fixed points
    for idx_Iext_e in range(dataStatus.shape[0]):
        for idx_eps in range(dataStatus.shape[1]):

            if dataStatus[idx_Iext_e,idx_eps] == 0:

                print(' -----------  Iext_e: ',dataVector_Iext_e[idx_Iext_e],
                      'eps: ',dataVector_eps[idx_eps],'  -----------')

                params['Iext_e'] = dataVector_Iext_e[idx_Iext_e]
                params['eps'] = dataVector_eps[idx_eps]

                initCond = InitCondEPHomogeneous(params)
                ICs[idx_Iext_e,idx_eps] = initCond
                
                matrixVaps = RoutineVaps(vapsConn, params, findEP=False, initCond=initCond)

                MaxRealVaps = np.amax(np.real(matrixVaps), axis=0)
                dataVapsReal[idx_Iext_e,idx_eps,:] = MaxRealVaps

                indicesMaxReal = np.argmax(np.real(matrixVaps), axis=0)
                dataVapsImaginary[idx_Iext_e,idx_eps,:] = \
                    np.imag(matrixVaps)[indicesMaxReal, np.arange(Npop)]

                TOL = 1e-4
                numPositive[idx_Iext_e,idx_eps] = np.sum(MaxRealVaps > TOL)

            else:
                dataVapsReal[idx_Iext_e,idx_eps] = np.nan*np.ones(Npop)
                dataVapsImaginary[idx_Iext_e,idx_eps] = np.nan*np.ones(Npop)
                numPositive[idx_Iext_e,idx_eps] = np.nan
                ICs[idx_Iext_e,idx_eps] = np.nan


    # Save data
    rows = []
    dataStatus = dataStatus.astype(int)

    for i in range(len(dataVector_Iext_e)):
        for j in range(len(dataVector_eps)):

            row = [
                dataVector_Iext_e[i],
                dataVector_eps[j],
                dataStatus[i, j],
            ]

            # Expand ICs
            if np.isnan(ICs[i, j]).all():
                row.extend([np.nan]*Nvariables)
            else:
                row.extend(ICs[i, j].tolist())

            # Characteristic exponents
            row.extend(dataVapsReal[i, j, :].tolist())

            # Number of unstable directions
            row.append(numPositive[i, j])

            rows.append(row)

    columns = (
        ["Iext_e", "eps", "status"] +
        [f"x{k}" for k in range(Nvariables)] +
        [f"lambda_{k}" for k in range(Npop)] +
        ["numPositive"]
    )

    df = pd.DataFrame(rows, columns=columns)

    df.to_csv(
        root / "scripts" / f"EqPointsBifDiagram_Npop={Npop}.txt",
        sep=" ",
        index=False
    )



    np.savez(root / "scripts" / f'EqPointsBifDiagram_Npop={Npop}.npz',vector_Iext_e=dataVector_Iext_e,vector_eps=dataVector_eps,dataVapsReal=dataVapsReal,dataStatus=dataStatus, numPositive=numPositive, ICs=ICs)




