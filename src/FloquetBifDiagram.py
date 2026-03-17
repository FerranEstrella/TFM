import numpy as np

from InitPO_2 import InitCondPOHomogeneous_2
from FloquetExponentsVariationals import ComputeFloquetExponents

def FloquetBifDiagram(W_file="NormalizedMatrix.npy", stability = False):
    # Load structural connectivity matrix
   
    root = Path(__file__).resolve().parent.parent

    W = np.load( root / "data" / W_file)
    p = {'scalar_params': scalar_params, 'matrix_params': W}

    Npop = W.shape[0]
    Nvariables = 6
    norm_matrix = W
    

    #Compute eigenvalues and eigenvectors of connectivity matrix W
    vapsConn,vepsConn = np.linalg.eig(norm_matrix)

    # Define discretization
    h = 0.05

    # Set some parameters
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
                Iext_i=0)

    # Define boundaries for Iext_e
    min_Iext_e = 0
    max_Iext_e = 16


    # Define boundaries for eps
    min_eps = 0
    max_eps = 30

    # Define number of points in Iext_e axis
    long_Iext_e = int((max_Iext_e-min_Iext_e)/h)
    # Define number of points in eps axis
    long_eps = int((max_eps-min_eps)/h)

    # Initialize empty (long_Iext_e,long_eps,Npop) array structure to store Floquet exponents
    dataFloquetReal = np.zeros((long_Iext_e+1,long_eps+1,Npop))
    dataFloquetImaginary = np.zeros((long_Iext_e+1,long_eps+1,Npop))

    # Initialize empty (long_Iext_e,long_eps) array structure to store status of each point
    dataStatus = np.zeros((long_Iext_e+1,long_eps+1))

    # Define vectors for Iext_e and eps axis
    vector_Iext_e = np.linspace(min_Iext_e,max_Iext_e,long_Iext_e+1)
    vector_eps = np.linspace(min_eps,max_eps,long_eps+1)

    for idx_Iext_e in range(len(vector_Iext_e)):
        for idx_eps in range(len(vector_eps)):
            # Show progress
            print(' -----------  Iext_e: ',vector_Iext_e[idx_Iext_e],'eps: ',vector_eps[idx_eps],'  -----------')

            # Update parameters for current point
            params['Iext_e'] = vector_Iext_e[idx_Iext_e]
            params['eps'] = vector_eps[idx_eps]

            # 1. Check if model produces oscillations
            # 2. Check if oscillations are periodic
            # 3. Compute Floquet exponents

            #Compute initial condition for periodic orbit and period with these params
            status, initCond, T = InitCondPOHomogeneous_2(Nvariables,params)

            print('InitCond status:', status)

            #Update dataStatus accordingly
            dataStatus[idx_Iext_e,idx_eps] = status

            #Initialize empty NvariablesxNpop matrices to store Nvariables-FloquetExp and Nvariables-FloquetMult of each dimension alpha 
            matrixFloquetExp = np.zeros((Nvariables,Npop),dtype=np.complex128)
            matrixFloquetMult = np.zeros((Nvariables,Npop),dtype=np.complex128)

            # Periodic Oscillations
            if status == 2 or status == 4:
                for idx in range(len(vapsConn)):
                    #print('------ Iteration: '+str(idx)+' ------')
                    #Set eigenvalue for current iteration
                    eig = vapsConn[idx]

                    #Compute Nvariables Floquet exponents and multipliers corresponding to current eigenvalue alpha
                    FloquetExp,FloquetMult,veps = ComputeFloquetExponents(Nvariables,eig,initCond,T,params)

                    #Save Floquet exponents and multipliers in corresponding position of the matrices
                    #print(FloquetExp)

                    matrixFloquetExp[:,idx] = FloquetExp
                    matrixFloquetMult[:,idx] = FloquetMult

                #Save maximum of the REAL part among the Nvariables FloquetExponents per each eigenvalue of W
                MaxRealFloquetExp  = np.amax(np.real(matrixFloquetExp),axis=0)
                #Save indices corresponding to the previous maximum values
                indicesMaxReal = np.argmax(np.real(matrixFloquetExp),axis=0)
                #Save imaginary part of the previous maximum values
                MaxImagFloquetExp = np.imag(matrixFloquetExp)[indicesMaxReal,np.arange(Npop)]

                # Check that first Floquet exponent is zero
                if np.abs(MaxImagFloquetExp[0])>10**(-4):
                    # Status 5 encodes Floquet exponents errors
                    status = 5
                    dataFloquetReal[idx_Iext_e,idx_eps]=np.nan*np.ones(Npop)
                    dataFloquetImaginary[idx_Iext_e,idx_eps]=np.nan*np.ones(Npop)


                    #Update dataStatus accordingly
                    dataStatus[idx_Iext_e,idx_eps] = status

                
                #Update dataFloquet accordingly
                dataFloquetReal[idx_Iext_e,idx_eps]=MaxRealFloquetExp
                dataFloquetImaginary[idx_Iext_e,idx_eps]=MaxImagFloquetExp

                # Show some info
                print('Rectified Status: ',status)
                print(MaxRealFloquetExp)
                

            # Status 0: No oscillations at all --> Fixed Point
            # Status 1: Chaotic region
            # Status 4: Convergence problems
            else:
                dataFloquetReal[idx_Iext_e,idx_eps]=np.nan*np.ones(Npop)
                dataFloquetImaginary[idx_Iext_e,idx_eps]=np.nan*np.ones(Npop)

    # =============== SAVE ALL DATA  ================
    
    np.savez(root / "scripts" / 'FloquetBifDiagram_Npop={Npop}.npz',vector_Iext_e=vector_Iext_e,vector_eps=vector_eps,dataFloquetReal=dataFloquetReal,dataFloquetImaginary=dataFloquetImaginary,dataStatus=dataStatus)
    
    return vector_Iext_e,vector_eps,dataFloquetReal,dataFloquetImaginary,dataStatus
