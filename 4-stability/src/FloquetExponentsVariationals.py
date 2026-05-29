import numpy as np
from scipy.integrate import solve_ivp

from VariationalsHomogeneous import VariationalsHomogeneous

def ComputeFloquetExponents(InitCond,params,eig,T):
    Nvariables=6
    FloquetExp = np.zeros(Nvariables,dtype=np.complex128)
    FloquetMult = np.zeros(Nvariables,dtype=np.complex128)

    # 2 Initial condition of variational equations
    initVariationals = np.eye(Nvariables).flatten()

    # 3. Define parameters for integration
    #Initial and final times of integration
    t0 = 0
    tf = T
    #Discretization used for integration
    #h = 0.0001
    #Number of points to evaluate the time integration
    #N = int((tf-t0)/h)
    #time_eval = np.linspace(t0, tf, N)
    #Complete initial condition for integration
    X0 = np.concatenate((InitCond,initVariationals),axis=0)

    # 5. Integrate the system up to the Poincare section
    #integrationT = solve_ivp(VariationalsHomogeneous, [t0,tf], X0, t_eval=time_eval, method='RK45', rtol=1e-6, atol=1e-9, args=(params,eig))

    integrationT = solve_ivp(
        VariationalsHomogeneous,
        [t0, tf],
        X0,
        method='DOP853',   
        rtol=1e-4,
        atol=1e-7,
        args=(params, eig))

    # 6. Monodromy matrix: Get solution of the variational equations at time T
    Monodromy = integrationT.y[Nvariables:Nvariables+Nvariables*Nvariables,len(integrationT.t)-1].reshape((Nvariables,Nvariables))

    # 7. Compute Characteristic Floquet Multipliers
    FloquetMult[:],veps = np.linalg.eig(Monodromy) 

    # 8. Compute Floquet exponents ---> WARNING: The period is in ms
    FloquetExp[:] = (1/T)*np.log(FloquetMult)

    return FloquetExp,FloquetMult,veps
