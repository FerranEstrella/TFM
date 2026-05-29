import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from HomogeneousSystem import HomogeneousSystem, HomogeneousSystemZero

def InitCondEPHomogeneous(params, exact=False):
        
    Nvariables=6

    # Define discretization
    #h = 0.01

    # Set time of integration
    t0 = 0
    tf = 2500
    #N = int((tf-t0)/h)
    #time = np.arange(t0,tf,h)

    # Set initial condition
    x0 = np.zeros(Nvariables)

    # Integrate system a long time to tend to the equilibrium point
    sol = solve_ivp(HomogeneousSystem, [t0,tf], x0, method='DOP853', rtol=1e-4, atol=1e-7,args=(params,))
    final_point = sol.y[:,len(sol.t)-1]

    if exact:
        # Compute zeros system to find the exact fixed point 
        eqPoint,infodict,ier,msg = fsolve(HomogeneousSystemZero, final_point,xtol=10**(-4), args=(params,),full_output=True) #maxfev = 20
        print('Exit status: ',ier)
        print('Exit message: ',msg)
        print('Function evaluations: ',infodict['nfev'])
        # print('Number jacobian calls: ',infodict['njev'])
        print('Final residuals: ',infodict['fvec'])
        
        return eqPoint

    else:
        return final_point
