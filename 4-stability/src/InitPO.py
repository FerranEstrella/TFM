import numpy as np
from scipy.integrate import solve_ivp

from HomogeneousSystem import HomogeneousSystem

import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import fsolve
from scipy.signal import find_peaks

# Define Poincare section
def PoincareEvent(t,x,params,alpha):
    return x[2]-alpha

# Event should be triggered when g(x) == 0 <=> x[1]=0
PoincareEvent.terminal = False  # Stop the integration when the event occurs
PoincareEvent.direction = 1    # Trigger when the second component crosses alpha in ascending direction


# Define the function that we want to find the zero of (phi(T, x0) - x0)
def SystemPeriodicity(initCond,params,alpha):
    Nvariables=6
    #Unfold initCond
    T_approx = initCond[0]
    x0_approx = initCond[1:len(initCond)]

    # 1. Set parameters 
    #Initial and final times of integration
    t0 = 0
    tf = T_approx
    #Discretization used for integration
    h = 0.001
    #Number of points to evaluate the time integration
    N = int((tf-t0)/h)
    #print(N)
    time_eval = np.linspace(t0, tf, N)

    # Integrate the system from initial condition x0_guess for time T_guess
    sol = solve_ivp(HomogeneousSystem, [t0,tf], x0_approx, t_eval=time_eval, method='DOP853' , rtol=1e-4, atol=1e-7, args=(params,))
    
    # Compute the difference between the final state and the initial state   g(x)=phi(T,x)-alpha=0// phi(T,x)=x
    diff = np.append(sol.y[2, -1]-alpha,sol.y[:, -1]-x0_approx)

    return diff


def InitCondPOHomogeneous(params, exact=False):
    Nvariables=6
    ## ==============================  1.  FIND FIRST APPROXIMATION  ===================================
    # 1. Initial and final times of integration
    t0 = 0
    tf = 1000    #1500
    x0 = np.zeros(Nvariables)
    
    # 1a. Initial integration to check that the model oscillates
   
    sol = solve_ivp(HomogeneousSystem, [t0,tf], x0, method='DOP853', rtol=1e-6, atol=1e-9, args=(params,))
     
    # 1b. Check if there are oscillations
    # Look at the peaks of the v_i voltage eliminating the first part of the integration

    t_tail = 100
    mask = sol.t > (sol.t[-1] - t_tail)
    difference = np.max(sol.y[4, mask]) - np.min(sol.y[4, mask])
    
    #Here we check that the model oscillates for this parameters --> WE ARE SURE WITH THIS CONDITION
    if difference < 0.1:
        # Status 0 encodes no oscillations in the model; we do not obtain e.p in this code
        status = 0
        return status, np.zeros(Nvariables), 0

    # 2. Compute the mean s_e of the previous integration
    #mean = np.mean(sol.y[2,len(sol.t)-int(idx_ms/h):len(sol.t)])
    mean = np.mean(sol.y[2, mask])
    #print(mean)
    PoincareEventAlpha = partial(PoincareEvent, alpha=mean)
    PoincareEventAlpha.direction = 1  

    tf = 500
    sol = solve_ivp(HomogeneousSystem, [t0,tf], sol.y[:,-1], method='DOP853', rtol=1e-4, atol=1e-7,events = PoincareEventAlpha, args=(params,))


    # 2b. Check if the oscillations are periodic

    vector_periods = sol.t_events[0]
    diff_periods = np.diff(vector_periods)
    equal_diff_periods = np.diff(diff_periods)
    
    check = np.where(equal_diff_periods>5*10**(-3)*np.ones(len(equal_diff_periods)))
    

    if len(check[0])>0:
        # This condition encodes a possible chaotic region. If this condition is satisfied, we integrate again with more precision to check             if it is really a chaotic region

        t0 = 0
        tf = 2500    #1500
        x0 = np.zeros(Nvariables)

        #Discretization used for integration
        h = 0.001 #0.001
        #Number of points to evaluate the time integration
        N = int((tf-t0)/h)
        time_eval = np.linspace(t0, tf, N)
        sol = solve_ivp(HomogeneousSystem, [t0,tf], x0, t_eval=time_eval,  method='DOP853', rtol=1e-6, atol=1e-9, args=(params,))


        #We repeat the same computation with the more precise integration:

        t_tail = 100
        mask = sol.t > (sol.t[-1] - t_tail)
        difference = np.max(sol.y[4, mask]) - np.min(sol.y[4, mask])
        
        if difference < 0.1:
            status = 0
            return status, np.zeros(Nvariables), 0

        mean = np.mean(sol.y[2, mask])
        PoincareEventAlpha = partial(PoincareEvent, alpha=mean)
        PoincareEventAlpha.direction = 1  

        tf = 500
        #Number of points to evaluate the time integration
        N = int((tf-t0)/h)
        time_eval = np.linspace(t0, tf, N)

        sol = solve_ivp(HomogeneousSystem, [t0,tf], sol.y[:,-1],t_eval=time_eval, method='DOP853', rtol=1e-4, atol=1e-7,events = PoincareEventAlpha, args=(params,))

        vector_periods = sol.t_events[0]
        diff_periods = np.diff(vector_periods)
        equal_diff_periods = np.diff(diff_periods)
        
        check = np.where(equal_diff_periods>5*10**(-3)*np.ones(len(equal_diff_periods)))
        
        if len(check[0])>0:
            #Now we are more convinced it is chaotic
            status = 1
            return status, np.zeros(Nvariables), 0


    
    # Status 2 encodes periodic oscillation region
    status = 2

    # 3. Take final point of the integration --> should be approx to periodic orbit
    x0_approx = sol.y_events[0][-1,:]
    T_approx = np.abs(sol.t_events[0][-2:][1]-sol.t_events[0][-2:][0])

    if exact:
        # Use fsolve to find the solution
        T_solution,infodict,ier,msg = fsolve(SystemPeriodicity, np.append(T_approx,x0_approx),xtol=10**(-4), args=(params,mean),full_output=True) #maxfev = 20
        print('Exit status: ',ier)
        print('Exit message: ',msg)
        print('Function evaluations: ',infodict['nfev'])
        print('Final residuals: ',infodict['fvec'])

        if ier != 1 or T_solution[0] <= 0:
            # Status 4 encodes convergence problems 
            status = 4
            return status, x0_approx, T_approx

        return status,T_solution[1:len(T_solution)],T_solution[0]
    
    else:
        return status, x0_approx, T_approx
    

