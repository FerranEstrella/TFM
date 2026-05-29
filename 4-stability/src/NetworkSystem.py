import numpy as np

def NetworkSystem(t0,u,params,W):

    Npop = W.shape[0]

    # Indices
    idx_ve, idx_se, idx_ri, idx_vi, idx_si = 1, 2, 3, 4, 5

    # State vector slices
    r_e_vector = u[0:Npop]
    v_e_vector = u[Npop*idx_ve:Npop*idx_se]
    s_e_vector = u[Npop*idx_se:Npop*idx_ri]
    r_i_vector = u[Npop*idx_ri:Npop*idx_vi]
    v_i_vector = u[Npop*idx_vi:Npop*idx_si]
    s_i_vector = u[Npop*idx_si:]

    
    #---------  PARAMETERS MODEL  ---------
    tau_e = params['tau_e']
    tau_i = params['tau_i']
    tau_se = params['tau_se']
    tau_si = params['tau_si']
    Delta_e = params['Delta_e']
    Delta_i = params['Delta_i']
    nu_e = params['nu_e']
    nu_i = params['nu_i']
    Jee = params['Jee']
    Jei = params['Jei']
    Jii = params['Jii']
    Jie = params['Jie']
    Iext_e = params['Iext_e']
    Iext_i = params['Iext_i']
    eps = params['eps']


 
    # Compute coupling sum
    coupling_sum = W @ s_e_vector

    # Update derivatives in-place
    du=[0]*(Npop*6)
    du[0:Npop] = (Delta_e/(tau_e*np.pi) + 2*r_e_vector*v_e_vector)/tau_e
    du[Npop*idx_ve:Npop*idx_se] = (v_e_vector**2 + nu_e - (np.pi*r_e_vector*tau_e)**2 + Iext_e + tau_e*Jee*s_e_vector + tau_e*eps*coupling_sum - tau_e*Jei*s_i_vector)/tau_e
    du[Npop*idx_se:Npop*idx_ri] = (-s_e_vector + r_e_vector)/tau_se
    du[Npop*idx_ri:Npop*idx_vi] = (Delta_i/(tau_i*np.pi) + 2*r_i_vector*v_i_vector)/tau_i
    du[Npop*idx_vi:Npop*idx_si] = (v_i_vector**2 + nu_i - (np.pi*r_i_vector*tau_i)**2 + Iext_i + tau_i*Jie*s_e_vector + tau_i*eps*coupling_sum - tau_i*Jii*s_i_vector)/tau_i
    du[Npop*idx_si:] = (-s_i_vector + r_i_vector)/tau_si

    return du


def JacobianNetworkSystemEigenmode(x,params,eig):

        #---------  VARIABLES MODEL  ---------
        #Firing rates E-I population
        r_e = x[0]
        r_i = x[3]
        #Membrane potential voltage E-I population
        v_e = x[1]
        v_i = x[4]
        #Dynamics synapses
        s_e = x[2]
        s_i = x[5]

        #---------  PARAMETERS MODEL  ---------
        tau_e = params['tau_e']
        tau_i = params['tau_i']
        tau_se = params['tau_se']
        tau_si = params['tau_si']
        Delta_e = params['Delta_e']
        Delta_i = params['Delta_i']
        nu_e = params['nu_e']
        nu_i = params['nu_i']
        Jee = params['Jee']
        Jei = params['Jei']
        Jii = params['Jii']
        Jie = params['Jie']
        Iext_e = params['Iext_e']
        Iext_i = params['Iext_i']
        eps = params['eps']

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  DIFFERENTIAL FIELD MODEL EQUATIONS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        calJ = np.array([[(2*v_e)/tau_e, (2*r_e)/tau_e, 0, 0, 0, 0],[(-2*r_e*(tau_e*np.pi)**2)/tau_e, (2*v_e)/tau_e, Jee+eig*eps, 0, 0, -Jei],[1/tau_se, 0, -1/tau_se, 0, 0, 0],[0, 0, 0, 2*v_i/tau_i, (2*r_i)/tau_i, 0],[0, 0, Jie+eig*eps, (-2*r_i*(tau_i*np.pi)**2)/tau_i, (2*v_i)/tau_i,-Jii],[0, 0, 0, 1/tau_si, 0, -1/tau_si]])
        
        return calJ                                  
