import numpy as np

def VariationalsHomogeneous(t0,x,Nvariables,eig,param):
        #Function to build Next Generation model of the population idx_pop of the network

        #InputsModel:
            #t0: (Float) Time
            #x: (1xNvariables Float) Array defining the vector field x=np.array([r_e,v_e,s_e,r_i,v_i,s_i])
            #W: (NpopxNpop Integer) Normalised structural connectivity matrix
            #Nvariables: (Integer) Number of variables of the model
            #eigenvalue: (Complex or real) Eigenvalue of the connectivity matrix for which we are computing the Floquet exponents
            #Parameters: (1x15 Float Dictionary) Array containing the parameters of the model
                #tau_e,tau_i, tau_se, tau_si: Time constants of the model
                #nu_e,nu_i: Baseline constant current for excitatory,inhibitory neurons
                #Delta_e,Delta_i: Mean neuron noise intensity over excitatory,inhibitory
                #Jpq: For p,q in {e,i} Synaptic strength between E-I populations
                #Iext_e,Iext_i: External currents
                #eps: Strength of connection from other populations
        
        #vector to store the field that describes the model
        dx = np.zeros(Nvariables+Nvariables*Nvariables)

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
        
        #--------------  Variational equations  --------------
        variationals = x[Nvariables:Nvariables+Nvariables*Nvariables]
        variationals_reshape = variationals.reshape((Nvariables,Nvariables))

        #---------  PARAMETERS MODEL  ---------
        tau_e = param['tau_e']
        tau_i = param['tau_i']
        tau_se = param['tau_se']
        tau_si = param['tau_si']
        Delta_e = param['Delta_e']
        Delta_i = param['Delta_i']
        nu_e = param['nu_e']
        nu_i = param['nu_i']
        Jee = param['Jee']
        Jei = param['Jei']
        Jii = param['Jii']
        Jie = param['Jie']
        Iext_e = param['Iext_e']
        Iext_i = param['Iext_i']
        eps = param['eps']

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  VARIATIONAL EQUATIONS MODEL  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #Vector field homogeneous system
        dx[0]=(Delta_e/(tau_e*np.pi)+2*r_e*v_e)/tau_e
        dx[1]=(v_e**2+nu_e-(np.pi*r_e*tau_e)**2+Iext_e+tau_e*(Jee+eps)*s_e-tau_e*Jei*s_i)/tau_e
        dx[2]=(-s_e+r_e)/tau_se
        dx[3]=(Delta_i/(tau_i*np.pi)+2*r_i*v_i)/tau_i
        dx[4]=(v_i**2+nu_i-(np.pi*r_i*tau_i)**2+Iext_i+tau_i*(Jie+eps)*s_e-tau_i*Jii*s_i)/tau_i
        dx[5]=(-s_i+r_i)/tau_si
        #Variational equations part
        calJ = np.array([[(2*v_e)/tau_e, (2*r_e)/tau_e, 0, 0, 0, 0],[(-2*r_e*(tau_e*np.pi)**2)/tau_e, (2*v_e)/tau_e, Jee+eig*eps, 0, 0, -Jei],[1/tau_se, 0, -1/tau_se, 0, 0, 0],[0, 0, 0, 2*v_i/tau_i, (2*r_i)/tau_i, 0],[0, 0, Jie+eig*eps, (-2*r_i*(tau_i*np.pi)**2)/tau_i, (2*v_i)/tau_i,-Jii],[0, 0, 0, 1/tau_si, 0, -1/tau_si]])
        
        dx_variationals = np.matmul(calJ,variationals_reshape)
        dx[Nvariables:Nvariables+Nvariables*Nvariables] = dx_variationals.flatten()

        return dx
