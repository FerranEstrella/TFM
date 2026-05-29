import numpy as np
#import numba as nb

def HomogeneousSystem(t0,x,params):
        #Function to build Next Generation model of the population idx_pop of the network

        #InputsModel:
            #t0: (Float) Time. The system is autonomous so it is not use. However, is required for the function being used in integrators.
            #x: (1xNvariables Float) Array defining the vector field x=np.array([r_e,v_e,r_i,v_i])
                #Parameters: (1x15 Float) Array containing the parameters of the model
                    #tau_e,tau_i, tau_se, tau_si: Time constants of the model
                    #nu_e,nu_i: Baseline constant current for excitatory,inhibitory neurons
                    #Delta_e,Delta_i: Mean neuron noise intensity over excitatory,inhibitory
                    #Jpq: For p,q in {e,i} Synaptic strength between E-I populations
                    #Iext_e,Iext_i: External currents
                    #eps
        
        #vector to store the field that describes the model
        dx = np.zeros(6)

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
        dx[0]=(Delta_e/(tau_e*np.pi)+2*r_e*v_e)/tau_e
        dx[1]=(v_e**2+nu_e-(np.pi*r_e*tau_e)**2+Iext_e+tau_e*(Jee+eps)*s_e-tau_e*Jei*s_i)/tau_e
        dx[2]=(-s_e+r_e)/tau_se
        dx[3]=(Delta_i/(tau_i*np.pi)+2*r_i*v_i)/tau_i
        dx[4]=(v_i**2+nu_i-(np.pi*r_i*tau_i)**2+Iext_i+tau_i*(Jie+eps)*s_e-tau_i*Jii*s_i)/tau_i
        dx[5]=(-s_i+r_i)/tau_si

        return dx

def HomogeneousSystemZero(x, params):
    #Same function but without explicit t dependence
    return HomogeneousSystem(0, x, params)


def JacobianHomogeneousSystem(x,params):
        #Function to build Next Generation model of the population idx_pop of the network

        #InputsModel:
            #x: (1xNvariables Float) Array defining the vector field x=np.array([r_e,v_e,r_i,v_i])
            #Parameters: (1x15 Float) Array containing the parameters of the model
                    #tau_e,tau_i, tau_se, tau_si: Time constants of the model
                    #nu_e,nu_i: Baseline constant current for excitatory,inhibitory neurons
                    #Delta_e,Delta_i: Mean neuron noise intensity over excitatory,inhibitory
                    #Jpq: For p,q in {e,i} Synaptic strength between E-I populations
                    #Iext_e,Iext_i: External currents
                    #eps
        

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
        calJ = np.array([[(2*v_e)/tau_e, (2*r_e)/tau_e, 0, 0, 0, 0],[(-2*r_e*(tau_e*np.pi)**2)/tau_e, (2*v_e)/tau_e, Jee+eps, 0, 0, -Jei],[1/tau_se, 0, -1/tau_se, 0, 0, 0],[0, 0, 0, 2*v_i/tau_i, (2*r_i)/tau_i, 0],[0, 0, Jie+eps, (-2*r_i*(tau_i*np.pi)**2)/tau_i, (2*v_i)/tau_i,-Jii],[0, 0, 0, 1/tau_si, 0, -1/tau_si]])

        return calJ
