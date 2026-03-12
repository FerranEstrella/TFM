
import numpy as np
from scipy.integrate import solve_ivp

def homogeneous_system(t, u, p):
    """
    Define the system du/dt = f(u, t)
    
    u: state vector [r_e, v_e, s_e, r_i, v_i, s_i]
    p: parameters [tau_e, tau_i, tau_se, tau_si, nu_e, nu_i, Delta_e, Delta_i, 
                    Jee, Jei, Jii, Jie, Iext_i, Iext_e, eps]
    """
    du = np.zeros_like(u)

    r_e, v_e, s_e, r_i, v_i, s_i = u
    (tau_e, tau_i, tau_se, tau_si, nu_e, nu_i, Delta_e, Delta_i,
     Jee, Jei, Jii, Jie, Iext_i, Iext_e, eps) = p

    du[0] = (Delta_e / (tau_e * np.pi) + 2 * r_e * v_e) / tau_e
    du[1] = (v_e**2 + nu_e - (np.pi * r_e * tau_e)**2 + Iext_e + tau_e * (Jee + eps) * s_e - tau_e * Jei * s_i) / tau_e
    du[2] = (-s_e + r_e) / tau_se
    du[3] = (Delta_i / (tau_i * np.pi) + 2 * r_i * v_i) / tau_i
    du[4] = (v_i**2 + nu_i - (np.pi * r_i * tau_i)**2 + Iext_i + tau_i * (Jie + eps) * s_e - tau_i * Jii * s_i) / tau_i
    du[5] = (-s_i + r_i) / tau_si

    return du

def jacobian_homogeneous(u, p):
    """
    Compute the Jacobian matrix of the system at state u.
    
    u: state vector
    p: parameters
    """
    M = np.zeros((6, 6))

    r_e, v_e, s_e, r_i, v_i, s_i = u
    (tau_e, tau_i, tau_se, tau_si, nu_e, nu_i, Delta_e, Delta_i,
     Jee, Jei, Jii, Jie, Iext_i, Iext_e, eps) = p

    # r_e equation
    M[0,0] = 2*v_e / tau_e
    M[0,1] = 2*r_e / tau_e

    # v_e equation
    M[1,0] = -2 * r_e * tau_e * np.pi**2
    M[1,1] = 2*v_e / tau_e
    M[1,2] = Jee + eps
    M[1,5] = -Jei

    # s_e equation
    M[2,0] = 1 / tau_se
    M[2,2] = -1 / tau_se

    # r_i equation
    M[3,3] = 2*v_i / tau_i
    M[3,4] = 2*r_i / tau_i

    # v_i equation
    M[4,2] = Jie + eps
    M[4,3] = -2 * r_i * tau_i * np.pi**2
    M[4,4] = 2*v_i / tau_i
    M[4,5] = -Jii

    # s_i equation
    M[5,3] = 1 / tau_si
    M[5,5] = -1 / tau_si

    return M

