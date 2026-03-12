# Network.py

import numpy as np

def network( u, p, t):
    """
    du: derivative vector (in-place update)
    u: state vector (contains in order the vectors r_e, v_e, s_e, r_i, v_i, s_i)
    p: dictionary with keys 'scalar_params' and 'matrix_params'
    t: time
    """
    scalar_params = p['scalar_params']
    W = p['matrix_params']
    
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

    # Parameters
    (tau_e, tau_i, tau_se, tau_si, nu_e, nu_i, Delta_e, Delta_i,
     Jee, Jei, Jii, Jie, Iext_i, Iext_e, eps) = scalar_params

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


def jacobian_network(M, u, p, t):
    """
    M: Jacobian matrix (in-place update)
    u: state vector
    p: dictionary
    t: time
    """
    scalar_params = p['scalar_params']
    W = p['matrix_params']
    
    Npop = W.shape[0]
    idx_ve, idx_se, idx_ri, idx_vi, idx_si = 1,2,3,4,5

    r_e_vector = u[0:Npop]
    v_e_vector = u[Npop*idx_ve:Npop*idx_se]
    s_e_vector = u[Npop*idx_se:Npop*idx_ri]
    r_i_vector = u[Npop*idx_ri:Npop*idx_vi]
    v_i_vector = u[Npop*idx_vi:Npop*idx_si]
    s_i_vector = u[Npop*idx_si:]

    (tau_e, tau_i, tau_se, tau_si, nu_e, nu_i, Delta_e, Delta_i,
     Jee, Jei, Jii, Jie, Iext_i, Iext_e, eps) = scalar_params

    # Clear matrix
    M.fill(0)

    # Diagonal blocks
    M[0:Npop, 0:Npop] = 0  # r_e derivatives
    # Partial r_e
    M[0:Npop, Npop*idx_ve:Npop*idx_se] = (2/tau_e)*r_e_vector[:,None]
    M[0:Npop, 0:Npop] = (2/tau_e)*v_e_vector[:,None]

    # Partial v_e
    M[Npop*idx_ve:Npop*idx_se, 0:Npop] = -2*r_e_vector*tau_e*np.pi**2
    M[Npop*idx_ve:Npop*idx_se, Npop*idx_ve:Npop*idx_se] = 2*v_e_vector/tau_e
    M[Npop*idx_ve:Npop*idx_se, Npop*idx_se:Npop*idx_ri] = Jee
    M[Npop*idx_ve:Npop*idx_se, Npop*idx_si:] = -Jei
    # Coupling sum contribution
    M[Npop*idx_ve:Npop*idx_se, Npop*idx_se:Npop*idx_ri] += eps * W

    # Partial s_e
    M[Npop*idx_se:Npop*idx_ri, 0:Npop] = 1/tau_se
    M[Npop*idx_se:Npop*idx_ri, Npop*idx_se:Npop*idx_ri] = -1/tau_se

    # Partial r_i
    M[Npop*idx_ri:Npop*idx_vi, Npop*idx_ri:Npop*idx_vi] = (2/tau_i)*v_i_vector[:,None]
    M[Npop*idx_ri:Npop*idx_vi, Npop*idx_vi:Npop*idx_si] = (2/tau_i)*r_i_vector[:,None]

    # Partial v_i
    M[Npop*idx_vi:Npop*idx_si, Npop*idx_ri:Npop*idx_vi] = -2*r_i_vector*tau_i*np.pi**2
    M[Npop*idx_vi:Npop*idx_si, Npop*idx_vi:Npop*idx_si] = (2/tau_i)*v_i_vector[:,None]
    M[Npop*idx_vi:Npop*idx_si, Npop*idx_se:Npop*idx_ri] = Jie + eps*W
    M[Npop*idx_vi:Npop*idx_si, Npop*idx_si:] = -Jii

    # Partial s_i
    M[Npop*idx_si:, Npop*idx_ri:Npop*idx_vi] = 1/tau_si
    M[Npop*idx_si:, Npop*idx_si:] = -1/tau_si

