import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

from NetworkSystem import network  

def network_system_plot(u0, scalar_params, W_file="NormalizedMatrix.npy",ve=False):
    """
    Simulate the network system and save heatmaps for r_e, v_e, s_e, r_i, v_i, s_i
    directly in the scripts folder.
    
    Parameters
    ----------
    u0 : np.ndarray
        Initial condition vector of size Npop*Nvariables
    scalar_params : list or np.ndarray
        List of scalar parameters [tau_e, tau_i, tau_se, ...]
    W_file : str
        Path to .npz file containing the connectivity matrix under key 'normalized_matrix'
    """
    
    # ------------------- Load connectivity -------------------
   
    
    root = Path(__file__).resolve().parent.parent
 
    W = np.load( root / "data" / W_file)
    p = {'scalar_params': scalar_params, 'matrix_params': W}
    
    Npop = W.shape[0]
    Nvariables = 6
    
    # ------------------- Time span -------------------
    t0, tf, dt = 0, 1000, 0.01
    t_eval = np.arange(t0, tf, dt)
    
    # ------------------- Solve ODE -------------------
    sol = solve_ivp(lambda t, u: network(u, p, t),
                    (t0, tf), u0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)
    
    # ------------------- Unpack variables -------------------
    idx_ve, idx_se, idx_ri, idx_vi, idx_si = 1, 2, 3, 4, 5
    r_e = sol.y[0:Npop, :]
    v_e = sol.y[Npop*idx_ve:Npop*idx_se, :]
    s_e = sol.y[Npop*idx_se:Npop*idx_ri, :]
    r_i = sol.y[Npop*idx_ri:Npop*idx_vi, :]
    v_i = sol.y[Npop*idx_vi:Npop*idx_si, :]
    s_i = sol.y[Npop*idx_si:, :]
    
    # ------------------- Helper to save heatmap -------------------
    def save_heatmap(data_matrix, t, fname, colorbar,  clim=(-2,2), xlabel="Time (ms)", ylabel="Population i", title=""):
        plt.figure(figsize=(10,5))
        plt.imshow(data_matrix, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], 1, data_matrix.shape[0]],
                   cmap='RdBu', vmin=clim[0], vmax=clim[1])
        plt.colorbar(label=colorbar)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        save_dir = root / "scripts" / "NetTraj"
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()
    
    # ------------------- Save heatmaps -------------------
    last_ms = 100
    idx_steps = int(last_ms/dt)
    
    var_dict = {"r_e": r_e, "v_e": v_e, "s_e": s_e,
                "r_i": r_i, "v_i": v_i, "s_i": s_i}
    
    if ve:
        var_dict = {"v_e": v_e}
        
    for name, matrix in var_dict.items():
        data_window = matrix[:, -idx_steps:]
        W_name = os.path.splitext(os.path.basename(W_file))[0]
        fname = f"{name}_u0first={u0[0]:.3f}_u0last={u0[-1]:.3f}_eps={scalar_params[14]}_Iext_e={scalar_params[13]}_{W_name}.png"
        title = rf"{name}, $\epsilon={scalar_params[14]}$, $I_{{ext}}^e={scalar_params[13]}$"
        save_heatmap(data_window, sol.t[-idx_steps:], fname, title=title, colorbar=name)
    
