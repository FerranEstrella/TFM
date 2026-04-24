import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from TrainedModel import predict_state, dict_to_array



def NetworkSystemPlot_DeepONet(u0, params, W_file="NormalizedMatrix.npy", ve=False):
    # ------------------- Load connectivity & Determine Npop -------------------
    
    
    root = Path(__file__).resolve().parent.parent
    # Note: Using Path(__file__) might fail in interactive environments like Kaggle; 
    # using Path.cwd() or a relative string is often safer there.
    #root = Path.cwd() 
    #W_path = root / "data" / W_file
    
    #if not W_path.exists():
        # Fallback for standard Kaggle structure if needed
        #W_path = Path("/kaggle/working/data") / W_file

    #W = np.load(W_path)
    
    # Homogeneous case:
    W = np.array([[1.0]])

    Npop = W.shape[0]  # Implicitly determine Npop from the matrix
    total_dim = 6 * Npop

    # ------------------- Time span -------------------
    t0, tf, dt = 0, 10, 0.01  # 1000
    t_eval = np.arange(t0, tf, dt)
    t_eval_jax = jnp.array(t_eval)

    # ------------------- DeepONet Prediction -------------------
    # Prepare inputs
    mu_jax = dict_to_array(params)
    u0_jax = jnp.array(u0)

    # Forward pass: predict_state returns (N_steps, total_dim)
    # Note: Your model's 'system_dimension' must match total_dim
    print("Starting system prediction")
    prediction = predict_state(mu_jax, u0_jax, t_eval_jax)
    
    # Convert to NumPy and transpose to (total_dim, N_steps)
    y_pred = np.array(prediction).T 

    # ------------------- Dynamic Unpacking -------------------
    # We use Npop to slice the total_dim correctly
    r_e = y_pred[0 : Npop, :]
    v_e = y_pred[Npop : 2*Npop, :]
    s_e = y_pred[2*Npop : 3*Npop, :]
    r_i = y_pred[3*Npop : 4*Npop, :]
    v_i = y_pred[4*Npop : 5*Npop, :]
    s_i = y_pred[5*Npop : , :]

    # ------------------- Helper to save heatmap -------------------
    def save_heatmap(data_matrix, t, fname, colorbar, clim=(-2,2),
                     xlabel="Time (ms)", ylabel="Population i", title=""):
        plt.figure(figsize=(10, 5))
        plt.imshow(
            data_matrix,
            aspect='auto',
            origin='lower',
            extent=[t[0], t[-1], 0, data_matrix.shape[0]],
            cmap='RdBu',
            vmin=clim[0],
            vmax=clim[1],
            interpolation='nearest'
        )
        plt.colorbar(label=colorbar)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks(fontsize=12)
        
        # Dynamic Y-axis labels based on Npop
        plt.yticks(
            np.arange(Npop) + 0.5,
            [f"Pop {i}" for i in range(1, Npop + 1)],
            fontsize=10
        )
        plt.title(title, fontsize=16)
        plt.tight_layout()

        save_dir = root / "scripts" / "NetTraj"
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()

    # ------------------- Save heatmaps -------------------
    last_ms = 10 #100
    idx_steps = int(last_ms / dt)

    var_dict = {"r_e": r_e, "v_e": v_e, "s_e": s_e,
                "r_i": r_i, "v_i": v_i, "s_i": s_i}

    if ve:
        var_dict = {"v_e": v_e}

    W_name = os.path.splitext(os.path.basename(W_file))[0]
    
    for name, matrix in var_dict.items():
        data_window = matrix[:, -idx_steps:]
        
        fname = f"DeepONet_{name}_eps={params['eps']}_Ie={params['Iext_e']}_{W_name}.png"
        title = rf"DeepONet {name}, $\epsilon={params['eps']}$, $I_{{ext}}^e={params['Iext_e']}$ ($N_{{pop}}={Npop}$)"
        
        save_heatmap(data_window, t_eval[-idx_steps:], fname, title=title, colorbar=name)
