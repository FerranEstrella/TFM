import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
from HomogeneousSystem import homogeneous_system

def homogeneous_system_plot(u0,p):
    scripts_folder = Path(__file__).resolve().parent.parent / "scripts"

    t0, tf = 0, 200
    t_eval = np.arange(t0, tf, 0.001)

    sol = solve_ivp(lambda t,u: homogeneous_system(t,u,p), (t0, tf), u0,
                method='RK45', rtol=1e-6, atol=1e-6, t_eval=t_eval)

    def save_plot(x, ys, labels, colors, xlabel, ylabel, filename, title):
        plt.figure(figsize=(8,4))
        for y, label, color in zip(ys, labels, colors):
            plt.plot(x, y, color=color, linewidth=3, label=label)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12, ncol=2, loc='upper right')
        plt.title(title, fontsize=20)
        plt.tight_layout()
        plt.savefig(scripts_folder / filename, dpi=300, bbox_inches='tight')
        plt.close()

    # Save r_i and r_e
    save_plot(sol.t, [sol.y[3,:], sol.y[0,:]], ['r_i', 'r_e'], ['green', 'blue'],
            'Time (ms)', 'Voltage', 'ri_re.png',
            r"$\epsilon = {}$, $I_{{ext}}^e = {}$".format(p[14], p[13]))

    # Save v_i and v_e
    save_plot(sol.t, [sol.y[4,:], sol.y[1,:]], ['v_i', 'v_e'], ['magenta', 'red'],
          'Time (ms)', 'Voltage', 'vi_ve.png',
          r"$\epsilon = {}$, $I_{{ext}}^e = {}$".format(p[14], p[13]))

    # Save s_i and s_e
    save_plot(sol.t, [sol.y[5,:], sol.y[2,:]], ['s_i', 's_e'], ['yellow', 'orange'],
          'Time (ms)', 'Synapsis', 'si_se.png',
          r"$\epsilon = {}$, $I_{{ext}}^e = {}$".format(p[14], p[13]))
