import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap


def BifDiagramPlot(W_file="normalized_matrix_4cluster.npy"):

    root = Path(__file__).resolve().parent.parent
    W = np.load(root / "data" / W_file)
    Npop = W.shape[0]

    data = np.load(root / "scripts" / f"BifDiagram_Npop={Npop}.npz")

    vector_Iext_e = data["vector_Iext_e"]
    vector_eps = data["vector_eps"]

    dataStatus = data["dataStatus"]
    exponents = data["exponents"]
    numPositive = data["numPositive"]

    # Axes
    dataStatus = dataStatus.T
    numPositive = numPositive.T
    exponents = np.transpose(exponents, (1, 0, 2))

    # Grid
    I_grid, eps_grid = np.meshgrid(vector_Iext_e, vector_eps)

    I_flat = I_grid.flatten()
    eps_flat = eps_grid.flatten()
    status_flat = dataStatus.flatten()
    numPositive_flat = numPositive.flatten()   

    # Max real exponent
    max_exp = np.max(np.real(exponents), axis=2)
    max_exp_flat = max_exp.flatten()


    
    # FIGURE 1: EXPONENTS 

    # Masks
    mask_eq = (status_flat == 0)
    mask_chaos = (status_flat == 1)         
    mask_po = (status_flat == 2) 
    mask_error= (status_flat == 4) | (status_flat == 5)

    norm = TwoSlopeNorm(vmin=-0.025, vcenter=0.0, vmax=0.025)
    fig, ax = plt.subplots(figsize=(15, 7))

    
    # Equilibria
    sc_eq = plt.scatter(
        I_flat[mask_eq],
        eps_flat[mask_eq],
        c=max_exp_flat[mask_eq],
        cmap=LinearSegmentedColormap.from_list("cyan_blue_magenta",["#00cfff", "#1f4ed8", "#ff00aa"]),
        norm=norm,
        s=5
    )

    # Chaos
    plt.scatter(
        I_flat[mask_chaos],
        eps_flat[mask_chaos],
        color="grey",
        s=5,
        edgecolors="none",
        label="Chaos (1)"
    )


    # Periodic orbits
    sc_po = plt.scatter(
        I_flat[mask_po],
        eps_flat[mask_po],
        c=max_exp_flat[mask_po],
        cmap=LinearSegmentedColormap.from_list("green_brown_red",["#1a9850", "#8c6d31", "#d73027"]),
        norm=norm,
        s=5
    )
    
    # Errors 
    plt.scatter(
        I_flat[mask_error],
        eps_flat[mask_error],
        color="black",
        s=5,
        edgecolors="none",
        label="Errors (4,5)"
    )


    # Horizontal colorbars 

    cbar_eq = plt.colorbar(
        ScalarMappable(norm=norm, cmap=LinearSegmentedColormap.from_list("cyan_blue_magenta",["#00cfff", "#1f4ed8", "#ff00aa"])),
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.12,
        location="bottom"
    )
    cbar_eq.set_label(r"Equilibrium: $\max \Re(\lambda)$")

    cbar_po = plt.colorbar(
        ScalarMappable(norm=norm, cmap=LinearSegmentedColormap.from_list("green_brown_red",["#1a9850", "#8c6d31", "#d73027"])),
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.22,
        location="bottom"
    )
    cbar_po.set_label(r"Periodic orbit: $\max \Re(\lambda)$")

    # Labels and styling
    plt.xlabel(r"$I_{ext}^E$")
    plt.ylabel(r"$\varepsilon$")
    plt.title(f"Exponents — Npop={Npop}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(root / "scripts" / "BifDiagramPlot1.png", dpi=300, bbox_inches='tight')
    plt.close()


    # FIGURE 2: UNSTABLE DIRECTIONS
    
    # Masks
    mask_eq = (status_flat == 0)
    mask_chaos = (status_flat == 1)            
    mask_po = (status_flat == 2)

    mask_other = ~(mask_eq | mask_po | mask_chaos)  # optional but cleaner

    plt.figure(figsize=(15, 7))

    #colors_eq = ["#252525", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    #colors_chaos = ["#808080","#808080","#808080","#808080","#808080"]
    #colors_po = ["#808080", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"]

    colors_eq = ["#deebf7","#9ecae1","#6baed6","#3182bd","#08519c"] 
    colors_chaos = ["#f0f0f0","#bdbdbd","#969696","#636363","#252525"] 
    colors_po = ["#fee5d9","#fcae91","#fb6a4a","#de2d26","#a50f15"]



    for k in range(5):
        mask = mask_eq & (numPositive_flat == k)
        plt.scatter(I_flat[mask], eps_flat[mask],
                    color=colors_eq[k], s=8,
                    label=f"Eq: {k} unstable", alpha=0.9)

    for k in range(5):
        mask = mask_chaos & (numPositive_flat == k)
        plt.scatter(I_flat[mask], eps_flat[mask],
                    color=colors_chaos[k], s=8,
                    marker="^")

    for k in range(5):
        mask = mask_po & (numPositive_flat == k)
        plt.scatter(I_flat[mask], eps_flat[mask],
                    color=colors_po[k], s=8,
                    marker="s",
                    label=f"PO: {k} unstable", alpha=0.9)

    plt.xlabel(r"$I_{ext}^E$")
    plt.ylabel(r"$\varepsilon$")
    plt.title(f"Unstable directions — Npop={Npop}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "scripts" / "BifDiagramPlot2.png", dpi=300, bbox_inches='tight')
    plt.close()

