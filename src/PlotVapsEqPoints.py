from RoutineEqPoints import RoutineVaps

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox


def PlotVaps(params,NetworkParams,W,save):

    Npop=W.shape[0]

    #Compute eigenvalues and eigenvectors of connectivity matrix W
    vapsConn,vepsConn = np.linalg.eig(W)

    #Define tuple of invented vaps connectivity matrix
    vapsCurve = np.arange(-1,1+0.005,0.005)

    #Define colormap and vector of colors
    colormap = cm.Set1
    colors_vector = colormap(np.linspace(0, 0.5, len(NetworkParams)))

    #Initialize matrices
    matrixMaxRealVaps = np.zeros((len(NetworkParams),Npop))
    matrixMaxImagVaps = np.zeros((len(NetworkParams),Npop))
    InventedMatrixMaxRealVaps = np.zeros((len(NetworkParams),len(vapsCurve)))
    InventedMatrixMaxImagVaps = np.zeros((len(NetworkParams),len(vapsCurve)))

    for idx in range(len(NetworkParams)):

        tuple = NetworkParams[idx]

        Iext_e = tuple[0]
        eps = tuple[1]

        params['Iext_e'] = Iext_e
        params['eps'] = eps

        matrixVaps = RoutineVaps(vapsConn,params)
        InventedMatrixVaps = RoutineVaps(vapsCurve,params)

        matrixMaxRealVaps[idx,:]  = np.amax(np.real(matrixVaps),axis=0)
        InventedMatrixMaxRealVaps[idx,:]  = np.amax(np.real(InventedMatrixVaps),axis=0)

        indicesMaxReal = np.argmax(np.real(matrixVaps),axis=0)
        InventedIndicesMaxReal = np.argmax(np.real(InventedMatrixVaps),axis=0)

        matrixMaxImagVaps[idx,:] = np.imag(matrixVaps)[indicesMaxReal,np.arange(Npop)]
        InventedMatrixMaxImagVaps[idx,:] = np.imag(InventedMatrixVaps)[InventedIndicesMaxReal,np.arange(len(vapsCurve))]

    # Enable LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })

    #Real part plot
    fig1=plt.figure(figsize=(12,7))
    ax=plt.axes()
    plt.title(r'$\varepsilon = '+ f'{NetworkParams[idx][1]}$',fontsize=40,fontname='Times New Roman',loc="left",pad=20)
    plt.xlabel(r'$\Lambda_{\alpha}$',fontsize=40,fontname='Times New Roman')
    plt.ylabel(r'$\mu_{max}^{(\alpha)}$',fontsize=40,fontname='Times New Roman')
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    legend_handles = []
    for idx in range(len(NetworkParams)):
        plt.plot(vapsCurve,InventedMatrixMaxRealVaps[idx,:],linewidth = 1,color=colors_vector[idx])
        plt.scatter(vapsConn,matrixMaxRealVaps[idx,:],
                    facecolors='none',edgecolors=colors_vector[idx], s=30)

        legend_handles.append(
            Line2D([0], [0],
                   marker='o', color='w',
                   markerfacecolor='none',
                   markeredgecolor=colors_vector[idx],
                   markersize=6,
                   label=r'$I_{ext}^E = ' + f'{NetworkParams[idx][0]}$')
        )

    plt.legend(handles=legend_handles,
               ncol=4,
               fontsize=22,
               loc='upper right',
               bbox_to_anchor=(1.02, 1.16),
               columnspacing=0.1,
               handletextpad=0.5,
               frameon=True)

    plt.axhline(0,color="black", ls="-")
    plt.xlim([-1.01,1.01])
    plt.xticks([-1,-0.5,0,0.5,1])
    plt.yticks([-0.3,-0.2,-0.1,0,0.1])
    plt.ylim(-0.3,0.15)

    if save:
        root = Path(__file__).resolve().parent.parent
        save_dir = root / "scripts" / "DispersionRelations"
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"Vaps__Npop={Npop}_eps={eps}.png"
        plt.savefig(save_dir / filename, dpi=500, bbox_inches="tight")

    #Imaginary part plot
    if False:

        fig2=plt.figure(figsize=(12,7))
        ax=plt.axes()
        plt.title(r'Imaginary part Vaps $\varepsilon = '+ f'{NetworkParams[idx][1]}$',fontsize=30,fontname='Times New Roman')
        plt.xlabel(r'$\Lambda_{\alpha}$',fontsize=30,fontname='Times New Roman')
        plt.ylabel(r'$\beta$',fontsize=30,fontname='Times New Roman')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        legend_handles = []
        for idx in range(len(NetworkParams)):
            plt.scatter(vapsConn,matrixMaxImagVaps[idx,:],
                        facecolors='none',edgecolors=colors_vector[idx], s=30)
            plt.plot(vapsConn,matrixMaxImagVaps[idx,:],
                     color=colors_vector[idx],linewidth=1)

            legend_handles.append(
                Line2D([0], [0],
                       marker='o', color='w',
                       markerfacecolor='none',
                       markeredgecolor=colors_vector[idx],
                       markersize=6,
                       label=r'$I_{ext}^E = ' + f'{NetworkParams[idx][0]}$')
            )

        plt.legend(handles=legend_handles,
                   ncol=4,
                   fontsize=22,
                   loc='upper right',
                   bbox_to_anchor=(1.02, 1.16),
                   columnspacing=0.1,
                   handletextpad=0.5,
                   frameon=True)

        plt.axhline(0,color="black", ls="-")
        plt.xlim([-1.01,1.01])
        plt.xticks([-1,-0.5,0,0.5,1])

        if save:
            filename = f"ImaginaryVaps__Npop={Npop}_eps={eps}.png"
            plt.savefig(save_dir / filename, dpi=500, bbox_inches="tight")

    return fig1
