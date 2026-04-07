from RoutineFloquet import RoutineFloquet

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

def PlotFloquetExponents(params,NetworkParams,W,save):
    
    #NetworkParams: Set of (tuples) of explored (pairs of) parameters

    Npop=W.shape[0]
    
    #Compute eigenvalues and eigenvectors of connectivity matrix W
    vapsConn,vepsConn = np.linalg.eig(W)

    #Define tuple of invented vaps connectivity matrix
    vapsCurve = np.arange(-1,1+0.005,0.005)

    #Define colormap and vector of colors
    colormap = cm.Set1
    colors_vector = colormap(np.linspace(0, 0.5, len(NetworkParams)))

    #Initialize matrices to store FloquetExponents
    matrixMaxRealFloquetExp = np.zeros((len(NetworkParams),Npop))
    matrixMaxImagFloquetExp = np.zeros((len(NetworkParams),Npop))
    InventedMatrixMaxRealFloquetExp = np.zeros((len(NetworkParams),len(vapsCurve)))

    for idx in range(len(NetworkParams)):
        #Get current pair of parameters for the simulation
        tuple = NetworkParams[idx]

        #Get pairs of parameters for network simulation
        Iext_e = tuple[0]
        eps = tuple[1]
                
        #Update parameters model with networks params for current simulation
        params['Iext_e'] = Iext_e
        params['eps'] = eps

        #Compute FloquetExponents and Multipliers for current pair of parameters (Iext_e,eps)
        matrixFloquetExp,matrixFloquetMult = RoutineFloquet(vapsConn,params)

        #Compute Continuous curve of FloquetExponents and Multipliers for current pair of parameters (Iext_e,eps)
        InventedMatrixFloquetExp,InventedMatrixFloquetMult = RoutineFloquet(vapsCurve,params)

        #Save maximum of the REAL part among the Nvariables FloquetExponents per each eigenvalue of W
        matrixMaxRealFloquetExp[idx,:]  = np.amax(np.real(matrixFloquetExp),axis=0)
        InventedMatrixMaxRealFloquetExp[idx,:]  = np.amax(np.real(InventedMatrixFloquetExp),axis=0)
        #Save indices corresponding to the previous maximum values
        indicesMaxReal = np.argmax(np.real(matrixFloquetExp),axis=0)
        #Save imaginary part of the previous maximum values
        matrixMaxImagFloquetExp[idx,:] = np.imag(matrixFloquetExp)[indicesMaxReal,np.arange(Npop)]

    # Enable LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for all text
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })


    fig1=plt.figure(figsize=(12,7))
    ax=plt.axes()
    plt.title(r'$\varepsilon = '+ f'{NetworkParams[idx][1]}$',fontsize=40,fontname='Times New Roman',loc="left",pad=20)
    plt.xlabel(r'$\Lambda_{\alpha}$',fontsize=40,fontname='Times New Roman')
    plt.ylabel(r'$\mu_{max}^{(\alpha)}$',fontsize=40,fontname='Times New Roman')
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # Create custom legend handles for each row
    legend_handles = []
    for idx in range(len(NetworkParams)):
        plt.plot(vapsCurve,InventedMatrixMaxRealFloquetExp[idx,:],linewidth = 1,color=colors_vector[idx])
        plt.scatter(vapsConn,matrixMaxRealFloquetExp[idx,:],color=colors_vector[idx],facecolors='none',edgecolors=colors_vector[idx], s=30)
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor=colors_vector[idx], markersize=6, label=r'$I_{ext}^E = ' + f'{NetworkParams[idx][0]}$'))
    # Add the legend
    plt.legend(handles=legend_handles,ncol=4, fontsize=22,loc='upper right', bbox_to_anchor=(1.02, 1.16),columnspacing=0.1,handletextpad=0.5,frameon=True)
    plt.axhline(0,color="black", ls="-")
    plt.xlim([-1.01,1.01])
    plt.xticks([-1,-0.5,0,0.5,1])
    plt.yticks([-0.2,-0.15,-0.1,-0.05,0])
    if save:
        root = Path(__file__).resolve().parent.parent
        save_dir = root / "scripts" / "MasterStabilityFunctions"
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"FloquetExponents__Npop={Npop}_eps={eps}.png"
        plt.savefig(save_dir / filename, dpi=500, bbox_inches="tight")


    if False:

        fig2=plt.figure(figsize=(12,7))
        ax=plt.axes()
        plt.title(r'Imaginary part Floquet Exponents $\epsilon = '+ f'{NetworkParams[idx][1]}$',fontsize=30,fontname='Times New Roman')
        plt.xlabel(r'$\mathcal{\Lambda}_{\alpha}$',fontsize=30,fontname='Times New Roman')
        plt.ylabel(r'$\beta$',fontsize=30,fontname='Times New Roman')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        legend_handles = []
        for idx in range(len(NetworkParams)):
            plt.scatter(vapsConn,matrixMaxImagFloquetExp[idx,:],color=colors_vector[idx],facecolors='none',edgecolors=colors_vector[idx], s=30)   
            plt.plot(vapsConn,matrixMaxImagFloquetExp[idx,:],color=colors_vector[idx],linewidth=1)
            legend_handles.append(Line2D([0], [0], marker='o', color='w',markerfacecolor='none', markeredgecolor=colors_vector[idx], markersize=6, label=r'$I_{ext}^e = ' + f'{NetworkParams[idx][0]}$'))
        # Add the legend
        plt.legend(handles=legend_handles,ncol=4, fontsize=15, title='Legend', title_fontsize=15,loc='lower left')
        plt.axhline(0,color="black", ls="-")
        plt.xlim([-1.01,1.01])
        plt.xticks([-1,-0.5,0,0.5,1])
        if save:
            filename = f"ImaginaryFloquetExponents__Npop={Npop}_eps={eps}.png"
            plt.savefig(save_dir / filename, dpi=500, bbox_inches="tight")

    return fig1
