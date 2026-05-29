from pathlib import Path
import os
import sys
import numpy as np
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root / "src"))
from NetworkSystemSimulation import NetworkSystemPlot

from PlotFloquetExponents import PlotFloquetExponents
from FloquetBifDiagram import FloquetBifDiagram
from InitPO import InitCondPOHomogeneous

from PlotVapsEqPoints import PlotVaps
from InitEP import InitCondEPHomogeneous
from EqPointsBifDiagram import EqPointsBifDiagram

from BifDiagram import BifDiagram
from BifDiagramPlot import BifDiagramPlot

from NetworkSystemSimulationnn import NetworkSystemPlot_DeepONet 

plotMSF=0
tableMSF=0

plotDispRel=0
tableDispRel=0

tableMerge=0
plotBifDiagram=0

plotDeep=1


params = dict(tau_e = 8,
                    tau_i = 8,
                    tau_se=1,
                    tau_si=5,
                    nu_e = -5,
                    nu_i = -5,
                    Delta_e = 1,
                    Delta_i = 1,
                    Jee = 5,
                    Jei = 13,
                    Jii = 5,
                    Jie = 13,
                    Iext_i=0,
                    Iext_e=12,
                    eps=8)



############################## MASTER STABILITY FUNCTION PLOTS (periodic orbits destabilization) #####################################

if plotMSF==1:

    root = Path(__file__).resolve().parent.parent
    W_orig = np.load( root / "data" / "normalized_matrix.npy")
    W_redu = np.load( root / "data" / "normalized_matrix_4cluster.npy")

    MSF=0
    if MSF==1:    
        I_eps_plots=[[[7,5],[9,5],[11,5],[13,5]],
            [[6,9],[8,9],[9.5,9],[12,9]],
            [[4,12],[6,12],[8,12],[12,12]],
            [[5,20],[7,20],[7.5,20],[8,20]],
            [[7,10],[8,10],[9,10],[10,10]],
            [[6,16],[8,16],[10,16],[11,16]]]
         
        for I_eps in I_eps_plots:
            PlotFloquetExponents(params,I_eps,W_redu,True)

    plottrajectories=1
    if plottrajectories==1:
        I_eps = [[12,9],[9.5,9],[8,9],[6,12],[8,12],[12,12],[11,16]]

        for i in I_eps:
            params['Iext_e'] = i[0] 
            params['eps'] = i[1]
            status,u0,T = InitCondPOHomogeneous(params)
            u0 = np.tile(u0, 4)
            NetworkSystemPlot(u0, params, W_file="normalized_matrix_4cluster.npy",ve=True)


############################## MASTER STABILITY FUNCTION TABLE #####################################

if tableMSF==1:
    FloquetBifDiagram()


############################# DISPERSION RELATION PLOTS (equilibrium points destabilization) ##################################################


if plotDispRel==1:

    root = Path(__file__).resolve().parent.parent
    W_orig = np.load( root / "data" / "normalized_matrix.npy")
    W_redu = np.load( root / "data" / "normalized_matrix_4cluster.npy")

    DispRel=0
    if DispRel==1:    
        I_eps_plots=[[[1,8],[3,8],[16,8],[25,8]],
            [[1,10],[3,10],[16,10],[35,10]],
            [[2,18],[12,18],[16,18],[35,18]],
            [[1,30],[14,30],[60,30],[80,30]]]
         
        for I_eps in I_eps_plots:
            PlotVaps(params,I_eps,W_redu,True)

    plottrajectories=1
    if plottrajectories==1:
        I_eps = [[3,8],[16,8],[16,10],[14,30]]

        for i in I_eps:
            params['Iext_e'] = i[0] 
            params['eps'] = i[1]
            u0 = InitCondEPHomogeneous(params)
            u0 = np.tile(u0, 4)
            NetworkSystemPlot(u0, params, W_file="normalized_matrix_4cluster.npy",ve=True)



############################## DISPERSION RELATION (E.P) TABLE #####################################

if tableDispRel==1:
    EqPointsBifDiagram()

############################## MERGE TABLE #####################################

if tableMerge==1:
    BifDiagram()

############################## BIFURCATION DIAGRAM PLOT #####################################

if plotBifDiagram==1:
    BifDiagramPlot()


############################### DeepONet PLOTS ###############################################3

if plotDeep==1:
    I_eps = [[3,8],[6,12],[8,9]]
    
    for i in I_eps:
        params['Iext_e'] = i[0] 
        params['eps'] = i[1]
    
        u0 = np.zeros(6)
        NetworkSystemPlot_DeepONet(u0, params, W_file="normalized_matrix_4cluster.npy",ve=True)
        NetworkSystemPlot(u0, params, W_file="normalized_matrix_4cluster.npy",ve=True)


