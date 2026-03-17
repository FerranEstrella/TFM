from pathlib import Path
import os
import sys
import numpy as np
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root / "src"))
from NetworkSystemSimulation import network_system_plot
from PlotFloquetExponents import PlotFloquetExponents

ploto=0
#################################### TRAJECTORY INTEGRATIONS ###############################################
    
if ploto==1:
    #For reference homogeneous solutions exploration, recall checking the bifurcation diagram
    p = [8,8,1,5,-5,-5,1,1,5,13,5,13,0,12,8] 
    u0 = np.zeros(6*90) 
    network_system_plot(u0, p, W_file="normalized_matrix.npy", ve=True)
    u0 = np.zeros(6*4)
    network_system_plot(u0, p, W_file="normalized_matrix_4cluster.npy", ve=True)
    #Periodic homogeneous solutions

    u0 = np.zeros(6*90)
    u0[0] = 0.1
    network_system_plot(u0, p, W_file="normalized_matrix.npy", ve=True)
    u0 = np.zeros(6*4)
    u0[0] = 0.1
    network_system_plot(u0, p, W_file="normalized_matrix_4cluster.npy", ve=True)


    p = [8,8,1,5,-5,-5,1,1,5,13,5,13,0,12,12]
    u0 = np.zeros(6*90)
    network_system_plot(u0, p, W_file="normalized_matrix.npy", ve=True)
    u0 = np.zeros(6*4)
    network_system_plot(u0, p, W_file="normalized_matrix_4cluster.npy", ve=True)
    #Periodic homogeneous solutions

    u0 = np.zeros(6*90)
    u0[0] = 0.1
    network_system_plot(u0, p, W_file="normalized_matrix.npy", ve=True)
    u0 = np.zeros(6*4)
    u0[0] = 0.1
    network_system_plot(u0, p, W_file="normalized_matrix_4cluster.npy", ve=True)


    p = [8,8,1,5,-5,-5,1,1,5,13,5,13,0,13,5]
    u0 = np.zeros(6*90)
    network_system_plot(u0, p, W_file="normalized_matrix.npy", ve=True)
    u0 = np.zeros(6*4)
    network_system_plot(u0, p, W_file="normalized_matrix_4cluster.npy", ve=True)
    #Periodic homogeneous solutions

    u0 = np.zeros(6*90)
    u0[0] = 0.1
    network_system_plot(u0, p, W_file="normalized_matrix.npy", ve=True)
    u0 = np.zeros(6*4)
    u0[0] = 0.1
    network_system_plot(u0, p, W_file="normalized_matrix_4cluster.npy", ve=True)


    p = [8,8,1,5,-5,-5,1,1,5,13,5,13,0,7,20]
    u0 = np.zeros(6*90)
    network_system_plot(u0, p, W_file="normalized_matrix.npy", ve=True)
    u0 = np.zeros(6*4)
    network_system_plot(u0, p, W_file="normalized_matrix_4cluster.npy", ve=True)
    #Periodic homogeneous solutions

    u0 = np.zeros(6*90)
    u0[0] = 0.1
    network_system_plot(u0, p, W_file="normalized_matrix.npy", ve=True)
    u0 = np.zeros(6*4)
    u0[0] = 0.1
    network_system_plot(u0, p, W_file="normalized_matrix_4cluster.npy", ve=True)








############################## MASTER STABILITY FUNCTION PLOTS (periodic orbits destabilization) #####################################


root = Path(__file__).resolve().parent.parent
W_orig = np.load( root / "data" / "normalized_matrix.npy")
W_redu = np.load( root / "data" / "normalized_matrix_4cluster.npy")
Nvariables = 6

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
                Iext_i=0)

I_eps_plots=[[[7,5],[9,5],[11,5],[13,5]],[[6,9],[8,9],[9.5,9],[12,9]],[[4,12],[6,12],[8,12],[12,12]],[[5,20],[7,20],[7.5,20],[8,20]]]

#for I_eps in I_eps_plots:
    #PlotFloquetExponents(W_redu,Nvariables,4,params,I_eps,True)

I_eps = [[12,9],[9.5,9],[8,12],[12,12]]
p = [8,8,1,5,-5,-5,1,1,5,13,5,13,0]
u0 = np.zeros(6*4)
for i in I_eps:
    network_system_plot(u0, p+i, W_file="normalized_matrix_4cluster.npy",ve=True)




