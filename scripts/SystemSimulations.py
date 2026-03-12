from pathlib import Path
import os
import sys
import numpy as np
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root / "src"))
from NetworkSystemSimulation import network_system_plot


os.chdir(root / "data")
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
#




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
#






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
#




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
#
