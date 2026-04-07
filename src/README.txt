CONNECTIVITY CLUSTERING
1. SpectralClustering: Functions to compute the spectral clustering of the original SC matrix and representation.

SYSTEM VECTOR FIELDS AND TRAJECTORY SIMULATION
2. HomogeneousSystem: Function to generate the homogeneous vector field and its Jacobian.
3. HomogeneousSystemSimulation: Functions to simulate trajectories on the homogeneous system. 
4. NetworkSystem: Functions to generate the brain-scale vector field and the Jacobian for each eigenmode.
5. NetworkSystemSimulation: Functions to simulate trajectories on the brain-scale system.

DESTABILIZATION STATIONARY P.O OF HOMOGENEOUS SYSTEM
6. InitPO: Functions to classify the omega limit of the point 0 in the homogeneous system, and determine it with precision in case it is a periodic orbit.

7. VariationalsHomogeneous: Function to generate the variational equations of the homogeneous system. 
8. FloquetExponentsVariationals: Function to integrate the variational equations of the network system to compute the Floquet exponents and Floquet multipliers for a given parameter combination.

9. RoutineFloquet: Function to compute the Npop x Nvariables Floquet exponents of the large-scale brain model for a fixed set of parameter combinations, for which a p.o is the stationary solution. Takes the real and imaginary parts.
10. PlotFloquetExponents: Function to plot the real part of the Npop max Floquet exponents computed with RoutineFloquet for a set of pairs (Iext,eps)

11. FloquetBifDiagram: Function to compute the N_pop Floquet exponents with maximum real part among the considered range of parameters, for those parameters in which a periodic orbit has been determined (and discarding the remaining regimes). A lattice of parameters is considered.

DESTABILIZATION STATIONARY E.P OF HOMOGENEOUS SYSTEM
12. InitEP: Function to compute with precision the omega limit of the point 0 in the homogeneous system in case it is an equilibrium point.

13. RoutineEqPoints: Function to compute the Npop x Nvariables characteristic exponents of the large-scale brain model for a fixed set of parameter combinations, for which an e.p is the stationary solution. 
14. PlotVapsEqPoints: Function to plot the real part of the Npop max characteristic exponents computed with RoutineEqPoints for a set of pairs (Iext,eps)

15. EqPointsBifDiagram: Function to compute the N_pop characteristic exponents with maximum real part among the considered range of parameters, for those parameters in which an equilibrium point has been determined. Based on the computation done in FloquetBifDiagram

16. BifDiagram: Function which merges the data from FloquetBifDiagram and EqPointsBifDiagram

17. BifDiagramPlot: Function to plot the maximum Floquet/characteristic exponents and number of positive Floquet/characteristic exponents for the considered grid of parameters in the generated dataset by BifDiagram




