CONNECTIVITY CLUSTERING
1. SpectralClustering: Functions to compute the spectral clustering of the original SC matrix and representation.

SYSTEM VECTOR FIELDS AND TRAJECTORY SIMULATION
2. HomogeneousSystem: Function to generate the homogeneous vector field.
3. HomogeneousSystemSimulation: Functions to simulate trajectories on the homogeneous system. 
4. NetworkSystem: Functions to generate the brain-scale vector field and the corresponding Jacobian.
5. NetworkSystemSimulation: Functions to simulate trajectories on the brain-scale system.

STATIONARY P.O OF HOMOGENEOUS SYSTEM
6. InitPO_2: Functions to classify the omega limit of the point 0 in the homogeneous system, and determine it with precision in case it is a periodic orbit.

DESTABILIZATION STATIONARY P.O OF HOMOGENEOUS SYSTEM
7. VariationalHomogeneous: Function to generate the variational equations of the homogeneous system. 
8. FloquetExponentsVariationals.py: Function to integrate the variational equations of the NextGenDynSyn model to compute the Floquet exponents and Floquet multipliers for a given parameter combination.

9*. RoutineFloquet: Function to compute the Npop Floquet exponents with maximum real part of the large-scale brain model for a fixed set of parameter combinations, for which a p.o is the stationary solution. Take the real and imaginary parts.
10*. PlotFloquetExponents.py: Function to plot the real and imaginary parts of the Npop Floquet exponents computed with RoutineFloquet.


11. FloquetBifDiagram.py: Function to compute the N_pop Floquet exponents with maximum real part among the considered range of parameters, for those parameters in which a periodic orbit has been determined (and discarding the remaining regimes). A lattice of parameters is considered.