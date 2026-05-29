import numpy as np
from NetworkSystem import JacobianNetworkSystemEigenmode
from HomogeneousSystem import HomogeneousSystem

def VariationalsHomogeneous(t0,X,params,eig):
    
    #Augmented system: state + variational equations

    #X = [x, vec(Y)] with:
        #x ∈ R^6          (state variables)
        #Y ∈ R^{6×6}      (variational matrix, flattened)

    Nvariables = 6
    var_size = Nvariables * Nvariables

    # --- Split state ---
    x = X[:Nvariables]
    Y = X[Nvariables:Nvariables + var_size].reshape((Nvariables, Nvariables))

    # --- Allocate output ---
    dX = np.zeros(Nvariables + var_size)

    # --- Original system ---
    dX[:Nvariables] = HomogeneousSystem(t0, x, params)

    # --- Jacobian ---
    J = JacobianNetworkSystemEigenmode(x, params, eig)

    # --- Variational equations ---
    dX[Nvariables:] = (J @ Y).flatten()

    return dX
