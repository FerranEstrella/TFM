import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from TrainedModel import predict_state, dict_to_array

# 1. Re-create the skeleton (MUST match your training parameters)
model_path = "pi_deeponet_model.eqx" 
model_skeleton = MultiInputDeepONet(
    mu_dim=15, x0_dim=6, latent_dim=64, 
    basis_size=32, system_dimension=6, t_max=2500.0, 
    key=jax.random.PRNGKey(0)
)

# 2. Load the trained weights
model = eqx.tree_deserialise_leaves(model_path, model_skeleton)

# 3. The "Forward Prop" function
@jax.jit
def predict_state(params, x0, t):
    """
    Evaluates the model. 
    params: (15,) array
    x0: (6,) array
    t: scalar or (N,) array
    """
    # If t is a single value, it returns (6,)
    # If t is an array, we vmap it to get (N, 6)
    if t.ndim > 0:
        return jax.vmap(lambda time: model(params, x0, time))(t)
    return model(params, x0, t)


# THIS MUST BE THE EXACT ORDER FROM YOUR TRAINING:
# te, ti, tse, tsi, De, Di, ne, ni, Jee, Jei, Jii, Jie, Ie, Ii, eps
PARAM_KEYS = [
    'tau_e',   # te
    'tau_i',   # ti
    'tau_se',  # tse
    'tau_si',  # tsi
    'Delta_e', # De
    'Delta_i', # Di
    'nu_e',    # ne
    'nu_i',    # ni
    'Jee',     # Jee
    'Jei',     # Jei
    'Jii',     # Jii
    'Jie',     # Jie
    'Iext_e',  # Ie
    'Iext_i',  # Ii
    'eps'      # eps
]

def dict_to_array(p_dict):
    """Safely converts the dict to the training-order JAX array."""
    try:
        return jnp.array([p_dict[key] for key in PARAM_KEYS])
    except KeyError as e:
        print(f"Error: Missing parameter in dictionary: {e}")
        return None


def SystemPeriodicity_DeepONet(initCond, params_dict, alpha):
    # 1. Unfold initial condition [Period, r_e, v_e, s_e, r_i, v_i, s_i]
    T_approx = initCond[0]
    x0_approx = initCond[1:]

    # 2. TRANSFORM DICTIONARY TO JAX VECTOR
    mu_jax = dict_to_array(params_dict)
    x0_jax = jnp.array(x0_approx)
    
    # 3. Predict State at Time T
    # No integration needed, just evaluate
    final_state = predict_state(mu_jax, x0_jax, jnp.array(T_approx))
    
    # 4. Compute differences
    diff = np.zeros(7)
    diff[0] = final_state[2] - alpha   # Poincare condition (s_e == alpha)
    diff[1:] = final_state - x0_approx  # Periodicity condition (phi(T,x0) == x0)
    
    return diff


def InitCondPO_DeepONet(params_dict, exact=False):
    Nvariables = 6
    
    # TRANSFORM DICTIONARY TO JAX VECTOR
    mu_jax = dict_to_array(params_dict)
    x0_start = jnp.zeros(Nvariables)
    
    # 1. Check for oscillations (Evaluate 0 to 1000 ms)
    time_eval = jnp.linspace(0, 1000, 1000)
    trajectory = predict_state(mu_jax, x0_start, time_eval)
    
    # Check v_i (index 4) for oscillations in the last 100ms
    tail_signal = trajectory[-100:, 4] 
    amp_difference = jnp.max(tail_signal) - jnp.min(tail_signal)

    if amp_difference < 0.1:
        return 0, np.zeros(Nvariables), 0 # Status 0: No oscillations

    # 2. Find Poincare Crossing using s_e (index 2)
    mean_se = jnp.mean(trajectory[-100:, 2])
    se_signal = trajectory[:, 2]
    
    # Detect ascending crossings
    crossings = np.where((se_signal[:-1] < mean_se) & (se_signal[1:] > mean_se))[0]
    
    if len(crossings) < 2:
        return 1, np.zeros(Nvariables), 0 # Status 1: Chaotic/Complex

    # 3. Initial Guess from the trajectory
    idx1 = crossings[-2]
    idx2 = crossings[-1]
    
    T_approx = float(time_eval[idx2] - time_eval[idx1])
    x0_approx = np.array(trajectory[idx2])

    if exact:
        # Pass the dictionary directly to the new periodicity function
        res, infodict, ier, msg = fsolve(
            SystemPeriodicity_DeepONet, 
            np.append(T_approx, x0_approx), 
            args=(params_dict, mean_se), 
            xtol=1e-4,
            full_output=True
        )
        
        if ier != 1:
            return 4, x0_approx, T_approx # Status 4: Refinement failed
            
        return 2, res[1:], res[0] # Status 2: Periodic Orbit found
    
    return 2, x0_approx, T_approx
