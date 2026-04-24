import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np


from PIDeepONet import MultiInputDeepONet

# 1. Re-create the skeleton (MUST match your training parameters)
model_path = "pi_deeponet_model.eqx"
model_skeleton = MultiInputDeepONet(
    mu_dim=15, x0_dim=6, latent_dim=128, basis_size=128, system_dimension=6, t_max=10.0, mu_active_idx=[12,14], 
    mu_min=jnp.array([0.0, 0.0]), 
    mu_max=jnp.array([16.0, 35.0]), 
    u_center=[ 2.4801437e+01,  4.0550064e+02,  9.5484853e-03, -2.8033929e+02,
  3.5323181e+03, -7.7897832e-02], 
    u_scale=[6.6561813e+01, 1.2011915e+03, 9.5495701e-01, 5.2514038e+02, 7.8766201e+03,
 5.1334763e-01], 
    key=jax.random.PRNGKey(0)
)

# 2. Load the trained weights
model = eqx.tree_deserialise_leaves(model_path, model_skeleton)
print("Loaded trained model")
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

