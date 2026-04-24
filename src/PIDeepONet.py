# Version 7 blockxblock:
    #Normalization
    #Error function control

!pip install equinox optax diffrax
import jax
import jax.numpy as jnp
import equinox as eqx  # A great library for JAX-based neural nets
import optax # JAX optimizer library
from tqdm import tqdm
import jax.random as jrandom
import matplotlib.pyplot as plt
import diffrax as dfx
from jax import vmap
import os

#################################################################
#DeepONet MODEL
##################################################################


# FOURIER FEATURES: Instead of raw t, trunk input are fourier features for better capturing oscillatory dynamics

def fourier_features(t):
    freqs = jnp.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    t = jnp.atleast_1d(t)
    return jnp.concatenate([jnp.sin(freqs * t), jnp.cos(freqs * t)])


# Late fusion (params-IC) DeepONet architecture

class MultiInputDeepONet(eqx.Module):
    branch_mu: eqx.nn.MLP
    branch_x0: eqx.nn.MLP
    trunk_t: eqx.nn.MLP
    merger: eqx.nn.MLP  # The "Late Fusion" MLP
    
    mu_active_idx: jnp.ndarray = eqx.field(static=True) # Index of the active params in training
    t_max: float = eqx.field(static=True)  # Normalization of time
    basis_size: int = eqx.field(static=True)
    system_dimension: int = eqx.field(static=True)

    # Scaling arrays. No x0 normalization bc is very close to zero. 
    mu_min: jnp.ndarray = eqx.field(static=True)
    mu_max: jnp.ndarray = eqx.field(static=True)
    u_center: jnp.ndarray = eqx.field(static=True)
    u_scale: jnp.ndarray = eqx.field(static=True)

    def __init__(self, mu_dim, x0_dim, latent_dim, basis_size,
                 system_dimension, t_max, mu_active_idx,
                 mu_min, mu_max, u_center, u_scale, key):
        
        keys = jax.random.split(key, 4)
        
        self.t_max = t_max
        self.basis_size = basis_size
        self.system_dimension = system_dimension
        self.mu_active_idx = jnp.array(mu_active_idx)

        self.mu_min = jnp.array(mu_min)
        self.mu_max = jnp.array(mu_max)
        self.u_center = jnp.array(u_center)
        self.u_scale = jnp.array(u_scale)


        active_dim = len(mu_active_idx)

        # Branch 1: Extracts features from parameters
        self.branch_mu = eqx.nn.MLP(active_dim, latent_dim, width_size=128, depth=2, # Width size dimension hidden layer, latent layer is the output
                                   activation=jax.nn.tanh, key=keys[0])

        # Branch 2: Extracts features from initial conditions
        self.branch_x0 = eqx.nn.MLP(x0_dim, latent_dim, width_size=128, depth=2,
                                   activation=jax.nn.tanh, key=keys[1])

        # Merger: Combines the two latent vectors
        # Input size is latent_dim * 2 because of concatenation
        self.merger = eqx.nn.MLP(
            latent_dim * 2, system_dimension * basis_size, width_size=128, depth=2, activation=jax.nn.tanh, key=keys[2]
        )

        # Trunk: Learns the temporal basis, with input the 8 fourier features 
        self.trunk_t = eqx.nn.MLP(
            14, basis_size, width_size=128, depth=4, key=keys[3])

    
    def __call__(self, mu, x0, t):

        mu_active = mu[self.mu_active_idx] # Only active parameters in training
        mu_norm = 2.0 * (mu_active - self.mu_min) / (self.mu_max - self.mu_min) - 1.0 # Map parameters to [-1, 1] range for better Tanh performance - 1.0

        t_norm = t / self.t_max # Time normalization
        t_feat = fourier_features(t_norm) # Extract time features
        
        # 1. Get Coefficients (Branch)
        c = self.merger(jnp.concatenate([
            self.branch_mu(mu_norm),
            self.branch_x0(x0)
        ]))

        # 2. Get Basis Functions (Trunk)
        phi = self.trunk_t(t_feat)

        # 3. Precision Reshape: (d, K)
        c = c.reshape(self.system_dimension, self.basis_size)

        # 4. Matrix-Vector product: (d, K) @ (K,) -> (d,)
        raw_out = c @ phi
        return (raw_out * self.u_scale) + self.u_center # Physics scaling in output


# ODE

def f_model(u, mu, W):

    N = W.shape[0]

    r_e = u[0:N]
    v_e = u[N:2*N]
    s_e = u[2*N:3*N]
    r_i = u[3*N:4*N]
    v_i = u[4*N:5*N]
    s_i = u[5*N:6*N]

    te, ti, tse, tsi, De, Di, ne, ni, Jee, Jei, Jii, Jie, Ie, Ii, eps = mu

    c = W @ s_e

    dre = (De / (te * jnp.pi) + 2 * r_e * v_e) / te
    dve = (v_e**2 + ne - (jnp.pi * r_e * te)**2 + Ie +
           te*Jee*s_e + te*eps*c - te*Jei*s_i) / te
    dse = (-s_e + r_e) / tse

    dri = (Di / (ti * jnp.pi) + 2 * r_i * v_i) / ti
    dvi = (v_i**2 + ni - (jnp.pi * r_i * ti)**2 + Ii +
           ti*Jie*s_e + ti*eps*c - ti*Jii*s_i) / ti
    dsi = (-s_i + r_i) / tsi

    return jnp.concatenate([dre, dve, dse, dri, dvi, dsi])


###############################################
#LOSS
##############################################
# This is the crucial part. There are different sources of error, and we will add them gradually 
    # 1) Supervision error: Error wrt real trajectories obtained by integration
    # 2) IC constrain error
    # 3) ODE error: Error of the predicted trajectory when adjusting to the ODE.
      # BIG PROBLEM: Stiffness. System is highly nonlinear, is fast--slow, so this error can grow very fast. Initially, is better to try the system to learn by supervision, and then add this loss constriction.
      # Possible solution: Normalization
    # 4) Var. Eq error: Similar to the previous one...
    # 5) Integrability consistency (rollout): Error of the predicted trajectory when adjusting to integration.

#Stiff systems often have physics losses that are orders of magnitude larger than the data losses. Clipping helps, but you might also need Adaptive Loss Weighting (like the NTK-based weighting or simple Uncertainty Weighting) to keep the components in balance.
# Curriculum learning ("Pre-training with Data (phase 1), Fine-tuning with Physics (phase 2).")

# LOSS FOR PHASE 1
def loss_fn(model, mu, x0, t, u_true):

    def loss_single(mu, x0, t, u_true):

        # Model for fixed params and IC
        def model_t(tt):
            return model(mu, x0, tt)

        # Prediction by the model 
        u_pred = jax.vmap(model_t)(t)
        
        #1) Supervision error
        loss_data = jnp.mean( ((u_pred - u_true) / model.u_scale)**2 ) # u_true precomputed and passed as argument. We compute relative MSE to consider that variables can take values in different scales.

        #2) IC error
        loss_ic = jnp.mean( ((model_t(0.0) - x0) / model.u_scale)**2 )

        return (
            loss_data +
            10.0 * loss_ic 
        )

    return jnp.mean(jax.vmap(loss_single)(mu, x0, t, u_true))




# LOSS FOR PHASE 2
def loss_fn2(model, mu, x0, t, u_true, W, weights):

    def loss_single2(mu, x0, t, u_true, w):

        # Model for fixed params and IC
        def model_t(tt):
            return model(mu, x0, tt)

        # Prediction by the model 
        u_pred = jax.vmap(model_t)(t)
        
        #1) Supervision error
        loss_data = w*jnp.mean( ((u_pred - u_true) / model.u_scale)**2 ) # u_true precomputed and passed as argument, when w=1 (supervision)

        #2) IC error
        loss_ic = jnp.mean( ((model_t(0.0) - x0) / model.u_scale)**2 )

        #3) ODE error
        def res(tt): # ODE residual by comparing prediction with real ODE
            u, du = jax.jvp(model_t, (tt,), (1.0,)) # Jacobian-Vector product, returns prediction + time derivative (jacobian vector produt with time direction) 
            physical_residual = du - f_model(u, mu, W)
            return physical_residual / model.u_scale # Relative scaling of the residual, such that error is not dominated by variable with larger scale
        
        loss_ode = jnp.mean(jax.vmap(res)(t)**2)

        #4) Var eq error
        def var_res(tt):
            phi = jax.jacobian(lambda x: model(mu, x, tt))(x0) #Phi = du(t)/du0
            _, dphi_dt = jax.jvp(lambda t_eval: jax.jacobian(lambda x: model(mu, x, t_eval))(x0), (tt,), (1.0,)) #d/dt(Phi)
            u_pred = model(mu, x0, tt)
            Df = jax.jacobian(lambda u_in: f_model(u_in, mu, W))(u_pred) #Df = df/du evaluated at current predicted state
            physical_var_res = dphi_dt - (Df @ phi)
            return physical_var_res / model.u_scale[:, jnp.newaxis]

        loss_var = jnp.mean(jax.vmap(var_res)(t)**2)

        #5) Rollout consistency (...)
        
        return (loss_data + 10.0 * loss_ic + 0.1 * loss_ode + 0.01 * loss_var)

    return jnp.mean(jax.vmap(loss_single2)(mu, x0, t, u_true, weights))


######################################################
# DATA GENERATION (including supervised trajectories)
#Homogeneous system experiment
######################################################

# ODE to be time integrated 

def f_diffrax(t, u, args):
    mu, W = args
    return f_model(u, mu, W)

# Jax integrator

def integrate(mu, x0, t, W):

    t = jnp.sort(t)

    sol = dfx.diffeqsolve(
        dfx.ODETerm(f_diffrax),
        dfx.Tsit5(),
        t0=t[0],
        t1=t[-1],
        dt0=1e-3,
        y0=x0,
        args=(mu, W),
        saveat=dfx.SaveAt(ts=t),
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5)
    )

    return sol.ys


# Batch sample, for the considered range of Iext_e and eps (rest of params fixed)
def sample_batch(key, B): # B is the number of elements in the batch (i.e. total number of elements in the sample; not the minibatch during training)

    k1, k2, k3 = jax.random.split(key, 3)

    ie = jax.random.uniform(k1, (B,), minval=0.0, maxval=16.0) # Sampled variable params, chosen randomly in their range
    ep = jax.random.uniform(k2, (B,), minval=0.0, maxval=35.0)

    mu0 = jnp.array([8,8,1,5,1,1,-5,-5,5,13,5,13,0,0,0]) # Active params idx: 13 and 15
    
    mu = jnp.tile(mu0, (B,1)) # Batch parameters modif.
    mu = mu.at[:,12].set(ie)
    mu = mu.at[:,14].set(ep)

    x0 = 0.1 * jax.random.normal(k3, (B,6)) # Sample IC around zero, for the homog system (use 6)
    return mu, x0 # Return batch of mus and ICs


def sample_times(key, B, T): # T is the number of timesteps per batch
    k1, k2 = jax.random.split(key)
    t = jax.random.uniform(k1, (B, T), minval=0.0, maxval=10.0) # Just 10ms
    t = jnp.sort(t, axis=1)
    return t


# Precompute sample of params, ICs, times and corresponding trajectories for supervision

key = jax.random.PRNGKey(0)
W = jnp.array([[1.0]]) # Homogeneous system

N_traj = 8192 # Number of supervised trajectories, that is, considered number of active parameter vectors and IC (in the considered range)
N_timesteps = 512 # Number of timesteps (in the considered range)

print(f"Precomputing dataset with {N_traj} trajectories...")

key, sub = jrandom.split(key)

mu_b, x0_b = sample_batch(sub, N_traj) 
t_b = sample_times(sub, N_traj, N_timesteps) #Sample N_traj different time-grids (one per trajectory). This ensures the model doesn't just learn a single time-sequence

u_b = vmap(lambda m, x, t: integrate(m, x, t, W))(mu_b, x0_b, t_b)

print(f"Data generation done! Shape of u_b: {u_b.shape}") # Expected shape: (N_traj, N_timesteps, d) = (1024, 128, 6)

# Compute min and max across batches (axis 0) and time (axis 1)
u_min = u_b.min(axis=(0, 1))
u_max = u_b.max(axis=(0, 1))

# Compute the Shift and Stretch factors
u_center = (u_max + u_min) / 2.0
u_scale = (u_max - u_min) / 2.0

# Prevent division by zero just in case a variable is completely constant
u_scale = jnp.where(u_scale == 0.0, 1.0, u_scale)

print(f"Minimum of components: {u_min}")
print(f"Maximum of components: {u_max}")
print(f"Center (shift): {u_center}")
print(f"Scale (stretch): {u_scale}")



# Initialization

model = MultiInputDeepONet(
    mu_dim=15, x0_dim=6, latent_dim=128, basis_size=128, system_dimension=6, t_max=10.0, mu_active_idx=[12,14], 
    mu_min=jnp.array([0.0, 0.0]), 
    mu_max=jnp.array([16.0, 35.0]), 
    u_center=u_center, 
    u_scale=u_scale, 
    key=key
)



# Optimization setup (phase 1)

scheduler = optax.exponential_decay(
    init_value=1e-3, 
    transition_steps=20000, # Every 20 000 steps learning rate is 
    decay_rate=0.5 # cut in half to keep falling to narrow minima
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), # Clipping essential for stiff systems
    optax.adam(learning_rate=scheduler)
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


# Optimization setup (phase 2)

scheduler_phase2 = optax.exponential_decay(init_value=1e-5, transition_steps=10000, decay_rate=0.5) # Lower learning rate
optimizer_phase2 = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=scheduler_phase2))
opt_state_phase2 = optimizer_phase2.init(eqx.filter(model, eqx.is_inexact_array))






########################################################
# TRAINING
########################################################

# Training step for supervised training (phase 1)

@eqx.filter_jit
def make_step(model, opt_state, mu, x0, t, u_true):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, mu, x0, t, u_true)
    
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss_val



# Training step for PI (phase 2)

@eqx.filter_jit
def make_step2(model, opt_state, mu, x0, t, u_true, W, weights):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn2)(model, mu, x0, t, u_true, W, weights)
    updates, opt_state = optimizer_phase2.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


    
# PHASE 1: SUPERVISED PRE-TRAINING


print("Starting Phase 1: Supervised Pre-training...")
loss_hist = []
pbar = tqdm(range(100001), desc="Training DeepONet (phase 1, supervision)")
for step in pbar:

    key, sub = jrandom.split(key)

    idx = jax.random.randint(sub, (32,), 0, N_traj) # Mini-batch, size 32 
        
    mu = mu_b[idx]
    x0 = x0_b[idx]
    t = t_b[idx]
    u = u_b[idx]

    model, opt_state, loss = make_step(
        model, opt_state, mu, x0, t, u
    )

    loss_hist.append(loss)

    if step % 5000 == 0:
        eqx.tree_serialise_leaves(f"model_{step}.eqx", model)

    if step % 100 == 0:
        pbar.set_postfix({"loss": f"{loss:.4e}"})


# Save Phase 1 model
eqx.tree_serialise_leaves("model_phase1.eqx", model)


# PHASE 2: PI TRAINING

print("Starting Phase 2: Physics Fine-tuning...")
BATCH_SIZE = 32
N_sup = 16   # Number of supervised samples (from precomputed data)
N_coll = 16  # Number of collocation samples (freshly sampled)


pbar = tqdm(range(50000), desc="Training DeepONet (phase 2, PI)")
for step in pbar:
    
    key, sub_idx, sub_coll = jax.random.split(key, 3)

    # Supervised Batch 
    idx = jax.random.randint(sub_idx, (N_sup,), 0, N_traj)
    
    mu_sup = mu_b[idx]
    x0_sup = x0_b[idx]
    t_sup = t_b[idx]
    u_sup = u_b[idx]
    weights_sup = jnp.ones(N_sup) # Enable data loss

    # Collocation Batch (Freshly Sampled) 
    mu_coll, x0_coll = sample_batch(sub_coll, N_coll)
    t_coll = sample_times(sub_coll, N_coll, 512)
    u_coll = jnp.zeros((N_coll, 512, 6)) # Placeholder (not used because weight=0)
    weights_coll = jnp.zeros(N_coll)    # Disable data loss

    #  Concatenate Batches 
    mu = jnp.concatenate([mu_sup, mu_coll], axis=0)
    x0 = jnp.concatenate([x0_sup, x0_coll], axis=0)
    t = jnp.concatenate([t_sup, t_coll], axis=0)
    u = jnp.concatenate([u_sup, u_coll], axis=0)
    weights = jnp.concatenate([weights_sup, weights_coll], axis=0)

    # Training step
    model, opt_state, loss = make_step2(
        model, opt_state, mu, x0, t, u, W, weights
    )

    loss_hist.append(float(loss))

    if step % 5000 == 0:
        eqx.tree_serialise_leaves(f"model_p2_{step}.eqx", model)

    if step % 100 == 0:
        pbar.set_postfix({"loss": f"{loss:.4e}"})


# Save Phase 2 model
eqx.tree_serialise_leaves("model_phase2.eqx", model)



plt.figure(figsize=(8, 5))
plt.semilogy(loss_hist) # Use log scale for loss
plt.title("Training Loss (Physics-Informed)")
plt.xlabel("Step")
plt.ylabel("Mean Squared Error")
plt.grid(True, which="both", ls="-")
plt.savefig("loss_plot.png")
plt.show()



