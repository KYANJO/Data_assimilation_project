# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-11-06
# @description: Synthetic ice stream with data assimilation
# =============================================================================

# --- Synthetic ice stream example ---
print(" Importing firedrake and other packages ... ")
import firedrake
import sys, os
import h5py
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import copy

import warnings
warnings.filterwarnings("ignore")

# --- Geometry and input data ---
# an elongated fjord-like geometry (12km wide and 50km from the inflow boundary to the ice front)
# Lx, Ly = 50e3, 12e3
# nx, ny = 48, 32
Lx, Ly = 50e2, 12e2
nx, ny = 16, 10
mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)

Q = firedrake.FunctionSpace(mesh, "CG", 2)
V = firedrake.VectorFunctionSpace(mesh, "CG", 2)

x, y = firedrake.SpatialCoordinate(mesh)

# the bedrock slopes down from 200m ABS at the inflow boundary to -400m at the terminus
b_in, b_out = 200, -400
b = firedrake.interpolate(b_in - (b_in - b_out) * x / Lx, Q)

s_in, s_out = 850, 50
s0 = firedrake.interpolate(s_in - (s_in - s_out) * x / Lx, Q)

h0 = firedrake.interpolate(s0 - b, Q)

# --- Evaluate the driving stress ---
from icepack.constants import (
    ice_density as rho_I,
    water_density as rho_W,
    gravity as g,
)

h_in = s_in - b_in
ds_dx = (s_out - s_in) / Lx
tau_D = -rho_I * g * h_in * ds_dx
print(f"Driving stress = {1000*tau_D} kPa")

# --- Guess for the initial velocity ---
u_in, u_out = 20, 2400
velocity_x = u_in + (u_out - u_in) * (x / Lx) ** 2
u0 = firedrake.interpolate(firedrake.as_vector((velocity_x, 0)), V)

# --- Choosing the friciton coefficient ---
print("Importing icepack ...")
import icepack

T = firedrake.Constant(255.0)
A = icepack.rate_factor(T)

from icepack.constants import weertman_sliding_law as m

expr = (0.95 - 0.05 * x / Lx) * tau_D / u_in**(1 / m)
C = firedrake.interpolate(expr, Q)

p_W = rho_W * g * firedrake.max_value(0, h0 - s0)
p_I = rho_I * g * h0
phi = 1 - p_W / p_I

# --- Define the friction law ---
import icepack.models.friction

# wrapper fuction around the default parametisation
def weertman_friction_with_ramp(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    s = kwargs["surface"]
    C = kwargs["friction"]

    p_W = rho_W * g * firedrake.max_value(0, h - s)
    p_I = rho_I * g * h
    phi = 1 - p_W / p_I
    return icepack.models.friction.bed_friction(
        velocity = u,
        friction = C*phi,
    )

# --- Define the model ---
model_weertman = icepack.models.IceStream(friction = weertman_friction_with_ramp)
opts = {"dirichlet_ids": [1], "side_wall_ids": [3,4]}
solver_weertman = icepack.solvers.FlowSolver(model_weertman, **opts)

u0 = solver_weertman.diagnostic_solve(
                            velocity = u0,
                            thickness = h0,
                            surface = s0,
                            fluidity = A,
                            friction = C
)

from firedrake import sqrt, inner

expr = -1e3*C*phi*sqrt(inner(u0, u0))**(1/m-1)*u0
tau_b = firedrake.interpolate(expr, V)

# --- Accumulation ---
print("Setting accumulation ...")
a_in = firedrake.Constant(1.7)
a_in_p = firedrake.Constant(1.7+1.7*0.1)
da = firedrake.Constant(-2.7)
da_p = firedrake.Constant(-2.7 + -2.7*0.05)
a  = firedrake.interpolate(a_in + da * x / Lx, Q)  # accumulation 
a_p = firedrake.interpolate(a_in_p + da_p * x / Lx, Q)  # accumulation nurged

h = h0.copy(deepcopy=True)
u = u0.copy(deepcopy=True)

# set variables and parameters
num_years = 2
# dt = 2.0
# num_timesteps = int(num_years / dt)
timesteps_per_year = 1
dt = 1.0 / timesteps_per_year
num_timesteps = num_years * timesteps_per_year
t = np.linspace(0, num_years, num_timesteps + 1)

# Dimension of model state
num_state_vars = 3
hdim = h0.dat.data.size
nd = num_state_vars * hdim

sig_model = 0.1
sig_obs   = 1e-2
sig_Q     = 1e-2

Cov_model = sig_model**2 * np.eye(nd) # model error covariance
Q_err     = sig_Q**2 * np.eye(nd)     # process noise covariance

N = 30      # ensemble size
m_obs = 10  # number of observations
freq_obs = 1.5
ind_m = (np.linspace(int(freq_obs/dt),int(num_timesteps),m_obs)).astype(int)
t_m = t[ind_m] # time instatnces of observations

inflation_factor = 1.00

params = {"nt": num_timesteps,
           "dt":dt, "num_state_vars":num_state_vars,
           "nd":nd, "sig_model":sig_model,
           "sig_obs":sig_obs, "sig_Q":sig_Q, "Nens":N,
           "m_obs":m_obs, "inflation_factor":inflation_factor,
           "nt_m": m_obs,"dt_m":freq_obs}

# --- true state ---
print("Generating true state ...")
sys.path.insert(0,'../../src/models')
from icepack_model.run_icepack_da import generate_true_state

statevec_true = np.zeros([params["nd"],params["nt"]+1])
solver = solver_weertman

seed = 1 # random seed
kwargs = {"a":a, "h0":h0, "u0":u0, "C":C, "A":A,"Q":Q,"V":V,
          "b":b, "dt":dt,"seed":seed}
statevec_true = generate_true_state(solver,statevec_true,params,**kwargs)

# --- Observations ---
sys.path.insert(0,'../../src/utils')

from utils import UtilsFunctions
utils_funs = UtilsFunctions(params,statevec_true)

# create synthetic observations
print("Generating synthetic observations ...")
hu_obs = utils_funs._create_synthetic_observations(statevec_true)
# hu_obs

# --- initialize the ensemble ---
print("Initializing the ensemble ...")
statevec_bg         = np.zeros([params["nd"],params["nt"]+1])
statevec_ens_mean   = np.zeros_like(statevec_bg)
statevec_ens        = np.zeros([params["nd"],params["Nens"]])
statevec_ens_full   = np.zeros([params["nd"],params["Nens"],params["nt"]+1])

from icepack_model.run_icepack_da import initialize_ensemble

# add and entry to kwargs disctionery
kwargs["h_nurge_ic"] = 100
kwargs["u_nurge_ic"] = 2.5
kwargs["nurged_entries"] = 500
kwargs["a"] = a_p # nurged accumulation
kwargs['t'] = t
kwargs['x'] = x 
kwargs['Lx'] = Lx
statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full = initialize_ensemble(solver, statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full, params,**kwargs)

# --- Run the model with Data Assimilation ---
print("Running the model with Data Assimilation ...")
sys.path.insert(0,'../../src/run_model_da')
from run_models_da import run_model_with_filter
import tools

model_name   = "icepack"
model_solver = solver_weertman
filter_type  = "EnRSKF"  # EnKF, DEnKF, EnTKF, EnRSKF
parallel_flag = "Serial" # serial, Multiprocessing, Dask, Ray, MPI; only serial is supported for now
num_procs = 1
commandlinerun = True # this script is ran through the terminal

da_args = [parallel_flag, params, Q_err, hu_obs, statevec_ens, statevec_bg, statevec_ens_mean, statevec_ens_full,commandlinerun]

if parallel_flag == "Serial":
    python_script = "../../src/run_model_da/run_models_da.py"
    # tools.run_with_mpi(python_script, num_procs, model_name, model_solver, filter_type, da_args, kwargs)
    run_model_with_filter(model_name, model_solver, filter_type, *da_args, **kwargs)
    # load saved data
    # filename = f"results/{model_name}.h5"
    # with h5py.File(filename, "r") as f:
    #     statevec_ens_full = f["statevec_ens_full"][:]
    #     statevec_ens_mean = f["statevec_ens_mean"][:]
    #     statevec_bg = f["statevec_bg"][:]
else:
    statevec_ens_full, statevec_ens_mean, statevec_bg = run_model_with_filter(model_name, model_solver, filter_type, *da_args, **kwargs)
