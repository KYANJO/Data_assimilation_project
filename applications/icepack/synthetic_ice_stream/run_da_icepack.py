# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-11-06
# @description: Synthetic ice stream with data assimilation
# =============================================================================

# --- Synthetic ice stream example ---
import firedrake
import sys, os
import h5py
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import copy
import subprocess
from firedrake import *
from firedrake.petsc import PETSc

import warnings
warnings.filterwarnings("ignore")

os.environ["PETSC_CONFIGURE_OPTIONS"] = "--download-mpich-device=ch3:sock"
PETSc.Sys.Print('setting up mesh across %d processes' % COMM_WORLD.size)
# --- Geometry and input data ---
# an elongated fjord-like geometry (12km wide and 50km from the inflow boundary to the ice front)
# Lx, Ly = 50e3, 12e3
# nx, ny = 48, 32
Lx, Ly = 25e2, 6e2
nx, ny = 18, 12

comm = COMM_WORLD.Split(COMM_WORLD.rank % 2)
mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, comm=comm)

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
PETSc.Sys.Print(f"Driving stress = {1000*tau_D} kPa")

# --- Guess for the initial velocity ---
u_in, u_out = 20, 2400
velocity_x = u_in + (u_out - u_in) * (x / Lx) ** 2
u0 = firedrake.interpolate(firedrake.as_vector((velocity_x, 0)), V)

# --- Choosing the friciton coefficient ---
PETSc.Sys.Print("Importing icepack ...")
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

# optimiztion options: 1. Uisng icepack default solver
opts = {"dirichlet_ids": [1], "side_wall_ids": [3,4]}

# optimiztion options: 2. Using PETSc with LU
fast_opts = {
    "dirichlet_ids": [1],
    "side_wall_ids": [3, 4],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_line_search_type": "cp",
    },
}

# optimiztion options: 3. Using PETSc with multigrid
faster_opts = {
    "dirichlet_ids": [1],
    "side_wall_ids": [3, 4],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "ksp_type": "cg",
        "pc_type": "mg",
        "pc_mg_cycle_type": "w",
        "snes_line_search_type": "cp",
    },
    "prognostic_solver_parameters": {
        "ksp_type": "gmres",
        "pc_type": "ilu",
    },
}
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
PETSc.Sys.Print("Setting accumulation ...")
a_in = firedrake.Constant(1.7)
a_in_p = firedrake.Constant(1.7+1.7*0.1)
da = firedrake.Constant(-2.7)
da_p = firedrake.Constant(-2.7 + -2.7*0.05)
a  = firedrake.interpolate(a_in + da * x / Lx, Q)  # accumulation 
a_p = firedrake.interpolate(a_in_p + da_p * x / Lx, Q)  # accumulation nurged

h = h0.copy(deepcopy=True)
u = u0.copy(deepcopy=True)

# set variables and parameters
num_years = 250
# dt = 2.0
# num_timesteps = int(num_years / dt)
timesteps_per_year = 2
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

model_name   = "icepack"
model_solver = solver_weertman
filter_type  = "EnRSKF"  # EnKF, DEnKF, EnTKF, EnRSKF
parallel_flag = "Serial" # serial, Multiprocessing, Dask, Ray, MPI; only serial is supported for now
commandlinerun = True # this script is ran through the terminal

# add the utils directory to path
sys.path.insert(0, os.path.abspath('../../src/utils'))
import tools

# --- true state ---
PETSc.Sys.Print("Generating true state ...")
sys.path.insert(0,'../../src/models')
from icepack_model.run_icepack_da import generate_true_state

statevec_true = np.zeros([params["nd"],params["nt"]+1])
solver = solver_weertman

seed = 1 # random seed
kwargs = {"a":a, "h0":h0, "u0":u0, "C":C, "A":A,"Q":Q,"V":V,
          "b":b, "dt":dt,"seed":seed, "a":a_p, "t":t, "x":x, 
          "y":y, "Lx":Lx, "Ly":Ly, "nx":nx, "ny":ny }
statevec_true = generate_true_state(solver,statevec_true,params,**kwargs)

# --- wrong state ---
PETSc.Sys.Print("Generating wrong state ...")
sys.path.insert(0,'../../src/models')
from icepack_model.run_icepack_da import generate_nurged_state

statevec_nurged = np.zeros([params["nd"],params["nt"]+1])
solver = solver_weertman

# add and entry to kwargs disctionery
kwargs["h_nurge_ic"] = 100
kwargs["u_nurge_ic"] = 2.5
kwargs["nurged_entries"] = 500

statevec_nurged = generate_nurged_state(solver,statevec_nurged,params,**kwargs)

# save both true and wrong datasets
datasets = tools.save_arrays_to_h5(
    filter_type="true-wrong",
    model=model_name,
    parallel_flag=parallel_flag,
    commandlinerun=commandlinerun,
    t=t, b_io=np.array([b_in,b_out]),
    Lxy=np.array([Lx,Ly]),nxy=np.array([nx,ny]),
    statevec_true=statevec_true,
    statevec_nurged= statevec_nurged
    )

# --- Observations ---
sys.path.insert(0,'../../src/utils')

from utils import UtilsFunctions
utils_funs = UtilsFunctions(params,statevec_true)

# create synthetic observations
PETSc.Sys.Print("Generating synthetic observations ...")
hu_obs = utils_funs._create_synthetic_observations(statevec_true)
# hu_obs

# --- initialize the ensemble ---
PETSc.Sys.Print("Initializing the ensemble ...")
statevec_bg         = np.zeros([params["nd"],params["nt"]+1])
statevec_ens_mean   = np.zeros_like(statevec_bg)
statevec_ens        = np.zeros([params["nd"],params["Nens"]])
statevec_ens_full   = np.zeros([params["nd"],params["Nens"],params["nt"]+1])

from icepack_model.run_icepack_da import initialize_ensemble

statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full = initialize_ensemble(solver, statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full, params,**kwargs)

# --- Run the model with Data Assimilation ---
PETSc.Sys.Print("Running the model with Data Assimilation ...")
sys.path.insert(0,'../../src/run_model_da')
from run_models_da import run_model_with_filter
import tools

da_args = [parallel_flag, params, Q_err, hu_obs, statevec_ens, statevec_bg, statevec_ens_mean, statevec_ens_full,commandlinerun]

datasets = run_model_with_filter(model_name, model_solver, filter_type, *da_args, **kwargs)
