# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-11-06
# @description: Synthetic ice stream with data assimilation
# =============================================================================

# --- Imports ---
import os
import sys
import h5py
import numpy as np
import warnings
from firedrake import *
from firedrake.petsc import PETSc
from icepack.constants import (
    ice_density as rho_I,
    water_density as rho_W,
    gravity as g,
)
import icepack
import icepack.models.friction
from scipy.stats import norm, multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Utility Imports ---
sys.path.insert(0, os.path.abspath('../../../src/utils'))
from tools import save_arrays_to_h5
from utils import UtilsFunctions

sys.path.insert(0, '../../../src/models')
from icepack_model.run_icepack_da import generate_true_state, generate_nurged_state, initialize_ensemble

sys.path.insert(0, '../../../src/run_model_da')
from run_models_da import run_model_with_filter

sys.path.insert(0, '../../../config/icepack_config')
from config_loader import load_yaml_to_dict, get_section

# --- Configuration ---
os.environ["PETSC_CONFIGURE_OPTIONS"] = "--download-mpich-device=ch3:sock"
os.environ["OMP_NUM_THREADS"] = "1"
PETSc.Sys.Print('Setting up mesh across %d processes' % COMM_WORLD.size)

# --- Load Parameters ---
# Load parameters from a YAML file
parameters_file = "params.yaml"
parameters = load_yaml_to_dict(parameters_file)

physical_params = get_section(parameters, "physical-parameters")
modeling_params = get_section(parameters, "modeling-parameters")
enkf_params = get_section(parameters, "enkf-parameters")

# --- Geometry and Mesh ---
Lx, Ly = int(float(physical_params["Lx"])), int(float(physical_params["Ly"]))
nx, ny = int(float(physical_params["nx"])), int(float(physical_params["ny"]))
print(f"Mesh dimensions: {Lx} x {Ly} with {nx} x {ny} elements")

comm = COMM_WORLD.Split(COMM_WORLD.rank % 2)
mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly, comm=comm)

Q = firedrake.FunctionSpace(mesh, "CG", int(float(physical_params["degree"])))
V = firedrake.VectorFunctionSpace(mesh, "CG", int(float(physical_params["degree"])))

x, y = firedrake.SpatialCoordinate(mesh)

# --- Bedrock and Surface Elevations ---
b_in, b_out = (float(physical_params["b_in"])), (float(physical_params["b_out"]))
s_in, s_out = (float(physical_params["s_in"])), (float(physical_params["s_out"]))

b = firedrake.interpolate(b_in - (b_in - b_out) * x / Lx, Q)
s0 = firedrake.interpolate(s_in - (s_in - s_out) * x / Lx, Q)
h0 = firedrake.interpolate(s0 - b, Q)

# --- Driving Stress ---
h_in = s_in - b_in
ds_dx = (s_out - s_in) / Lx
tau_D = -rho_I * g * h_in * ds_dx
PETSc.Sys.Print(f"Driving stress = {1000*tau_D} kPa")

# --- Initial Velocity ---
u_in, u_out = float(physical_params["u_in"]), float(physical_params["u_out"])
velocity_x = u_in + (u_out - u_in) * (x / Lx) ** 2
u0 = firedrake.interpolate(firedrake.as_vector((velocity_x, 0)), V)

# --- Friction Coefficient ---
PETSc.Sys.Print("Importing icepack ...")
T = firedrake.Constant(float(modeling_params["T"]))
A = icepack.rate_factor(T)

from icepack.constants import weertman_sliding_law as m
expr = (0.95 - 0.05 * x / Lx) * tau_D / u_in**(1 / m)
C = firedrake.interpolate(expr, Q)

p_W = rho_W * g * firedrake.max_value(0, h0 - s0)
p_I = rho_I * g * h0
phi = 1 - p_W / p_I

# --- Friction Law ---
def weertman_friction_with_ramp(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    s = kwargs["surface"]
    C = kwargs["friction"]

    p_W = rho_W * g * firedrake.max_value(0, h - s)
    p_I = rho_I * g * h
    phi = 1 - p_W / p_I
    return icepack.models.friction.bed_friction(
        velocity=u,
        friction=C * phi,
    )

# --- Ice Stream Model ---
model_weertman = icepack.models.IceStream(friction=weertman_friction_with_ramp)

opts = {"dirichlet_ids": [1], "side_wall_ids": [3, 4]}
solver_weertman = icepack.solvers.FlowSolver(model_weertman, **opts)

u0 = solver_weertman.diagnostic_solve(
    velocity=u0,
    thickness=h0,
    surface=s0,
    fluidity=A,
    friction=C,
)

expr = -1e3 * C * phi * sqrt(inner(u0, u0)) ** (1 / m - 1) * u0
tau_b = firedrake.interpolate(expr, V)

# --- Accumulation ---
a_in = firedrake.Constant(float(modeling_params["a_in"]))
da   = firedrake.Constant(float(modeling_params["da"]))
a    = firedrake.interpolate(a_in + da * x / Lx, Q)

# nurged accumulation
a_in_p  = firedrake.Constant(float(modeling_params["a_in_p"]))
da_p    = firedrake.Constant(float(modeling_params["da_p"]))
a_p     = firedrake.interpolate(a_in_p + da_p * x / Lx, Q)

# --- Update h and u ---
h = h0.copy(deepcopy=True)
u = u0.copy(deepcopy=True)

# --- Ensemble Parameters ---
params = {
    "nt": int(float(modeling_params["num_years"])) * int(float(modeling_params["timesteps_per_year"])),
    "dt": 1.0 / float(modeling_params["timesteps_per_year"]),
    "num_state_vars": int(float(enkf_params["num_state_vars"])),
    "nd": h0.dat.data.size * int(float(enkf_params["num_state_vars"])),
    "Nens": int(float(enkf_params["Nens"])),
    "m_obs": int(float(enkf_params["m_obs"])),
    "inflation_factor": float(enkf_params["inflation_factor"]),
    "sig_model": float(enkf_params["sig_model"]),
    "sig_obs": float(enkf_params["sig_obs"]),
    "sig_Q": float(enkf_params["sig_Q"]),
    "nt_m": int(float(enkf_params["m_obs"])),
    "dt_m": int(float(enkf_params["freq_obs"])),
}

kwargs = {"a":a, "h0":h0, "u0":u0, "C":C, "A":A,"Q":Q,"V":V, "da":float(modeling_params["da"]),
          "b":b, "dt":params["dt"], "seed":float(enkf_params["seed"]), "x":x, "y":y,
          "t":np.linspace(0, int(float(modeling_params["num_years"])), params["nt"] + 1),
          "Lx":Lx, "Ly":Ly, "nx":nx, "ny":ny, "h_nurge_ic":float(enkf_params["h_nurge_ic"]), 
          "u_nurge_ic":float(enkf_params["u_nurge_ic"]),"nurged_entries":float(enkf_params["nurged_entries"]),
         "a_in_p":float(modeling_params["a_in_p"]), "da_p":float(modeling_params["da_p"]),
}

# --- Generate True and Nurged States ---
PETSc.Sys.Print("Generating true state ...")
statevec_true = generate_true_state(
    solver_weertman,
    np.zeros([params["nd"], params["nt"] + 1]), 
    params,  
    **kwargs  
)

PETSc.Sys.Print("Generating nurged state ...")
kwargs["a"] = a_p # Update accumulation with nurged accumulation
statevec_nurged = generate_nurged_state(
    solver_weertman, 
    np.zeros([params["nd"], params["nt"] + 1]), 
    params, 
    **kwargs  
)

# --- Save True and Nurged States ---
save_arrays_to_h5(
    filter_type="true-wrong",
    model=enkf_params["model_name"],
    parallel_flag=enkf_params["parallel_flag"],
    commandlinerun=enkf_params["commandlinerun"],
    degree=np.array([int(float(physical_params["degree"]))]),
    t=kwargs["t"], b_io=np.array([b_in,b_out]),
    Lxy=np.array([Lx,Ly]),nxy=np.array([nx,ny]),
    statevec_true=statevec_true,
    statevec_nurged=statevec_nurged,
)

# --- Synthetic Observations ---
PETSc.Sys.Print("Generating synthetic observations ...")
utils_funs = UtilsFunctions(params, statevec_true)
hu_obs = utils_funs._create_synthetic_observations(statevec_true)

# --- Initialize Ensemble ---
PETSc.Sys.Print("Initializing the ensemble ...")
statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full = initialize_ensemble(
    solver_weertman,
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"]]),
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"], params["nt"] + 1]),
    params, **kwargs
)

# --- Run Data Assimilation ---
PETSc.Sys.Print("Running the model with Data Assimilation ...")

# Additional arguments for the EnKF
da_args = [
    enkf_params["parallel_flag"],
    params,
    np.eye(params["nd"]) * params["sig_Q"] ** 2,
    hu_obs,
    statevec_ens,
    statevec_bg,
    statevec_ens_mean,
    statevec_ens_full,
    enkf_params["commandlinerun"],
]

run_model_with_filter(
    enkf_params["model_name"],
    solver_weertman,
    enkf_params["filter_type"],
    *da_args,
    **kwargs  
)

