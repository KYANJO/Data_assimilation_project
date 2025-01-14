# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-01-13
# @description: Lorenz96 model with data assimilation
# =============================================================================

# --- Imports ---
import sys
import os
import numpy as np

# --- Configuration ---
sys.path.insert(0, '../../config')
from _utility_imports import *

# --- Utility Functions ---
from lorenz96_model.run_lorenz96_da import generate_true_state,  initialize_ensemble
from run_models_da import run_model_with_filter

# --- Local MPI implementation ---
comm = MPI.COMM_WORLD   # Initialize MPI
rank = comm.Get_rank()  # Get rank of current MPI process
size = comm.Get_size()  # Get total number of MPI processes

# --- Load Parameters ---
# Load parameters from a YAML file
parameters_file = "params.yaml"
parameters = load_yaml_to_dict(parameters_file)

physical_params = get_section(parameters, "physical-parameters")
modeling_params = get_section(parameters, "modeling-parameters")
enkf_params = get_section(parameters, "enkf-parameters")

# --- Ensemble Parameters ---
params = {
    "nt": int(float(modeling_params["num_years"])/float(modeling_params["dt"])),
    "dt": float(modeling_params["dt"]),
    "nd": int(float(enkf_params["num_state_vars"])),
    "num_state_vars": int(float(enkf_params["num_state_vars"])),
    "Nens": int(float(enkf_params["Nens"])),
    "m_obs": int(float(enkf_params["m_obs"])),
    "inflation_factor": float(enkf_params["inflation_factor"]),
    "sig_model": float(enkf_params["sig_model"]),
    "sig_obs": float(enkf_params["sig_obs"]),
    "sig_Q": float(enkf_params["sig_Q"]),
    "nt_m": int(float(enkf_params["m_obs"])),
    "dt_m": int(float(enkf_params["freq_obs"])),
}

kwargs = { "dt":params["dt"], "seed":float(enkf_params["seed"]),
          "t":np.linspace(0, int(float(modeling_params["num_years"])), params["nt"] + 1), 
          "u0True": np.array([1,1,1]), "u0b": np.array([2.0,3.0,4.0]), 
          "sigma":float(physical_params["sigma"]), "beta":eval(physical_params["beta"]),
          "rho":float(physical_params["rho"])
}

# --- Generate True and Nurged States ---
print("Generating true state ...")
statevec_true = generate_true_state(
    np.zeros([params["nd"], params["nt"] + 1]), 
    params,  
    **kwargs  
)

# print("Generating nurged state ...")
# statevec_nurged = generate_nurged_state(
#     np.zeros([params["nd"], params["nt"] + 1]), 
#     params, 
#     **kwargs  
# )

comm.Barrier()
if rank == 0:
    # --- Save True and Nurged States ---
    save_arrays_to_h5(
        filter_type="true-wrong",
        model=enkf_params["model_name"],
        parallel_flag=enkf_params["parallel_flag"],
        commandlinerun=enkf_params["commandlinerun"],
        t=kwargs["t"], 
        statevec_true=statevec_true,
    )

# --- Synthetic Observations ---
print("Generating synthetic observations ...")
utils_funs = UtilsFunctions(params, statevec_true)
hu_obs = utils_funs._create_synthetic_observations(statevec_true)

# --- Initialize Ensemble ---
print("Initializing the ensemble ...")
statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full = initialize_ensemble(
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"]]),
    np.zeros([params["nd"], params["nt"] + 1]),
    np.zeros([params["nd"], params["Nens"], params["nt"] + 1]),
    params, **kwargs
)

# --- Run Data Assimilation ---
print("Running the model with Data Assimilation ...")

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

statevec_ens_full, statevec_ens_mean, statevec_bg = run_model_with_filter(
    enkf_params["model_name"],
    enkf_params["filter_type"],
    *da_args,
    **kwargs  
)

# Only rank 0 writes to file
comm.Barrier()
if rank == 0:
    save_arrays_to_h5(
    filter_type=enkf_params["filter_type"],
    model=enkf_params["model_name"],
    parallel_flag=enkf_params["parallel_flag"],
    commandlinerun=enkf_params["commandlinerun"],
    statevec_ens_full=statevec_ens_full,
    statevec_ens_mean=statevec_ens_mean,
    statevec_bg=statevec_bg
    )