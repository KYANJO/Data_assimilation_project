# ==============================================================================
# @des: This file contains run functions for any model with data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2024-11-4
# @author: Brian Kyanjo
# ==============================================================================

import sys
import os
import tqdm
import numpy as np
from scipy.stats import multivariate_normal,norm

# class instance of the observation operator and its Jacobian
sys.path.insert(0, os.path.abspath('../src'))
from utils import *
from EnKF.python_enkf.EnKF import EnsembleKalmanFilter as EnKF

os.environ["OMP_NUM_THREADS"] = "all_cores"


# ---- Run model with EnKF ----
def run_model_with_filter(model, model_solver, filter_type, *da_args, **model_kwargs): 
    """ General function to run any kind of model with the Ensemble Kalman Filter """

    # unpack the data assimilation arguments
    parallel_flag     = da_args[0]   # parallel flag
    params            = da_args[1]   # parameters
    Q_err             = da_args[2]   # process noise
    hu_obs            = da_args[3]   # observation vector
    statevec_ens      = da_args[4]   # ensemble of model state
    statevec_bg       = da_args[5]   # background state
    statevec_ens_mean = da_args[6]   # ensemble mean
    statevec_ens_full = da_args[7]   # full ensemble


    nd, N = statevec_ens.shape
    hdim = nd // params["num_state_vars"]

    # current implemented models inlcude: icepack, Lorenz96, flowline, ISSM yet to be implemented
    if model == "icepack":
        import icepack
        import firedrake
        from icepack_model.run_icepack_da import background_step, forecast_step_single
    else:
        raise ValueError("Other models are not yet implemented")
    
    km = 0
    for k in tqdm.trange(params["nt"]):
        # background step
        statevec_bg = background_step(k,model_solver,statevec_bg, hdim, **model_kwargs)

        EnKFclass = EnKF(parameters=params, parallel_flag = parallel_flag)

        # forecast step
        statevec_ens = EnKFclass.forecast_step(statevec_ens, model_solver, forecast_step_single, \
                                               Q_err, **model_kwargs)

        # Compute the ensemble mean
        statevec_ens_mean[:,k+1] = np.mean(statevec_ens, axis=1)

        # Compute the model covariance
        diff = statevec_ens - np.tile(statevec_ens_mean[:,k+1].reshape(-1,1),N)
        if filter_type == "EnKF" or filter_type == "DEnKF":
            Cov_model = 1/(N-1) * diff @ diff.T
        elif filter_type == "EnRSKF" or filter_type == "EnTKF":
            Cov_model = 1/(N-1) * diff 

        # Analysis step
        # if ts[k+1] in ts_obs:
        ind_m = params["ind_m"]
        if (km < params["nt_m"]) and (k+1 == ind_m[km]):
            # idx_obs = np.where(ts[k+1] == ts_obs)[0]

            Cov_obs = params["sig_obs"]**2 * np.eye(2*params["m_obs"]+1)

            utils_functions = UtilsFunctions(params, statevec_ens)
    
            hu_ob = utils_functions.Obs_fun(hu_obs[:,km])

            # Compute the analysis ensemble
            if filter_type == "EnKF":
                analysis  = EnKF(Observation_vec= hu_ob, Cov_obs=Cov_obs, \
                                 Cov_model= Cov_model, \
                                 Observation_function=utils_functions.Obs_fun, \
                                 Obs_Jacobian=utils_functions.JObs_fun, \
                                 parameters=  params,\
                                 parallel_flag=   parallel_flag)
                
                statevec_ens, Cov_model = analysis.EnKF_Analysis(statevec_ens)
            elif filter_type == "DEnKF":
                analysis  = EnKF(Observation_vec= hu_ob, Cov_obs=Cov_obs, \
                                 Cov_model= Cov_model, \
                                 Observation_function=utils_functions.Obs_fun, \
                                 Obs_Jacobian=utils_functions.JObs_fun, \
                                 parameters=  params,\
                                 parallel_flag=   parallel_flag)
                
                statevec_ens, Cov_model = analysis.DEnKF_Analysis(statevec_ens)
            elif filter_type == "EnRSKF":
                analysis  = EnKF(Observation_vec= hu_ob, Cov_obs=Cov_obs, \
                                 Cov_model= Cov_model, \
                                 Observation_function=utils_functions.Obs_fun, \
                                 parameters=  params,\
                                 parallel_flag=   parallel_flag)
                
                statevec_ens, Cov_model = analysis.EnRSKF_Analysis(statevec_ens)
            elif filter_type == "EnTKF":
                analysis  = EnKF(Observation_vec= hu_ob, Cov_obs=Cov_obs, \
                                 Cov_model= Cov_model, \
                                 Observation_function=utils_functions.Obs_fun, \
                                 parameters=  params,\
                                 parallel_flag=   parallel_flag)
                
                statevec_ens, Cov_model = analysis.EnTKF_Analysis(statevec_ens)

            statevec_ens_mean[:,k+1] = np.mean(statevec_ens, axis=1)

            # apply localization
            # coefficients 

            km += 1

            # inflate the ensemble
            statevec_ens = utils_functions.inflate_ensemble(in_place=True)

        # Save the ensemble
        statevec_ens_full[:,:,k+1] = statevec_ens

    return statevec_ens_full, statevec_ens_mean, statevec_bg