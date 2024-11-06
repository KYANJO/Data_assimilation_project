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

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# class instance of the observation operator and its Jacobian
from utils.utils import UtilsFunctions
from EnkF.python_enkf.EnKF import Analysis as Analysis_step


# ---- Run model with EnKF ----
def run_model_with_filter(model, model_solver, filter_type, *da_args, **model_kwargs): 
    """ General function to run any kind of model with the Ensemble Kalman Filter """

    # unpack the data assimilation arguments
    nt                = da_args[0]   # number of time steps
    num_state_vars    = da_args[1]   # number of state variables
    Q_err             = da_args[2]   # process noise covariance matrix
    ts                = da_args[3]   # time steps
    ts_obs            = da_args[4]   # observation time steps
    ind_m             = da_args[5]   # indices of the observation time steps
    nt_m              = da_args[6]   # number of observation time steps
    params            = da_args[7]   # parameters
    statevec_ens      = da_args[8]   # ensemble of model state
    statevec_bg       = da_args[9]   # background state
    statevec_ens_mean = da_args[10]   # ensemble mean
    statevec_ens_full = da_args[11]   # full ensemble


    nd, N = statevec_ens.shape
    hdim = nd // num_state_vars

    # current implemented models inlcude: icepack, Lorenz96, flowline, ISSM yet to be implemented
    if model == "icepack":
        import icepack
        import firedrake
        from icepack_model.run_icepack_da import background_step, forecast_step
    else:
        raise ValueError("Other models are not yet implemented")
    
    km = 0
    for k in tqdm.trange(nt):
        # background step
        statevec_bg = background_step(k,model_solver,statevec_bg, hdim, **model_kwargs)

        # forecast step
        statevec_ens = forecast_step(model_solver, statevec_ens, num_state_vars, **model_kwargs)

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
        if (km < nt_m) and (k+1 == ind_m[km]):
            idx_obs = np.where(ts[k+1] == ts_obs)[0]

            Cov_obs = params["sig_obs"]**2 * np.eye(2*params["m_obs"]+1)

            utils_functions = UtilsFunctions(params, statevec_ens)
            analysis  = Analysis_step(Cov_obs, Cov_model, utils_functions.Obs_fun, utils_functions.JObs_fun, \
                                 params)

            hu_ob = utils_functions.Obs_fun(hu_obs[:,km], params["m_obs"])

            # Compute the analysis ensemble
            if filter_type == "EnKF":
                statevec_ens, Cov_model = analysis.EnKF_analysis(statevec_ens, hu_ob, utils_functions.Obs_fun, \
                                                        utils_functions.JObs_fun, Cov_obs, Cov_model, params)
            elif filter_type == "DEnKF":
                statevec_ens, Cov_model = analysis.DEnKF_analysis(statevec_ens, hu_ob, utils_functions.Obs_fun, \
                                                         utils_functions.JObs_fun, Cov_obs, Cov_model, params)
            elif filter_type == "EnRSKF":
                statevec_ens, Cov_model = analysis.EnRSKF_analysis(statevec_ens, hu_ob, utils_functions.Obs_fun, \
                                                           utils_functions.JObs_fun, Cov_obs, Cov_model, params)
            elif filter_type == "EnTKF":
                statevec_ens, Cov_model = analysis.EnTKF_analysis(statevec_ens, hu_ob, utils_functions.Obs_fun, \
                                                         utils_functions.JObs_fun, Cov_obs, Cov_model, params)

            statevec_ens_mean[:,k+1] = np.mean(statevec_ens, axis=1)

            km += 1

            # inflate the ensemble
            statevec_ens = utils_functions.inflate_ensemble(statevec_ens, params["inflation_factor"], in_place=True)

        # Save the ensemble
        statevec_ens_full[:,:,k+1] = statevec_ens

    return statevec_ens_full, statevec_ens_mean, statevec_bg