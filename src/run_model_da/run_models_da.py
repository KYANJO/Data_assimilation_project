# ==============================================================================
# @des: This file contains run functions for any model with data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2024-11-4
# @author: Brian Kyanjo
# ==============================================================================
    
# --- Imports ---
from _utility_imports import *
import tqdm

src_dir = os.path.join(project_root, 'src')
applications_dir = os.path.join(project_root, 'applications')
sys.path.insert(0, src_dir)
sys.path.insert(0, applications_dir)

# class instance of the observation operator and its Jacobian
from utils import *
import re
from EnKF.python_enkf.EnKF import EnsembleKalmanFilter as EnKF
from supported_models import SupportedModels

# ---- Run model with EnKF ----
# @njit
def run_model_with_filter(model=None, filter_type=None, *da_args, **model_kwargs): 
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
    commandlinerun    = da_args[8]   # run through the terminal

    nd, N = statevec_ens.shape
    hdim = nd // params["num_state_vars"]

    # call curently supported model Class
    models = SupportedModels(model=model)
    model_module = models.call_model()
    
    km = 0
    radius = 2
    for k in tqdm.trange(params["nt"]):
        # background step
        statevec_bg = model_module.background_step(k,statevec_bg, hdim, **model_kwargs)

        EnKFclass = EnKF(parameters=params, parallel_flag = parallel_flag)
        
        statevec_ens = EnKFclass.forecast_step(statevec_ens, \
                                               model_module.forecast_step_single, \
                                                Q_err, **model_kwargs)

        # Compute the ensemble mean
        statevec_ens_mean[:,k+1] = np.mean(statevec_ens, axis=1)

        # Analysis step
        obs_index = model_kwargs["obs_index"]
        if (km < params["number_obs_instants"]) and (k+1 == obs_index[km]):

            # Compute the model covariance
            diff = statevec_ens - np.tile(statevec_ens_mean[:,k+1].reshape(-1,1),N)
            if filter_type == "EnKF" or filter_type == "DEnKF":
                Cov_model = 1/(N-1) * diff @ diff.T
            elif filter_type == "EnRSKF" or filter_type == "EnTKF":
                Cov_model = 1/(N-1) * diff 

            # convariance matrix for measurement noise
            Cov_obs = params["sig_obs"][k+1]**2 * np.eye(2*params["number_obs_instants"]+1)

            utils_functions = UtilsFunctions(params, statevec_ens) # Needed to update the ensembles for utils functions
    
            hu_ob = utils_functions.Obs_fun(hu_obs[:,km])
            
            # Compute the analysis ensemble
            if re.match(r"\AEnKF\Z", filter_type, re.IGNORECASE):
                analysis  = EnKF(Observation_vec= hu_ob, Cov_obs=Cov_obs, \
                                 Cov_model= Cov_model, \
                                 Observation_function=utils_functions.Obs_fun, \
                                 Obs_Jacobian=utils_functions.JObs_fun, \
                                 parameters=  params,\
                                 parallel_flag=   parallel_flag)
                
                statevec_ens, Cov_model = analysis.EnKF_Analysis(statevec_ens)
            elif re.match(r"\ADEnKF\Z", filter_type, re.IGNORECASE):
                analysis  = EnKF(Observation_vec= hu_ob, Cov_obs=Cov_obs, \
                                 Cov_model= Cov_model, \
                                 Observation_function=utils_functions.Obs_fun, \
                                 Obs_Jacobian=utils_functions.JObs_fun, \
                                 parameters=  params,\
                                 parallel_flag=   parallel_flag)
                
                statevec_ens, Cov_model = analysis.DEnKF_Analysis(statevec_ens)
            elif re.match(r"\AEnRSKF\Z", filter_type, re.IGNORECASE):
                analysis  = EnKF(Observation_vec= hu_ob, Cov_obs=Cov_obs, \
                                 Cov_model= Cov_model, \
                                 Observation_function=utils_functions.Obs_fun, \
                                 parameters=  params,\
                                 parallel_flag=   parallel_flag)
                
                statevec_ens, Cov_model = analysis.EnRSKF_Analysis(statevec_ens)
            elif re.match(r"\AEnTKF\Z", filter_type, re.IGNORECASE):
                analysis  = EnKF(Observation_vec= hu_ob, Cov_obs=Cov_obs, \
                                 Cov_model= Cov_model, \
                                 Observation_function=utils_functions.Obs_fun, \
                                 parameters=  params,\
                                 parallel_flag=   parallel_flag)
                
                statevec_ens, Cov_model = analysis.EnTKF_Analysis(statevec_ens)

            statevec_ens_mean[:,k+1] = np.mean(statevec_ens, axis=1)

            # Adaptive localization
            # radius = 2
            # calculate the correlation coefficient with the background ensembles
            # R = (np.corrcoef(statevec_ens))**2
            # #  compute the euclidean distance between the grid points
            # cutoff_distance = np.linspace(0, 5e3, Cov_model.shape[0])
            # #  the distance at which the correlation coefficient is less than 1/sqrt(N-1) is the radius
            # # radius = 
            # method = "Gaspari-Cohn"
            # localization_weights = localization_matrix(radius, cutoff_distance, method)
            # # perform a schur product to localize the covariance matrix
            # Cov_model = np.multiply(Cov_model, localization_weights)
            
            # update the ensemble with observations instants
            km += 1

            # inflate the ensemble
            utils_functions = UtilsFunctions(params, statevec_ens) # Needed to update the ensembles for inflation
            statevec_ens = utils_functions.inflate_ensemble(in_place=True) 

        # Save the ensemble
        statevec_ens_full[:,:,k+1] = statevec_ens

    return statevec_ens_full, statevec_ens_mean, statevec_bg


