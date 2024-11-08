# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-11-06
# @description: Conatins the class for analysis steps all Ensemble Kalman Filter (EnKF)
#              EnKF include both Stochastic and Deterministic EnKF options: EnKF,
#              EnSRF, DEnKF, and EnTKF.
#             - The class further includes parallelized forecast and analysis steps with
#               MPI, Dask, Ray, and Multiprocessing.
# =============================================================================

import os, sys
import numpy as np
from scipy.stats import multivariate_normal

class EnsembleKalmanFilter:
    def __init__(self, Observation_vec=None, Cov_obs=None, Cov_model=None, \
                 Observation_function=None, Obs_Jacobian=None, parameters=None, taper_matrix=None, parallel_flag="serial"):
        """
        Initializes the Analysis class for the Ensemble Kalman Filter (EnKF).
        
        Parameters:
        Observation_function: Callable - Observation function mapping state space to observation space.
        observation_vec: ndarray - Observation vector (m x 1).
        Obs_Jacobian: Callable - Jacobian of the observation function.
        Cov_obs: ndarray - Observation covariance matrix (m x m).
        Cov_model: ndarray - Model covariance matrix (n x n).
        taper: ndarray - Covariance taper matrix (n x n).
        params: dict - Dictionary containing parameters like "m_obs" and others.\
        parallel_flag: str - Flag for parallelization (serial,MPI, Dask, Ray, Multiprocessing).
        """
        self.Observation_vec        = Observation_vec
        self.Cov_obs                = Cov_obs
        self.Cov_model              = Cov_model
        self.Obs_Jacobian           = Obs_Jacobian
        self.parameters             = parameters
        self.taper_matrix           = taper_matrix
        self.Observation_function   = Observation_function
        self.parallel_flag          = parallel_flag

    def _compute_kalman_gain(self):
        """Compute the Kalman gain based on the Jacobian of the observation function."""
        Jobs = self.Obs_Jacobian(self.Cov_model.shape[0])
        inv_matrix = np.linalg.inv(Jobs @ self.Cov_model @ Jobs.T + self.Cov_obs)
        KalGain = self.Cov_model @ Jobs.T @ inv_matrix
        return KalGain
    
    # Forecast step
    def forecast_step(self, ensemble, solver, forecast_step_single, Q_err, **model_kwags):
        """
        Forecast step for the Ensemble Kalman Filter (EnKF).
        
        Parameters:
            forecast_step_single: Callable - Function for the forecast step of each ensemble member.
            Q_err: ndarray - Process noise matrix.
            parallel_flag: str - Flag for parallelization (serial,MPI, Dask, Ray, Multiprocessing).
            **model_kwags: dict - Keyword arguments for the model.
        
        Returns:
            ensemble: ndarray - Updated ensemble matrix.
        """
        if self.parallel_flag == "serial":
            # Serial forecast step
            _, Nens = ensemble.shape # Get the number of ensemble members

            # Loop over the ensemble members
            for ens in range(Nens):
                ensemble[:,ens] = forecast_step_single(solver, ensemble[:,ens],\
                                             Q_err, self.parameters, **model_kwags)
            return ensemble
        elif self.parallel_flag == "MPI":
            pass
        elif self.parallel_flag == "Dask":
            # Parallel forecast step using Dask
            import dask
            import dask.array as da

            # Convert the ensemble to a dask array
            ensemble_dask = da.from_array(ensemble, chunks=(ensemble.shape[0], ensemble.shape[1]//2))

            # distribute the ensemble members to the workers
            ensemble_tasks = [dask.delayed(forecast_step_single)(solver, ensemble_dask[:,i], Q_err, self.parameters, **model_kwags) for i in range(ensemble.shape[1])]

            # compute all ensemble members concurrently
            ensemble = da.compute(*ensemble_tasks)

            # update the ensemble matrix
            for ens in range(ensemble.shape[1]):
                ensemble[:,ens] = ensemble[ens]

            return ensemble

        elif self.parallel_flag == "Ray":
            pass
        elif self.parallel_flag == "Multiprocessing":
            pass
        else:
            raise ValueError("Invalid parallel flag. Choose from 'serial', 'MPI', 'Dask', 'Ray', 'Multiprocessing'.")

    # Analysis steps
    def EnKF_Analysis(self, ensemble):
        """
        Stochastic Ensemble Kalman Filter (EnKF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        # Compute the Kalman gain
        KalGain = self._compute_kalman_gain()

        n,N = ensemble.shape
        m   =self.Observation_vec.shape[0] 

        # compute virtual observations and ensemble analysis
        virtual_observations  = np.zeros((m,N))
        ensemble_analysis     = np.zeros_like(ensemble)

        for i in range(N):
            # Generate virtual observations
            virtual_observations[:,i] =  self.Observation_vec+ multivariate_normal.rvs(mean=np.zeros(m), cov=self.Cov_obs)
            # Compute the analysis step
            ensemble_analysis[:,i] = ensemble[:,i] + KalGain @ (virtual_observations[:,i] - self.Observation_function(ensemble[:,i]))

        # compute the analysis error covariance
        difference = ensemble_analysis - np.mean(ensemble_analysis, axis=1,keepdims=True)
        analysis_error_cov =(1/(N-1)) * difference @ difference.T

        return ensemble_analysis, analysis_error_cov
       
    def DEnKF_Analysis(self, ensemble):
        """
        Deterministic Ensemble Kalman Filter (DEnKF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        # Compute the Kalman gain
        KalGain = self._compute_kalman_gain()

        n,N = ensemble.shape
        m   =self.Observation_vec.shape[0] 

        # compute ensemble mean
        ensemble_forecast_mean = np.mean(ensemble, axis=1)

        # compute the anlysis mean
        analysis_mean = ensemble_forecast_mean + KalGain @ (self.Observation_vec- self.Observation_function(ensemble_forecast_mean))

        # compute the forecast and analysis anomalies
        forecast_anomalies = np.zeros_like(ensemble)
        analysis_anomalies = np.zeros_like(ensemble)

        for i in range(N):
            forecast_anomalies[:,i] = ensemble[:,i] - ensemble_forecast_mean
            analysis_anomalies[:,i] = forecast_anomalies[:,i] - 0.5 * KalGain @ self.Observation_function(forecast_anomalies[:,i])

        # compute the analysis ensemble
        ensemble_analysis = analysis_anomalies + analysis_mean.reshape(-1,1)

        # compute the analysis error covariance
        analysis_error_cov =(1/(N-1)) * analysis_anomalies @ analysis_anomalies.T

        return ensemble_analysis, analysis_error_cov
    
    def EnRSKF_Analysis(self, ensemble):
        """
        Deterministic Ensemble Square Root Filter (EnSRF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        _,N = ensemble.shape
        I = np.eye(N)

        # compute the mean of the forecast ensemble
        ensemble_forecast_mean = np.mean(ensemble, axis=1)

        obs_anomaly =self.Observation_vec.reshape(-1,1) - self.Observation_function(ensemble)

        V =  self.Observation_function(self.Cov_model)
        IN = self.Cov_obs + V@V.T

        # compute the analysis ensemble
        ensemble_analysis  = ensemble_forecast_mean.reshape(-1,1) + self.Cov_model @ V.T @ np.linalg.solve(IN, obs_anomaly)

        #  singular value decomposition
        U,S,Vt = np.linalg.svd(I - V.T @ np.linalg.solve(IN, V))

        # compute the analysis error covariance
        analysis_error_cov = ensemble_analysis + self.Cov_model@(U@np.diag(np.sqrt(S))@U.T)

        return ensemble_analysis, analysis_error_cov
    
    def EnTKF_Analysis(self, ensemble):
        """
        Ensemble Transform Kalman Filter (EnTKF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        
        # compute the mean of the forecast ensemble
        ensemble_forecast_mean = np.mean(ensemble, axis=1)

        obs_anomaly =self.Observation_vec.reshape(-1,1) - self.Observation_function(ensemble)

        _obs_operator = self.Observation_function(self.Cov_model)

        # perform singular value decomposition
        U,S, _ = np.linalg.svd(_obs_operator.T @ np.linalg.solve(self.Cov_obs, _obs_operator))

        # compute the inverse of the analysis error covariance
        inv_analysis_error_cov = U @ np.diag(1/(S+1)) @ U.T

        # Right hand side of the analysis equation
        rhs = _obs_operator.T @ np.linalg.solve(self.Cov_obs, obs_anomaly)

        # compute the analysis ensemble
        ensemble_analysis = ensemble_forecast_mean.reshape(-1,1) + self.Cov_model@inv_analysis_error_cov@rhs

        # compute the analysis ensemble increment
        ensemble_increment = self.Cov_model @ U @ np.diag(np.sqrt(1/(S+1))) @ U.T

        # compute the analysis error covariance
        analysis_error_cov = ensemble_analysis + ensemble_increment

        return ensemble_analysis, analysis_error_cov