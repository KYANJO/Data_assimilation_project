# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-11-06
# @description: Conatins the class for analysis steps all Ensemble Kalman Filter (EnKF)
#              EnKF include both Stochastic and Deterministic EnKF options: EnKF,
#              EnSRF, DEnKF, and EnTKF.
# =============================================================================

import numpy as np
from scipy.stats import multivariate_normal

class Analysis:
    def __init__(self, Cov_obs, Cov_model, Observation_function, Obs_Jacobian=None, parameters=None, taper_matrix=None):
        """
        Initializes the Analysis class for the Ensemble Kalman Filter (EnKF).
        
        Parameters:
        Observation_function: Callable - Observation function mapping state space to observation space.
        Obs_Jacobian: Callable - Jacobian of the observation function.
        Cov_obs: ndarray - Observation covariance matrix (m x m).
        Cov_model: ndarray - Model covariance matrix (n x n).
        taper: ndarray - Covariance taper matrix (n x n).
        params: dict - Dictionary containing parameters like "m_obs" and others.
        """
        self.Cov_obs                = Cov_obs
        self.Cov_model              = Cov_model
        self.Obs_Jacobian           = Obs_Jacobian
        self.parameters             = parameters
        self.taper_matrix           = taper_matrix
        self.Observation_function   = Observation_function

    def _compute_kalman_gain(self):
        """Compute the Kalman gain based on the Jacobian of the observation function."""
        Jobs = self.Obs_Jacobian(self.Cov_model.shape[0])
        inv_matrix = np.linalg.inv(Jobs @ self.Cov_model @ Jobs.T + self.Cov_obs)
        KalGain = self.Cov_model @ Jobs.T @ inv_matrix
        return KalGain
    
    def EnKF_Analysis(self, ensemble, Observation_vec):
        """
        Stochastic Ensemble Kalman Filter (EnKF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
            Observation_vec: ndarray - Observation vector (m x 1).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        # Compute the Kalman gain
        KalGain = self._compute_kalman_gain()

        n,N = ensemble.shape
        m   = Observation_vec.shape[0] 

        # compute virtual observations and ensemble analysis
        virtual_observations  = np.zeros((m,N))
        ensemble_analysis     = np.zeros_like(ensemble)

        for i in range(N):
            # Generate virtual observations
            virtual_observations[:,i] =  Observation_vec + multivariate_normal.rvs(mean=np.zeros(m), cov=self.Cov_obs)
            # Compute the analysis step
            ensemble_analysis[:,i] = ensemble[:,i] + KalGain @ (virtual_observations[:,i] - self.Observation_function(ensemble[:,i]))

        # compute the analysis error covariance
        difference = ensemble_analysis - np.mean(ensemble_analysis, axis=1,keepdims=True)
        analysis_error_cov =(1/(N-1)) * difference @ difference.T

        return ensemble_analysis, analysis_error_cov
       
    def DEnKF_Analysis(self, ensemble, Observation_vec):
        """
        Deterministic Ensemble Kalman Filter (DEnKF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
            Observation_vec: ndarray - Observation vector (m x 1).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        # Compute the Kalman gain
        KalGain = self._compute_kalman_gain()

        n,N = ensemble.shape
        m   = Observation_vec.shape[0] 

        # compute ensemble mean
        ensemble_forecast_mean = np.mean(ensemble, axis=1)

        # compute the anlysis mean
        analysis_mean = ensemble_forecast_mean + KalGain @ (Observation_vec - self.Observation_function(ensemble_forecast_mean))

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
    
    def EnRSKF_Analysis(self, ensemble, Observation_vec):
        """
        Deterministic Ensemble Square Root Filter (EnSRF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
            Observation_vec: ndarray - Observation vector (m x 1).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        _,N = ensemble.shape
        I = np.eye(N)

        # compute the mean of the forecast ensemble
        ensemble_forecast_mean = np.mean(ensemble, axis=1)

        obs_anomaly = Observation_vec.reshape(-1,1) - self.Observation_function(ensemble)

        V =  self.Observation_function(self.Cov_model)
        IN = self.Cov_obs + V@V.T

        # compute the analysis ensemble
        ensemble_analysis  = ensemble_forecast_mean.reshape(-1,1) + self.Cov_model @ V.T @ np.linalg.solve(IN, obs_anomaly)

        #  singular value decomposition
        U,S,Vt = np.linalg.svd(I - V.T @ np.linalg.solve(IN, V))

        # compute the analysis error covariance
        analysis_error_cov = ensemble_analysis + self.Cov_model@(U@np.diag(np.sqrt(S))@U.T)

        return ensemble_analysis, analysis_error_cov
    
    def EnTKF_Analysis(self, ensemble, Observation_vec):
        """
        Ensemble Transform Kalman Filter (EnTKF) analysis step.
        
        Parameters:
            ensemble: ndarray - Ensemble matrix (n x N).
            Observation_vec: ndarray - Observation vector (m x 1).
        
        Returns:
            ensemble_analysis: updated ensemble matrix (n x N).
            analysis_error_cov: ndarray - Analysis error covariance matrix (n x n).
        """
        
        # compute the mean of the forecast ensemble
        ensemble_forecast_mean = np.mean(ensemble, axis=1)

        obs_anomaly = Observation_vec.reshape(-1,1) - self.Observation_function(ensemble)

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