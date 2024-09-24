# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-09-24
# @description: This script includes the Ensemble Kalman Filter (EnKF) class for data assimilation.
# =============================================================================

# import libraries ========================
import numpy as np
from scipy.stats import multivariate_normal

class EnsembleKalmanFilter:
    def __init__(self, ObsFun, JObsFun, Cov_obs, Cov_model, taper, params):
        """
        Initializes the Ensemble Kalman Filter (EnKF).
        
        Parameters:
        ObsFun: Function - The observation function.
        JObsFun: Function - The Jacobian of the observation function.
        Cov_obs: ndarray (m x m) - Observation covariance matrix.
        Cov_model: ndarray (n x n) - Model covariance matrix.
        taper: ndarray (n x n) - Covariance taper matrix.
        params: dict - Dictionary containing additional parameters like "m_obs".
        """
        self.ObsFun = ObsFun
        self.JObsFun = JObsFun
        self.Cov_obs = Cov_obs
        self.Cov_model = Cov_model
        self.taper = taper
        self.params = params
        self.validate_input()

    def validate_input(self):
        """
        Validate the input matrices for correct dimensions and consistency.
        """
        if not isinstance(self.Cov_obs, np.ndarray) or not isinstance(self.Cov_model, np.ndarray):
            raise ValueError("Covariance matrices must be numpy arrays.")
        if self.Cov_obs.shape[0] != self.Cov_obs.shape[1]:
            raise ValueError("Observation covariance matrix must be square.")
        if self.Cov_model.shape[0] != self.Cov_model.shape[1]:
            raise ValueError("Model covariance matrix must be square.")
        if self.taper.shape != self.Cov_model.shape:
            raise ValueError("Taper matrix must have the same dimensions as the model covariance matrix.")
    
    def compute_kalman_gain(self):
        """
        Compute the Kalman gain based on the Jacobian of the observation function.
        """
        Jobs = self.JObsFun(self.Cov_model.shape[0], self.params["m_obs"])
        KalGain = self.Cov_model @ Jobs.T @ np.linalg.inv(Jobs @ self.Cov_model @ Jobs.T + self.Cov_obs)
        return KalGain

    def analyze(self, huxg_ens, huxg_obs):
        """
        Perform the analysis step of the Ensemble Kalman Filter (EnKF).

        Parameters:
        huxg_ens: ndarray (n x N) - The ensemble matrix of model states (n is state size, N is ensemble size).
        huxg_obs: ndarray (m,) - The observation vector (m is measurement size).

        Returns:
        analysis_ens: ndarray (n x N) - The updated ensemble after analysis.
        analysis_cov: ndarray (n x n) - The updated covariance after analysis.
        """
        if huxg_ens.ndim != 2 or huxg_obs.ndim != 1:
            raise ValueError("The ensemble matrix must be 2D, and the observation vector must be 1D.")
        
        n, N = huxg_ens.shape  # n is the state size, N is the ensemble size
        m = huxg_obs.shape[0]  # Measurement size

        # Compute the ensemble mean
        huxg_ens_mean = np.mean(huxg_ens, axis=1, keepdims=True)

        # Compute the Kalman Gain
        KalGain = self.compute_kalman_gain()

        # Initialize virtual observations and analysis ensemble
        obs_virtual = np.zeros((m, N))
        analysis_ens = np.zeros_like(huxg_ens)

        # Perform the analysis for each ensemble member
        for i in range(N):
            # Generate virtual observations using multivariate normal distribution
            obs_virtual[:, i] = huxg_obs + multivariate_normal.rvs(mean=np.zeros(m), cov=self.Cov_obs)

            # Update the ensemble member with the Kalman gain
            analysis_ens[:, i] = huxg_ens[:, i] + KalGain @ (obs_virtual[:, i] - self.ObsFun(huxg_ens[:, i], self.params["m_obs"]))

        # Compute the mean of the analysis ensemble
        analysis_ens_mean = np.mean(analysis_ens, axis=1, keepdims=True)

        # Compute the analysis error covariance
        analysis_cov = (1 / (N - 1)) * (analysis_ens - analysis_ens_mean) @ (analysis_ens - analysis_ens_mean).T
        analysis_cov = analysis_cov * self.taper  # Apply covariance tapering

        return analysis_ens, analysis_cov

    def set_cov_model(self, Cov_model):
        """
        Update the model covariance matrix.

        Parameters:
        Cov_model: ndarray (n x n) - New model covariance matrix.
        """
        if Cov_model.shape != self.Cov_model.shape:
            raise ValueError("New model covariance matrix must have the same dimensions as the current one.")
        self.Cov_model = Cov_model
    
    def set_cov_obs(self, Cov_obs):
        """
        Update the observation covariance matrix.

        Parameters:
        Cov_obs: ndarray (m x m) - New observation covariance matrix.
        """
        if Cov_obs.shape != self.Cov_obs.shape:
            raise ValueError("New observation covariance matrix must have the same dimensions as the current one.")
        self.Cov_obs = Cov_obs
