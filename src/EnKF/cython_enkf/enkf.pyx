# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-09-24
# @description: Optimized and Robust Ensemble Kalman Filter (EnKF) class with 
#               Parallelization.
#               - The class includes the analysis step of the EnKF with both
#                 Stochastic and Deterministic EnKF options.
# ==============================================================================

# Import necessary Cython and numpy modules 
import numpy as np  # Python numpy for high-level functions
cimport numpy as cnp  # Cython numpy for low-level array operations
from scipy.stats import multivariate_normal

# Declare the numpy array types for Cython
ctypedef cnp.float64_t DTYPE_t  # Define a floating-point type for numpy arrays

cdef class EnsembleKalmanFilter:
    cdef: 
        object Cov_obs  
        object Cov_model
        object taper
        dict params
        object ObsFun
        object JObsFun
    
    def __init__(self, object ObsFun, object JObsFun, cnp.ndarray Cov_obs, cnp.ndarray Cov_model, cnp.ndarray taper, dict params):
        """
        Initializes the Ensemble Kalman Filter (EnKF) with observation function, Jacobian, and covariance matrices.
        
        ObsFun: Python callable for the observation function.
        JObsFun: Python callable for the Jacobian of the observation function.
        Cov_obs: (m x m) numpy array for observation covariance matrix.
        Cov_model: (n x n) numpy array for model covariance matrix.
        taper: (n x n) numpy array for taper matrix.
        params: Dictionary containing filter parameters.
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
    
    cpdef cnp.ndarray compute_kalman_gain(self):
        """
        Compute the Kalman gain based on the Jacobian of the observation function.
        """
        cdef cnp.ndarray Jobs = self.JObsFun(self.Cov_model.shape[0], self.params["m_obs"])
        cdef cnp.ndarray KalGain = self.Cov_model @ Jobs.T @ np.linalg.inv(Jobs @ self.Cov_model @ Jobs.T + self.Cov_obs)
        return KalGain

    cpdef tuple analyze(self, cnp.ndarray huxg_ens, cnp.ndarray huxg_obs):
        """
        Perform the analysis step of the Ensemble Kalman Filter (EnKF).

        huxg_ens: (n x N) numpy array of the ensemble of model states.
        huxg_obs: (m,) numpy array of observations.
        
        Returns:
        Tuple containing:
        - analysis_ens: (n x N) numpy array of the updated ensemble after analysis.
        - analysis_cov: (n x n) numpy array of the updated covariance after analysis.
        """
        cdef int n, N, m, i
        cdef cnp.ndarray huxg_ens_mean
        cdef cnp.ndarray obs_virtual
        cdef cnp.ndarray analysis_ens
        cdef cnp.ndarray analysis_ens_mean
        cdef cnp.ndarray analysis_cov
        cdef cnp.ndarray KalGain

        # Validate dimensions of input arrays
        if huxg_ens.ndim != 2 or huxg_obs.ndim != 1:
            raise ValueError("The ensemble matrix must be 2D, and the observation vector must be 1D.")
        
        #n, N = huxg_ens.shape  # n is state size, N is ensemble size
        #m = huxg_obs.shape[0]  # Observation size
        n = int(huxg_ens.shape[0])
        N = int(huxg_ens.shape[1])
        m = int(huxg_obs.shape[0])

        # Compute the ensemble mean (use Python numpy functions)
        huxg_ens_mean = np.mean(huxg_ens, axis=1, keepdims=True)

        # Compute the Kalman Gain
        KalGain = self.compute_kalman_gain()

        # Initialize variables
        obs_virtual = np.zeros((m, N), dtype=np.float64)
        analysis_ens = np.zeros_like(huxg_ens)

        # Perform the analysis for each ensemble member
        for i in range(N):
            # Generate virtual observations using multivariate normal distribution
            obs_virtual[:, i] = huxg_obs + multivariate_normal.rvs(mean=np.zeros(m), cov=self.Cov_obs)

            # Update the ensemble member with the Kalman gain
            analysis_ens[:, i] = huxg_ens[:, i] + KalGain @ (obs_virtual[:, i] - self.ObsFun(huxg_ens[:, i], self.params["m_obs"]))

        # Compute the mean of the analysis ensemble (use Python numpy functions)
        analysis_ens_mean = np.mean(analysis_ens, axis=1, keepdims=True)

        # Compute the analysis error covariance
        analysis_cov = (1 / (N - 1)) * (analysis_ens - analysis_ens_mean) @ (analysis_ens - analysis_ens_mean).T
        analysis_cov = analysis_cov * self.taper  # Apply covariance tapering

        return analysis_ens, analysis_cov

    '''     

    cpdef set_cov_model(self, cnp.ndarray Cov_model):
        """
        Update the model covariance matrix.

        Cov_model: (n x n) numpy array for the new model covariance matrix.
        """
        # Convert shapes to tuples before comparison
        if tuple(Cov_model.shape) != tuple(self.Cov_model.shape):
            raise ValueError("New model covariance matrix must have the same dimensions as the current one.")
        self.Cov_model = Cov_model

    cpdef set_cov_obs(self, cnp.ndarray Cov_obs):
        """
        Update the observation covariance matrix.

        Cov_obs: (m x m) numpy array for the new observation covariance matrix.
        """
        # Convert shapes to tuples before comparison
        if tuple(Cov_obs.shape) != tuple(self.Cov_obs.shape):
            raise ValueError("New observation covariance matrix must have the same dimensions as the current one.")
        self.Cov_obs = Cov_obs

    '''