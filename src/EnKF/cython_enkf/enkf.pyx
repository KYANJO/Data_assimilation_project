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
from dask import delayed, compute

# Declare the numpy array types for Cython
ctypedef cnp.float64_t DTYPE_t  # Define a floating-point type for numpy arrays

# Define the ensemble forecast function outside of the Cython class
def forecast_single_member(i, k, statevec_ens, params, grid, bedfun, modelfun, run_modelfun, nd, Q):
    """
    Function to forecast a single ensemble member. This function is used by Dask in parallel.
    """
    huxg_temp = np.squeeze(run_modelfun(statevec_ens[:-1, i], params, grid, bedfun, modelfun))  # Run model

    nos = np.random.multivariate_normal(np.zeros(nd), Q)  # Process noise

    # Update state ensemble with noise and forecast
    statevec_ens[:, i] = np.concatenate([huxg_temp, [params["facemelt"][k+1] / params["uscale"]]]) + nos
    
    return statevec_ens[:, i]

# Define the Ensemble Kalman Filter (EnKF) class
cdef class EnsembleKalmanFilter:
    cdef:
        #object Cov_obs
        #object Cov_model
        object taper
        dict params
        object ObsFun
        object JObsFun

    def __init__(self, object ObsFun, object JObsFun, cnp.ndarray taper, dict params):
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
        #self.Cov_obs = Cov_obs
        #self.Cov_model = Cov_model
        self.taper = taper
        self.params = params
        #self.validate_input(Cov_model, Cov_obs)

    def validate_input(self, Cov_model, Cov_obs):
        """ Validate the input matrices for correct dimensions and consistency. """
        if not isinstance(Cov_obs, np.ndarray) or not isinstance(Cov_model, np.ndarray):
            raise ValueError("Covariance matrices must be numpy arrays.")
        if Cov_obs.shape[0] != Cov_obs.shape[1]:
            raise ValueError("Observation covariance matrix must be square.")
        if Cov_model.shape[0] != Cov_model.shape[1]:
            raise ValueError("Model covariance matrix must be square.")
        if self.taper.shape != Cov_model.shape:
            raise ValueError("Taper matrix must have the same dimensions as the model covariance matrix.")

    cpdef cnp.ndarray compute_kalman_gain(self, cnp.ndarray Cov_obs, cnp.ndarray Cov_model):
        """ Compute the Kalman gain based on the Jacobian of the observation function. """
        cdef cnp.ndarray Jobs = self.JObsFun(Cov_model.shape[0], self.params["m_obs"])
        cdef cnp.ndarray KalGain = Cov_model @ Jobs.T @ np.linalg.inv(Jobs @ Cov_model @ Jobs.T + Cov_obs)
        return KalGain

    cpdef tuple analyze(self, cnp.ndarray huxg_ens, cnp.ndarray huxg_obs, cnp.ndarray Cov_obs, cnp.ndarray Cov_model):
        """ Perform the analysis step of the Ensemble Kalman Filter (EnKF). """
        cdef int n, N, m, i
        cdef cnp.ndarray huxg_ens_mean
        cdef cnp.ndarray obs_virtual
        cdef cnp.ndarray analysis_ens
        cdef cnp.ndarray analysis_ens_mean
        cdef cnp.ndarray analysis_cov
        cdef cnp.ndarray KalGain

        # n, N = huxg_ens.shape  # n is state size, N is ensemble size
        # m = huxg_obs.shape[0]  # Observation size
        n = int(huxg_ens.shape[0])
        N = int(huxg_ens.shape[1])
        m = int(huxg_obs.shape[0])
        
        # Compute the ensemble mean
        huxg_ens_mean = np.mean(huxg_ens, axis=1, keepdims=True)

        # Compute the Kalman Gain
        KalGain = self.compute_kalman_gain(Cov_obs, Cov_model)

        # Initialize variables
        obs_virtual = np.zeros((m, N), dtype=np.float64)
        analysis_ens = np.zeros_like(huxg_ens)

        # Perform the analysis for each ensemble member
        for i in range(N):
            # Generate virtual observations using multivariate normal distribution
            obs_virtual[:, i] = huxg_obs + multivariate_normal.rvs(mean=np.zeros(m), cov=Cov_obs)

            # Update the ensemble member with the Kalman gain
            analysis_ens[:, i] = huxg_ens[:, i] + KalGain @ (obs_virtual[:, i] - self.ObsFun(huxg_ens[:, i], self.params["m_obs"]))

        # Compute the mean of the analysis ensemble
        analysis_ens_mean = np.mean(analysis_ens, axis=1, keepdims=True)

        # Compute the analysis error covariance
        analysis_cov = (1 / (N - 1)) * (analysis_ens - analysis_ens_mean) @ (analysis_ens - analysis_ens_mean).T
        analysis_cov = analysis_cov * self.taper  # Apply covariance tapering

        return analysis_ens, analysis_cov

    def EnKF_forecast(self, int k, int N, cnp.ndarray statevec_bg, cnp.ndarray statevec_ens, cnp.ndarray statevec_ens_mean, object grid, object bedfun, object modelfun, object run_modelfun, int nd, cnp.ndarray Q):
        """
        Perform the forecast step of the Ensemble Kalman Filter (EnKF) using Dask for parallelization.
        
        Parameters:
        k - Current time step index
        N - Number of ensemble members
        statevec_bg - Background state vector
        statevec_ens - State vector ensemble
        grid, bedfun, modelfun - Model functions
        run_modelfun - Function to run the model
        nd - Dimensionality of the noise vector
        Q - Covariance matrix for process noise
        
        Returns:
        - Updated state vector and forecasted ensemble
        """
        # Step 1: Update the background state vector
        statevec_bg[:-1, k+1] = np.squeeze(run_modelfun(statevec_bg[:-1, k], self.params, grid, bedfun, modelfun))
        statevec_bg[-1, k+1] = self.params["facemelt"][k+1] / self.params["uscale"]

        # Step 2: create a lazy Dask task for each ensemble member
        ensemble_tasks = [delayed(forecast_single_member)(i, k, statevec_ens, self.params, grid, bedfun, modelfun, run_modelfun, nd, Q) for i in range(N)]

        # execute the tasks in parallel
        updated_ensemble = compute(*ensemble_tasks)  

        # convert the result back to a numpy array
        statevec_ens = np.array(updated_ensemble).T
        #statevec_ens = np.array(updated_ensemble)

        # Step 3: Update the ensemble state
        # for i in range(N):
        #     statevec_ens[:, i] = updated_ensemble[i]
            #huxg_temp = np.squeeze(run_modelfun(statevec_ens[:-1, i], self.params, grid, bedfun, modelfun))

            #nos = np.random.multivariate_normal(np.zeros(nd), Q)  # Process noise

            # Update state ensemble with noise and forecast
            #statevec_ens[:, i] = np.concatenate([huxg_temp, [self.params["facemelt"][k+1] / self.params["uscale"]]]) + nos

        # Step 4: Compute the mean of the forecasted ensemble
        #statevec_ens_mean = np.zeros_like(statevec_bg)
        statevec_ens_mean[:, k + 1] = np.mean(statevec_ens, axis=1)

        # Step 5: Forecast error covariance matrix
        #diff = statevec_ens - statevec_ens_mean[:, k + 1].reshape(-1, 1)
        diff = statevec_ens - np.tile(statevec_ens_mean[:, k+1].reshape(-1, 1), N)
        Cov_model = (1 / (N - 1)) * diff @ diff.T

        return statevec_bg, statevec_ens, statevec_ens_mean, Cov_model

    def model_run_with_EnKF(self, int nt,  int N, cnp.ndarray statevec_bg,\
                         cnp.ndarray statevec_ens, cnp.ndarray statevec_ens_mean, \
                         cnp.ndarray statevec_ens_full, object grid, object bedfun, \
                         object modelfun, object run_modelfun, int nd, cnp.ndarray Q, \
                         cnp.ndarray ts, cnp.ndarray ts_obs, cnp.ndarray huxg_virtual_obs, \
                         cnp.float sig_obs):
        """
        Run the model with the EnKF.
        """

        for k in range(nt):
            self.params["tcurrent"] = k + 1
            print(f"Step {k+1}\n")

            # Forecast step
            statevec_bg, statevec_ens, statevec_ens_mean, Cov_model = self.EnKF_forecast(k, N, statevec_bg, statevec_ens, statevec_ens_mean, grid, bedfun, modelfun, run_modelfun, nd, Q)

            # Check for observations at time step k+1
            if ts[k+1] in ts_obs:
                idx_obs = np.where(ts[k+1] == ts_obs)[0]

                # taper the covariance matrix
                Cov_model *= self.taper

                # Measurement noise covariance
                Cov_obs = (sig_obs**2) * np.eye(2 * self.params["m_obs"] + 1)

                # Subsample virtual observations to actual measurement locations
                huxg_obs = self.ObsFun(huxg_virtual_obs[:, idx_obs], self.params["m_obs"])

                # flatten huxg_obs
                huxg_obs = huxg_obs.ravel()

                # Analysis step
                # Analysis corrections
                statevec_ens_temp, Cov_model = self.analyze(statevec_ens, huxg_obs, Cov_obs, Cov_model)

                statevec_ens = statevec_ens_temp
                statevec_ens_mean[:, k+1] = np.mean(statevec_ens, axis=1)

                # Inflate ensemble spread
                statevec_ens = np.tile(statevec_ens_mean[:, k+1].reshape(-1, 1), N) + self.params["inflation"] * (statevec_ens - np.tile(statevec_ens_mean[:, k+1].reshape(-1, 1), N))

                # Update facemelt parameter for future steps
                self.params["facemelt"][k+1:] = statevec_ens_mean[-1, k+1] * self.params["uscale"] * np.ones_like(self.params["facemelt"][k+1:])

            # Store full ensemble for the current time step
            statevec_ens_full[:, :, k+1] = statevec_ens

        return statevec_ens_full, statevec_ens_mean


        




