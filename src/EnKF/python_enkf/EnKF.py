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
import re
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
    def forecast_step(self, ensemble=None, forecast_step_single=None, Q_err=None, **model_kwags):
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
        
        if re.match(r"\Aserial\Z", self.parallel_flag, re.IGNORECASE):
            # Serial forecast step
            nd, Nens = ensemble.shape # Get the number of ensemble members

            # Loop over the ensemble members
            for ens in range(Nens):
                ensemble[:,ens] = forecast_step_single(ens, ensemble, nd, \
                                             Q_err, self.parameters, **model_kwags)
            return ensemble

        # Parallel forecast step using MPI
        elif re.match(r"\AMPI\Z", self.parallel_flag, re.IGNORECASE):
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            # Get the number of ensemble members
            nd, Nens = ensemble.shape

            size = min(comm.Get_size(), Nens)  # Limit processes to ensemble size

            # Balanced workload
            chunk_sizes = [(Nens // size) + (1 if i < (Nens % size) else 0) for i in range(size)]
            displacements = [sum(chunk_sizes[:i]) for i in range(size)]
            start_idx = displacements[rank]
            end_idx = start_idx + chunk_sizes[rank]

            # Local chunk
            local_ensemble = ensemble[:, start_idx:end_idx]

            # Perform forecast step
            for ens in range(local_ensemble.shape[1]):
                local_ensemble[:, ens] = forecast_step_single( ens, local_ensemble, nd, Q_err, self.parameters, **model_kwags)

            # Avoid gather; update ensemble in place
            gathered_ensemble = comm.allgather(local_ensemble)

            if rank == 0:
                # print(f"Gathered ensemble shape: {np.hstack(gathered_ensemble).shape}")
                ensemble = np.hstack(gathered_ensemble)

            return ensemble


        # Parallel forecast step using Dask
        elif re.match(r"\ADask\Z", self.parallel_flag, re.IGNORECASE):
            import dask
            import dask.array as da
            from dask import compute, delayed
            import copy

            nd, Nens = ensemble.shape

            # Create delayed tasks for each ensemble member
            tasks = [
                    delayed(forecast_step_single)(ens, ensemble, nd, Q_err, self.parameters, **model_kwags)
                    for ens in range(Nens)
                ]

            # Compute all tasks in parallel and collect the results
            results = compute(*tasks)

            # Update the ensemble matrix
            ensemble = np.array(results).T

            return ensemble

        # Parallel forecast step using Ray
        elif re.match(r"\ARay\Z", self.parallel_flag, re.IGNORECASE):
            import ray

            nd, Nens = ensemble.shape

            # Initialize Ray
            ray.init(ignore_reinit_error=True)

            @ray.remote
            def ray_worker( ensemble_member, Q_err, parameters, model_kwargs):
                """
                Remote function to perform forecast step for a single ensemble member.
                This function will be executed in parallel by Ray workers.
                """
                return forecast_step_single(ens, ensemble_member, nd, Q_err, parameters, **model_kwargs)
            
            _, Nens = ensemble.shape

            # Launch tasks in parallel using Ray
            futures = [
                ray_worker.remote(ens, ensemble[:, ens], Q_err, self.parameters, **model_kwargs)
                for ens in range(Nens)
            ]

            # Collect results from all Ray tasks
            results = ray.get(futures)

            # Convert the list of results back to a numpy array and transpose
            ensemble = np.array(results).T
            return ensemble
        
        # Parallel forecast step using Multiprocessing
        elif re.match(r"\AMultiprocessing\Z", self.parallel_flag, re.IGNORECASE):
            import multiprocessing as mp

            nd, Nens = ensemble.shape

            # Define a helper function to handle arguments for each worker
            def worker(ens_idx):
                return forecast_step_single(ens, ensemble[:, ens_idx], nd, Q_err, self.parameters, **model_kwargs)

            # Create a pool of workers
            with mp.Pool(mp.cpu_count()) as pool:
                # Use pool.map to parallelize the forecast step
                results = pool.map(worker, range(Nens))

            # Convert the list of results back to a numpy array and transpose
            ensemble = np.array(results).T
            return ensemble
        elif re.match(r"\APyomp\Z", self.parallel_flag, re.IGNORECASE):
            # python openmp parallelization
            raise NotImplementedError("Pyomp parallelization is not yet implemented.")
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