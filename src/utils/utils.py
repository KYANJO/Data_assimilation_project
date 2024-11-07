# =============================================================================
# @Author: Brian Kyanjo
# @Date: 2024-09-24
# @Description: This script includes the observation operator and its Jacobian 
#               for the EnKF data assimilation scheme. It also includes the bed
#               topography function used in the model.
# =============================================================================

# import libraries
import numpy as np
import jax.numpy as jnp
from collections.abc import Iterable
from scipy.stats import norm

# --- helper functions ---
def isiterable(obj):
    return isinstance(obj, Iterable)

class UtilsFunctions:
    def __init__(self, params,ensemble=None):
        """
        Initialize the utility functions with model parameters.
        
        Parameters:
        params (dict): Model parameters, including those used for bed topography.
        """
        self.params   = params
        self.ensemble = ensemble

    def Obs_fun(self, virtual_obs):
        """
        Observation operator that reduces the full observation vector into a smaller subset.
        
        Parameters:
        huxg_virtual_obs (numpy array): The virtual observation vector (n-dimensional).
        m_obs (int): The number of observations.
        
        Returns:
        numpy array: The reduced observation vector.
        """
        n = virtual_obs.shape[0]

        # Initialize the H matrix
        H = np.zeros((self.params["m_obs"] * 2 + 1, n))

        # Calculate distance between measurements
        di = int((n - 2) / (2 * self.params["m_obs"]))

        # Fill the H matrix
        for i in range(self.params["m_obs"]):
            H[i, i * di] = 1
            H[self.params["m_obs"] + i, int((n - 2) / 2) + i * di] = 1

        H[self.params["m_obs"] * 2, n - 2] = 1  # Final element

        # Perform matrix multiplication
        z = H @ virtual_obs
        return z

    def JObs_fun(self, n_model):
        """
        Jacobian of the observation operator.
        
        Parameters:
        n_model (int): The size of the model state vector.
        m_obs (int): The number of observations.
        
        Returns:
        numpy array: The Jacobian matrix of the observation operator.
        """
        n = n_model

        # Initialize the H matrix
        H = np.zeros((self.params["m_obs"] * 2 + 1, n))

        # Calculate distance between measurements
        di = int((n - 2) / (2 * self.params["m_obs"]))

        # Fill the H matrix
        for i in range(self.params["m_obs"]):
            H[i, i * di] = 1
            H[self.params["m_obs"] + i, int((n - 2) / 2) + i * di] = 1

        H[self.params["m_obs"] * 2, n - 2] = 1  # Final element

        return H
    
    def bed(self, x):
        """
        Bed topography function, which computes the bed shape based on input x and model parameters.
        
        Parameters:
        x (jax.numpy array): Input spatial grid points.
        
        Returns:
        jax.numpy array: The bed topography values at each x location.
        """
        # Ensure parameters are floats
        params     = self.params
        sillamp    = float(params['sillamp'])
        sillsmooth = float(params['sillsmooth'])
        xsill      = float(params['xsill'])

        # Compute the bed topography
        b = sillamp * (-2 * jnp.arccos((1 - sillsmooth) * jnp.sin(jnp.pi * x / (2 * xsill))) / jnp.pi - 1)
        return b
    
    def inflate_ensemble(self,in_place=True):
        """
        Inflate ensemble members by a given factor.
        
        Args: 
            ensemble: ndarray (n x N) - The ensemble matrix of model states (n is state size, N is ensemble size).
            inflation_factor: float - scalar or iterable length equal to model states
            in_place: bool - whether to update the ensemble in place
        Returns:
            inflated_ensemble: ndarray (n x N) - The inflated ensemble.
        """
        # check if the inflation factor is scalar
        if np.isscalar(self.params['inflation_factor']):
            _scalar = True
            _inflation_factor = float(self.params['inflation_factor'])
        elif isiterable(self.params['inflation_factor']):
            if len(self.params['inflation_factor']) == self.ensemble.shape[0]:
                _inflation_factor[:] = self.params['inflation_factor'][:]
                _scalar = False
            else:
                raise ValueError("Inflation factor length must be equal to the state size")
        
        # check if we need inflation
        if _scalar:
            if _inflation_factor == 1.0:
                return self.ensemble
            elif _inflation_factor < 0.0:
                raise ValueError("Inflation factor must be positive scalar")
        else:
            _inf = False
            for i in _inflation_factor:
                if i>1.0:
                    _inf = True
                    break
            if not _inf:
                return self.ensemble
        
        ens_size = self.ensemble.shape[1]
        mean_vec = np.mean(self.ensemble, axis=1)
        if in_place:
            inflated_ensemble = self.ensemble
            for ens_idx in range(ens_size):
                state = inflated_ensemble[:, ens_idx]
                if _scalar:
                    # state = (state.axpy(-1.0, mean_vec)).scale(_inflation_factor)
                    state = (state - mean_vec) * _inflation_factor
                else:
                    # state = (state.axpy(-1.0, mean_vec)).multiply(_inflation_factor)
                    state = (state - mean_vec) * _inflation_factor

                # inflated_ensemble[:, ens_idx] = state.add(mean_vec)
                inflated_ensemble[:, ens_idx] = state + mean_vec
        else:
            inflated_ensemble = np.zeros(self.ensemble.shape)
            for ens_idx in range(ens_size):
                state = self.ensemble[:, ens_idx].copy()
                if _scalar:
                    # state = (state.axpy(-1.0, mean_vec)).scale(_inflation_factor)
                    state = (state - mean_vec) * _inflation_factor
                else:
                    # state = (state.axpy(-1.0, mean_vec)).multiply(_inflation_factor)
                    state = (state - mean_vec) * _inflation_factor

                # inflated_ensemble[:, ens_idx] = state.add(mean_vec)
                inflated_ensemble[:, ens_idx] = state + mean_vec
        
        return inflated_ensemble

    def rmse(self,truth, estimate):
        """
        Calculate the Root Mean Squared Error (RMSE) between the true and estimated values.
        
        Parameters:
        truth (numpy array): The true values.
        estimate (numpy array): The estimated values.
        
        Returns:
        float: The RMSE value.
        """
        return np.sqrt(np.mean((truth - estimate) ** 2))

    # --- Create synthetic observations ---
    def _create_synthetic_observations(self,statevec_true):
        """create synthetic observations"""
        nd, nt = statevec_true.shape
        hdim = nd // self.params["num_state_vars"]

        # create synthetic observations
        hu_obs = np.zeros((nd,self.params["nt_m"]))
        ind_m = self.params["ind_m"]
        km = 0
        for step in range(nt):
            if (km<self.params["nt_m"]) and (step+1 == ind_m[km]):
                hu_obs[:,km] = statevec_true[:,step+1]
                km += 1

        obs_dist = norm(loc=0,scale=self.params["sig_obs"])
        hu_obs = hu_obs + obs_dist.rvs(size=hu_obs.shape)

        return hu_obs