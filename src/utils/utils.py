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

class UtilsFunctions:
    def __init__(self, params):
        """
        Initialize the utility functions with model parameters.
        
        Parameters:
        params (dict): Model parameters, including those used for bed topography.
        """
        self.params = params

    def Obs(self, huxg_virtual_obs, m_obs):
        """
        Observation operator that reduces the full observation vector into a smaller subset.
        
        Parameters:
        huxg_virtual_obs (numpy array): The virtual observation vector (n-dimensional).
        m_obs (int): The number of observations.
        
        Returns:
        numpy array: The reduced observation vector.
        """
        n = huxg_virtual_obs.shape[0]

        # Initialize the H matrix
        H = np.zeros((m_obs * 2 + 1, n))

        # Calculate distance between measurements
        di = int((n - 2) / (2 * m_obs))

        # Fill the H matrix
        for i in range(m_obs):
            H[i, i * di] = 1
            H[m_obs + i, int((n - 2) / 2) + i * di] = 1

        H[m_obs * 2, n - 2] = 1  # Final element

        # Perform matrix multiplication
        z = H @ huxg_virtual_obs
        return z

    def JObs(self, n_model, m_obs):
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
        H = np.zeros((m_obs * 2 + 1, n))

        # Calculate distance between measurements
        di = int((n - 2) / (2 * m_obs))

        # Fill the H matrix
        for i in range(m_obs):
            H[i, i * di] = 1
            H[m_obs + i, int((n - 2) / 2) + i * di] = 1

        H[m_obs * 2, n - 2] = 1  # Final element

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

