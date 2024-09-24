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

class ObservationOperator:
    def __init__(self, m_obs, params):
        self.m_obs = m_obs
        self.params = params

    def Obs(self, huxg_virtual_obs):
        """
        Observation operator that reduces the full observation vector into a smaller subset.
        
        Parameters:
        huxg_virtual_obs (numpy array): The virtual observation vector (n-dimensional).
        
        Returns:
        numpy array: The reduced observation vector.
        """
        n = huxg_virtual_obs.shape[0]
        m = self.m_obs

        # Initialize the H matrix
        H = np.zeros((m * 2 + 1, n))

        # Calculate distance between measurements
        di = int((n - 2) / (2 * m))

        # Fill the H matrix
        for i in range(m):
            H[i, i * di] = 1
            H[m + i, int((n - 2) / 2) + i * di] = 1

        H[m * 2, n - 2] = 1  # Final element

        # Perform matrix multiplication
        z = H @ huxg_virtual_obs
        return z

    def JObs(self, n_model):
        """
        Jacobian of the observation operator.
        
        Parameters:
        n_model (int): The size of the model state vector.
        
        Returns:
        numpy array: The Jacobian matrix of the observation operator.
        """
        n = n_model
        m = self.m_obs

        # Initialize the H matrix
        H = np.zeros((m * 2 + 1, n))

        # Calculate distance between measurements
        di = int((n - 2) / (2 * m))

        # Fill the H matrix
        for i in range(m):
            H[i, i * di] = 1
            H[m + i, int((n - 2) / 2) + i * di] = 1

        H[m * 2, n - 2] = 1  # Final element

        return H

    def bed(self, x):
        """
        Bed topography function, which computes the bed shape based on input x and model parameters.
        
        Parameters:
        x (jax.numpy array): Input spatial grid points.
        
        Returns:
        jax.numpy array: The bed topography values at each x location.
        """
        params = self.params
        sillamp = params['sillamp']
        sillsmooth = params['sillsmooth']
        xsill = params['xsill']

        # Compute the bed topography
        b = sillamp * (-2 * jnp.arccos((1 - sillsmooth) * jnp.sin(jnp.pi * x / (2 * xsill))) / jnp.pi - 1)
        return b
