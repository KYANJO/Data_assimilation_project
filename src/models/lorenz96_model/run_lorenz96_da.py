# ==============================================================================
# @des: This file contains run functions for icepack data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2025-01-13
# @author: Brian Kyanjo
# ==============================================================================

import sys
import os
import numpy as np
from scipy.stats import multivariate_normal,norm

# add the path to the utils.py file
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# --- run function for the lorenz96 model ---
def run_simulation(state, **kwargs):
    """des: Lorenz96 model function
        inputs: state - current state of the model
                **kwargs - additional arguments for the model
       outputs: f - the derivative of the state vector
    """
    # Unpack the arguments
    sigma = kwargs.get('sigma', None)
    beta  = kwargs.get('beta', None)
    rho   = kwargs.get('rho', None)

    x,y,z = state # Unpack the state vector
    f = np.zeros(3) # Create an empty vector to store the derivatives
    f[0] = sigma*(y-x)  
    f[1] = x*(rho-z)-y
    f[2] = x*y - beta*z
    return f

# --- 4th order Runge-Kutta integrator --- 
def RK4(rhs, state, **kwargs):
    """des: 4th order Runge-Kutta integrator
        inputs: rhs - function that computes the right-hand side of the ODE
                state - current state of the model
                dt - time step
                *args - additional arguments for the model
        outputs: state - updated state of the model after one time step
    """
    dt = kwargs.get('dt', None)
    k1 = rhs(state, **kwargs)
    k2 = rhs(state + 0.5*dt*k1, **kwargs)
    k3 = rhs(state + 0.5*dt*k2, **kwargs)
    k4 = rhs(state + dt*k3, **kwargs)
    return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)



# --- Forecast step for the Lorenz96 model ---
def forecast_step_single(ens=None, ensemble=None, nd=None, Q_err=None, params=None, **kwargs):
    """inputs: run_simulation - function that runs the model
                ensemble - current state of the model
                dt - time step
                *args - additional arguments for the model
         outputs: uai - updated state of the model after one time step
    """

    # Run the RK4 integrator to push the state forward in time
    ensemble = RK4(run_simulation, ensemble, **kwargs)

    # add noise to the state variables
    noise = multivariate_normal.rvs(mean=np.zeros(nd), cov=Q_err)

    # update the ensemble with the noise
    ndim = nd//params['num_state_vars']
    ensemble[:,ens] = ensemble[:,ens] + noise

    return ensemble[:,ens]

# --- Background step for the Lorenz96 model ---
def background_step(k=None,statevec_bg=None, hdim=None, **kwargs):
    """inputs: k - current time step
                run_simulation - function that runs the model
                state - current state of the model
                dt - time step
                *args - additional arguments for the model
        outputs: state - updated state of the model after one time step
    """
    # Run the RK4 integrator to push the state forward in time
    statevec_bg = RK4(run_simulation, statevec_bg, **kwargs)# Run the RK4 integrator to push the state forward in time
    return statevec_bg

# --- generate true state ---
def generate_true_state(statevec_true=None,params=None, **kwargs):
    """inputs: statevec_true - true state of the model
                params - parameters of the model
                *args - additional arguments for the model
        outputs: statevec_true - updated true state of the model after one time step
    """
    # Unpack the parameters
    nd = params['nd']
    nt = params['nt']
    dt = params['dt']
    num_state_vars = params['num_state_vars']
    u0True = kwargs.get('u0True', None)


    # Set the initial condition
    statevec_true[:, 0] = u0True

    # Run the model forward in time
    for k in range(nt):
        statevec_true[:, k + 1] = RK4(run_simulation, statevec_true[:, k], **kwargs)

    return statevec_true

# --- initialize the ensemble members ---
def initialize_ensemble(statevec_bg=None, statevec_ens=None, \
                        statevec_ens_mean=None, statevec_ens_full=None, params=None,**kwargs):
    """initialize the ensemble members"""
    nd, N = statevec_ens.shape
    hdim = nd // params["num_state_vars"]

    u0b = kwargs.get('u0b', None)

    for ens in range(N):
        statevec_ens[:, ens] = u0b + np.random.normal(0, params['sig_b'], [nd,])

    statevec_bg = u0b
    statevec_ens_mean = u0b
    statevec_ens_full[:,:,0] = statevec_ens
    return statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full