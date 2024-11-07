# ==============================================================================
# @des: This file contains run functions for icepack data assimilation.
#       - contains different options of the EnKF data assimilation schemes.
# @date: 2024-11-4
# @author: Brian Kyanjo
# ==============================================================================

import sys
import os
import numpy as np
from scipy.stats import multivariate_normal,norm

import firedrake
import icepack
import tqdm

from firedrake import *

# add the path to the utils.py file
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# --- run functions ---
def run_simualtion(solver, h, u, a, b, dt, h0, **kwargs):
    # for i in tqdm.trange(num_timesteps):
    h = solver.prognostic_solve(
        dt = dt,
        thickness = h,
        velocity = u,
        accumulation = a,
        thickness_inflow = h0,
    )

    s = icepack.compute_surface(thickness = h, bed = b)

    u = solver.diagnostic_solve(
        velocity = u,
        thickness = h,
        surface = s,
        **kwargs
    )

    return h, u

# --- Forecast step ---
def forecast_step(solver, ensemble, Q_err,params, **kwags):
    """ensemble: packs the state variables:h,u,v of the ensemble members
                 where h is thickness, u and v are the x and y components 
                 of the velocity field
        dt: time step
        Nens: number of ensemble members
        ndim: number of dimensions of the state variables
        a: accumulation rate
        args: other arguments: b, ela, max_a, da_ds

    Returns: ensemble: updated ensemble members
             a: updated accumulation rate
    """

    # unpack the **kwargs
    a = kwags.get('a', None)
    b = kwags.get('b', None)
    dt = kwags.get('dt', None)
    h0 = kwags.get('h0', None)
    A = kwags.get('A', None)
    C = kwags.get('C', None)
    Q = kwags.get('Q', None)
    V = kwags.get('V', None)

    nd, Nens = ensemble.shape
    ndim = nd // params["num_state_vars"]
    
    # iterate over the ensemble members
    for ens in range(Nens):
        h_vec = ensemble[:ndim, ens]
        u_vec = ensemble[ndim:2*ndim, ens]
        v_vec = ensemble[2*ndim:, ens]

        # create firedrake functions from the ensemble members
        h = Function(Q)
        h.dat.data[:] = h_vec[:]

        u = Function(V)
        u.dat.data[:,0] = u_vec[:]
        u.dat.data[:,1] = v_vec[:]

        # call the ice stream model to update the state variables
        h, u = run_simualtion(solver, h, u, a, b, dt, h0, fluidity = A, friction = C)

        nos = np.random.multivariate_normal(np.zeros(nd), Q_err)

        # update the ensemble members with the new state variables and noise 
        ensemble[:ndim, ens]        = h.dat.data_ro      + nos[:ndim]
        ensemble[ndim:2*ndim, ens]  = u.dat.data_ro[:,0] + nos[ndim:2*ndim]
        ensemble[2*ndim:, ens]      = u.dat.data_ro[:,1] + nos[2*ndim:]
    return ensemble

# --- Background step ---
def background_step(k,solver,statevec_bg, hdim, **kwags):
    """ computes the background state of the model"""
    # unpack the **kwargs
    a = kwags.get('a', None)
    b = kwags.get('b', None)
    A = kwags.get('A', None)
    C = kwags.get('C', None)
    Q = kwags.get('Q', None)
    V = kwags.get('V', None)
    dt = kwags.get('dt', None)
    h0 = kwags.get('h0', None)

    hb = Function(Q)
    ub = Function(V)
    hb.dat.data[:]   = statevec_bg[:hdim,k]
    ub.dat.data[:,0] = statevec_bg[hdim:2*hdim,k]
    ub.dat.data[:,1] = statevec_bg[2*hdim:,k]
    hb, ub = run_simualtion(solver, hb, ub, a, b, dt, h0, fluidity = A, friction = C)
    statevec_bg[:hdim,k+1] = hb.dat.data_ro
    statevec_bg[hdim:2*hdim,k+1] = ub.dat.data_ro[:,0]
    statevec_bg[2*hdim:,k+1] = ub.dat.data_ro[:,1]
    return statevec_bg


# --- initialize the ensemble members ---
def initialize_ensemble(statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full, Cov_model, params,**kwags):
    """initialize the ensemble members"""
    nd, N = statevec_ens.shape
    hdim = nd // params["num_state_vars"]

    # unpack the **kwargs
    h0 = kwags.get('h0', None)
    u0 = kwags.get('u0', None)

    #statevec_bg[:hdim,0] = h0.dat.data_ro
    statevec_bg[hdim:2*hdim,0] = u0.dat.data_ro[:,0]
    statevec_bg[2*hdim:,0] = u0.dat.data_ro[:,1]

    # statevec_ens_mean[:,0] = h0.dat.data_ro
    statevec_ens_mean[:hdim,0] = h0.dat.data_ro
    statevec_ens_mean[hdim:2*hdim,0] = u0.dat.data_ro[:,0]
    statevec_ens_mean[2*hdim:,0] = u0.dat.data_ro[:,1]

    # Intialize ensemble thickness and velocity
    # h_ens = np.array([h0.dat.data_ro for _ in range(N)])
    # u_ens = np.array([u0.dat.data_ro[:,0] for _ in range(N)])
    # v_ens = np.array([u0.dat.data_ro[:,1] for _ in range(N)])

    for i in range(N):
        perturbed_state = multivariate_normal.rvs(mean=np.zeros(nd-1), cov=Cov_model[:-1,:-1])
        
        statevec_ens[:hdim-1,i]         = h0.dat.data_ro[:-1] + perturbed_state[:hdim-1]
        statevec_ens[hdim:2*hdim-1,i]   = u0.dat.data_ro[:-1,0] + perturbed_state[hdim:2*hdim-1]
        statevec_ens[2*hdim:nd-1,i]     = u0.dat.data_ro[:-1,1] + perturbed_state[2*hdim:nd-1]

        statevec_ens[hdim-1,i]   = h0.dat.data_ro[-1]
        statevec_ens[2*hdim-1,i] = u0.dat.data_ro[-1,0]
        statevec_ens[-1,i]       = u0.dat.data_ro[-1,1]

    statevec_ens_full[:,:,0] = statevec_ens

    return statevec_bg, statevec_ens, statevec_ens_mean, statevec_ens_full


# --- generate true state ---
def generate_true_state(solver, statevec_true,params, **kwags):
    """generate the true state of the model"""
    nd, nt = statevec_true.shape
    nt = nt - 1
    hdim = nd // params["num_state_vars"]

    # unpack the **kwargs
    a = kwags.get('a', None)
    b = kwags.get('b', None)
    dt = kwags.get('dt', None)
    A = kwags.get('A', None)
    C = kwags.get('C', None)
    Q = kwags.get('Q', None)
    V = kwags.get('V', None)
    h0 = kwags.get('h0', None)
    u0 = kwags.get('u0', None)

    statevec_true[:hdim,0]       = h0.dat.data_ro
    statevec_true[hdim:2*hdim,0] = u0.dat.data_ro[:,0]
    statevec_true[2*hdim:,0]     = u0.dat.data_ro[:,1]

    h = h0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)
    for k in tqdm.trange(nt):
        # call the ice stream model to update the state variables
        h, u = run_simualtion(solver, h, u, a, b, dt, h0, fluidity = A, friction = C)

        statevec_true[:hdim,k+1]        = h.dat.data_ro
        statevec_true[hdim:2*hdim,k+1]  = u.dat.data_ro[:,0]
        statevec_true[2*hdim:,k+1]      = u.dat.data_ro[:,1]

    return statevec_true
