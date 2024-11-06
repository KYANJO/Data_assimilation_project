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
def forecast_step(solver, ensemble, num_state_vars, Q_err, **kwags):
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
    A = kwags.get('A', None)
    C = kwags.get('C', None)
    Q = kwags.get('Q', None)
    V = kwags.get('V', None)

    nd, Nens = ensemble.shape
    ndim = nd // num_state_vars
    
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
        h, u = run_simualtion(solver, h, u, a, fluidity = A, friction = C)

        nos = np.random.multivariate_normal(np.zeros(nd), Q_err)

        # update the ensemble members with the new state variables and noise 
        ensemble[:ndim, ens]        = h.dat.data_ro      + nos[:ndim]
        ensemble[ndim:2*ndim, ens]  = u.dat.data_ro[:,0] + nos[ndim:2*ndim]
        ensemble[2*ndim:, ens]      = u.dat.data_ro[:,1] + nos[2*ndim:]
    return ensemble

# --- Background step ---
def background_step(k,solver,statevec_bg, hdim, a = None, A= None, C= None, Q=None, V=None):
    """ computes the background state of the model"""
    hb = Function(Q)
    ub = Function(V)
    hb.dat.data[:]   = statevec_bg[:hdim,k]
    ub.dat.data[:,0] = statevec_bg[hdim:2*hdim,k]
    ub.dat.data[:,1] = statevec_bg[2*hdim:,k]
    hb, ub = run_simualtion(solver, hb, ub, a, fluidity = A, friction = C)
    statevec_bg[:hdim,k+1] = hb.dat.data_ro
    statevec_bg[hdim:2*hdim,k+1] = ub.dat.data_ro[:,0]
    statevec_bg[2*hdim:,k+1] = ub.dat.data_ro[:,1]
    return statevec_bg


# --- Run the model with the Ensemble Kalman Filter ---
# def run_model_with_filter(solver, filter_type,*args, a = None, A= None, C= None,Q=None,V=None):


