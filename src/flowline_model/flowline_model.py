# =============================================================================
# Implicit 1D flowline model (Jax version)
# @author: Brian Kyanjo
# @date: 2024-09-24
# @description: This script includes the flowline model using JAX, 
#               - Jacobian computation using JAX
#               - Implicit solver using JAX
# =============================================================================

# Import libraries ====
import jax
import numpy as np
from jax import jacfwd
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.stats import norm, multivariate_normal

jax.config.update("jax_enable_x64", True) # Set the precision in JAX to use float64
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root

class FlowlineModel:
    def __init__(self, params, grid, bedfun):
        self.params = params
        self.grid = grid
        self.bedfun = bedfun

    def flowline(self, varin, varin_old):
        # Unpack grid and parameters
        NX, N1, dt, ds, sigma, sigma_elem = (self.params["NX"], self.params["N1"], 
                                             self.params["dt"] / self.params["tscale"],
                                             self.grid["dsigma"], self.grid["sigma"], 
                                             self.grid["sigma_elem"])
        tcurrent, xscale, hscale, lambd, m, n = (self.params["tcurrent"], self.params["xscale"],
                                                 self.params["hscale"], self.params["lambda"],
                                                 self.params["m"], self.params["n"])
        a = self.params["accum"] / self.params["ascale"]
        eps = self.params["eps"]
        transient = self.params["transient"]
        
        # facemelt could be a scalar or an array
        if isinstance(self.params["facemelt"], (int, float)):
            mdot = self.params["facemelt"] / self.params["uscale"]
        else:
            mdot = self.params["facemelt"][tcurrent] / self.params["uscale"]

        # Unpack variables
        h = varin[:NX]
        u = varin[NX:2 * NX]
        xg = varin[2 * NX]

        h_old = varin_old[:NX]
        xg_old = varin_old[2 * NX]

        # Calculate bed
        hf = -self.bedfun(xg * xscale, self.params) / (hscale * (1 - lambd))
        hfm = -self.bedfun(xg * sigma_elem[-1] * xscale, self.params) / (hscale * (1 - lambd))
        b = -self.bedfun(xg * sigma * xscale, self.params) / hscale

        # Initialize the residual vector
        F = jnp.zeros(2 * NX + 1, dtype=jnp.float64)

        # Calculate thickness and velocity functions (reusing existing logic)
        F = self._calculate_thickness_functions(F, h, u, xg, h_old, xg_old, hf, ds, sigma_elem, a, transient, NX, N1, mdot, b)
        F = self._calculate_velocity_functions(F, h, u, xg, ds, eps, n, m, hf, b, NX, N1)

        # Grounding line function
        F = F.at[2 * NX].set(3 * h[NX-1] - h[NX-2] - 2 * hf)

        return F

    def _calculate_thickness_functions(self, F, h, u, xg, h_old, xg_old, hf, ds, sigma_elem, a, transient, NX, N1, mdot, b):
        # Calculate thickness functions (split for modularity)
        F = F.at[0].set(transient * (h[0] - h_old[0]) / self.params["dt"] + (2 * h[0] * u[0]) / (ds[0] * xg) - a)
        F = F.at[1].set(
            transient * (h[1] - h_old[1]) / self.params["dt"]
            - transient * sigma_elem[1] * (xg - xg_old) * (h[2] - h[0]) / (2 * self.params["dt"] * ds[1] * xg)
            + (h[1] * (u[1] + u[0])) / (2 * xg * ds[1]) - a
        )
        F = F.at[2:NX-1].set(
            transient * (h[2:NX-1] - h_old[2:NX-1]) / self.params["dt"]
            - transient * sigma_elem[2:NX-1] * (xg - xg_old) * (h[3:NX] - h[1:NX-2]) / (2 * self.params["dt"] * ds[2:NX-1] * xg)
            + (h[2:NX-1] * (u[2:NX-1] + u[1:NX-2]) - h[1:NX-2] * (u[1:NX-2] + u[0:NX-3])) / (2 * xg * ds[2:NX-1]) - a
        )
        F = F.at[N1-1].set(
            (1 + 0.5 * (1 + (ds[N1-1] / ds[N1-2]))) * h[N1-1]
            - 0.5 * (1 + (ds[N1-1] / ds[N1-2])) * h[N1-2]
            - h[N1]
        )
        F = F.at[NX-1].set(
            transient * (h[NX-1] - h_old[NX-1]) / self.params["dt"]
            - transient * sigma_elem[NX-1] * (xg - xg_old) * (h[NX-1] - h[NX-2]) / (self.params["dt"] * ds[NX-2] * xg)
            + (h[NX-1] * (u[NX-1] + mdot * hf / h[NX-1] + u[NX-2]) - h[NX-2] * (u[NX-2] + u[NX-3])) / (2 * xg * ds[NX-2])
            - a
        )
        return F

    def _calculate_velocity_functions(self, F, h, u, xg, ds, eps, n, m, hf, b, NX, N1):
        # Velocity functions
        F = F.at[NX].set(
            ((4 * eps / (xg * ds[0]) ** ((1 / n) + 1)) * (h[1] * (u[1] - u[0]) * abs(u[1] - u[0]) ** ((1 / n) - 1)
            - h[0] * (2 * u[0]) * abs(2 * u[0]) ** ((1 / n) - 1)))
            - u[0] * abs(u[0]) ** (m - 1)
            - 0.5 * (h[0] + h[1]) * (h[1] - b[1] - h[0] + b[0]) / (xg * ds[0])
        )
        F = F.at[NX+1:2*NX-1].set(
            (4 * eps / (xg * ds[1:NX-1]) ** ((1 / n) + 1))
            * (h[2:NX] * (u[2:NX] - u[1:NX-1]) * abs(u[2:NX] - u[1:NX-1]) ** ((1 / n) - 1)
               - h[1:NX-1] * (u[1:NX-1] - u[0:NX-2]) * abs(u[1:NX-1] - u[0:NX-2]) ** ((1 / n) - 1))
            - u[1:NX-1] * abs(u[1:NX-1]) ** (m - 1)
            - 0.5 * (h[1:NX-1] + h[2:NX]) * (h[2:NX] - b[2:NX] - h[1:NX-1] + b[1:NX-1]) / (xg * ds[1:NX-1])
        )
        F = F.at[NX + N1 - 1].set((u[N1] - u[N1 - 1]) / ds[N1 - 1] - (u[N1 - 1] - u[N1 - 2]) / ds[N1 - 2])
        F = F.at[2 * NX - 1].set(
            (1 / (xg * ds[NX-2]) ** (1 / n)) * (abs(u[NX-1] - u[NX-2]) ** ((1 / n) - 1)) * (u[NX-1] - u[NX-2])
            - self.params["lambda"] * hf / (8 * eps)
        )
        return F

    def jacobian(self, huxg_old, flowlinefun):
        """ Calculate the Jacobian of the flowline function using automatic differentiation. """
        def f(varin):
            return flowlinefun(varin, huxg_old)

        # Use JAX's automatic differentiation to calculate the Jacobian
        return jax.jacfwd(f)

    def run_flowline(self, varin, flowlinefun):
        nt = self.params["NT"]
        huxg_old = varin
        huxg_all = np.zeros((huxg_old.shape[0], nt))

        for i in range(nt):
            if not self.params["assim"]:
                self.params["tcurrent"] = i + 1  # Adjust for 1-based indexing
            
            # Jacobian calculation
            Jf = self.jacobian(huxg_old, flowlinefun)
            
            # Solve the system of nonlinear equations using a root-finding algorithm
            solve_result = root(
                lambda varin: flowlinefun(varin, huxg_old), 
                huxg_old, 
                jac=Jf, 
                method='hybr',
                options={'maxiter': 100}
            )
            
            # Update the solution
            huxg_old = solve_result.x
            
            # Store the result for this time step
            huxg_all[:, i] = huxg_old

            if not self.params["assim"]:
                print(f"Step {i + 1}\n")

        return huxg_all
