# =============================================================================
# @author: Brian Kyanjo
# @date: 2025-01-17
# @description: Add or remove models to be supported by the application here
#              Currently supported models include: - icepack
#                                                  - Lorenz96
#                                                  - flowline (integration still underway)
#                                                  - ISSM    (development still underway)
# =============================================================================

# --- Imports ---
import sys
import os
import re
import numpy as np

# --- Configuration ---
sys.path.insert(0, '../config')
from _utility_imports import *

#  add the application directory to the path
application_dir = os.path.join(project_root, 'applications')
sys.path.insert(0, application_dir)


class SupportedModels:
    """ Class to call the supported
        models in the application
    """
    def __init__(self, model=None):
        self.model = model

        def call_models(self):
            if re.match(r"\Aicepack\Z", self.model, re.IGNORECASE):
                import icepack
                import firedrake
                from icepack.icepack_enkf import background_step, forecast_step_single
            elif re.match(r"\Alorenz96\Z",  self.model, re.IGNORECASE):
                from lorenz96.lorenz96_enkf import background_step, forecast_step_single
            elif re.match(r"\Aflowline\Z",  self.model, re.IGNORECASE):
                # import jax.numpy as jnp
                from flowline.flowline_enkf import background_step, forecast_step_single
                raise ValueError("model development still underway")
            elif re.match(r"\Aissm\Z",  self.model, re.IGNORECASE):
                from issm.issm_enkf import background_step, forecast_step_single
                raise ValueError("model development still underway")
            else:
                raise ValueError("Other models are not yet implemented")
            
            
