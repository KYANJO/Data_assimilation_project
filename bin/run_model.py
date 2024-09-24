# bin/run_model.py
# @author: Brian Kyanjo
# @date: 2024-09-24
# @description: This script runs the model.

# import libraries
import numpy as np
import os
import sys

# Add the root directory (parent directory of 'bin') to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from config
from config.config_loader import ParamsLoader

# Load the parameters from the YAML file into a dictionary
params_loader = ParamsLoader("config/params.yaml")
params = params_loader.get_params()

# Now `params` is a dictionary and you can access it as you did before
print("A:", params["A"])
print("uscale:", params["uscale"])
print("Grid sigma values:", params["grid"]["sigma"])
