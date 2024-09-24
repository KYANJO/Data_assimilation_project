# src/project_name/config_loader.py
# @author: Brian Kyanjo
# @date: 2024-09-24
# @description: This file is used to load the parameters from the YAML file

import yaml
import numpy as np

class ParamsLoader:
    def __init__(self, config_path):
        # Load the YAML file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['parameters']
        
        # Initialize the parameters dictionary
        self.params = self.config.copy()
        
        # Ensure all necessary values are cast to proper types
        self._cast_parameters()
        
        # Compute derived parameters
        self._compute_derived_parameters()

    def _cast_parameters(self):
        # Cast specific parameters to ensure they are floats or integers
        self.params["A"] = float(self.params["A"])
        self.params["n"] = int(self.params["n"])
        self.params["C"] = float(self.params["C"])
        self.params["rho_i"] = float(self.params["rho_i"])
        self.params["rho_w"] = float(self.params["rho_w"])
        self.params["g"] = float(self.params["g"])
        self.params["accum"] = float(self.params["accum"]) / self.params["year"]  # Convert to per second
        self.params["facemelt"] = float(self.params["facemelt"]) / self.params["year"]  # Convert to per second

    def _compute_derived_parameters(self):
        # Compute dependent parameters
        self.params["m"] = 1 / self.params["n"]
        self.params["B"] = self.params["A"] ** (-1 / self.params["n"])
        
        # Compute scaling parameters using eval to evaluate formulas from YAML
        self.params["uscale"] = eval(self.config['uscale_formula'], {}, self.params)
        self.params["xscale"] = eval(self.config['xscale_formula'], {}, self.params)
        self.params["tscale"] = eval(self.config['tscale_formula'], {}, self.params)
        self.params["eps"] = eval(self.config['eps_formula'], {}, self.params)
        self.params["lambda"] = eval(self.config['lambda_formula'], {}, self.params)

        # Grid parameters
        self.params["NX"] = eval(self.config['NX_formula'], {}, self.params)
        self.params["dt"] = eval(self.config['dt_formula'], {}, self.params)
        
        # Generate sigma values for the grid
        sigma1 = np.linspace(self.params["sigGZ"] / (self.params["N1"] + 0.5), self.params["sigGZ"], self.params["N1"])
        sigma2 = np.linspace(self.params["sigGZ"], 1, self.params["N2"] + 1)
        sigma = np.concatenate((sigma1, sigma2[1:self.params["N2"] + 1]))

        # Create the grid dictionary within params
        self.params["grid"] = {
            "sigma": sigma,
            "sigma_elem": np.concatenate(([0], (sigma[:-1] + sigma[1:]) / 2)),
            "dsigma": np.diff(sigma)
        }

    def get_params(self):
        return self.params
