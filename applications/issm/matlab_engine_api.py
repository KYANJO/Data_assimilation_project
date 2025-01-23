# ==============================================================
# @Author: Brian Kyanjo
# @Date: 2025-01-21
# @Description: Call matlab function or code from python
#               requires matlab to be installed on the machine
# ==============================================================

# --- Imports ---
import matlab.engine

# Start the MATLAB engine
eng = matlab.engine.start_matlab()


# call the matlab function
def call_matlab_function():
    """des: Call the matlab function from python"""
    # Call the MATLAB

    # Define the MATLAB function
    eng.eval("disp('Hello, MATLAB!')", nargout=0)

# Quit the MATLAB engine
eng.quit()