# ==============================================================================
# @des: This file contains helper functions that are used in the main script.
# @date: 2024-10-4
# @author: Brian Kyanjo
# ==============================================================================

import os
import sys
import subprocess


# Function to safely change directory
def safe_chdir(main_directory,target_directory):
    # Get the absolute path of the target directory
    target_path = os.path.abspath(target_directory)

    # Check if the target path starts with the main directory path
    if target_path.startswith(main_directory):
        os.chdir(target_directory)
    # else:
    #     print(f"Error: Attempted to leave the main directory '{main_directory}'.")


def install_requirements():
    """
    Install dependencies listed in the requirements.txt file if not already installed.
    """
    try:
        # Run the command to install the requirements from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "../requirements.txt"])
    except subprocess.CalledProcessError as e:
        # Print the error and raise a more meaningful exception
        print(f"Error occurred while installing dependencies: {e}")
        raise RuntimeError("Failed to install dependencies from requirements.txt. Please check the file and try again.")