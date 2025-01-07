# Building the Firedrake IcePack Container

## Prerequisites
Before building or running the container, ensure all MPI and GCC modules are unloaded to prevent conflicts.

## Available Definitions
This repository includes two container definition files:

1. **`firedrake-icepack-docker.def`**: Uses a pre-built Docker base image from Docker Hub for a faster and simpler build process.
2. **`firedrake-icepack-ubuntu.def`**: Builds the container from scratch using a minimal Ubuntu image, including the full installation of dependencies such as Firedrake and Ice-Pack. *(Note: This process may take a significant amount of time.)*

## Build Steps
1. Clone this repository if you haven’t already.
2. Grant execution permissions to the setup script:
   ```bash
   chmod 777 set_tmp_cache_dir.sh
   ```
3. Run the script to set up cache directories and configure the environment:
   ```bash
   ./set_tmp_cache_dir.sh
   ```
4. Build and interact with the container:
   - **Build the container**:
     ```bash
     apptainer build icepack.sif firedrake-icepack-docker.def
     ```
     This generates the `.sif` container image.
   - **Start a shell inside the container**:
     ```bash
     apptainer shell icepack.sif
     ```
   - **Run a script using the container**:
     ```bash
     apptainer exec icepack.sif script.py
     ```

## Running on SLURM

### Create a Job Script
After building the container, create a SLURM job script (`job.sh`) similar to the example below:

```bash
#!/bin/bash
#SBATCH -J Icepack                    # Job name
#SBATCH --account=gts-gburdell3       # Charge account
#SBATCH -N 1                          # Number of nodes
#SBATCH --ntasks-per-node=4           # Tasks per node
#SBATCH --mem-per-cpu=1G              # Memory per CPU
#SBATCH -t 15                         # Job duration (e.g., 15 minutes)
#SBATCH -q inferno                    # QOS name
#SBATCH -o Report-%j.out              # Output file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL    # Email notifications
#SBATCH --mail-user=gburdell3@gatech.edu  # Email address for notifications

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Load required modules
module load gcc
module load mvapich2

# Run the application using the container
mpiexec -n 4 apptainer exec icepack.sif python3 test.py
```

### Job Submission
To submit the job, execute the following command:
```bash
sbatch job.sh
```
