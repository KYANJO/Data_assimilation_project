# Building the Firedrake IcePack Container

## Prerequisites
Before building or running the container, ensure all MPI and GCC modules are unloaded to prevent conflicts.

## Available Definitions
This repository includes two container definition files:

1. **[firedrake-icepack-docker.def](./firedrake-icepack-docker.def)**: Uses a pre-built Docker base image from Docker Hub for a faster and simpler build process.
2. **[firedrake-icepack-ubuntu.def](./firedrake-icepack-ubuntu.def)**: Builds the container from scratch using a minimal Ubuntu image, including the full installation of dependencies such as Firedrake and Ice-Pack. *(Note: This process may take a significant amount of time.)*

## Build Steps
1. Clone this repository if you haven’t already.
2. Grant execution permissions to the setup script:
   ```bash
   chmod 777 [set_tmp_cache_dir.sh](./set_tmp_cache_dir.sh)
   ```
3. Run the script to set up cache directories and configure the environment:
   ```bash
   ./[set_tmp_cache_dir.sh](./set_tmp_cache_dir.sh)
   ```
4. Build and interact with the container:
   - **Build the container**:
     ```bash
     apptainer build icepack.sif [firedrake-icepack-ubuntu.def](./firedrake-icepack-ubuntu.def)
     ```
     This generates the `.sif` container image.
   - **Start a shell inside the container**:
     ```bash
     apptainer shell icepack.sif
     ```
   - **Run a script using the container**:
     ```bash
     apptainer exec icepack.sif [test.py](./test.py)
     ```

Here’s the refined version for better clarity, formatting, and consistency:

---

## Running on SLURM

### **A Simple Test**
To verify the compatibility of the container with the slurm environment, test a simple [mpi_hello_world.c](./mpi_hello_world.c) code by following these steps:

```bash
module purge
apptainer exec icepack_working.sif mpicc mpi_hello_world.c -o mpi_hello
module load gcc/12 && module load mvapich2
srun --mpi=pmi2 -n 4 apptainer exec icepack_working.sif ./mpi_hello
```

#### **Expected Output**
The output should resemble the following:

```
Hello world! Processor atl1-1-02-002-23-2.pace.gatech.edu, Rank 0 of 4, CPU 0, NUMA node 0, Namespace mnt:[4026534439]
Hello world! Processor atl1-1-02-002-23-2.pace.gatech.edu, Rank 1 of 4, CPU 1, NUMA node 0, Namespace mnt:[4026534442]
Hello world! Processor atl1-1-02-002-23-2.pace.gatech.edu, Rank 2 of 4, CPU 2, NUMA node 0, Namespace mnt:[4026534440]
Hello world! Processor atl1-1-02-002-23-2.pace.gatech.edu, Rank 3 of 4, CPU 3, NUMA node 0, Namespace mnt:[4026534441]
```

---

### **Create a Job Script**
After building the container, create a SLURM job script (e.g., `job.sh`) with the following content:

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

module load gcc/12 && module load mvapich2

# Run the application using the container
srun --mpi=pmi2 apptainer exec icepack.sif python3 test.py
```

---

### **Submit the Job**
Submit your job to the SLURM scheduler using the following command:

```bash
sbatch run_script.sh
```

---
