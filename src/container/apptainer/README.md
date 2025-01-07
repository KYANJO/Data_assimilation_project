```markdown
# Building the Firedrake Ice-Pack Container

## Prerequisites
Before proceeding, ensure all MPI and GCC modules are unloaded to avoid conflicts during the container build and runtime.

## Available Definitions
This repository provides two container definition files:
1. **`firedrake-icepack-docker.def`**: Uses a pre-built Docker base image from Docker Hub for a faster and simpler build.
2. **`firedrake-icepack-ubuntu.def`**: Builds the container from scratch starting with a minimal Ubuntu image. This includes reinstalling all dependencies, including Firedrake and Ice-Pack. (taake some time to build)

## Build Steps
1. Clone this repository (if not already done).
2. Grant execution permissions to the setup script:
   ```bash
   chmod 777 set_tmp_cache_dir.sh
   ```
3. Run the script to configure cache directories and environment settings:
   ```bash
   ./set_tmp_cache_dir.sh
   ```
4. Build and interact with the container:
   - **To build the container**:
     ```bash
     apptainer build icepack.sif `firedrake-icepack-docker.def`
     ```
     This generates the `.sif` image file.
   - **To start a shell inside the container**:
     ```bash
     apptainer shell icepack.sif
     ```
   - **To execute a script using the container**:
     ```bash
     apptainer exec icepack.sif script.py
     ```

Follow these steps to successfully set up and use the Firedrake Ice-Pack container.
``` 

