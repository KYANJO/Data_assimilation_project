#!/bin/bash
#!/bin/bash
#SBATCH -JIcepack                   # Job name
#SBATCH --account=gts-arobel3-atlas     # charge account
#SBATCH -N1 --ntasks-per-node=4     # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G                  # Memory per core
#SBATCH -t15                              # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                         # QOS Name
#SBATCH -oReport-%j.out                   # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail preferences
#SBATCH --mail-user=bkyanjo3@gatech.edu  # E-mail address for notifications


# Load module dependencies 
module load gcc
module load mvapich2

mpiexec -n 4 apptainer exec  icepack_working.sif python3 run_da_icepack.py