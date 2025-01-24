#!/bin/bash

#Name of the job
#SBATCH --job-name=precompute_sttc_arrays

#sciCORE account running the job
#SBATCH --account=donafl00

#Number of CPU cores reserved
#SBATCH --cpus-per-task=128

#Memory reserved per core.
#SBATCH --mem=448G

#Time during which the task will run
#SBATCH --time=00:30:00
#SBATCH --qos=30min

#Paths to STDOUT or STDERR files should be absolute or relative to the current
#working directory
#SBATCH --output=logs/precompute_sttc_arrays.o
#SBATCH --error=logs/precompute_sttc_arrays.e

#Notify via email when the task ends or fails
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=rodrigo.cerqueiragonzalezpena@unibas.ch

#This job runs from the current working directory

#Load the required modules
##########################
eval "$(conda shell.bash hook)"
conda activate ngmd

#Export the required environment variables
##########################################
n_sessions_in_parallel=128
# Set up other variables in ../scripts/precompute_sttc_arrays_config.py

#Run the desired commands
#########################
python scripts/precompute_sttc_arrays.py --data_dir $NGMD_DATA --n_sessions_in_parallel $n_sessions_in_parallel
