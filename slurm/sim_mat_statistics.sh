#!/bin/bash

#Name of the job
#SBATCH --job-name=sim_mat_statistics

#sciCORE account running the job
#SBATCH --account=donafl00

#Number of CPU cores reserved
#SBATCH --cpus-per-task=1

#Memory reserved per core.
#SBATCH --mem=32G

#Time during which the task will run
#SBATCH --time=00:30:00
#SBATCH --qos=30min

#Create array job to execute the same script with different parameters
#SBATCH --array=1-2

#Paths to STDOUT or STDERR files should be absolute or relative to the current
#working directory
#SBATCH --output=logs/sim_mat_statistics.o
#SBATCH --error=logs/sim_mat_statistics.e

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
array_job_cmds=slurm/sim_mat_statistics_cmds.sh

#Run the desired commands
#########################
eval $(sed -n ${SLURM_ARRAY_TASK_ID}p $array_job_cmds)
