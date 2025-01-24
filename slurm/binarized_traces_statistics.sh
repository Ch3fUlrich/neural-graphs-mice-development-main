#!/bin/bash

#Name of the job
#SBATCH --job-name=binarized_traces_statistics

#sciCORE account running the job
#SBATCH --account=donafl00

#Number of CPU cores reserved
#SBATCH --cpus-per-task=1

#Memory reserved per core.
#SBATCH --mem=32G

#Time during which the task will run
#SBATCH --time=00:30:00
#SBATCH --qos=30min

#Paths to STDOUT or STDERR files should be absolute or relative to the current
#working directory
#SBATCH --output=logs/binarized_traces_statistics.o
#SBATCH --error=logs/binarized_traces_statistics.e

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
data_dir=$NGMD_DATA
save_dir=$NGMD_DATA/analysis_files

#Run the desired commands
#########################
python scripts/binarized_traces_statistics.py --data_dir $data_dir --save_dir $save_dir
