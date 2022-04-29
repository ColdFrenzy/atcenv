#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=atcenv-sweep
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate atc

python --version
export SDL_AUDIODRIVER='dsp'

#running the actual code
echo "Starting the process..."
wandb agent dizzibus/atc-sweep/9cwxjhf5

