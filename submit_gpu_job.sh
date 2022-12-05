#!/bin/bash
#SBATCH --job-name=demoscript
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michaelmacisaac@ufl.edu
#SBATCH --output j_%j.out
#SBATCH --error j_%j.err
#SBATCH --account=spearot
#SBATCH --qos=spearot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --time=5-0:00:00

cd $SLURM_SUBMIT_DIR
pwd;
echo 'GPU Job started - '$(pwd)
date;
nvidia-smi

export PATH=/blue/subhash/michaelmacisaac/SiC/envs/deepmd/bin:$PATH
python demoscript.py
echo 'Done.'
date;
