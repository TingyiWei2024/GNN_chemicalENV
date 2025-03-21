#!/bin/bash
#SBATCH --job-name=gnn_mix_lc          
#SBATCH --account=def-nrpop     
#SBATCH --gres=gpu:4              
#SBATCH --cpus-per-task=4          
#SBATCH --mem=32G                  
#SBATCH --time=0-24:00              
#SBATCH --output=gnn_mix_lc.out       
#SBATCH --error=gnn_mix_lc.err        

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn_env

# Run the Python scripts sequentially:
python gnn_mix_lc01_bde.py
python gnn_mix_lc01_engergy.py
python gnn_mix_lc01_Cbnum.py
