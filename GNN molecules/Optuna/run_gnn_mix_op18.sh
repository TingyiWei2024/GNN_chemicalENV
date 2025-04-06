#!/bin/bash
#SBATCH --job-name=gnn_mix_lc          
#SBATCH --account=def-nrpop     
#SBATCH --partition=gpubase_bygpu_b6   # 选择支持单节点至少有4个GPU的分区
#SBATCH --cpus-per-task=4          
#SBATCH --mem=32G                  
#SBATCH --time=7-00:00:00             
#SBATCH --output=gnn_mix_op18.out       
#SBATCH --error=gnn_mix_op18.err        

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn_env

# 启动4个并行进程，每个进程在单个节点上分配4个GPU：
srun --nodes=1 --gres=gpu:4 python gnn_mix_op17.py & 
srun --nodes=1 --gres=gpu:4 python gnn_mix_op17.py & 
srun --nodes=1 --gres=gpu:4 python gnn_mix_op17.py & 
srun --nodes=1 --gres=gpu:4 python gnn_mix_op17.py & 
wait
