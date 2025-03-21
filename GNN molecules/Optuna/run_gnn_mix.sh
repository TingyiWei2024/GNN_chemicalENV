#!/bin/bash
#SBATCH --job-name=gnn_mix          
#SBATCH --account=def-nrpop     
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=4          
#SBATCH --mem=32G                  
#SBATCH --time=0-03:00             
#SBATCH --output=gnn_mix.out       
#SBATCH --error=gnn_mix.err        

# 1) 加载Python或CUDA模块（根据需要调整版本）
#module load python/3.9

# 2) 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn_env

# 3) 执行Notebook并生成带有执行结果的新Notebook
jupyter nbconvert --to notebook --execute gnn_mix_op13.ipynb --output gnn_mix_op13_executed.ipynb
