#!/bin/bash
#SBATCH --job-name=gnn_mix_lc          
#SBATCH --account=def-nrpop     
#SBATCH --partition=gpubase_bygpu_b6   # 使用支持单节点至少4个GPU的分区
#SBATCH --cpus-per-task=4          
#SBATCH --mem=32G                  
#SBATCH --time=3-00:00:00             # 设置作业最长运行时间为3天
#SBATCH --output=gnn_mix_op17.out       
#SBATCH --error=gnn_mix_op17.err        

# 加载 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn_env

# 启动24个并行进程，每个进程在单个节点上分配4个GPU
for i in {1..24}; do
    srun --nodes=1 --gres=gpu:4 python gnn_mix_op17.py &
done
wait
