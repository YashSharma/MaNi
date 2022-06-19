#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -A GutIntelligenceLab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/ys5hd/EoE/CRC/finetune/model_log_tcia_tnbc.out
module load anaconda3
source activate pytorch_yash
python -u /scratch/ys5hd/EoE/CRC/finetune/orchestrator.py 0. 1. 'tcia' 'tnbc'