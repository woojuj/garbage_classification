#!/bin/bash
# ====================================
#SBATCH --job-name=final_result
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=final_output_assignment2.out
#SBATCH --mail-type=END
#SBATCH --mail-user=felipe.castanogonzal@ucalgary.ca
# ====================================

# Initialize Conda for this session
eval "$(/home/felipe.castanogonzal/software/miniconda3/bin/conda shell.bash hook)"

conda activate felipe_finalproject

python finalassignment2.py
