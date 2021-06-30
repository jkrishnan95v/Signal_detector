#! /bin/bash

#SBATCH --ntasks=8
#SBATCH --time=48:00:00
#SBATCH --partition=singleGPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,FALI,END
#SBATCH --mail-user=jayakrishnan@unm.edu
#SBATCH --job-name=Sig_est

module load anaconda3

source  activate pytorch
cd      $SLURM_SUBMIT_DIR


 
cd      ../Lab1



echo ' TRAINING CNN ON ARRAY COVS '
python main.py "configs/onedcnn_exp_0.json" 



