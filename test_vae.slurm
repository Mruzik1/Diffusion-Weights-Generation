#!/bin/bash

#SBATCH --job-name=test_vae
#SBATCH --time=2:00:00
#SBATCH --output=test_vae.out
#SBATCH --error=test_vae.err
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd /home/mr752zh/Diffusion-Weights-Generation
source .venv/bin/activate
python test_vae_reconstruction.py