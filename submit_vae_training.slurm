#!/bin/bash

#SBATCH --job-name=vae_zoo_training
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/vae_training_%j.out
#SBATCH --error=logs/vae_training_%j.err

# Print job information
echo "Starting VAE training job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories
mkdir -p logs
mkdir -p vae_checkpoints
mkdir -p vae_logs

# Print system information
echo "=== System Information ==="
nvidia-smi
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo

# Run the training
echo "=== Starting VAE Training ==="
python train_vae_enhanced.py \
    --data_root model_data \
    --dataset joint \
    --batch_size 32 \
    --num_workers 4 \
    --learning_rate 1e-3 \
    --max_epochs 500 \
    --min_epochs 10 \
    --embed_dim 4 \
    --z_channels 4 \
    --kl_weight 1e-6 \
    --save_path vae_checkpoints \
    --log_dir vae_logs \
    --name vae_zoo_weights \
    --gpus 1 \
    --precision 32 \
    --config weights_encoding/configs/base_config.yaml

# Check training completion
exit_code=$?
echo
echo "=== Training Completed ==="
echo "Exit code: $exit_code"
echo "End time: $(date)"

# Show final checkpoint information
if [ -d "vae_checkpoints" ]; then
    echo "=== Checkpoint Information ==="
    ls -la vae_checkpoints/
fi

exit $exit_code