#!/usr/bin/env python3
"""
Enhanced VAE training script adapted for ZooDataset
"""

import argparse
import os
import sys
import datetime
import torch
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from utils.util import instantiate_from_config


# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="VAE Training with ZooDataset")
    
    # Data arguments
    parser.add_argument("--data_root", default="model_data", type=str, 
                       help="Root directory containing the weights folder")
    parser.add_argument("--dataset", default="joint", type=str, 
                       help="Dataset choice among [mnist, svhn, cifar10, stl10, joint]")
    parser.add_argument("--topk", default=None, type=int, 
                       help="Number of top samples per dataset")
    parser.add_argument("--scale", default=1.0, type=float, 
                       help="Scale factor for weight values")
    parser.add_argument("--normalize", default=False, type=str2bool, 
                       help="Whether to normalize weights")
    
    # Training arguments
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of data loader workers")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--max_epochs", default=1000, type=int, help="Maximum epochs")
    parser.add_argument("--min_epochs", default=50, type=int, help="Minimum epochs")
    
    # Model arguments
    parser.add_argument("--embed_dim", default=4, type=int, help="Embedding dimension")
    parser.add_argument("--z_channels", default=4, type=int, help="Latent channels")
    parser.add_argument("--kl_weight", default=1e-6, type=float, help="KL divergence weight")
    
    # Checkpointing and logging
    parser.add_argument("--save_path", default="vae_checkpoints", type=str, 
                       help="Checkpoint save directory")
    parser.add_argument("--log_dir", default="vae_logs", type=str, 
                       help="Logging directory")
    parser.add_argument("--resume", default="", type=str, 
                       help="Resume from checkpoint path")
    parser.add_argument("--name", default="vae_zoo", type=str, 
                       help="Experiment name")
    
    # Hardware
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--precision", default=32, type=int, 
                       help="Training precision (16 or 32)")
    
    # Config
    parser.add_argument("--config", default="weights_encoding/configs/base_config.yaml", 
                       type=str, help="Path to base config file")
    
    return parser


def create_dynamic_config(args):
    """Create configuration based on arguments"""
    config = OmegaConf.create({
        'model': {
            'base_learning_rate': args.learning_rate,
            'target': 'weights_encoding.vae.VAENoDiscModel',
            'params': {
                'embed_dim': args.embed_dim,
                'input_key': 'weight',
                'learning_rate': args.learning_rate,
                'lossconfig': {
                    'target': 'weights_encoding.losses.custom_losses.MyLoss',
                    'params': {
                        'kl_weight': args.kl_weight
                    }
                },
                'ddconfig': {
                    'double_z': True,
                    'z_channels': args.z_channels,
                    'resolution': 64,
                    'in_channels': 1,
                    'my_channels': 1,
                    'out_ch': 1,
                    'ch': 128,
                    'ch_mult': [1, 1, 2],
                    'num_res_blocks': 2,
                    'attn_resolutions': [2, 4],
                    'dropout': 0.0,
                    'in_dim': 2864,
                    'fdim': 4096
                }
            }
        },
        'data': {
            'target': 'zooloaders.autoloader.ZooDataModule',
            'params': {
                'data_dir': args.data_root,
                'data_root': os.path.join(args.data_root, 'data'),
                'batch_size': args.batch_size,
                'num_workers': args.num_workers,
                'scale': args.scale,
                'dataset': args.dataset,
                'topk': args.topk,
                'normalize': args.normalize,
                'num_sample': 5
            }
        }
    })
    
    return config


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Create timestamp for this run
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{args.name}_{now}"
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    print("=== VAE Training with ZooDataset ===")
    print(f"Run name: {run_name}")
    print(f"Data root: {args.data_root}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.max_epochs}")
    print()
    
    # Create configuration
    if os.path.exists(args.config):
        print(f"Loading config from: {args.config}")
        config = OmegaConf.load(args.config)
        # Update with command line arguments
        config.data.params.data_dir = args.data_root
        config.data.params.batch_size = args.batch_size
        config.data.params.num_workers = args.num_workers
        config.data.params.dataset = args.dataset
        config.data.params.scale = args.scale
        config.data.params.topk = args.topk
        config.data.params.normalize = args.normalize
        config.model.params.learning_rate = args.learning_rate
        config.model.params.embed_dim = args.embed_dim
        config.model.params.lossconfig.params.kl_weight = args.kl_weight
    else:
        print("Creating dynamic configuration from arguments")
        config = create_dynamic_config(args)
    
    # Instantiate model and data
    print("Creating model and data module...")
    model = instantiate_from_config(config.model)
    datamodule = instantiate_from_config(config.data)
    
    # Setup data
    print("Setting up data...")
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    print("Data Summary:")
    print(f"  Train samples: {len(datamodule.trainset)}")
    print(f"  Val samples: {len(datamodule.valset)}")
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="train/aeloss",
        dirpath=args.save_path,
        filename=f"{run_name}_{{epoch:03d}}",
        every_n_epochs=10,
        save_top_k=3,
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=run_name,
        version=""
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        precision=args.precision,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Resume from checkpoint if specified
    ckpt_path = None
    if args.resume and os.path.exists(args.resume):
        ckpt_path = args.resume
        print(f"Resuming from checkpoint: {ckpt_path}")
    
    # Train
    print("Starting training...")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    
    print("Training completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()