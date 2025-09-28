#!/usr/bin/env python3
"""
Test script to validate VAE integration with ZooDataset
"""

import torch
import sys
import os
from omegaconf import OmegaConf

# Add current directory to path
sys.path.append(os.getcwd())

from zoodatasets.basedatasets import ZooDataset
from zooloaders.autoloader import ZooDataModule
from utils.util import instantiate_from_config

def test_zoodataset():
    """Test basic ZooDataset functionality"""
    print("Testing ZooDataset...")
    
    try:
        # Test if we can create the dataset
        dataset = ZooDataset(root="model_data", split="train")
        print(f"✓ ZooDataset created successfully")
        print(f"✓ Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample keys: {sample.keys()}")
            print(f"✓ Weight shape: {sample['weight'].shape}")
            print(f"✓ Model properties type: {type(sample['model_properties'])}")
            
            # Test weight reconstruction if layer_info is available
            if "layer_info" in sample["model_properties"]:
                from zoodatasets.basedatasets import reconstruct_weights
                flattened_weights = sample["weight"].squeeze(0)
                layer_info = sample["model_properties"]["layer_info"]
                
                reconstructed = reconstruct_weights(flattened_weights, layer_info)
                print(f"✓ Weight reconstruction successful: {len(reconstructed)} layers")
        
        return True
        
    except Exception as e:
        print(f"✗ ZooDataset test failed: {e}")
        return False

def test_datamodule():
    """Test ZooDataModule functionality"""
    print("\nTesting ZooDataModule...")
    
    try:
        # Load config and create data module
        config = OmegaConf.load("weights_encoding/configs/base_config.yaml")
        datamodule = instantiate_from_config(config.data)
        
        print(f"✓ ZooDataModule created successfully")
        
        # Test setup
        datamodule.setup("fit")
        print(f"✓ DataModule setup completed")
        print(f"✓ Train dataset length: {len(datamodule.trainset)}")
        print(f"✓ Val dataset length: {len(datamodule.valset)}")
        
        # Test dataloader
        train_loader = datamodule.train_dataloader()
        print(f"✓ Train dataloader created")
        
        # Test getting a batch
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded successfully")
        print(f"✓ Batch keys: {batch.keys()}")
        print(f"✓ Weight batch shape: {batch['weight'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ ZooDataModule test failed: {e}")
        return False

def test_vae_forward():
    """Test VAE forward pass with ZooDataset batch"""
    print("\nTesting VAE integration...")
    
    try:
        # Load config and create model
        config = OmegaConf.load("weights_encoding/configs/base_config.yaml")
        model = instantiate_from_config(config.model)
        
        print(f"✓ VAE model created successfully")
        
        # Create a dummy batch like ZooDataset produces
        batch_size = 2
        max_len = 2864
        
        dummy_batch = {
            'weight': torch.randn(batch_size, max_len),
            'model_properties': [
                {'layer_info': [('fc.weight', [10, 5]), ('fc.bias', [10])] } for _ in range(batch_size)
            ],
            'dataset': []
        }
        
        print(f"✓ Dummy batch created")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            result = model(dummy_batch)
            
        print(f"✓ VAE forward pass successful")
        print(f"✓ Forward result length: {len(result)}")
        
        if len(result) >= 3:
            inputs, reconstructions, posterior = result[0], result[1], result[2]
            print(f"✓ Input shape: {inputs.shape}")
            print(f"✓ Reconstruction shape: {reconstructions.shape}")
            print(f"✓ Posterior type: {type(posterior)}")
        
        return True
        
    except Exception as e:
        print(f"✗ VAE integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== VAE-ZooDataset Integration Test ===\n")
    
    results = []
    results.append(test_zoodataset())
    results.append(test_datamodule())
    results.append(test_vae_forward())
    
    print(f"\n=== Test Results ===")
    print(f"ZooDataset: {'✓ PASS' if results[0] else '✗ FAIL'}")
    print(f"ZooDataModule: {'✓ PASS' if results[1] else '✗ FAIL'}")
    print(f"VAE Integration: {'✓ PASS' if results[2] else '✗ FAIL'}")
    
    if all(results):
        print(f"\n🎉 All tests passed! The VAE is ready to train with ZooDataset.")
    else:
        print(f"\n❌ Some tests failed. Please review the errors above.")

if __name__ == "__main__":
    main()