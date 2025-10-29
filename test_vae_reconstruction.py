"""
Script to test VAE reconstruction accuracy on neural network weights.

This script:
1. Loads a trained VAE from checkpoint
2. Samples weights from ZooDataset
3. Encodes and decodes weights through the VAE
4. Reconstructs the weights into model format using existing functions
5. Evaluates accuracy on the respective dataset
6. Compares original vs reconstructed model accuracy
"""
import torch
import torch.nn as nn
import sys
import os
from collections import OrderedDict
import yaml

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "model_data"))

from zoodatasets.basedatasets import ZooDataset, reconstruct_from_chunks, reconstruct_weights
from weights_encoding.vae import AutoencoderKL
from model_data.model_definitions.def_net import NNmodule


def load_vae(checkpoint_path, config_path='weights_encoding/configs/base_config.yaml'):
    """Load VAE model from checkpoint"""
    print(f"Loading VAE from: {checkpoint_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create VAE model with full config
    model_config = config['model']['params']
    vae = AutoencoderKL(
        ddconfig=model_config['ddconfig'],
        lossconfig=model_config['lossconfig'],
        embed_dim=model_config['embed_dim'],
        learning_rate=model_config['learning_rate'],
        input_key=model_config.get('input_key', 'weight'),
        cond_key='dataset',
        device=model_config.get('device', 'cpu')
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    vae.load_state_dict(state_dict)
    vae.eval()
    
    print(f"✓ VAE loaded successfully")
    return vae


def load_test_data(dataset_name):
    """Load test dataset for accuracy evaluation"""
    import torchvision
    import torchvision.transforms as transforms
    
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'fashionmnist' or dataset_name.lower() == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    return testloader


def evaluate_model(model, testloader, device='cpu'):
    """Evaluate model accuracy on test data"""
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def infer_dataset_from_properties(model_properties):
    """Infer dataset name from model properties or return unknown"""
    # Try to get from properties
    dataset = model_properties.get('dataset', 'unknown')
    if dataset != 'unknown':
        return dataset
    
    # Try to infer from other properties
    channels = model_properties.get('model::channels_in', 1)
    if channels == 3:
        return 'cifar10'  # Default for 3 channels
    else:
        return 'mnist'  # Default for 1 channel


def test_vae_reconstruction(vae, dataset, file_idx, sample_idx, device='cpu'):
    """
    Test VAE reconstruction for a single model using ZooDataset.
    
    Args:
        vae: Trained VAE model
        dataset: ZooDataset instance
        file_idx: Index of the file in dataset
        sample_idx: Index of sample within the file
        device: Device to run on
    
    Returns:
        dict: Results containing original and reconstructed accuracies
    """
    print(f"\n{'='*80}")
    
    # Get max_len and scale from dataset
    max_len = dataset.max_len
    scale = dataset.scale
    
    # Get model properties and weight from dataset
    # Note: In ZooDataset with chunking, this gets a single chunk
    model_properties, weight_chunk = dataset.file_data[file_idx][sample_idx]
    
    # Get metadata about chunking
    num_chunks = model_properties.get('num_chunks', 1)
    original_length = model_properties.get('original_length', 0)
    layer_info = model_properties.get('layer_info')
    chunk_idx = model_properties.get('chunk_idx', 0)
    
    # Infer dataset name from filename or properties
    dataset_name = infer_dataset_from_properties(model_properties)
    
    print(f"Testing Model: {dataset_name} | Chunks: {num_chunks} | Params: {original_length:,}")
    print(f"{'='*80}")
    
    # Load test data
    try:
        testloader = load_test_data(dataset_name)
    except Exception as e:
        print(f"⚠ Could not load test data for {dataset_name}: {e}")
        return None
    
    # Step 1: Reconstruct original model and evaluate
    print("\n1. Reconstructing and evaluating ORIGINAL model...")
    try:
        # Collect all chunks if this is a multi-chunk model
        if num_chunks == 1:
            # Single chunk - just remove padding
            original_weights = weight_chunk.squeeze()[:original_length]
        else:
            # Multi-chunk - collect all chunks
            all_chunks = []
            for i, (props, w) in enumerate(dataset.file_data[file_idx]):
                if (props.get('layer_info') == layer_info and
                    props.get('original_length') == original_length):
                    chunk_id = props.get('chunk_idx', 0)
                    all_chunks.append((chunk_id, w))
            
            # Sort by chunk index
            all_chunks = sorted(all_chunks, key=lambda x: x[0])
            chunk_weights = [w for _, w in all_chunks]
            
            # Reconstruct full weights
            original_weights = reconstruct_from_chunks(chunk_weights, original_length)
        
        # Reconstruct state_dict from weights
        original_state_dict = reconstruct_weights(original_weights, layer_info)
        
        # Create model using the model properties config (not VAE config!)
        original_model = NNmodule(model_properties)
        original_model.load_state_dict(original_state_dict)
        
        original_accuracy = evaluate_model(original_model, testloader, device)
        print(f"   ✓ Original accuracy: {original_accuracy:.2f}%")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 2: Encode and decode through VAE (process all chunks)
    print(f"\n2. Encoding and decoding through VAE ({num_chunks} chunk(s))...")
    vae.to(device)
    
    decoded_chunks = []
    
    # Collect and process all chunks
    if num_chunks == 1:
        # Single chunk - use the weight directly from dataset
        weight = weight_chunk.squeeze()  # Already padded to max_len by ZooDataset
        weight = weight.to(torch.float32) / dataset.scale
        
        with torch.no_grad():
            # Prepare weight tensor for VAE
            # The encoder expects [batch, fch, in_dim] where fch=1
            weight_tensor = weight.to(device).to(torch.float32)
            if len(weight_tensor.shape) == 1:
                weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
            elif len(weight_tensor.shape) == 2:
                weight_tensor = weight_tensor.unsqueeze(1)  # [batch, 1, seq_len]
            
            # Encode directly (skip get_input as weight is already correctly shaped)
            posterior = vae.encode(weight_tensor)
            z = posterior.sample()
            
            # Decode
            decoded = vae.decode(z)
            
            # Flatten back to 1D
            decoded = decoded.view(-1)[:max_len]  # Get first max_len elements
            decoded = decoded.cpu() * scale
            
        decoded_chunks.append(decoded)
        print(f"   ✓ Encoded and decoded 1 chunk")
        
    else:
        # Multiple chunks - collect from dataset
        with torch.no_grad():
            chunks_to_process = []
            for i, (props, w) in enumerate(dataset.file_data[file_idx]):
                if (props.get('layer_info') == layer_info and
                    props.get('original_length') == original_length):
                    chunk_id = props.get('chunk_idx', 0)
                    chunks_to_process.append((chunk_id, w))
            
            # Sort by chunk index
            chunks_to_process = sorted(chunks_to_process, key=lambda x: x[0])
            
            for chunk_id, weight_chunk in chunks_to_process:
                weight = weight_chunk.squeeze()  # Already padded by ZooDataset
                weight = weight.to(torch.float32) / dataset.scale
                weight_tensor = weight.to(device)
                
                # The encoder expects [batch, fch, in_dim] where fch=1
                if len(weight_tensor.shape) == 1:
                    weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
                elif len(weight_tensor.shape) == 2:
                    weight_tensor = weight_tensor.unsqueeze(1)  # [batch, 1, seq_len]
                
                # Encode directly
                posterior = vae.encode(weight_tensor)
                z = posterior.sample()
                
                # Decode
                decoded = vae.decode(z)
                
                # Flatten back to 1D
                decoded = decoded.view(-1)[:max_len]
                decoded = decoded.cpu() * scale
                
                decoded_chunks.append(decoded)
        
        print(f"   ✓ Encoded and decoded {len(decoded_chunks)} chunks")
    
    # Step 3: Reconstruct full weights
    print("\n3. Reconstructing full weights from decoded chunks...")
    if num_chunks == 1:
        # Single chunk - just remove padding
        reconstructed_weights = decoded_chunks[0][:original_length]
    else:
        # Multiple chunks - concatenate and remove padding
        reconstructed_weights = reconstruct_from_chunks(decoded_chunks, original_length)
    
    print(f"   ✓ Reconstructed weights shape: {reconstructed_weights.shape}")
    print(f"   ✓ Expected length: {original_length}")
    
    # Step 4: Reconstruct state_dict
    print("\n4. Reconstructing state_dict from weights...")
    try:
        reconstructed_state_dict = reconstruct_weights(reconstructed_weights, layer_info)
        print(f"   ✓ Reconstructed {len(reconstructed_state_dict)} layers")
    except Exception as e:
        print(f"   ✗ Error reconstructing state_dict: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 5: Load into model and evaluate
    print("\n5. Evaluating RECONSTRUCTED model...")
    try:
        # Create model using the same model properties (this is the model config, not VAE config)
        reconstructed_model = NNmodule(model_properties)
        reconstructed_model.load_state_dict(reconstructed_state_dict)
        
        reconstructed_accuracy = evaluate_model(reconstructed_model, testloader, device)
        print(f"   ✓ Reconstructed accuracy: {reconstructed_accuracy:.2f}%")
        
    except Exception as e:
        print(f"   ✗ Error evaluating reconstructed model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 6: Compare results
    diff = abs(original_accuracy - reconstructed_accuracy)
    recovery_rate = (reconstructed_accuracy / original_accuracy * 100) if original_accuracy > 0 else 0
    
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"  Original accuracy:      {original_accuracy:.2f}%")
    print(f"  Reconstructed accuracy: {reconstructed_accuracy:.2f}%")
    print(f"  Absolute difference:    {diff:.2f}%")
    print(f"  Recovery rate:          {recovery_rate:.1f}%")
    
    if recovery_rate >= 90:
        print(f"  ✓ EXCELLENT: >90% recovery")
        status = "EXCELLENT"
    elif recovery_rate >= 70:
        print(f"  ✓ GOOD: 70-90% recovery")
        status = "GOOD"
    elif recovery_rate >= 50:
        print(f"  ⚠ FAIR: 50-70% recovery")
        status = "FAIR"
    else:
        print(f"  ✗ POOR: <50% recovery")
        status = "POOR"
    print(f"{'='*80}")
    
    return {
        'dataset': dataset_name,
        'num_chunks': num_chunks,
        'params': original_length,
        'original_accuracy': original_accuracy,
        'reconstructed_accuracy': reconstructed_accuracy,
        'difference': diff,
        'recovery_rate': recovery_rate,
        'status': status
    }


def main():
    """Main testing function"""
    print("="*80)
    print("VAE WEIGHT RECONSTRUCTION ACCURACY TEST")
    print("="*80)
    print("\nThis script tests the full VAE reconstruction pipeline:")
    print("  Original → Encode → Decode → Reconstruct → Evaluate")
    
    # Configuration
    vae_checkpoint = 'vae_checkpoints/last.ckpt'
    config_path = 'weights_encoding/configs/base_config.yaml'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"VAE checkpoint: {vae_checkpoint}")
    
    # Check if checkpoint exists
    if not os.path.exists(vae_checkpoint):
        print(f"\n✗ Error: VAE checkpoint not found at {vae_checkpoint}")
        print("Please train the VAE first or provide correct checkpoint path.")
        return
    
    # Load VAE
    print("\n" + "-"*80)
    try:
        vae = load_vae(vae_checkpoint, config_path)
    except Exception as e:
        print(f"✗ Error loading VAE: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load dataset using existing ZooDataset
    print("\n" + "-"*80)
    print("Loading dataset...")
    
    # Load config to get dataset parameters
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']['params']
    max_len = config['model']['params']['ddconfig']['in_dim']
    
    # Create dataset with same parameters as training
    train_dataset = ZooDataset(
        root=data_config.get('data_dir', 'model_data'),
        dataset=data_config.get('dataset', 'joint'),
        split='train',
        scale=data_config.get('scale', 1.0),
        max_len=max_len
    )
    
    val_dataset = ZooDataset(
        root=data_config.get('data_dir', 'model_data'),
        dataset=data_config.get('dataset', 'joint'),
        split='val',
        scale=data_config.get('scale', 1.0),
        max_len=max_len
    )
    
    print(f"✓ Loaded datasets")
    print(f"  Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"  Scale: {data_config.get('scale', 1.0)}, Max length: {max_len}")
    
    # Test on multiple samples
    print("\n" + "="*80)
    print("TESTING VAE RECONSTRUCTION ON SAMPLES")
    print("="*80)
    
    results = []
    
    # Test samples from train split
    print(f"\n--- Testing 2 samples from TRAIN split ---")
    for file_idx in range(min(2, len(train_dataset.file_data))):
        if len(train_dataset.file_data[file_idx]) > 0:
            sample_idx = 0  # Test first sample from each file
            result = test_vae_reconstruction(vae, train_dataset, file_idx, sample_idx, device)
            if result is not None:
                results.append(result)
    
    # Test samples from val split
    print(f"\n--- Testing 2 samples from VAL split ---")
    for file_idx in range(min(2, len(val_dataset.file_data))):
        if len(val_dataset.file_data[file_idx]) > 0:
            sample_idx = 0
            result = test_vae_reconstruction(vae, val_dataset, file_idx, sample_idx, device)
            if result is not None:
                results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL TESTS")
    print("="*80)
    
    if not results:
        print("⚠ No models were successfully tested")
        return
    
    print(f"\nTested {len(results)} model(s):\n")
    print(f"{'Dataset':<15} {'Chunks':<8} {'Params':<12} {'Original%':<12} {'Reconstructed%':<16} {'Recovery%':<12} {'Status':<10}")
    print("-"*95)
    
    for r in results:
        print(f"{r['dataset']:<15} {r['num_chunks']:<8} {r['params']:<12,} "
              f"{r['original_accuracy']:<12.2f} {r['reconstructed_accuracy']:<16.2f} "
              f"{r['recovery_rate']:<12.1f} {r['status']:<10}")
    
    # Statistics
    avg_recovery = sum(r['recovery_rate'] for r in results) / len(results)
    excellent = sum(1 for r in results if r['status'] == 'EXCELLENT')
    good = sum(1 for r in results if r['status'] == 'GOOD')
    fair = sum(1 for r in results if r['status'] == 'FAIR')
    poor = sum(1 for r in results if r['status'] == 'POOR')
    
    print("\n" + "-"*95)
    print(f"Average recovery rate: {avg_recovery:.1f}%")
    print(f"Status distribution: {excellent} excellent, {good} good, {fair} fair, {poor} poor")
    
    print("\n" + "="*80)
    if avg_recovery >= 70:
        print("✓ OVERALL: GOOD - VAE reconstruction is working well")
    elif avg_recovery >= 50:
        print("⚠ OVERALL: FAIR - VAE reconstruction needs improvement")
    else:
        print("✗ OVERALL: POOR - VAE reconstruction has significant issues")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
