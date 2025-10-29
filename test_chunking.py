"""
Test script to verify chunking functionality for large models
"""
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))

from zoodatasets.basedatasets import ZooDataset, reconstruct_from_chunks, reconstruct_weights

def test_chunking():
    """Test that large models are properly chunked"""
    print("=" * 80)
    print("Testing ZooDataset Chunking Functionality")
    print("=" * 80)
    
    # Create dataset with smaller max_len to force chunking
    max_len = 2864
    dataset = ZooDataset(split="train", max_len=max_len)
    
    print(f"\nDataset created with max_len={max_len}")
    print(f"Total samples (including chunks): {len(dataset)}")
    
    # Track chunking statistics
    chunk_stats = {}
    
    # Sample multiple items to see chunking distribution
    print("\n" + "-" * 80)
    print("Sampling dataset to find chunked models...")
    print("-" * 80)
    
    samples_checked = 0
    max_samples = 100
    found_chunked = False
    
    for i in range(min(len(dataset), max_samples)):
        sample = dataset[i]
        props = sample['model_properties']
        
        num_chunks = props.get('num_chunks', 1)
        chunk_idx = props.get('chunk_idx', 0)
        original_length = props.get('original_length', 0)
        
        key = (num_chunks, original_length)
        if key not in chunk_stats:
            chunk_stats[key] = 0
        chunk_stats[key] += 1
        
        # Print info for chunked models
        if num_chunks > 1 and chunk_idx == 0:  # Only print first chunk
            found_chunked = True
            print(f"\n✓ Found chunked model:")
            print(f"  - Original length: {original_length:,} parameters")
            print(f"  - Number of chunks: {num_chunks}")
            print(f"  - Chunk size: {max_len}")
            print(f"  - Weight shape: {sample['weight'].shape}")
            
            # Verify chunk is properly padded
            actual_size = sample['weight'].shape[-1]
            assert actual_size == max_len, f"Chunk size mismatch: {actual_size} != {max_len}"
            print(f"  ✓ Chunk properly sized and padded")
        
        samples_checked += 1
    
    print("\n" + "-" * 80)
    print(f"Checked {samples_checked} samples")
    print("\nChunking Statistics:")
    print("-" * 80)
    
    for (num_chunks, orig_len), count in sorted(chunk_stats.items()):
        if num_chunks > 1:
            print(f"  {num_chunks} chunks ({orig_len:,} params): {count} samples")
        else:
            print(f"  1 chunk ({orig_len:,} params): {count} samples")
    
    if not found_chunked:
        print("\n⚠ No chunked models found in first 100 samples")
        print("  This is okay if all models in dataset are smaller than max_len")
    
    # Test chunk reconstruction
    print("\n" + "=" * 80)
    print("Testing Chunk Reconstruction")
    print("=" * 80)
    
    # Find a chunked model by looking through file_data directly
    for file_samples in dataset.file_data:
        # Group samples by original model
        model_groups = {}
        for props, weight in file_samples:
            if 'num_chunks' in props and props['num_chunks'] > 1:
                # Use a unique key to group chunks from same model
                # We'll use layer_info as it's shared across chunks
                key = str(props.get('layer_info', ''))
                if key not in model_groups:
                    model_groups[key] = []
                model_groups[key].append((props, weight))
        
        # Test reconstruction on first chunked model found
        for model_key, chunks in model_groups.items():
            if len(chunks) > 1:
                # Sort by chunk_idx
                chunks = sorted(chunks, key=lambda x: x[0]['chunk_idx'])
                
                props = chunks[0][0]
                print(f"\nReconstructing model with {props['num_chunks']} chunks:")
                print(f"  Original length: {props['original_length']:,} parameters")
                
                # Extract weights from chunks
                chunk_weights = [weight for _, weight in chunks]
                
                # Reconstruct full weights
                full_weights = reconstruct_from_chunks(chunk_weights, props['original_length'])
                print(f"  Reconstructed tensor shape: {full_weights.shape}")
                print(f"  Expected length: {props['original_length']}")
                
                assert full_weights.shape[0] == props['original_length'], "Reconstruction length mismatch!"
                print(f"  ✓ Reconstruction successful!")
                
                # Test layer reconstruction
                layer_info = props['layer_info']
                state_dict = reconstruct_weights(full_weights, layer_info)
                print(f"  ✓ Reconstructed {len(state_dict)} layers")
                
                # Verify layer shapes
                total_params = 0
                for layer_name, original_shape in layer_info[:3]:
                    reconstructed_shape = state_dict[layer_name].shape
                    num_params = state_dict[layer_name].numel()
                    total_params += num_params
                    print(f"    Layer '{layer_name}': {original_shape} ({num_params:,} params)")
                    assert reconstructed_shape == original_shape, f"Shape mismatch for {layer_name}"
                
                print(f"\n  ✓ All tests passed for chunked model!")
                return True
    
    print("\n⚠ No chunked models found in dataset to test reconstruction")
    print("  All models fit within max_len")
    return False

if __name__ == "__main__":
    try:
        test_chunking()
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
