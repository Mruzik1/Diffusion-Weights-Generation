import os
import sys
import yaml
import hashlib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

for p in ["..", "../model_data"]:
    sys.path.append(os.path.join(os.path.dirname(__file__), p))


################### Functions ###################
def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def matpadder(x, max_in=512):
    shape = x.shape
    delta2 = max_in - shape[1]
    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out


def reconstruct_weights(flattened_weights, layer_info):
    """
    Reconstruct flattened weights back to original state_dict structure
    
    Args:
        flattened_weights: 1D tensor of flattened weights
        layer_info: list of tuples containing (layer_name, original_shape) for each layer
    
    Returns:
        OrderedDict: reconstructed state_dict
    """
    from collections import OrderedDict
    
    reconstructed = OrderedDict()
    start_idx = 0
    
    for layer_name, original_shape in layer_info:
        num_elements = 1
        for dim in original_shape:
            num_elements *= dim
        
        end_idx = start_idx + num_elements
        layer_weights = flattened_weights[start_idx:end_idx]
        
        reconstructed[layer_name] = layer_weights.reshape(original_shape)
        
        start_idx = end_idx
    
    return reconstructed


################### Classes ###################
class ZooDataset(Dataset):
    def __init__(
        self,
        root="model_data",
        dataset="joint",
        split="train",
        scale=1.0,
        topk=None,
        transform=None,
        normalize=False,
        max_len=2864
    ):
        """ Weights dataset """
        super(ZooDataset, self).__init__()
        self.dataset = dataset
        self.topk = topk

        self.max_len = max_len
        self.normalize = normalize
        self.split = split
        self.scale = scale

        datapath = os.path.join(root, "weights")

        self.transform = transform
        all_file_data, _ = self.load_data(datapath)
        
        # apply deterministic split
        self.file_data, self.file_lengths = self.apply_split(all_file_data, split)

        total_samples = sum(self.file_lengths)
        print(f"Loaded {len(self.file_data)} files with {total_samples} total samples for {split} split")

    def __len__(self):
        return sum(self.file_lengths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # balanced sampling
        import random
        file_idx = random.randint(0, len(self.file_data) - 1)
        sample_idx = random.randint(0, self.file_lengths[file_idx] - 1)
        
        model_properties, weight = self.file_data[file_idx][sample_idx]
        weight = weight.to(torch.float32) / self.scale
        
        if self.transform:
            weight = self.transform(weight)
            
        sample = {"weight": weight, "model_properties": model_properties, "dataset": []}
        return sample

    def apply_split(self, all_file_data, split):        
        split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        
        filtered_file_data = []
        filtered_file_lengths = []
        
        for file_samples in all_file_data:
            split_samples = []
            
            for sample in file_samples:
                model_properties, _ = sample
                
                # create deterministic hash from model properties
                sample_str = str(model_properties)
                sample_hash = int(hashlib.md5(sample_str.encode()).hexdigest(), 16)
                hash_ratio = (sample_hash % 1000) / 1000.0
                
                # determine split assignment
                if split == "train" and hash_ratio < split_ratios["train"]:
                    split_samples.append(sample)
                elif split == "val" and split_ratios["train"] <= hash_ratio < (split_ratios["train"] + split_ratios["val"]):
                    split_samples.append(sample)
                elif split == "test" and hash_ratio >= (split_ratios["train"] + split_ratios["val"]):
                    split_samples.append(sample)
            
            if split_samples:
                filtered_file_data.append(split_samples)
                filtered_file_lengths.append(len(split_samples))
        
        return filtered_file_data, filtered_file_lengths

    def load_data(self, folder_path):
        # load data from all .pt files in the folder
        file_data = []
        file_lengths = []
        
        pt_files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]
        
        for pt_file in pt_files:
            file_path = os.path.join(folder_path, pt_file)
            data = torch.load(file_path, weights_only=False)
            
            processed_samples = []
            for sample in data:
                model_properties, model = sample
                model_weights = model.state_dict()

                weight_tensors = []
                layer_info = []
                
                for key, tensor in model_weights.items():
                    flattened = tensor.detach().cpu().flatten()
                    weight_tensors.append(flattened)
                    layer_info.append((key, tensor.shape))

                w = torch.cat(weight_tensors, dim=0)

                # Ensure tensor is 1D, then pad to max_len
                if len(w.shape) > 1:
                    w = w.flatten()
                
                # Pad or truncate to max_len
                if w.shape[0] < self.max_len:
                    # Pad with zeros
                    w = F.pad(w, (0, self.max_len - w.shape[0]), "constant", 0)
                elif w.shape[0] > self.max_len:
                    # Truncate to max_len
                    w = w[:self.max_len]
                
                # Add batch dimension
                w = w.unsqueeze(0)

                if self.topk is not None:
                    w = w[:self.topk]

                # store layer info in model_properties for reconstruction
                model_properties["layer_info"] = layer_info
                processed_samples.append((model_properties, w))
            
            file_data.append(processed_samples)
            file_lengths.append(len(processed_samples))
        
        return file_data, file_lengths


if __name__ == "__main__":
    # sanity test
    print("Testing ZooDataset...")

    try:
        # test train split
        train_dataset = ZooDataset(split="train")
        print(f"Train dataset length: {len(train_dataset)}")

        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Weight shape: {sample['weight'].shape}")
            print(f"Model properties type: {type(sample['model_properties'])}")
            
            # test reconstruction
            if "layer_info" in sample["model_properties"]:
                print("\nTesting weight reconstruction...")
                flattened_weights = sample["weight"].squeeze(0)
                layer_info = sample["model_properties"]["layer_info"]
                
                reconstructed = reconstruct_weights(flattened_weights, layer_info)
                print(f"Reconstructed state_dict keys: {list(reconstructed.keys())}")
                print(f"Number of layers reconstructed: {len(reconstructed)}")
                
                # check if shapes match original layer info
                for i, (layer_name, original_shape) in enumerate(layer_info):
                    if layer_name in reconstructed:
                        reconstructed_shape = reconstructed[layer_name].shape
                        print(f"Layer {layer_name}: original {original_shape} -> reconstructed {reconstructed_shape}")
                        assert reconstructed_shape == original_shape, f"Shape mismatch for {layer_name}"
                
                print("Weight reconstruction test passed!")

        # test val split
        val_dataset = ZooDataset(split="val")
        print(f"Val dataset length: {len(val_dataset)}")

        # test test split
        test_dataset = ZooDataset(split="test")
        print(f"Test dataset length: {len(test_dataset)}")

        print("Dataset sanity test passed!")

    except Exception as e:
        print(f"Dataset sanity test failed: {e}")
