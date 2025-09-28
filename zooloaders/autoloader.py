import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

from zoodatasets.basedatasets import ZooDataset


def custom_collate_fn(batch):
    """Custom collate function to handle ZooDataset batch structure"""
    # Separate different types of data
    weights = []
    model_properties = []
    datasets = []
    
    for sample in batch:
        weights.append(sample['weight'])
        model_properties.append(sample['model_properties'])
        datasets.append(sample['dataset'])
    
    # Stack tensors
    weights = torch.stack(weights, dim=0)
    
    return {
        'weight': weights,
        'model_properties': model_properties,
        'dataset': datasets
    }


class ZooDataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_dir, data_root, batch_size, num_workers, scale, num_sample, topk, normalize):
        super().__init__()
        self.data_dir = data_dir
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.topk = topk
        self.normalize = normalize
        self.num_sample = num_sample
        self.scale = scale

    def prepare_data(self):
        # Only download if needed, and check if data_root exists
        import os
        if os.path.exists(self.data_root):
            try:
                datasets.CIFAR10(self.data_root, train=True, download=True)
                datasets.CIFAR10(self.data_root, train=False, download=True)
            except:
                print(f"Warning: Could not download CIFAR10 to {self.data_root}")

    def setup(self, stage):
        if stage == "fit":
            self.trainset = ZooDataset(
                root=self.data_dir, 
                dataset=self.dataset, 
                split="train", 
                scale=self.scale, 
                topk=self.topk,
                normalize=self.normalize
            )
            self.valset = ZooDataset(
                root=self.data_dir, 
                dataset=self.dataset, 
                split="val", 
                scale=self.scale,
                normalize=self.normalize
            )

        if stage == "test":
            self.testset = ZooDataset(
                root=self.data_dir, 
                dataset=self.dataset, 
                split="test", 
                scale=self.scale,
                normalize=self.normalize
            )

        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

    # dataloader to evaluate the reconstruction performance on model zoo.
    def predict_dataloader(self):
        pass
