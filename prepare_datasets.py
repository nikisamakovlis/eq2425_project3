import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import Dataset

import utils

class ReturnIndexDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.subset)

class GetCIFAR():
    def __init__(self, dataset_params, transforms_aug, transforms_plain, normalize):
        self.dataset_params = dataset_params
        self.transforms_aug = transforms_aug
        self.transforms_plain = transforms_plain
        self.normalize = normalize

    def get_datasets(self, official_split):
        # Train: 50,000 images
        # Test: 10,000 images
        if official_split == 'train/' or official_split == 'val/':
            # Note that CIFAR dataset doesn't have its official validation set, so here we create our own
            train_dataset = datasets.CIFAR10(
                root=self.dataset_params['data_folder'],
                train=True,
                download=False,
                transform=None)

            val_dataset = datasets.CIFAR10(
                root=self.dataset_params['data_folder'],
                train=False,
                download=False,
                transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))
  
            train_set = ReturnIndexDataset(train_dataset, transform=torchvision.transforms.Compose([self.transforms_aug, self.normalize]))
            valid_set = ReturnIndexDataset(val_dataset, transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))

            if utils.is_main_process():
                print(f"There are {len(train_set)} samples in train split, on each rank. ")
                print(f"There are {len(valid_set)} samples in val split, on each rank. ")
            return train_set, valid_set
        else:
            dataset = datasets.CIFAR10(
                root=self.dataset_params['data_folder'],
                train=False,
                download=False,
                transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))
            dataset = ReturnIndexDataset(dataset, transform=None)
            return dataset