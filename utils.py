import os
import numpy as np
import random
import socket

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import Dataset


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
            original_train_dataset = datasets.CIFAR10(
                root=self.dataset_params['data_folder'],
                train=True,
                download=False,
                transform=None)

            num_train = len(original_train_dataset)
            valid_size = 0.02
            split = int(np.floor(valid_size * num_train))
            train_set, valid_set = torch.utils.data.random_split(original_train_dataset, [num_train-split, split],
                                                                 generator=torch.Generator().manual_seed(42))

            train_set = ReturnIndexDataset(train_set, transform=torchvision.transforms.Compose([self.transforms_aug, self.normalize]))
            valid_set = ReturnIndexDataset(valid_set, transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))

            if is_main_process():
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


def get_open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def set_sys_params(random_seed):
    # Set random seeds for reproducibility TODO: to figure out whether it is necessary to have different random seeds
    # on different ranks (DeiT uses different seeds)
    seed = random_seed  #+ get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True  # benchmark mode is good whenever your input sizes for your network do not vary.

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not ddp():
        return 0
    return dist.get_rank()

def ddp():
    world_size = dist.get_world_size()
    if not dist.is_available() or not dist.is_initialized() or world_size < 2:
        return False
    return True

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def run_distributed_workers(rank, main_func, world_size, dist_url, args):
    # Initialize the process group
    dist.init_process_group(backend="NCCL", init_method=dist_url, world_size=world_size, rank=rank)

    # Synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    if ddp():
        dist.barrier()

    torch.cuda.set_device(rank)
    # print('| distributed init (rank {}): {}'.format(
    #     rank, dist_url), flush=True)

    main_func(rank, args)

def launch(main_func, args=()):
    # Set gpu params
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args['system_params']['gpu_ids']

    world_size = args['system_params']['num_gpus']
    port = get_open_port()
    dist_url = f"tcp://127.0.0.1:{port}"
    # os.environ[
    #     "TORCH_DISTRIBUTED_DEBUG"
    # ] = "INFO"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    mp.spawn(
        run_distributed_workers,
        nprocs=world_size,
        args=(main_func, world_size, dist_url, args),
        daemon=False,
    )

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def synchronize():
    if not ddp():
        return
    dist.barrier()

