import os
import sys
import time
import datetime
import random
import math
import json
import yaml
import cv2
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import socket
from copy import deepcopy
from pathlib import Path
from collections import defaultdict, deque
from operator import itemgetter
from typing import Iterator, List, Optional, Union
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
from torchvision import transforms
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def constant_scheduler(base_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, step_epoch=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    if step_epoch < epochs:
        iters1 = step_epoch * niter_per_ep - warmup_iters
        iters2 = epochs * niter_per_ep - step_epoch * niter_per_ep
        schedule = [base_value]*iters1 + [base_value/10]*iters2
    else:
        iters1 = epochs * niter_per_ep - warmup_iters
        schedule = [base_value]*iters1

    if warmup_epochs > 0:
        schedule = np.concatenate((warmup_schedule, schedule))

    # print(schedule)
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                if is_main_process():
                    print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    if is_main_process():
                        print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    if is_main_process():
                        print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            if is_main_process():
                print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def layer_decay_get_params_groups(model, weight_decay, skip_list=(), get_num_layer=None, get_layer_scale=None):
    encoder_params, encoder_names = get_params_per_model(model['model'], weight_decay, skip_list, get_num_layer, get_layer_scale, if_encoder=True)
    if is_main_process():
        output_path = 'yaml/params/parameter_group_printed_all.yaml'
        with open(output_path, 'w') as outfile:
            yaml.dump({**encoder_names,}, outfile, sort_keys=False, default_flow_style=False)
    return list({**encoder_params, }.values())