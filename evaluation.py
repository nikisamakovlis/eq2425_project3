import os
import ast
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.distributed as dist
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, top_k_accuracy_score

from utils import accuracy
import utils


@torch.no_grad()
def cifar_validate_network(rank, val_dataloader, model, wandb, dataset_params, log_img=False):
    device = torch.device("cuda:{}".format(rank))
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    all_pred_list = []
    all_true_list = []
    all_loss_list = []
    count = 0

    for images, labels, ind in metric_logger.log_every(val_dataloader, 40, header):
        # Move images and labels to gpu
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).to(torch.int64)
        labels = torch.nn.functional.one_hot(labels, num_classes=10).float()

        # Forward
        with torch.no_grad():
            output = model(images)
            output = torch.exp(output)
            loss = -torch.mean(torch.sum(torch.log(output)*labels, dim=1))
            labels = torch.argmax(labels, dim=1)

        if isinstance(labels.squeeze().tolist(), list):
            all_true_list.extend(labels.squeeze().tolist())
        else:
            all_true_list.extend([labels.squeeze().tolist()])

        all_pred_list.extend(output)
        all_loss_list.extend([loss])
        count += 1

    # we have to create enough room to store the collected objects
    all_pred_list_outputs = [None for _ in range(utils.dist.get_world_size())]
    all_true_list_outputs = [None for _ in range(utils.dist.get_world_size())]
    all_loss_list_outputs = [None for _ in range(utils.dist.get_world_size())]

    # the first argument is the collected lists, the second argument is the data unique in each process
    dist.all_gather_object(all_pred_list_outputs, all_pred_list)
    dist.all_gather_object(all_true_list_outputs, all_true_list)
    dist.all_gather_object(all_loss_list_outputs, all_loss_list)

    all_pred_list_outputs = [item.tolist() for sublist in all_pred_list_outputs for item in sublist]
    all_true_list_outputs = [item for sublist in all_true_list_outputs for item in sublist]
    all_loss_list_outputs = [item.cpu() for sublist in all_loss_list_outputs for item in sublist]
    loss_to_return = sum(all_loss_list_outputs)/len(all_loss_list_outputs)

    acc1 = top_k_accuracy_score(all_true_list_outputs, np.array(all_pred_list_outputs), k=1)

    if rank == 0:
        print(f'Gathering image level - Number of validation images {len(all_true_list_outputs)}')

    return loss_to_return, acc1