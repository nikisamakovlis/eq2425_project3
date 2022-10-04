import os
import yaml

import torch
from torch.utils.data import Dataset, DataLoader

import utils
import prepare_augmentations
import prepare_models
import prepare_datasets

def prepare_args(default_params_path=None):
    # Load default params
    with open(default_params_path) as f:
        args = yaml.safe_load(f)
    return args

def set_globals(rank, args):
    global mode
    global save_params, dataset_params, system_params, dataloader_params, model_params
    global training_params, trainloader_params, valloader_params

    # Prepare mode, and other "_params"
    mode = args['mode']
    save_params = args['save_params']
    dataset_params = args['dataset_params']
    system_params = args['system_params']
    dataloader_params = args['dataloader_params']
    model_params = args['model_params']
    training_params = args['training_params']
    trainloader_params = dataloader_params['trainloader']
    valloader_params = dataloader_params['valloader']

    # ============ Writing model_path in args ... ============
    # Write model_path in args
    batch_size = trainloader_params['batch_size']
    lr = training_params['train']['optimizer']['sgd']['lr']
    shuffling = trainloader_params['shuffling']
    if shuffling:
        shuffling_label = 'shuffling'
    else:
        shuffling_label = 'notshuffling'

    output_dir = save_params["output_dir"]
    model_path = os.path.join(output_dir, f"saved_model_bs{batch_size}_lr{lr}_{shuffling_label}")
    if not os.path.exists(model_path):
        if rank == 0:
            os.makedirs(model_path)
    args["save_params"]["model_path"] = model_path
    return args

def get_data(rank):
    global CnnModel
    # Get augmentations
    augmentations = prepare_augmentations.PublicDataAugmentation(dataset_params)
    transforms_plain = augmentations.transforms_plain
    transforms_aug = augmentations.transforms_aug

    # Get dataset class
    # CnnModel = prepare_models.CNN()

    dataset_class = prepare_datasets.GetCIFAR(dataset_params, transforms_aug=transforms_aug, transforms_plain=transforms_plain, normalize=normalize)

    if mode == 'train':
        train_dataset, val_dataset = dataset_class.get_datasets('train/')
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=system_params['num_gpus'], rank=rank, shuffle=trainloader_params['shuffling'])  # shuffle=True to reduce monitor bias
    else:
        val_dataset = dataset_class.get_datasets('test/')

    # Set the data sampler with DistributedSampler
    val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=system_params['num_gpus'], rank=rank, shuffle=valloader_params['shuffling'])  # shuffle=True to reduce monitor bias

    # Build train and val data loaders
    train_batch_size = int(trainloader_params['batch_size'] / (system_params['num_gpus']*trainloader_params['accum_iter']))
    val_batch_size = int(valloader_params['batch_size'] / (system_params['num_gpus']*valloader_params['accum_iter']))
    if mode == 'train':  # if train
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=train_batch_size,
                                      num_workers=trainloader_params['num_workers'],
                                      pin_memory=trainloader_params['pin_memory'],
                                      drop_last=trainloader_params['drop_last'])
    else:
        train_dataloader = None
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler,
                                batch_size=val_batch_size,
                                num_workers=valloader_params['num_workers'],
                                pin_memory=valloader_params['pin_memory'],
                                drop_last=valloader_params['drop_last'])

    if rank == 0 and mode == 'train':
        print(f"There are {len(train_dataloader)} train_dataloaders on each rank. ")
        print(f"There are {len(val_dataloader)} val_dataloaders on each rank. ", end="\n\n")

    return rank, train_dataloader, val_dataloader


def main(rank, args):
    # ============ Define some parameters for easy access... ============
    _ = set_globals(rank, args)

    # ============ Setting up system configuration ... ============
    # Set gpu params and random seeds for reproducibility
    utils.set_sys_params(system_params['random_seed'])

    # ============ Getting data ready ... ============
    rank, train_dataloader, val_dataloader = get_data(rank)

    # # ============ Getting model and loss ready ... ============
    # rank, model, loss = get_model_loss(rank)
    #
    # if mode == 'train':
    #     # ============ Getting optimizer ready ... ============
    #     optimizer = get_optimizer(model)
    #
    #     # ============ Getting schedulers ready ... ============
    #     lr_schedule = get_schedulers(train_dataloader)
    #
    #     if rank == 0:
    #         print(f"Loss, optimizer and schedulers ready.", end="\n\n")
    #
    #     # ============ Start training process ... ============
    #     train_process(rank, train_dataloader, val_dataloader, model, loss, optimizer, lr_schedule)


if __name__ == '__main__':
    args = prepare_args(default_params_path='train_params.yaml')
    utils.launch(main, args)