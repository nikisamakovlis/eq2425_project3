import os
import ast
import yaml
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utils
import prepare_augmentations
import prepare_models
import prepare_datasets
import prepare_trainers
import evaluation

import warnings
warnings.filterwarnings("ignore")

import wandb
from wandb import AlertLevel
torch.autograd.set_detect_anomaly(True)

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
    # TODO: to change for Network Structure
    batch_size = trainloader_params['batch_size']
    lr = training_params['train']['optimizer']['sgd']['lr']
    shuffling = trainloader_params['shuffling']

    if shuffling:
        shuffling_label = 'shuffling'
    else:
        shuffling_label = 'notshuffling'

    output_dir = save_params["output_dir"]
    model_path = os.path.join(output_dir, f"saved_model_bs{batch_size}_lr{lr}_{shuffling_label}_{model_params['variant']}_{model_params['filter_num']}_{model_params['filter_size12']}")

    if not os.path.exists(model_path):
        if rank == 0:
            os.makedirs(model_path)
    args["save_params"]["model_path"] = model_path

    if mode == 'train':
        if args['resume_id'] != "None":
            wandb.init(project="visual_search", entity="yueliukth", name=os.path.basename(args["save_params"]["model_path"]), config=args, resume="must", id=args['resume_id'],
                       settings=wandb.Settings(start_method="fork"))
        else:
            wandb.init(project="visual_search", entity="yueliukth", name=os.path.basename(args["save_params"]["model_path"]), config=args)

    return args


def get_data(rank):
    # Get augmentations
    augmentations = prepare_augmentations.PublicDataAugmentation(dataset_params)
    transforms_plain = augmentations.transforms_plain
    transforms_aug = augmentations.transforms_aug

    dataset_class = prepare_datasets.GetCIFAR(dataset_params, transforms_aug=transforms_aug, transforms_plain=transforms_plain)

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


def get_model_loss(rank):
    # Get dataset class
    model = prepare_models.CNN(model_params)

    # ============ Preparing model ... ============
    # 'DefaultModel', 'ConnectedLayerModel', 'LeakyReLUModel', 'DropoutModel', 'BatchNormModel'
    filter1, filter2, filter3 = ast.literal_eval(str(model_params['filter_num']))
    kernel1, kernel2 = ast.literal_eval(str(model_params['filter_size12']))

    print(model)
    print(f'filter num: {filter1}, {filter2}, {filter3}')
    print(f'filter size: {kernel1}, {kernel2}')
    print(f"model variant: {model_params['variant']}")

    # Move the model to gpu. This step is necessary for DDP later
    device = torch.device("cuda:{}".format(rank))
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

    # Log the number of trainable parameters in Tensorboard
    if rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of trainable params in the model:', n_parameters, end='\n\n')

    # ============ Preparing loss and move it to gpu ... ============
    loss = nn.NLLLoss()

    return rank, {'model': model}, {'classification_loss': loss}


def get_optimizer(model):
    # ============ Preparing optimizer ... ============
    params_dict = utils.layer_decay_get_params_groups(model, weight_decay=0,
                                                      skip_list=(),
                                                      get_num_layer=None,
                                                      get_layer_scale=None)
    optimizer_choice = training_params[mode]['optimizer']['name']
    lr = training_params[mode]['optimizer'][optimizer_choice]['lr']
    optimizer = torch.optim.SGD(params_dict, lr=lr)
    return optimizer


def get_schedulers(train_dataloader):
    # ============ Initialize schedulers ... ============
    optimizer_choice = training_params[mode]['optimizer']['name']
    base_lr = training_params[mode]['optimizer'][optimizer_choice]['lr']

    lr_schedule = utils.constant_scheduler(base_value=base_lr, epochs=training_params[mode]['num_epochs'],
                                           niter_per_ep=len(train_dataloader), warmup_epochs=0,
                                           start_warmup_value=0, step_epoch=1000)
    return lr_schedule


def train_process(rank, train_dataloader, val_dataloader, model, loss, optimizer, lr_schedule):
    model = model['model']

    # # ============ Optionally resume training ... ==========
    to_restore = {'epoch': 0}
    utils.restart_from_checkpoint(
        os.path.join(save_params['model_path'], f'checkpoint.pth'),
        run_variables=to_restore, model=model, optimizer=optimizer)

    # # ============ Start training ... ============
    if rank == 0:
        print("Starting training !")

    best_avg_auc = 0.
    for epoch in range(to_restore['epoch'], training_params[mode]['num_epochs']):
        # In distributed mode, calling the :meth:`set_epoch` method at
        # the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        # is necessary to make shuffling work properly across multiple epochs. Otherwise,
        # the same ordering will be always used.
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        # ============ Evaluating the classification performance before training starts ... ============
        best_top1_recall = 0
        if epoch == to_restore['epoch']:
            val_loss, top1_recall = evaluation.cifar_validate_network(rank, val_dataloader, model, wandb, dataset_params, log_img=True)
            best_top1_recall = max(best_top1_recall, top1_recall)
            if rank == 0:
                print(f"Best best_top1_recall at epoch {epoch} of the network on the validation set: {best_top1_recall}")
                print(f"Top1_recall at epoch {epoch} of the network on the validation set: {top1_recall}")
                print(f"Val Loss at epoch {epoch} of the network on the validation set: {val_loss}")
                wandb.log({f"VAL-LOSS": val_loss,
                           f"Top1-Recall": top1_recall,
                           f"BEST-Top1-Recall": best_top1_recall},
                           step=epoch)

        # ============ Training one epoch of finetuning ... ============
        train_global_avg_stats = prepare_trainers.train_for_image_one_epoch(rank, epoch, training_params[mode]['num_epochs'],
                                                                            model, loss, train_dataloader, optimizer, lr_schedule)


        # Log the number of training loss in Tensorboard, at every epoch
        if rank == 0:
            print('Training one epoch is done, start writing loss and learning rate in wandb...')
            wandb.log({f"LOSS": train_global_avg_stats['loss'],
                       f"LEARNING RATE": train_global_avg_stats['lr'],
                       f"WEIGHT DECAY": train_global_avg_stats['wd']},
                      step=epoch+1)

            if train_global_avg_stats['loss'] > 10:
                wandb.alert(
                    title='High loss',
                    text=f"Loss {train_global_avg_stats['loss']} is above the acceptable threshold {10}",
                    level=AlertLevel.WARN,
                    wait_duration=timedelta(minutes=5)
                )

        # ============ Evaluating the classification performance ... ============
        if (epoch + 1) % training_params[mode]['val_freq'] == 0 or (epoch + 1) == training_params[mode]['num_epochs']:
            val_loss, top1_recall = evaluation.cifar_validate_network(rank, val_dataloader, model, wandb,
                                                                      dataset_params, log_img=False)
            best_top1_recall = max(best_top1_recall, top1_recall)
            if rank == 0:
                print(
                    f"Best best_top1_recall at epoch {epoch+1} of the network on the validation set: {best_top1_recall}")
                print(f"Top1_recall at epoch {epoch+1} of the network on the validation set: {top1_recall}")
                print(f"Val Loss at epoch {epoch+1} of the network on the validation set: {val_loss}")
                wandb.log({f"VAL-LOSS": val_loss,
                           f"Top1-Recall": top1_recall,
                           f"BEST-Top1-Recall": best_top1_recall},
                            step=epoch+1)

        # ============ Saving the model ... ============
        save_dict = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}

        if rank == 0:
            torch.save(save_dict,
                       os.path.join(save_params['model_path'], f'checkpoint.pth'))
            if save_params['saveckp_freq'] and (epoch + 1) % save_params['saveckp_freq'] == 0:
                torch.save(save_dict, os.path.join(save_params['model_path'],
                                                   f'checkpoint{epoch + 1:04}.pth'))


def main(rank, args):
    # ============ Define some parameters for easy access... ============
    _ = set_globals(rank, args)

    # ============ Setting up system configuration ... ============
    # Set gpu params and random seeds for reproducibility
    utils.set_sys_params(system_params['random_seed'])

    # ============ Getting data ready ... ============
    rank, train_dataloader, val_dataloader = get_data(rank)

    # ============ Getting model and loss ready ... ============
    rank, model, loss = get_model_loss(rank)

    if mode == 'train':
        # ============ Getting optimizer ready ... ============
        optimizer = get_optimizer(model)

        # ============ Getting schedulers ready ... ============
        lr_schedule = get_schedulers(train_dataloader)

        if rank == 0:
            print(f"Loss, optimizer and schedulers ready.", end="\n\n")

        # ============ Start training process ... ============
        train_process(rank, train_dataloader, val_dataloader, model, loss, optimizer, lr_schedule)


if __name__ == '__main__':
    # args = prepare_args(default_params_path='yaml/default_config.yaml')
    # args = prepare_args(default_params_path='yaml/default_config_change_filter_num.yaml')
    # args = prepare_args(default_params_path='yaml/connected_layer_config.yaml')
    # args = prepare_args(default_params_path='yaml/default_config_change_filter_num_size.yaml')
    # args = prepare_args(default_params_path='yaml/default_config_change_filter_num_leakyrelu.yaml')
    # args = prepare_args(default_params_path='yaml/default_config_change_filter_num_leakyrelu_dropout.yaml')
    # args = prepare_args(default_params_path='yaml/default_config_change_filter_num_leakyrelu_dropout_bn.yaml')
    args = prepare_args(default_params_path='yaml/default_config_change_filter_num_leakyrelu_dropout_bn_bs.yaml')
    # args = prepare_args(default_params_path='yaml/default_config_change_filter_num_leakyrelu_dropout_bn_lr.yaml')
    # args = prepare_args(default_params_path='yaml/default_config_change_filter_num_leakyrelu_dropout_bn_shuffling.yaml')
    utils.launch(main, args)