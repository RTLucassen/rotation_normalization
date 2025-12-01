"""
Contains model training loop.
"""

import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Set environment variable to allow expandable CUDA memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torchinfo
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from training_utils import (
    CSLoss,
    CSLoss2,
    BCELoss,
    CCELoss,
    MSELoss, 
    MAELoss,
    RotationDataset, 
    seed_worker, 
)
from ViT import ViT, convert_state_dict

if True:
    config = {
        "experiment_name": "001_experiment",
        "experiments_folder": "",
        "norm_image_folder": "",
        "norm_segmentation_folder": "",
        "unnorm_image_folder": "",
        "unnorm_segmentation_folder": "",
        "labels_path": [
            "image_rotations_HE.json",
        ],
        "selection_path": "assigned_split.json",
        "mapping_path": "patient_mapping.json",
        "seed": 1,
        "iterations": 750000,
        "iterations_per_update": 20,
        "iterations_per_checkpoint": 5000,
        "save_all_checkpoints": False,
        "label_type": "binary_prob",
        "gaussian_stdev": 2.5, # degrees
        "augmentation": {
            "oversample_small_angles": True,
            "small_angle_prob": 0.25,
            "small_angle_range": 30, # degrees
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.05,
        },
        "learning_rate": 2e-5,
        "weight_decay": 1e-2,
        "lr_patience": 5,
        "lr_factor": 0.5,
        "loss_function": "BCE",
        "model": {
            "patch_shape": 16,
            "input_dim": 3,
            "embed_dim": 256,
            "n_classes": 1,
            "depth": 14,
            "n_heads": 4,
            "mlp_ratio": 5,
            "pytorch_attn_imp": False,
            "init_values": 1e-5,   
            "max_tiles": 15000,
            "tile_dropout_prob": 0.0,
        },
        "pretrained_weights_path": "vit_wee_patch16_reg1_gap_256.sbb_in1k.pth",
        "num_workers": 4,
        "pin_memory": False,
    }

if __name__ == '__main__':

    # create experiment folder if it does not exist yet
    experiment_folder = os.path.join(config['experiments_folder'], config['experiment_name'])
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    else:
        raise FileExistsError('Experiment folder already exists')
    
    # configure logging
    start = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    logging.basicConfig(
        level=logging.INFO,
        filename=f'{start}_{config["experiment_name"]}_log.txt',
        format='%(asctime)s - %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        encoding='utf-8',
    )
    logger = logging.getLogger(__name__)

    # save training settings
    with open(os.path.join(experiment_folder, 'config.json'), 'w') as f:
        f.write(json.dumps(config))
    logger.info(json.dumps(config))

    # Set the random seed for reproducibility
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load subset selection
    with open(config['selection_path'], 'r') as f:
        selection_dict = json.loads(f.read())

    # load patient mapping
    with open(config['mapping_path'], 'r') as f:
        mapping_dict = json.loads(f.read())

    # load labels
    labels = {}
    for label_path in config['labels_path']:
        with open(label_path, 'r') as f:
            labels = {**labels, **json.loads(f.read())}
    # select the labels for the specific sets
    train_labels = {}
    val_labels = {}
    for name, angle in labels.items():
        case = '_'.join(name.split('_')[3:5])
        subset = selection_dict[mapping_dict[case]]
        if subset == 'train':
            train_labels[name] = angle
        elif subset == 'val':
            val_labels[name] = angle

    # load the images, masks and corresponding rotations
    train_dataset = RotationDataset(
        image_folder=config['norm_image_folder'], 
        segmentation_folder=config['norm_segmentation_folder'],
        labels = train_labels,
        iterations = config['iterations'],
        config = {
            'apply_augmentation': True,
            'label_type': config['label_type'],
            'gaussian_stdev': config['gaussian_stdev'],
            'loss_function': config['loss_function'],
            'n_classes': config['model']['n_classes'],
            **config['augmentation'],
        }
    )
    val_dataset = RotationDataset(
        image_folder=config['unnorm_image_folder'], 
        segmentation_folder=config['unnorm_segmentation_folder'],
        labels = val_labels,
        iterations = len(val_labels),
        config = {
            'apply_augmentation': False,
            'label_type': config['label_type'],
            'gaussian_stdev': config['gaussian_stdev'],
            'loss_function': config['loss_function'],
            'n_classes': config['model']['n_classes'],
        }
    )
    # create dataloader for training and validation set
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        sampler= RandomSampler(train_dataset, replacement=True),
        batch_size=1,
        worker_init_fn=seed_worker,
        num_workers=config['num_workers'], 
        pin_memory=config['pin_memory'], 
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        sampler= SequentialSampler(val_dataset),
        batch_size=1,
        worker_init_fn=seed_worker,
        num_workers=config['num_workers'], 
        pin_memory=config['pin_memory'], 
    ) 

    # define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')
    logger.info(f'Images in training set: {len(train_labels)}')
    logger.info(f'Images in validation set: {len(val_labels)}')

    # Initialize the model
    model = ViT(
        **config['model'],
    )
    # Load the pretrained weights
    if config['pretrained_weights_path'] is not None:
        state_dict = torch.load(config['pretrained_weights_path'])

        # Convert the state dict to match the model
        converted_state_dict = convert_state_dict(state_dict=state_dict)
        converted_state_dict['pos_embedder.pos_embed'] = model.state_dict()['pos_embedder.pos_embed']
        converted_state_dict.pop('classifier.weight', None)
        converted_state_dict.pop('classifier.bias', None)
        model.load_state_dict(converted_state_dict, strict=False)
    logger.info(torchinfo.summary(model))
    
    # define optimizer and learning rate scheduler
    learning_rate = config['learning_rate']
    optimizer = AdamW(
        params=model.parameters(), 
        lr=learning_rate, 
        weight_decay=config['weight_decay'],
    )
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer, 
        mode='min', 
        factor=config['lr_factor'], 
        patience=config['lr_patience'],
    )

    # bring model to device(s)
    model.to(device)

    # define loss functions
    if config['loss_function'] == 'MSE':
        loss_function = MSELoss()
    elif config['loss_function'] == 'MAE':
        loss_function = MAELoss()
    elif config['loss_function'] == 'CS':
        loss_function = CSLoss()
    elif config['loss_function'] == 'CS2':
        loss_function = CSLoss2()
    elif config['loss_function'] == 'BCE':
        loss_function = BCELoss(aggregation='mean')
    elif config['loss_function'] == 'CCE':
        loss_function = CCELoss(aggregation='sum')
    else:
        raise NotImplementedError

    accumulated_loss = []
    train_loss = []
    train_index = []
    val_loss = []
    val_index = []
    lr_values = []
    best_validation_loss = None
    for i, (image, label, pos) in enumerate(train_dataloader):
        index = i+1

        # ---------------- TRAINING -------------------

        # get the image, label and position matrix
        image = image.to(device)
        pos = pos.to(device)
        label = label.to(device)

        # forward pass, compute loss and backpropagate
        pred = model(image, pos)
        # compute loss
        loss = loss_function(pred, label) / config['iterations_per_update']
        accumulated_loss.append(loss.item())

        # perform the backwards pass
        loss.backward()
        
        # update weights
        if (index) % config['iterations_per_update'] == 0:
            # update the network parameters and reset the gradient
            optimizer.step()
            optimizer.zero_grad() # set the gradient to 0 again

            # calculate the training loss for the batch
            train_loss.append(sum(accumulated_loss))
            train_index.append(index)
            accumulated_loss = []
            logger.info(f'{index} - Training loss: {train_loss[-1]:0.4f}')

        # --------------- VALIDATION ------------------
        # periodically evaluate on the validation set
        if index % config['iterations_per_checkpoint'] == 0:

            # set the model in evaluation mode
            model.eval()
            prev_tile_dropout_prob = model.tile_dropout_prob
            model.tile_dropout_prob = 0.0

            loss_per_image = []
            # deactivate autograd engine (backpropagation not required here)
            with torch.no_grad():
                for j, (image, label, pos) in enumerate(val_dataloader):
                    # get the image, label, and position matrix
                    image = image.to(device)
                    pos = pos.to(device)
                    label = label.to(device)
                    # forward pass, compute loss, and backpropagate
                    pred = model(image, pos)
                    loss = loss_function(pred, label)
                    loss_per_image.append(loss.item())

            val_loss.append(sum(loss_per_image)/len(loss_per_image))
            val_index.append(index)
            logger.info(f'{index} - Validation loss: {val_loss[-1]:0.4f}')

            scheduler.step(val_loss[-1])
            lr_values.append(scheduler.optimizer.param_groups[0]['lr'])
            if len(lr_values) > 2 and lr_values[-2] != lr_values[-1]:
                logger.info(f'Learning rate was reduced to {lr_values[-1]}')

            # determine if the last model checkpoint achieved the best validation loss
            save_checkpoint = False
            if best_validation_loss is None:
                best_validation_loss = val_loss[-1]
                save_checkpoint = True
            elif best_validation_loss > val_loss[-1]:
                best_validation_loss = val_loss[-1]
                save_checkpoint = True
            elif 'save_all_checkpoints' in config:
                if config['save_all_checkpoints']:
                    save_checkpoint = True

            # save model checkpoint
            if save_checkpoint:
                torch.save({
                        'iteration': index,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_loss[-1],
                    },
                    os.path.join(experiment_folder, f'checkpoint_I{str(index).zfill(7)}.tar'),
                )
            # set the model to training mode
            model.train()
            model.tile_dropout_prob = prev_tile_dropout_prob

    # save training and validation loss as excel file
    train_loss_df = pd.DataFrame.from_dict({'index': train_index,
                                            'loss': train_loss})
    validation_loss_df = pd.DataFrame.from_dict({'index': val_index,
                                                 'loss': val_loss})
    
    # create a excel writer object
    with pd.ExcelWriter(os.path.join(experiment_folder, 'loss.xlsx')) as writer:
        train_loss_df.to_excel(writer, sheet_name='Training', index=False)
        validation_loss_df.to_excel(writer, sheet_name='Validation', index=False)

    # plot training and validation loss with linear and log y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # linear scale plot
    ax1.plot(train_index, train_loss, zorder=1,
            color='royalblue', alpha=0.40, label='Train')
    ax1.plot(val_index, val_loss,
            color='forestgreen', zorder=2, label='Validation')
    ax1.scatter(val_index, val_loss, zorder=2, marker='o',
                facecolor='white', edgecolor='forestgreen', lw=1.5, s=15)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Linear Scale')
    ax1.set_ylim(bottom=0)
    ax1.legend()

    # log scale plot
    ax2.plot(train_index, train_loss, zorder=1,
            color='royalblue', alpha=0.40, label='Train')
    ax2.plot(val_index, val_loss,
            color='forestgreen', zorder=2, label='Validation')
    ax2.scatter(val_index, val_loss, zorder=2, marker='o',
                facecolor='white', edgecolor='forestgreen', lw=1.5, s=15)
    ax2.set_xlabel('Iteration')
    ax2.set_yscale('log')
    ax2.set_title('Logarithmic Scale')
    ax2.legend()

    plt.xlim([-25, train_index[-1]+25])
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_folder, 'loss.png'), dpi=300)