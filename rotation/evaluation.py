import json
import logging
import os
import random
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
# Set environment variable to allow expandable CUDA memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.utils.data import DataLoader, SequentialSampler

from evaluation_utils import (
    convert_to_angle_deg, 
    mean_abs_error, 
    median_abs_error, 
    acc_less_than,
    bootstrapper
)
from training_utils import RotationDataset, seed_worker
from ViT import ViT

# define parameters and paths
SEED = 12345

subset = 'val'
experiment_folders = []

if __name__ == '__main__':

    # configure logging
    start = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    logging.basicConfig(
        level=logging.INFO,
        filename=f'{start}_evaluation_log.txt',
        format='%(asctime)s - %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        encoding='utf-8',
    )
    logger = logging.getLogger(__name__)

    # loop over experiment folders
    for experiment_folder in experiment_folders:

        # Set the random seed for reproducibility
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(f'Start evaluation for {os.path.split(experiment_folder)[1]}')
        # load training settings
        with open(os.path.join(experiment_folder, 'config.json'), 'r') as f:
            config = json.loads(f.read())
        config["model"]["tile_dropout_prob"] = 0.0
        logger.info(json.dumps(config))

        # load subset selection
        with open(config['selection_path'], 'r') as f:
            selection_dict = json.loads(f.read())

        # load patient mapping
        with open(config['mapping_path'], 'r') as f:
            mapping_dict = json.loads(f.read())

        # load labels
        for label_path in config['labels_path']:
            label_path = Path(label_path)
            with open(label_path, 'r') as f:
                labels = json.loads(f.read())
            # select the labels for the specific sets
            subset_labels = {}
            for name, angle in labels.items():
                case = '_'.join(name.split('_')[3:5])
                subset_case = selection_dict[mapping_dict[case]]
                if subset_case == subset:
                    subset_labels[name] = angle

            # load the images, masks, and corresponding rotations
            dataset = RotationDataset(
                image_folder=config['unnorm_image_folder'], 
                segmentation_folder=config['unnorm_segmentation_folder'],
                labels = subset_labels,
                iterations = len(subset_labels),
                return_filename=True,
                return_angle=True,
                config = {
                    'apply_augmentation': False,
                    'label_type': config['label_type'],
                    'gaussian_stdev': config['gaussian_stdev'],
                    'loss_function': config['loss_function'],
                    'n_classes': config['model']['n_classes'],
                },
            )
            dataloader = DataLoader(
                dataset=dataset, 
                sampler= SequentialSampler(dataset),
                batch_size=1,
                worker_init_fn=seed_worker,
                num_workers=0, 
                pin_memory=False, 
            )

            # define device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f'Device: {device}')
            logger.info(f'Images in {subset} set: {len(subset_labels)}')

            # Initialize the model
            model = ViT(
                **config['model'],
            )
            # Load the pretrained weights
            last_checkpoint = sorted([name for name in os.listdir(experiment_folder) if (name.endswith('.tar') or name.endswith('.pth'))])[-1]
            model_checkpoint = torch.load(os.path.join(experiment_folder, last_checkpoint))
            if 'model_state_dict' in model_checkpoint:
                state_dict = model_checkpoint['model_state_dict']
            else:
                state_dict = model_checkpoint
            model.load_state_dict(state_dict, strict=True)
            logger.info(f'Loaded checkpoint: {last_checkpoint}')

            # bring model to device(s)
            model.to(device)
            model.eval()
            
            filenames = []
            angles = []
            predictions = []
            angle_predictions = []
            abs_diffs = []
            
            # loop over the images in the test set
            with torch.no_grad():
                if config['label_type'] == 'binary_prob':
                    # loop over images in the dataset
                    for i in range(len(dataset)):
                        pred = []
                        # loop over the offsets
                        for offset in range(360):
                            if offset == 0:
                                filename, image, _, pos, angle = dataset.__getitem__(i, offset)
                            else:
                                _, image, _, pos, _ = dataset.__getitem__(i, offset)
                            # bring to device
                            image = image[None, ...].to(device)
                            pos = pos[None, ...].to(device)
                            # get model prediction
                            single_pred = model(image, pos).to('cpu')
                            pred.append(single_pred)
                        pred = torch.sigmoid(torch.tensor(pred)[None, ...])      

                        angle_pred = convert_to_angle_deg(pred, config['label_type'])
                        abs_diff = min(abs(angle-angle_pred), 360-abs(angle-angle_pred))

                        filenames.append(filename)
                        angles.append(angle)
                        predictions.append(pred)
                        angle_predictions.append(angle_pred)
                        abs_diffs.append(abs_diff)
                        logger.info(f'{i} - {filename} - Angle: {angle:0.2f}, Pred: {angle_pred:0.2f}')         
                else:
                    for i, (filename, image, _, pos, angle) in enumerate(dataloader):
                        # get the image name, image, position matrix, and angle
                        image = image.to(device)
                        pos = pos.to(device)
                        angle = angle.to(device).item()

                        # forward pass, compute loss and backpropagate
                        pred = model(image, pos).to('cpu')
                        if config['loss_function'] == 'BCE':
                            pred = torch.sigmoid(pred[0:int(pred.shape[1]/2)])
                        elif config['loss_function'] == 'CCE':
                            pred = torch.softmax(pred, dim=1)
                        
                        angle_pred = convert_to_angle_deg(pred, config['label_type'])
                        abs_diff = min(abs(angle-angle_pred), 360-abs(angle-angle_pred))

                        filenames.append(filename[0])
                        angles.append(angle)
                        predictions.append(pred)
                        angle_predictions.append(angle_pred)
                        abs_diffs.append(abs_diff)
                        logger.info(f'{i} - {filename} - Angle: {angle:0.2f}, Pred: {angle_pred:0.2f}')

            metrics = [
                mean_abs_error,
                median_abs_error,
                partial(acc_less_than, threshold=2.5),
                partial(acc_less_than, threshold=5),
                partial(acc_less_than, threshold=10),
            ]
            
            results = {
                'mean_abs_error': [mean_abs_error(angles, angle_predictions)],
                'median_abs_error': [median_abs_error(angles, angle_predictions)],
                'acc_less_than_2.5': [acc_less_than(angles, angle_predictions, threshold=2.5)],
                'acc_less_than_5': [acc_less_than(angles, angle_predictions, threshold=5)],
                'acc_less_than_10': [acc_less_than(angles, angle_predictions, threshold=10)],
            }
            bootstrap_results = bootstrapper(
                y_true=angles,
                y_pred=angle_predictions,
                eval_func=metrics,
                iterations=10000,
                confidence_level=0.95,
                seed=SEED,
                show_progress=False,
            )
            
            stain_name = label_path.stem.split('_')[-1]
            # prepare dataframes
            df = pd.DataFrame.from_dict({
                'filename': filenames,
                'prediction': predictions,
                'angle_prediction': angle_predictions,
                'angle': angles,
                'abs_diff': abs_diffs
            })
            df_results = pd.DataFrame.from_dict(results)
            df_bootstrap = pd.DataFrame.from_dict(bootstrap_results)

            # write results to Excel file
            with pd.ExcelWriter(os.path.join(experiment_folder, f'{subset}_{stain_name}_results.xlsx')) as writer:
                df.to_excel(writer, sheet_name='predictions', index=False)
                df_results.to_excel(writer, sheet_name='results', index=False)
                df_bootstrap.to_excel(writer, sheet_name='bootstrap_results', index=False)
            logger.info(f'Finished {stain_name} evaluation for {os.path.split(experiment_folder)[1]}')