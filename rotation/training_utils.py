import math
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class RotationDataset(Dataset):

    def __init__(
                self, 
        image_folder: Union[str, Path], 
        segmentation_folder: Union[str, Path], 
        labels: dict[str, float],
        iterations: int,
        config: dict[str, Any],
        return_filename: bool = False,
        return_angle: bool = False,
    ) -> None:
        """
        Initializes the Rotation Dataset instance with the specified parameters.

        Args:
            image_folder: The path to the folder containing images.
            segmentation_folder: The path to the folder containing segmentations.
            labels: A dictionary mapping filenames to their corresponding labels.
            iterations: The number of iterations for the training process.
            config: A configuration dictionary containing additional settings.
            return_filename: Flag indicating whether to return the filename. Defaults to False.
            return_angle: Flag indicating whether to return the angle. Defaults to False.
        """
        self.image_folder = Path(image_folder)
        self.segmentation_folder = Path(segmentation_folder)
        self.labels = labels
        self.filenames = list(self.labels.keys())
        self.iterations = iterations
        self.config = config
        self.return_filename = return_filename
        self.return_angle = return_angle

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx, predefined_angle=0):
        # get the selected filename
        filename = self.filenames[idx % len(self.filenames)]
        
        # construct the paths the to the image and segmentation
        image_path = self.image_folder/filename
        segmentation_path = self.segmentation_folder/filename
        
        # load the image and segmentation
        image = Image.open(image_path).convert("RGB")
        segmentation = Image.open(segmentation_path).convert("L")

        # apply random rotation, flipping, color augmentations if enabled,
        # otherwise only load the image and apply padding
        if self.config['apply_augmentation']:
            image, segmentation, angle_deg = self.augment_image(image, segmentation)
        else:
            angle_deg = (self.labels[filename]-predefined_angle) % 360
            image, segmentation = rotate_image_with_correct_padding(image, segmentation, predefined_angle)

        # Convert angle in degrees to vector
        angle_rad = angle_deg * math.pi / 180.0
        if self.config['label_type'] == 'radians':
            angle_vector = torch.tensor([angle_rad])
        elif self.config['label_type'] == 'coordinate':
            angle_vector = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=torch.float32)
        elif self.config['label_type'] == 'binary_prob':
            prob = math.exp(-0.5*(min(angle_deg, 360-angle_deg)**2)/(self.config['gaussian_stdev']**2))
            angle_vector = torch.tensor([prob], dtype=torch.float32)
        elif self.config['label_type'] == 'class_prob':
            classes = torch.linspace(0, 360, self.config['n_classes']+1)[:-1]-angle_deg
            angle_vector = torch.exp(-0.5*(torch.minimum(classes, 360-classes)**2)/(self.config['gaussian_stdev']**2))
            if self.config['loss_function'] == 'CCE':
                angle_vector /= torch.sum(angle_vector)
        else:
            raise NotImplementedError

        # convert to torch tensor and normalize
        image = torch.from_numpy(np.transpose(np.array(image), (2,0,1)))/255

        # Get the position of each patch in the image
        pos = get_pos(image, segmentation)      

        # prepare output
        output = [image, angle_vector, pos]
        if self.return_filename:
            output = [filename]+output
        if self.return_angle:
            output = output+[angle_deg]
        return tuple(output)

    def augment_image(self, image, segmentation):
        """ 
        Augment the image and segmentation by addjusting brightness, contrast, saturation and hue. Also the image is flipped vertically with
        a probability of 50%. The image and segmentation are rotated by. If the extra small angles are set to True, the angle is biased towards 0 
        degrees with a probability of 25%. Otherwise, the angle is uniformly distributed between 10 and 350 degrees.
        
        Parameters:
            img (PIL.Image): Image to be augmented.
            segmentation (PIL.Image): Segmentation to be augmented.
        
        Returns:
            img (PIL.Image): Augmented image.
            segmentation (PIL.Image): Augmented segmentation.
            angle (float): Rotation of image and segmentation. 
        """
        # Adjust brightness, contrast, saturation and hue
        transform = torchvision.transforms.ColorJitter(
            brightness=self.config['brightness'], 
            contrast=self.config['contrast'], 
            saturation=self.config['saturation'], 
            hue=self.config['hue'],
        )
        image = transform(image) 

        # Randomly flip the image and segmentation vertically
        flip_image_vertically = random.choice([True, False]) # Probability of setting Boolean to True is 50%
        if flip_image_vertically: 
            image = ImageOps.mirror(image) 
            segmentation = ImageOps.mirror(segmentation)

        # Rotate the image and segmentation based on the calculated angle
        if self.config['oversample_small_angles']:
            if random.random() < self.config['small_angle_prob']:
                # oversample near 0°
                if random.random() < 0.5:
                    angle = random.random()*(self.config['small_angle_range']/2)
                else:
                    angle = ((360-(self.config['small_angle_range']/2))
                             + (random.random()*(self.config['small_angle_range']/2)))
            else:
                # General angle range
                angle = ((self.config['small_angle_range']/2)
                         + (random.random()*(360-self.config['small_angle_range'])))
        else:
            angle = random.random()*360

        image, segmentation = rotate_image_with_correct_padding(image, segmentation, -angle)

        return image, segmentation, angle


def get_centroid_of_segmentation(segmentation):
    """ 
    Get the centroid of the segmentation.

    Parameters:
    segmentation (PIL.Image): The segmentation to find the centroid of.

    Returns:
    centroid (tuple): The (x, y) coordinates of the centroid. 
    """
    # Convert to numpy array
    segmentation_array = np.array(segmentation)

    # Get coordinates of foreground pixels (non-zero)
    y_indices, x_indices = np.where(segmentation_array > 0)

    # Compute centroid
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Segmentation is empty — no foreground pixels found.")

    # Compute centroid
    x_centroid = x_indices.mean()
    y_centroid = y_indices.mean()
    centroid = (x_centroid, y_centroid)
    return centroid


def crop_to_segmentation(image, segmentation):
    """ 
    Crop the image and segmentation to the bounding box of the segmentation.
    
    Parameters:
    image (PIL.Image): The image to be cropped.
    segmentation (PIL.Image): The segmentation to be used for cropping.
    
    Returns:
    cropped_image (PIL.Image): The cropped image.
    cropped_segmentation (PIL.Image): The cropped segmentation. 
    """
    # Calculate the bounding box of the segmentation
    np_segmentation = np.array(segmentation)
    y_indices, x_indices = np.where(np_segmentation > 0)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return image, segmentation  # Avoid cropping everything
    left, right = x_indices.min(), x_indices.max()
    top, bottom = y_indices.min(), y_indices.max()
    box = (left, top, right + 1, bottom + 1)

    # Crop the image and segmentation to the bounding box of the segmentation
    cropped_image = image.crop(box)
    cropped_segmentation = segmentation.crop(box)

    return cropped_image, cropped_segmentation


def pad_to_fit_patches_with_centroid(image, segmentation, centroid, patch_size=16, bg_color=(245, 245, 245, 255)):
    """ 
    Pad the image to patchifyable size, such the the centroid is in the middle of patch (0,0). 
    Since there is no center pixel in the patch (16, 16), the centroid is pixel (8,8) (right bottom of 4 center pixels).
    
    Parameters:
        image (PIL.Image): The image to be padded.
        segmentation (PIL.Image): The segmentation to be padded.
        centroid (tuple): The (x, y) coordinates of the centroid.
        patch_size (int): The size of the patches. Default is 16.
        bg_color (tuple): The background color to use for padding. Default is (245, 245, 245, 255).
    
    Returns:
        padded_image (PIL.Image): The padded image.
        padded_segmentation (PIL.Image): The padded segmentation. 
    """
    # get image size
    w, h = image.size

    # Get the centroid coordinates
    cx, cy = centroid

    # Convert to pixel value
    cx = math.floor(cx)
    cy = math.floor(cy)

    # Caclulate target image dimensions
    number_extra_pixels_left = cx % patch_size
    if number_extra_pixels_left < patch_size // 2:
        add_left = patch_size // 2 - number_extra_pixels_left
    else:
        add_left = patch_size + patch_size // 2 - number_extra_pixels_left

    number_extra_pixels_top = cy % patch_size
    if number_extra_pixels_top < patch_size // 2:
        add_top = patch_size // 2 - number_extra_pixels_top
    else:
        add_top = patch_size + patch_size // 2 - number_extra_pixels_top

    number_extra_pixels_right = (w - cx) % patch_size
    if number_extra_pixels_right < patch_size // 2:
        add_right = patch_size // 2 - number_extra_pixels_right
    else:
        add_right = patch_size + patch_size // 2 - number_extra_pixels_right

    number_extra_pixels_bottom = (h - cy) % patch_size
    if number_extra_pixels_bottom < patch_size // 2:
        add_bottom = patch_size // 2 - number_extra_pixels_bottom
    else:
        add_bottom = patch_size + patch_size // 2 - number_extra_pixels_bottom

    # Create new padded image and segmentation
    padded_image = ImageOps.expand(image, border=(add_left, add_top, add_right, add_bottom), fill=bg_color)
    padded_segmentation = ImageOps.expand(segmentation, border=(add_left, add_top, add_right, add_bottom), fill=0)  

    return padded_image, padded_segmentation


def rotate_image_with_correct_padding(image, segmentation, angle, bg_color=(245, 245, 245, 255)):
    """ 
    Rotate the image by the given angle and return the rotated image. The background 
    color is set to (245, 245, 245). The image is centered on a square canvas.

    Parameters:
    image (PIL.Image): The image to be rotated.
    segmentation (PIL.Image): The segmentation to be rotated.
    angle (float): The angle to rotate the image.
    bg_color (tuple): The background color to use for padding. Default is (245, 245, 245, 255).

    Returns:
    padded_image (PIL.Image): The rotated image with padding.
    padded_segmentation (PIL.Image): The rotated segmentation with padding. 
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # Remove background if not already remved
    background = Image.new("RGBA", image.size, bg_color)
    image = Image.composite(image, background, segmentation)
    
    # rotate the image if the angle is not equal to zero
    if angle == 0:
        cropped_image = image
        cropped_segmentation = segmentation
    else:
        rotated_image = image.rotate(-angle, expand=True, resample=Image.BILINEAR, fillcolor=bg_color)
        rotated_segmentation = segmentation.rotate(-angle, expand=True, resample=Image.NEAREST, fillcolor=0)

        # Trim the image and segmentation to remove unnecessary padding
        cropped_image, cropped_segmentation = crop_to_segmentation(rotated_image, rotated_segmentation)

    # Get the centroid of the segmentation
    centroid = get_centroid_of_segmentation(cropped_segmentation)

    # Pad the image and segmentation to next multiple of 16 with the centroid in the center of a patch
    padded_image, padded_segmentation = pad_to_fit_patches_with_centroid(cropped_image, cropped_segmentation, centroid, bg_color=bg_color)

    # Convert to RGB (remove alpha) and save
    padded_image = padded_image.convert("RGB")

    return padded_image, padded_segmentation


def get_pos(image, segmentation, patch_size=16):
    """ 
    Returns the position of each patch in the image. The position is calculated based on the centroid of the segmentation.

    Parameters:
        img (torch.Tensor): Image for position matrix needs to be calculated.
        segmentation (PIL.Image): Segmentation of the image.
        patch_size (int): Size of the patches. Default is 16.
    
    Returns:
        pos (torch.Tensor): Position matrix of the patches. 
    """
    # Calculate centroid and convert to pixel value
    centroid = get_centroid_of_segmentation(segmentation)
    cx = math.floor(centroid[0])
    cy = math.floor(centroid[1])

    # Get the size of the image and number of patches
    _, H, W = image.shape 
    h_patches = H // patch_size
    w_patches = W // patch_size

    # Convert pixel centroid to patch coords
    centroid_patch_x = int(cx // patch_size)
    centroid_patch_y = int(cy // patch_size)

    x_range = torch.arange(w_patches) - centroid_patch_x
    y_range = -(torch.arange(h_patches) - centroid_patch_y)

    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    pos = torch.stack([y_grid, x_grid], dim=2).reshape(-1,2)  # Stack into (H, W, 2) and then flatten to (H*W, 2)

    return pos


def seed_worker(worker_id):
    """ Set the random seed for each worker to ensure reproducibility. The seed is set based on the global RANDOM_SEED and the worker_id.
    
    Parameters:
        worker_id (int): ID of the worker.
    
    Returns:
        None. Sets the random seed for the worker. """
    np.random.seed(worker_id)
    random.seed(worker_id)


class CCELoss(nn.Module):

    def __init__(
        self,
        gamma: float = 0.0,
        aggregation='mean',
    ) -> None:
        """
        Initialize the cross-entropy loss.

        Args:
            gamma:  Parameter that governs the relative importance of incorrect 
                predictions. If gamma equals 0.0, the cross-entropy loss is equal to the 
                cross-entropy loss.
        """
        super().__init__()
        self.gamma = gamma
        if aggregation == 'mean':
            self.aggregation = torch.mean
        elif aggregation == 'sum':
            self.aggregation = torch.sum
        else:
            ValueError('Invalid aggregation function')

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            logit:  Logit predictions volumes of shape: (batch, logit).
            y_true:  True label volumes of matching shape: (batch, prob).
        
        Returns:
            loss:  Cross-entropy loss averaged over all instances in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
        
        # check if the values of y_true range between 0.0-1.0
        if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
            raise ValueError('Invalid values for y_true (outside the range 0.0-1.0).')
        
        # calculate the cross-entropy
        log_y_pred = F.log_softmax(logit, dim=1)
        cross_entropy = -(log_y_pred * y_true)
        instance_loss = self.aggregation(cross_entropy, dim=-1)

        # compute the mean loss over the batch
        loss = torch.mean(instance_loss)

        return loss


class BCELoss(nn.Module):

    def __init__(
        self,
        gamma: float = 0.0,
        aggregation='mean',
    ) -> None:
        """
        Initialize the cross-entropy loss.

        Args:
            gamma:  Parameter that governs the relative importance of incorrect 
                predictions. If gamma equals 0.0, the cross-entropy loss is equal to the 
                cross-entropy loss.
        """
        super().__init__()
        self.gamma = gamma
        if aggregation == 'mean':
            self.aggregation = torch.mean
        elif aggregation == 'sum':
            self.aggregation = torch.sum
        else:
            ValueError('Invalid aggregation function')

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            logit:  Logit predictions volumes of shape: (batch, logit).
            y_true:  True label volumes of matching shape: (batch, prob).
        
        Returns:
            loss:  Cross-entropy loss averaged over all instances in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
        
        # check if the values of y_true range between 0.0-1.0
        if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
            raise ValueError('Invalid values for y_true (outside the range 0.0-1.0).')
        
        # calculate the cross-entropy        
        binary_cross_entropy = -(y_true*torch.log(torch.sigmoid(logit)) 
                                 + (1-y_true)*torch.log(1-torch.sigmoid(logit)))
        instance_loss = self.aggregation(binary_cross_entropy, dim=-1)

        # compute the mean loss over the batch
        loss = torch.mean(instance_loss)

        return loss
    

class MSELoss(nn.Module):

    def __init__(self, aggregation='mean') -> None:
        super().__init__()
        if aggregation == 'mean':
            self.aggregation = torch.mean
        elif aggregation == 'sum':
            self.aggregation = torch.sum
        else:
            ValueError('Invalid aggregation function')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            y_pred:  Predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  MSE loss averaged over all images in the batch.
        """
        # check if the prediction and true labels are of equal shape
        if y_pred.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
                
        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the mean squared error
        MSE = self.aggregation((y_true_flat-y_pred_flat)**2, dim=-1)

        # compute the mean loss over the batch
        instance_loss = torch.sum(MSE, dim=1)
        loss = torch.mean(instance_loss)

        return loss
    

class MAELoss(nn.Module):

    def __init__(self, aggregation='mean') -> None:
        super().__init__()
        if aggregation == 'mean':
            self.aggregation = torch.mean
        elif aggregation == 'sum':
            self.aggregation = torch.sum
        else:
            ValueError('Invalid aggregation function')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            y_pred:  Predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  MSE loss averaged over all images in the batch.
        """
        # check if the prediction and true labels are of equal shape
        if y_pred.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
                
        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the mean squared error
        MAE = self.aggregation(torch.abs(y_true_flat-y_pred_flat), dim=-1)

        # compute the mean loss over the batch
        instance_loss = torch.sum(MAE, dim=1)
        loss = torch.mean(instance_loss)

        return loss


class CSLoss(nn.Module):

    def __init__(self, aggregation='mean') -> None:
        super().__init__()
        if aggregation == 'mean':
            self.aggregation = torch.mean
        elif aggregation == 'sum':
            self.aggregation = torch.sum
        else:
            ValueError('Invalid aggregation function')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            y_pred:  Predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  MSE loss averaged over all images in the batch.
        """
        # check if the prediction and true labels are of equal shape
        if y_pred.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
                
        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the mean cosine similarity loss
        CS = self.aggregation(1-torch.cos(y_true_flat-y_pred_flat), dim=-1)

        # compute the mean loss over the batch
        instance_loss = torch.sum(CS, dim=1)
        loss = torch.mean(instance_loss)

        return loss
    

class CSLoss2(nn.Module):

    def __init__(self, aggregation='mean') -> None:
        super().__init__()
        if aggregation == 'mean':
            self.aggregation = torch.mean
        elif aggregation == 'sum':
            self.aggregation = torch.sum
        else:
            ValueError('Invalid aggregation function')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            y_pred:  Predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  MSE loss averaged over all images in the batch.
        """
        # check if the prediction and true labels are of equal shape
        if y_pred.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')

        # add an additional dimension
        if len(y_true.shape) == 2:
            y_true = y_true[None, ...]
        if len(y_pred.shape) == 2:
            y_pred = y_pred[None, ...]

        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the mean cosine similarity loss
        CS = self.aggregation(1 - (
            torch.sum(y_true_flat*y_pred_flat, dim=-1, keepdim=True)
            / ((torch.sum(y_true_flat**2, dim=-1, keepdim=True)**0.5)
            * (torch.sum(y_pred_flat**2, dim=-1, keepdim=True)**0.5))
        ), dim=-1)

        # compute the mean loss over the batch
        instance_loss = torch.sum(CS, dim=1)
        loss = torch.mean(instance_loss)

        return loss