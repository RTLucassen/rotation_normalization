"""
Contains evaluation utility functions.
"""

import functools
import math
import random
from typing import Callable, Union

import numpy as np
import torch
from tqdm import tqdm


def coord_to_angle_deg(v):
    """ 
    Convert a 2D coordinate to an angle in degrees.

    Parameters:
        v (torch.Tensor): A 1D or 2D tensor representing the coordinate.
    
    Returns:
        float: The angle in degrees, normalized to the range [0, 360).
    """
    if v.dim() == 1:
        v = v / (v.norm() + 1e-8)
        return float((torch.atan2(v[1], v[0]) * 180 / torch.pi % 360).item())
    else:
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        return float((torch.atan2(v[:, 1], v[:, 0]) * 180 / torch.pi % 360).item())


def class_prob_to_angle_deg(pred) -> float:
    """
    Convert class probabilities to an angle in degrees using circular mean.
    Args:
        pred: Tensor of class probabilities with shape (batch, n_classes).
    Returns:
        float: Predicted angle in degrees.
    """
    pred = pred[0, ...]
    n_classes = len(pred)
    classes = torch.linspace(0, 360, n_classes+1)[:-1]
    # calculate the probability weighted coordinate
    sum_pred = torch.sum(pred)
    mean_x = torch.sum(torch.cos(classes/180*torch.pi)*pred)/sum_pred
    mean_y = torch.sum(torch.sin(classes/180*torch.pi)*pred)/sum_pred

    return coord_to_angle_deg(torch.tensor([mean_x, mean_y]))


def convert_to_angle_deg(pred, label_type: str) -> float:
    """
    Converts predictions to angle in degrees based on the label type.
    """
    if label_type == 'radians':
        return float(pred * 180 / torch.pi % 360)
    elif label_type == 'coordinate':
        return coord_to_angle_deg(pred)
    elif label_type == 'binary_prob':
        return class_prob_to_angle_deg(pred)
    elif label_type == 'class_prob':
        return class_prob_to_angle_deg(pred)
    else:
        raise NotImplementedError
    

def mean_abs_error(
    y_true: list[int], 
    y_pred: list[float],     
) -> float:
    abs_error = [min(abs(t-p), 360-abs(t-p)) for t, p in zip(y_true, y_pred)]
    return np.mean(abs_error)


def median_abs_error(
    y_true: list[int], 
    y_pred: list[float],     
) -> float:
    abs_error = [min(abs(t-p), 360-abs(t-p)) for t, p in zip(y_true, y_pred)]
    return np.median(abs_error)


def acc_less_than(
    y_true: list[int], 
    y_pred: list[float],
    threshold: float,     
) -> float:
    abs_error = [min(abs(t-p), 360-abs(t-p)) for t, p in zip(y_true, y_pred)]
    correct = [1 if error <= threshold else 0 for error in abs_error]
    return sum(correct)/len(correct)


def bootstrapper(
    y_true: list[int], 
    y_pred: list[float], 
    eval_func: Union[Callable, list[Callable]],
    iterations: int,
    confidence_level: float,
    seed: int,
    show_progress: bool = True,
) -> dict[str, list]:
    """
    Performs bootstrap resampling to evaluate the performance of specified metrics on predicted values.
    
    Parameters:
        y_true: The true labels.
        y_pred: The predicted values.
        eval_func: A single evaluation function or a list of functions to evaluate.
        iterations: The number of bootstrap iterations.
        confidence_level: The confidence level for the confidence intervals.
        seed: Random seed for reproducibility.
        show_progress: Flag to show progress during iterations (default is True).
    
    Returns:
        bootstrap_results: A dictionary containing the mean, confidence intervals, and sample sizes for each metric.
    """
    # set seed
    random.seed(seed)

    # prepare dictionary with evaluation functions
    if not isinstance(eval_func, list):
        eval_func = [eval_func]
    eval_funcs = {}

    for function in eval_func:
        if isinstance(function, functools.partial):
            name = function.func.__name__
            if name == 'acc_less_than':
                name = f'{name}_{function.keywords["threshold"]}'
            eval_funcs[name] = function
        else:
            eval_funcs[function.__name__] = function

    # initialize iterator
    if show_progress:
        iterator = tqdm(range(iterations))
    else:
        iterator = range(iterations)

    # sample with replacement with or without stratification          
    sample_results = {name: [] for name in eval_funcs.keys()}
    for _ in iterator:
        # generate indices
        sample_indices = [random.randint(0, len(y_true)-1) for _ in range(len(y_true))]
        # select cases with replacement based on generated indices
        y_true_sampled = [y_true[i] for i in sample_indices]
        y_pred_sampled = [y_pred[i] for i in sample_indices]
        for name, func in eval_funcs.items():
            sample_results[name].append(func(y_true_sampled, y_pred_sampled))

    # calculate mean and confidence intervals for bootstrap samples
    bootstrap_results = {}
    for metric, values in sample_results.items():
        # add items to dictionary
        bootstrap_results[f'mean {metric}'] = [np.mean(values)]
        bootstrap_results[f'{confidence_level}% CI lower {metric}'] = [np.quantile(values, (1-confidence_level)/2)]
        bootstrap_results[f'{confidence_level}% CI upper {metric}'] = [np.quantile(values, 1-((1-confidence_level)/2))]
        bootstrap_results[f'N {metric}'] = [len(values)]

    return bootstrap_results