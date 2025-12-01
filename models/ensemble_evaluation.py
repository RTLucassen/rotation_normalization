"""
Evaluate the performance of model ensembles.
"""

import os

import pandas as pd
from math import sin, cos, pi, atan2

from evaluation_utils import (
    mean_abs_error, 
    median_abs_error, 
    acc_less_than,
)

def circular_mean(angles):
    mean_x_coords = sum([cos(a) for a in angles])/len(angles)
    mean_y_coords = sum([sin(a) for a in angles])/len(angles)
    return atan2(mean_y_coords, mean_x_coords)

paths = []
output_folder = r""
output_file = "test_IHC_results.xlsx"

if __name__ == '__main__':

    dictionary = {}
    for i, path in enumerate(paths):
        dictionary[i] = pd.read_excel(path)

    filenames = dictionary[0]['filename'].tolist()
    angles = dictionary[0]['angle'].tolist()
    mean_angle_predictions = []
    for i in range(len(dictionary[0])):
        predictions = []
        for j in range(len(paths)):
            predictions.append(dictionary[j]['angle_prediction'][i]/180*pi)
        mean_angle_predictions.append((circular_mean(predictions)*180/pi)% 360)

    df = pd.DataFrame.from_dict({
        'filename': filenames,
        'angle_prediction': mean_angle_predictions,
        'angle': angles,
    })

    df_results = pd.DataFrame.from_dict({
        'mean_abs_error': [mean_abs_error(angles, mean_angle_predictions)],
        'median_abs_error': [median_abs_error(angles, mean_angle_predictions)],
        'acc_less_than_2.5': [acc_less_than(angles, mean_angle_predictions, threshold=2.5)],
        'acc_less_than_5': [acc_less_than(angles, mean_angle_predictions, threshold=5)],
        'acc_less_than_10': [acc_less_than(angles, mean_angle_predictions, threshold=10)],
    })

    # write results to Excel file
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with pd.ExcelWriter(os.path.join(output_folder, output_file)) as writer:
        df.to_excel(writer, sheet_name='predictions', index=False)
        df_results.to_excel(writer, sheet_name='results', index=False)