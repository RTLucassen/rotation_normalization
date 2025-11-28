import json
from pathlib import Path

from evaluation_utils import (
    mean_abs_error, 
    median_abs_error, 
    acc_less_than,
)

folder = r""
comparisons = [
    ('HE_ET_1', 'HE_ET_2'),
    ('HE_ET_1', 'HE_RL'),
    ('HE_EP', 'HE_ET_1'),
    ('HE_EP', 'HE_RL'),
    ('IHC_ET_1', 'IHC_ET_2'),
    ('IHC_ET_1', 'IHC_RL'),
    ('IHC_EP', 'IHC_ET_1'),
    ('IHC_EP', 'IHC_RL'),
]

if __name__ == '__main__':

    folder = Path(folder)

    for comparison in comparisons:
        with open(folder/f'image_rotations_{comparison[0]}.json') as f:
            obs1 = {Path(k).name: v for k, v in json.loads(f.read()).items()} 
        with open(folder/f'image_rotations_{comparison[1]}.json') as f:
            obs2 = {Path(k).name: v for k, v in json.loads(f.read()).items()} 
    
        if len(obs1) != len(obs2):
            raise ValueError
        
        obs1_angles = []
        obs2_angles = []
        for name, angle in obs1.items():
            obs1_angles.append(angle)
            obs2_angles.append(obs2[name])

        results = {
            'mean_abs_error': [mean_abs_error(obs1_angles, obs2_angles)],
            'median_abs_error': [median_abs_error(obs1_angles, obs2_angles)],
            'acc_less_than_2.5': [acc_less_than(obs1_angles, obs2_angles, threshold=2.5)],
            'acc_less_than_5': [acc_less_than(obs1_angles, obs2_angles, threshold=5)],
            'acc_less_than_10': [acc_less_than(obs1_angles, obs2_angles, threshold=10)],
        }
        results = {k:round(v[0],3) for k, v in results.items()}
        print(comparison)
        print(results)
        print('')