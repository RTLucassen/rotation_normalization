from slideloader import SlideLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# define paths and parameters
HE_folder = Path(r"")
IHC_folder = Path(r"")

HE_output_folder = Path(r"")
IHC_output_folder = Path(r"")

paths_file = r""

magnification = 1.25

if __name__ == '__main__':

    # initialize SlideLoader instance
    loader = SlideLoader()

    # load file with paths
    df = pd.read_json(paths_file)
    
    # load and save the HE and IHC images
    for i, row in tqdm(df.iterrows()):
        HE_path = [HE_folder/Path(p).name for p in row['HE_path']]
        loader.load_slide(HE_path)
        loader.write_image(magnification, HE_output_folder/f"{str(i).zfill(4)}_{row['IHC_stain']}_{row['pa_number']}_{row['specimen']}_{HE_path[0].stem}.png")
        
        IHC_path = [IHC_folder/Path(p).name for p in row['IHC_path']]
        loader.load_slide(IHC_path)
        loader.write_image(magnification, IHC_output_folder/f"{str(i).zfill(4)}_{row['IHC_stain']}_{row['pa_number']}_{row['specimen']}_{IHC_path[0].stem}.png")