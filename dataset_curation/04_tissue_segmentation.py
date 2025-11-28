import os
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from slidesegmenter import SlideSegmenter
from slidesegmenter._model_utils import ModifiedUNet
from tqdm import tqdm


# define paths
input_directory = Path(r"")
output_directory = Path(r"")

if __name__ == '__main__':

    # initialize SlideSegmenter instance
    segmenter = SlideSegmenter(
        device='cuda', 
        model_folder='latest', 
        tissue_segmentation=True, 
        separate_cross_sections=True,
        alternative_directory=r"..\cache",
    )
    # load custom model
    segmenter._load_model(
        ModifiedUNet, 
        r"..\2025-04-15\model_state_dict.pth", 
        r"..\2025-04-15\settings.json",
    )
    # loop over images and perform the tissue segmentation
    for i, path in tqdm(enumerate(input_directory.iterdir())):
        # get filename
        filename = path.stem
        if not (output_directory/f'{filename}.zip').exists():
            try:
                # load the image
                image = sitk.GetArrayFromImage(sitk.ReadImage(path))
                # segment the tissue and pen markings
                result = segmenter.segment(image/255, return_distance_maps=True)
            except:
                continue
            else:
                # save the segmentation
                segmentation = np.concatenate([result['tissue']], axis=-1)
                sitk.WriteImage(sitk.GetImageFromArray(((segmentation*255).transpose((2,0,1)))[..., None].astype(np.uint8)), output_directory/f'{filename}.tiff')
                # save the segmentation in a compressed format and remove the original image
                with zipfile.ZipFile(output_directory/f'{filename}.zip', 'w',
                                compression=zipfile.ZIP_DEFLATED,
                                compresslevel=9) as zf:
                    zf.write(output_directory/f'{filename}.tiff', arcname=f'{filename}.tiff')
                os.remove(output_directory/f'{filename}.tiff')