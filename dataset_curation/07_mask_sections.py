from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def save_masked_image(image_path, annotation_path, output_image_path):
    """
    Save a masked copy of an image with a background intensity of (245, 245, 245)
    using a binary annotation mask.

    Parameters:
        image_path : str or pathlib.Path
            Path to the input image file readable by SimpleITK.
        annotation_path : str or pathlib.Path
            Path to the annotation/mask image file readable by SimpleITK. The annotation is expected
            to be single-channel; any nonzero value is treated as foreground.
        output_image_path : str or pathlib.Path
            Path where the masked image will be written. If a file already exists at this path,
            the function returns immediately and does not overwrite it."""
    
    if not Path(output_image_path).exists():
        # load the image and annotation
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        annotation = np.tile(sitk.GetArrayFromImage(sitk.ReadImage(annotation_path))[..., None], (1,1,1,3))
        # create a background image with intensity of (245, 245, 245),
        # which is a representative WSI background color
        background = np.ones_like(image)*245
        # change the original background of the image with the homogeneous background
        masked_image = np.where(annotation, image, background)
        # save the masked image
        sitk.WriteImage(sitk.GetImageFromArray(masked_image), output_image_path)

# define paths
image_paths = sorted(list(Path(r"").iterdir()))
anno_paths = sorted(list(Path(r"").iterdir()))
out_image_paths = [p.as_posix().replace('HE_sections', 'HE_masked_sections') for p in image_paths]

if __name__ == '__main__':

    with ThreadPoolExecutor() as executor:
        executor.map(save_masked_image, image_paths, anno_paths, out_image_paths)