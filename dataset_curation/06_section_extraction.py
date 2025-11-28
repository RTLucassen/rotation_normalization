import io
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile
from skimage.morphology import opening, diamond
from tqdm import tqdm


def get_white_pixel_bounding_box(image, padding=0):
    """
    Given a binary or RGB image (NumPy array), return the bounding box
    (min_row, min_col, max_row, max_col) that contains all white pixels,
    with optional padding. Padding is clipped to image boundaries.

    Parameters:
        image (np.ndarray): Input image as a NumPy array (2D or 3D).
        padding (int): Number of pixels to pad the bounding box.

    Returns:
        tuple: (min_row, min_col, max_row, max_col) of the padded bounding box.
               Returns None if no white pixels are found.
    """
    height, width = image.shape[:2]

    if image.ndim == 3:
        white_mask = np.all(image == 255, axis=-1)
    else:
        white_mask = image == 255

    white_coords = np.argwhere(white_mask)
    if white_coords.size == 0:
        return None

    min_row, min_col = white_coords.min(axis=0)
    max_row, max_col = white_coords.max(axis=0)

    min_row = int(max(min_row - padding, 0))
    min_col = int(max(min_col - padding, 0))
    max_row = int(min(max_row + padding, height - 1))
    max_col = int(min(max_col + padding, width - 1))

    return (min_row, min_col, max_row, max_col)


# define paths and parameters
image_path = Path(r"")
anno_path = Path(r"")

out_image_path = Path(r"")
out_anno_path = Path(r"")
out_image_control_path = Path(r"")
out_anno_control_path = Path(r"")

start = 0
controls_present = True

if __name__ == '__main__':

    # get the paths to all files in the folders
    image_paths = sorted(list(image_path.iterdir()))
    anno_paths = sorted(list(anno_path.iterdir()))

    # loop over the pairs of images and annotations
    for index, (p1, p2) in tqdm(enumerate(zip(image_paths, anno_paths))):
        if index >= start:
            image = sitk.GetArrayFromImage(sitk.ReadImage(p1))
            if '.zip' in p2.as_posix():
                with zipfile.ZipFile(p2, 'r') as zf:
                    with zf.open(p2.name.replace('.zip', '.tiff')) as file:
                        annotation = tifffile.imread(io.BytesIO(file.read()))
            else:
                annotation = sitk.GetArrayFromImage(sitk.ReadImage(p2))                    
            
            # add extra dimension if necessary
            if len(annotation.shape) == 2:
                annotation = annotation[None, ...]
            annotation = annotation.transpose((1,2,0))

            # isolate the dimensions
            if controls_present:
                tissue = annotation[..., 0:1]
                control = annotation[..., 1:2]
                sections = annotation[..., 2:]
                sections = tissue * sections
            else:
                tissue = annotation[..., 0:1]
                sections = annotation[..., 1:]
                sections = tissue * sections

            # loop over the sections and extract them from the image and segmentation
            for i in range(sections.shape[-1]):
                parts = p1.name.split('_')
                parts.insert(1, str(i).zfill(2))
                new_image_name = '_'.join(parts)

                if not (out_image_path/new_image_name).exists():
                    section_annotation = (sections[..., i]*255).astype(np.uint8)
                    opened_section_annotation = opening(section_annotation, diamond(5))

                    out = get_white_pixel_bounding_box(opened_section_annotation, 30)
                    if out is None:
                        continue 
                    else:
                        min_row, min_col, max_row, max_col = out
                    
                    if controls_present:
                        control_frac = (np.sum(sections[..., i:i+1]*control)/np.sum(sections[..., i]))/255
                        if control_frac > 0.5:
                            sitk.WriteImage(sitk.GetImageFromArray(image[None, min_row:max_row, min_col:max_col, :]), out_image_control_path/new_image_name)
                            sitk.WriteImage(sitk.GetImageFromArray(section_annotation[None, min_row:max_row, min_col:max_col, None]), out_anno_control_path/new_image_name)
                        else:
                            sitk.WriteImage(sitk.GetImageFromArray(image[None, min_row:max_row, min_col:max_col, :]), out_image_path/new_image_name)
                            sitk.WriteImage(sitk.GetImageFromArray(section_annotation[None, min_row:max_row, min_col:max_col, None]), out_anno_path/new_image_name) 
                    else:
                        sitk.WriteImage(sitk.GetImageFromArray(image[None, min_row:max_row, min_col:max_col, :]), out_image_path/new_image_name)
                        sitk.WriteImage(sitk.GetImageFromArray(section_annotation[None, min_row:max_row, min_col:max_col, None]), out_anno_path/new_image_name)  