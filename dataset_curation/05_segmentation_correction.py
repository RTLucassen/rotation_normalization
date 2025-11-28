from pathlib import Path

from annotation_tool import AnnotationTool


# define the image paths and layer names / classes
image_directory = Path(r"")
annotation_directory = Path(r"")
output_directory = Path(r"")

image_paths = sorted([p for p in image_directory.iterdir() if '.png' in p.as_posix()])
annotation_paths = sorted([p for p in annotation_directory.iterdir() if '.tiff' in p.as_posix()])
paths = list(zip(image_paths, annotation_paths))

layer_names = ['tissue', 'control']

if __name__ == '__main__':

    # start annotating
    AnnotationTool(
        input_paths=paths, 
        layers=layer_names,
        output_directory=output_directory,
        autosave=True,
    )    