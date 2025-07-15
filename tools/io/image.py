from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Literal, Tuple, Union, TYPE_CHECKING  

import torch
from tools.util.format import parse_enum
from tools.util.reflection import class_name
from tools.util.torch import tensorify_image, as_tensors
from tools.util.typing import _DEFAULT, DEFAULT, VEC_TYPE
import numpy as np
from tools.transforms.to_numpy import numpyify
from tools.transforms.to_numpy_image import ToNumpyImage, numpyify_image
from tools.serialization.json_convertible import JsonConvertible
from tools.logger.logging import logger
from PIL.Image import Exif, Image
try:
    from PIL.ExifTags import Base as ExifTagsBase
except Exception as err:
    if not TYPE_CHECKING:
        if "cannot import name 'Base' from 'PIL.ExifTags'" in str(err):
            from PIL.ExifTags import TAGS as Exif_Tags
            from enum import Enum
            ExifTagsBase = Enum(
                'ExifTagsBase', {v: k for k, v in Exif_Tags.items()})
        else:
            ExifTagsBase = object

from tools.util.format import parse_format_string
from tqdm.auto import tqdm
from tools.util.path_tools import numerated_file_name, read_directory
import os
from torch.nn.functional import grid_sample
from tools.util.progress_factory import ProgressFactory
from tools.util.sized_generator import SizedGenerator, sized_generator

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Colormap
except Exception as err:
    if not TYPE_CHECKING:
        plt = None
        cm = None
        Colormap = None

def create_image_exif(metadata: Dict[Union[str, ExifTagsBase], Any]) -> Exif:
    """Creates an Exif object from the given metadata.
    Metadata can contain any key-value pair, if key is a regular Exif tag, it will be displayed as such, other keys will be wrapped
    within the MakerNote tag as a json string.

    Parameters
    ----------
    metadata : Dict[Union[str, ExifTagsBase], Any]
        Arbitrary metadata to save with the image.

    Returns
    -------
    Exif
        The Exif object containing the metadata.
    """
    from PIL.ExifTags import TAGS
    # if metadata contains make or MakerNote, wrap it.
    if ExifTagsBase.Make in metadata:
        raise ValueError("Make is a reserved Exif tag")
    if ExifTagsBase.MakerNote in metadata:
        raise ValueError("MakerNote is a reserved Exif tag")

    real_tags = {k: v for k, v in metadata.items() if k in TAGS}
    for k in real_tags:
        metadata.pop(k)
    metadata_json = JsonConvertible.convert_to_json_str(metadata,
                                                        handle_unmatched="raise",
                                                        indent=0)
    make = class_name(create_image_exif)
    exif = Exif()
    exif[ExifTagsBase.Make] = make
    exif[ExifTagsBase.MakerNote] = metadata_json

    for k, v in real_tags.items():
        exif[k] = v
    return exif


def load_image_exif(image: Image, safe_load: bool = True) -> Dict[Union[str, ExifTagsBase], Any]:
    """Loads the exif data from the given image.

    Parameters
    ----------
    image : Image
        Image to load the exif data from.
        If data was saved with create_image_exif, it will be loaded here
        and fully restored.

    safe_load : bool, optional
        If safe_load is true, the json module will be used for loading
        otherwise JsonConvertible will be used for loading which can be unsafe
        w.r.t. code execution, by default True

    Returns
    -------
    Dict[Union[str, ExifTagsBase], Any]
        Dictionary containing the exif data.
    """
    exif = image.getexif()
    if exif is None:
        return dict()
    metadata = dict()
    metadata_json = None
    # Get Make
    make = exif.get(ExifTagsBase.Make, None)
    if make is None:
        make = exif.get(ExifTagsBase.Make.value, None)
    if make is not None and make == class_name(create_image_exif):
        metadata_json = exif.get(ExifTagsBase.MakerNote, None)
        if metadata_json is None:
            metadata_json = exif.get(ExifTagsBase.MakerNote.value, None)
    other_tags = [k for k in exif.keys() if k !=
                  ExifTagsBase.Make and k != ExifTagsBase.MakerNote]
    for k in other_tags:
        try:
            en = parse_enum(ExifTagsBase, k)
            metadata[en] = exif[k]
        except ValueError:
            metadata[k] = exif[k]
    if metadata_json is not None:
        if safe_load:
            import json
            lm = json.loads(metadata_json)
            metadata.update(lm)
        else:
            lm = JsonConvertible.convert_from_json_str(metadata_json)
            metadata.update(lm)
    return metadata


def load_image(
    path: str,
    load_metadata: bool = False,
    safe_metadata_load: bool = True,
    max_size: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[Union[str, ExifTagsBase], Any]]]:
    """
    Loads an image from the given path.

    Parameters
    ----------
    path : str
        Path to the image.

    load_metadata : bool, optional
        Whether to load metadata, by default False
        This will return a tuple with the image and the metadata.
        Metadata usally contains exif data and other information.

    safe_metadata_load : bool, optional
        Whether the metadata should be loaded safely, by default True

    max_size : Optional[int], optional
        If the image should be resized to a maximum size, by default None
        The image will be resized to the maximum size while maintaining aspect ratio.

    size : Optional[Tuple[int, int]], optional
        If a specific size should be used instead of max size, by default None
        Is exclusive with max_size. (H, W)

    Returns
    -------
    np.ndarray
        Numpy array of the mask.

    OR:

    Tuple[np.ndarray, Dict[Union[str, ExifTagsBase], Any]]
        Tuple of the image and the metadata.
    """
    from PIL import Image
    base, ext = os.path.splitext(path)
    try:
        if ext.lower() == ".npy":
            return np.load(path)
        elif ext.lower() in [".tiff", ".tif"]:
            try:
                import cv2
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                # Convert to RGB if 3 channel
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if max_size is not None or size is not None:
                    channel_added = False
                    if len(image.shape) == 2:
                        image = np.expand_dims(image, axis=-1)
                        channel_added = True
                    image = resize_image(image, max_size=max_size, size=size)
                    if channel_added:
                        image = np.squeeze(image, axis=-1)
                return image
            except ImportError as err:
                raise ImportError(
                    "OpenCV is not installed, but is required to load tiff images.")
        else:
            mask_pil = Image.open(path)
            image = np.array(mask_pil)
            if max_size is not None or size is not None:
                image = resize_image(image, max_size=max_size, size=size)
            if not load_metadata:
                return image
            # Load metadata
            metadata = load_image_exif(mask_pil, safe_load=safe_metadata_load)
            return image, metadata
    except FileNotFoundError as err:
        raise err


def apply_colormap(image: VEC_TYPE,
                   colormap: Union[str, Colormap] = "viridis",
                   vmin: Optional[VEC_TYPE] = None,
                   vmax: Optional[VEC_TYPE] = None,
                   ) -> np.ndarray:
    """Applies a colormap to the given image.

    Parameters
    ----------
    image : VEC_TYPE
        The image to apply the colormap to. 
        Image should be in the shape (H, W[, C]) for numpy arrays or ([C,] H, W) for torch tensors.
        C should be 1.
    colormap : Union[str, Colormap], optional
        The colormap to apply, can be a string or a Colormap instance, by default "viridis"
    vmin : Optional[VEC_TYPE], optional
        The minimum value for the colormap if not specified, will be set to the min of the image, by default None
    vmax : Optional[VEC_TYPE], optional
        The maximum value for the colormap if not specified, will be set to the max of the image, by default None
dded
    Returns
    -------
    np.ndarray
        The image with the colormap applied. Shape will be (H, W, NC), where NC is the number of channels in the colormap.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    TypeError
        If the colormap is not a string or a Colormap instance.
    """
    from tools.transforms.to_numpy_image import numpyify_image
    from tools.transforms.numpy.min_max import MinMax
    image = numpyify_image(image)
    if isinstance(colormap, str):
        if plt is None or cm is None:
            raise ImportError(
                "Matplotlib is not installed, but is required to apply colormaps.")
        colormap = cm.get_cmap(colormap)
    if not isinstance(colormap, Colormap):
        raise TypeError(
            f"Colormap should be a string or a Colormap instance, got {type(colormap)}")
    if vmin is None:
        vmin = np.min(image)
    if vmax is None:    
        vmax = np.max(image)
    norm = MinMax(0, colormap.N -1)
    norm.min = vmin
    norm.max = vmax
    norm.fitted = True
    v = colormap(norm(image).round().astype(int))
    return v
    
def save_image(image: VEC_TYPE,
               path: Union[str, Path],
               override: bool = True,
               mkdirs: bool = True,
               metadata: Optional[Dict[Union[str, ExifTagsBase], Any]] = None) -> Union[str, Path]:
    """Saves numpy array or torch tensor as an image.

    Parameters
    ----------
    data : VEC_TYPE
        Data to be saved as an image. Should be in the shape (H, W[, C])
        for numpy arrays or ([C,] H, W) torch tensors.
    path : Union[str, Path]
        Path to save the image to.
    override : bool, optional
        If an existing image should be overriden, by default False
    mkdirs : bool, optional
        If the directories should be created if they do not exist, by default True
    metadata : Optional[Dict[Union[str, ExifTagsBase], Any]], optional
        Metadata to save with the image, by default None
        Will be saved as exif data.
    Returns
    -------
    Union[str, Path]
        The path where the image was saved.
    """
    from PIL import Image
    is_path = False
    if isinstance(path, Path):
        is_path = True
        path = str(path)
    img = numpyify_image(image)
    if not override:
        path = numerated_file_name(path)

    if mkdirs:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    args = dict()
    if metadata is not None and len(metadata) > 0:
        args['exif'] = create_image_exif(metadata)

    base, ext = os.path.splitext(path)
    # If tiff use opencv
    if ext.lower() == ".tiff" or ext.lower() == ".tif":
        if len(args) > 0:
            raise ValueError("Exif data is not supported for tiff images.")
        try:
            import cv2
            # If 3 channel, convert to BGR
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img)
        except ImportError as err:
            raise ImportError(
                "OpenCV is not installed, but is required to save tiff images.")
    # Numpy save
    elif ext.lower() == ".npy":
        if len(args) > 0:
            raise ValueError("Exif data is not supported for numpy images.")
        np.save(base, img)
    else:
        Image.fromarray(img).save(path, **args)
    if is_path:
        return Path(path)
    return path


def resample(
    image: np.ndarray,
    resampling_ratio: Optional[float] = 1.,
    mode: str = "bilinear",
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Resamples the image with the given resampling ratio.

    Parameters
    ----------
    image : np.ndarray
        Image to resample.
    resampling_ratio : float
        The resampling ratio. Should be greater than 0 as a fractional times the original size.
        Is exclusive with target_size, by default 1
    mode : str, optional
        Either grid_sample modes like bilinear or subsample to just subsample the grid, by default "bilinear"
    target_size : Optional[Tuple[int, int]], optional
        The target size of the image is exclusive with resampling ratio. If provided this will be used, by default None
        Format is (H, W)

    Returns
    -------
    np.ndarray
        The resampled image.
    """
    if resampling_ratio <= 0:
        raise ValueError("Resampling ratio should be greater than 0.")
    if mode != "subsample":
        H, W, C = image.shape

        if target_size is not None:
            H_s, W_s = target_size
        else:
            H_s, W_s = int(H * resampling_ratio), int(W * resampling_ratio)

        is_uint8 = image.dtype == np.uint8

        image_tensor = tensorify_image(image, dtype=torch.float32)
        grid = torch.nn.functional.affine_grid(torch.tensor(
            [[[1., 0, 0], [0, 1., 0]]]), (1, C, H_s, W_s), align_corners=True)
        # Move to GPU if available
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            grid = grid.cuda()

        # Convert to 0-1 range if image was uint8
        if is_uint8:
            image_tensor = image_tensor / 255.0

        resampled_image = grid_sample(image_tensor.unsqueeze(
            0), grid, mode=mode, align_corners=True)[0]

        if is_uint8:
            # Convert back to 0-255 range
            resampled_image = (resampled_image * 255.0).to(torch.uint8)
        return numpyify_image(resampled_image)
    else:
        # Subsampling
        skips = (1 / resampling_ratio)
        # Check if close to integer
        if not np.isclose(skips, int(skips), rtol=0.05):
            raise ValueError(
                "Resampling ratio should be close to 1/n, where n > 0 is an integer.")
        skips = int(skips)
        return image[::skips, ::skips, :]


def compute_new_size(image_shape: Tuple[int, int], max_size: Optional[int] = None, min_size: Optional[int] = None) -> Tuple[int, int]:
    """"Computes the new size of the image while maintaining aspect ratio.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        Shape of the image in format (H, W)

    max_size : Optional[int]
        Max size of the longest side of the image.
        Mutually exclusive with min_size.

    min_size : Optional[int]
        Min size of the shortest side of the image.
        Mutually exclusive with max_size.

    Returns
    -------
    Tuple[int, int]
        New size of the image in format (H, W)
    """
    aspect = image_shape[1] / image_shape[0]
    # Get if in landscape or portrait
    if max_size is not None:
        if image_shape[1] > image_shape[0]:
            # Landscape
            new_size = (int(max_size / aspect), max_size)
        else:
            # Portrait
            new_size = (max_size, int(max_size * aspect))
    elif min_size is not None:
        if image_shape[1] > image_shape[0]:
            # Landscape
            new_size = (min_size, int(min_size * aspect))
        else:
            # Portrait
            new_size = (int(min_size / aspect), min_size)
    else:
        raise ValueError("Either max_size or min_size should be provided.")
    return new_size


def compute_max_resolution(image_shape: Tuple[int, int], max_size: int) -> Tuple[int, int]:
    if image_shape[0] <= max_size and image_shape[1] <= max_size:
        return image_shape
    aspect = image_shape[1] / image_shape[0]
    # Get if in landscape or portrait
    if image_shape[1] > image_shape[0]:
        # Landscape
        new_size = (int(max_size / aspect), max_size)
    elif image_shape[1] <= image_shape[0]:
        # Portrait
        new_size = (max_size, int(max_size * aspect))
    return new_size


def resize_image(
        image: VEC_TYPE,
        max_size: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        factor: Optional[float] = None,
) -> np.ndarray:
    """Resizes the image to the given max size.
    Maintains aspect ratio.

    Parameters
    ----------
    image : np.ndarray
        Image to resize. Should be in the shape (H, W, C) or (B, H, W, C)
    max_size : int
        Max size of the longest side of the image.
    size : Optional[Tuple[int, int]], optional
        If a specific size should be used instead of max size, by default None
        Is exclusive with max_size. Has format (H, W)
    factor : Optional[float], optional
        If a specific factor should be used instead of max size, by default None
        The image will be resized to the factor times the original size.
    Returns
    -------
    np.ndarray
        Resized image.
    """
    from torchvision.transforms import Compose, Resize, ToTensor
    from tools.transforms.to_numpy_image import ToNumpyImage
    dtype = image.dtype
    is_tensor = torch.is_tensor(image)
    if len(image.shape) == 4:
        B, H, W, C = image.shape
    elif len(image.shape) == 3:
        H, W, C = image.shape
        B = 1
    else:
        raise ValueError("Image should have shape B x H x W x C or H x W x C")
    if max_size is None and size is None and factor is None:
        raise ValueError(
            "Either max_size, size, or factor should be provided.")
    if max_size is not None and size is not None and factor is None:
        raise ValueError(
            "Either max_size, size or factor should be provided.")
    if max_size is not None:
        new_size = compute_new_size((H, W), max_size)
        transforms = Compose(
                            ([ToTensor()] if not is_tensor else [])
            + ([Resize(new_size)]
                                if (H > max_size or W > max_size) else []) +
            ([ToNumpyImage(output_dtype=dtype)] if not is_tensor else [])
        )
    elif size is not None:
        transforms = Compose(
            ([ToTensor()] if not is_tensor else [])
            + [Resize(size)]
            + ([ToNumpyImage(output_dtype=dtype)] if not is_tensor else [])
        )
    elif factor is not None:
        new_size = (int(round(H * factor)), int(round(W * factor)))
        transforms = Compose(
            ([ToTensor()] if not is_tensor else [])
            + [Resize(new_size)]
            + ([ToNumpyImage(output_dtype=dtype)] if not is_tensor else [])
        )
    if len(transforms.transforms) == 0:
        return image
    return transforms(image)


def index_image_folder(path, filename_format=r"(?P<index>[0-9]+).png", return_dict: bool = False) -> Union[List[str], List[Dict[str, Any]]]:
    """Indexes the images in the given folder.

    Parameters
    ----------
    path : str
        Path to the folder.
    filename_format : str, optional
        Filename format regex, by default r"(?P<index>[0-9]+).png"
    return_dict : bool, optional
        If the return should be a list of dictionaries including the parsed values, by default False
        If False, only the paths will be returned.

    Returns
    -------
    Union[List[str], List[Dict[str, Any]]]
        List of indexed filenames if return_dict is False, otherwise list of dictionaries with the parsed values.
    """
    image_paths = read_directory(
        path, filename_format, parser=dict(index=int), path_key="path")
    sorted_index = sorted(image_paths, key=lambda x: x["index"])
    return [x["path"] for x in sorted_index] if not return_dict else sorted_index


def load_image_stack(
        path: Optional[str] = None,
        filename_format: str = r"(?P<index>[0-9]+)\.((png)|(jpg))",
        max_size: Optional[int] = None,
        sorted_image_paths: Optional[List[str]] = None,
        progress_bar: bool = True,
        progress_factory: Optional[ProgressFactory] = None,
        **kwargs
) -> np.ndarray:
    """Helper function to load images for a given path and filename format.

    Parameters
    ----------
    path : Optional[str], optional
        Path to the images.
        Must be specified if sorted_image_paths is None.

    filename_format : str, optional
        Filename format regex, by default r"(?P<index>[0-9]+).png"
        Images will be sorted by index.

    max_size : Optional[int], optional
        If the image should be resized to a maximum size, by default None
        The image will be resized to the maximum size while maintaining aspect ratio.

    sorted_image_paths : Optional[List[str]], optional
        Sorted image paths, by default None
        If provided, the images will be loaded from these paths in the given order.

    progress_bar : bool, optional
        If a progress bar should be shown, by default True

    progress_factory : Optional[ProgressFactory], optional
        Optional progress factory to use, by default None

    Returns
    -------
    Optional[np.ndarray]
        Image stack in shape B x H x W x C.
        B order is index order ascending.
        If no images are found, None will be returned.
    """
    if progress_bar and progress_factory is None:
        progress_factory = ProgressFactory()
    if sorted_image_paths is None:
        if path is None:
            raise ValueError(
                "Path must be specified if sorted_image_paths is None")
        sorted_image_paths = index_image_folder(
            path, filename_format=filename_format)

    if len(sorted_image_paths) == 0:
        return None

    img = load_image(sorted_image_paths[0], max_size=max_size, **kwargs)
    images = np.zeros((len(sorted_image_paths), *img.shape), dtype=img.dtype)
    images[0] = img

    it = None
    if progress_bar:
        it = progress_factory.bar(total=len(
            sorted_image_paths), desc="Loading images", is_reusable=True, tag="load_image_stack", delay=1)
        it.update(1)

    for i in range(1, len(sorted_image_paths)):
        images[i] = load_image(sorted_image_paths[i],
                               max_size=max_size, **kwargs)
        if progress_bar:
            it.update(1)
    return images


@sized_generator()
def load_image_stack_generator(
        path: Optional[str] = None,
        filename_format: str = r"(?P<index>[0-9]+)\.((png)|(jpg))",
        max_size: Optional[int] = None,
        sorted_image_paths: Optional[List[str]] = None,
        progress_bar: bool = True,
        progress_factory: Optional[ProgressFactory] = None,
        **kwargs
) -> Generator[np.ndarray, None, None]:
    """Helper function to load images for a given path and filename format.

    Parameters
    ----------
    path : Optional[str], optional
        Path to the images.
        Must be specified if sorted_image_paths is None.

    filename_format : str, optional
        Filename format regex, by default r"(?P<index>[0-9]+).png"
        Images will be sorted by index.

    max_size : Optional[int], optional
        If the image should be resized to a maximum size, by default None
        The image will be resized to the maximum size while maintaining aspect ratio.

    sorted_image_paths : Optional[List[str]], optional
        Sorted image paths, by default None
        If provided, the images will be loaded from these paths in the given order.

    progress_bar : bool, optional
        If a progress bar should be shown, by default True

    progress_factory : Optional[ProgressFactory], optional
        Optional progress factory to use, by default None

    Returns
    -------
    Optional[np.ndarray]
        Image stack in shape H x W x C.


        B order is index order ascending.
        If no images are found, None will be returned.
    """
    if progress_bar and progress_factory is None:
        progress_factory = ProgressFactory()
    if sorted_image_paths is None:
        if path is None:
            raise ValueError(
                "Path must be specified if sorted_image_paths is None")
        sorted_image_paths = index_image_folder(
            path, filename_format=filename_format)

    yield len(sorted_image_paths)  # Length for the generator

    if len(sorted_image_paths) == 0:
        return None

    img = load_image(sorted_image_paths[0], max_size=max_size, **kwargs)

    yield img

    it = None
    if progress_bar:
        it = progress_factory.bar(total=len(
            sorted_image_paths), desc="Loading images", is_reusable=True, tag="load_image_stack", delay=1)
        it.update(1)

    for i in range(1, len(sorted_image_paths)):
        yield load_image(sorted_image_paths[i],
                         max_size=max_size, **kwargs)
        if progress_bar:
            it.update(1)


def save_image_stack(images: VEC_TYPE,
                     format_string_path: Union[str, Path],
                     override: bool = False, mkdirs: bool = True,
                     metadata: Optional[Dict[Union[str,
                                                   ExifTagsBase], Any]] = None,
                     progress_bar: bool = False,
                     additional_filename_variables: Optional[Dict[str, Any]] = None,
                     additional_filename_variables_list: Optional[List[Dict[str, Any]]] = None
                     ) -> List[str]:
    """Saves the given image stack.

    Expects to get a stack of images in the shape B x H x W x C for numpy or B x C x H x W if tensor.

    Parameters
    ----------
    images : VEC_TYPE
        Image stack to save.
        Should be in the shape B x H x W x C for numpy or B x C x H x W if tensor.

    format_string_path : Union[str, Path]
        Path format string to save the images.
        Example: "path/to/save/image_{index}.png"
        Index will be replaced with the index of the image in the stack.

    metadata : Optional[dict], optional
        Optional metadata to save with the image as exif metadata. Will be saved as MakerNote in the exif tag.
        The metadata will be the same for all images.

    progress_bar : bool, optional
        Whether to show a progress bar when saving a batch of images, by default False

    additional_filename_variables : Optional[Dict[str, Any]], optional
        Additional variables to use in the filename format string, by default None

    additional_filename_variables_list : Optional[List[Dict[str, Any]], optional
        Additional variables to use in the filename format string, by default None
        If provided, the additional variables item corresponding to the index will be used.

    Returns
    -------
    List[str]
        List of saved filenames.
    """
    saved_files = []
    images = numpyify_image(images)
    if not len(images.shape) == 4:
        raise ValueError("Image stack should have shape B x H x W x C")
    if mkdirs:
        if not os.path.exists(os.path.dirname(format_string_path)):
            os.makedirs(os.path.dirname(format_string_path), exist_ok=True)
    args = dict()

    def from_optional_list_index(l: Optional[List[Dict[str, Any]]], index: int) -> Dict[str, Any]:
        if l is None:
            return dict()
        if len(l) <= index:
            return dict()
        d = l[index]
        if d is None:
            return dict()
        d = dict(d)
        d.pop("index", None)
        return d

    if isinstance(format_string_path, Path):
        format_string_path = str(format_string_path)

    filenames = parse_format_string(format_string_path, [
        dict(index=i, **from_optional_list_index(additional_filename_variables_list, i)) for i in range(images.shape[0])
    ], additional_variables=additional_filename_variables)
    if len(set(filenames)) != images.shape[0]:
        raise ValueError(
            f"Number of distinct filenames {len(set(filenames))} does not match number of masks {images.shape[0]} if you specified an index template?")
    it = enumerate(filenames)
    if progress_bar:
        it = tqdm(it, total=images.shape[0], desc="Saving images")
    for i, f in it:
        sf = save_image(images[i], f, override=override,
                        mkdirs=False, metadata=metadata)
        saved_files.append(sf)
    return saved_files


def get_origin(
        text: str,
        vertical_alignment: str = "center",
        horizontal_alignment: str = "center",
        family: int = DEFAULT,
        size: float = 1,
        thickness: int = 1,

) -> Tuple[float, float]:
    import cv2 as cv

    if family == DEFAULT:
        family = cv.FONT_HERSHEY_SIMPLEX

    # Opencv Assumes text origin to be bottom left
    text_width, text_height = cv.getTextSize(text, family, size, thickness)[0]

    x = 0
    y = 0

    if vertical_alignment == "center":
        # Add half of the text height to center the text
        y += text_height // 2
    elif vertical_alignment == "top":
        y += text_height
    elif vertical_alignment == "bottom":
        y = 0
    else:
        raise ValueError(f"Unknown vertical alignment: {vertical_alignment}")

    if horizontal_alignment == "center":
        # Add half of the text width to center the text
        x -= text_width // 2
    elif horizontal_alignment == "left":
        x = 0
    elif horizontal_alignment == "right":
        x -= text_width
    else:
        raise ValueError(
            f"Unknown horizontal alignment: {horizontal_alignment}")
    return x, y


def rgba_to_rgb(img: VEC_TYPE, base_color: VEC_TYPE) -> VEC_TYPE:
    """
    Converts an RGBA image to RGB using the given base color.
    by blending the image with the base color using the alpha channel (alpha-matting).

    Parameters
    ----------
    img : VEC_TYPE
        Image to convert. Should be in the shape H x W x C if numpy or C x H x W if tensor.
    base_color : VEC_TYPE
        Base color to use for the alpha channel. Should be in the shape C.

    Returns
    -------
    VEC_TYPE
        The converted image. Will be in the shape H x W x C if numpy or C x H x W if tensor
    """

    reorderd = False
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0)
    dtype = img.dtype
    base_color = numpyify(base_color)
    color_converted = False
    if dtype == np.uint8:
        img = img.astype(float) / 255
        base_color = base_color.astype(float) / 255
        color_converted = True
    if img.shape[-1] == 4:
        img = img[..., :3] * img[..., -1][..., None] + \
            (1 - img[..., -1][..., None]) * (base_color)
    if reorderd:
        img = img.permute(2, 0, 1)
    if color_converted:
        img = (img * 255).astype(np.uint8)
    return img


def gamma_correction(image: VEC_TYPE, gamma: Union[_DEFAULT, float] = DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies gamma correction to the given image.
    The gamma value is calculated based on the mean brightness of the image, or can be specified manually.

    Parameters
    ----------
    image : np.ndarray
        The image to apply gamma correction to.
        Should be in the shape H x W x C if numpy or C x H x W if tensor.

    gamma : Union[_DEFAULT, float], optional
        Gamma value or DEFAULT, by default DEFAULT
        If DEFAULT, the gamma value will be calculated based on the mean brightness of the image.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        1. The gamma corrected image. Will be in the shape H x W x C and dtype uint8.
        2. The gamma value used for correction.
    """
    import cv2
    import math
    from tools.transforms.to_numpy_image import ToNumpyImage

    numpyify = ToNumpyImage(output_dtype=np.uint8)
    image = numpyify(image)
    is_rgba = image.shape[-1] == 4
    rgba_image = None

    if is_rgba:
        # Convert to RGB
        rgba_image = image.copy()
        image = image[..., :3]
    # convert img to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    if gamma == DEFAULT:
        mid = 0.5
        mean = np.mean(val)
        gamma = math.log(mid*255)/math.log(mean)

    val_corrected = np.power(val, gamma).clip(0, 255).astype(np.uint8)

    hsv_corrected = cv2.merge([hue, sat, val_corrected])
    gamma_corrected = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

    if is_rgba:
        rgba_image[..., :3] = gamma_corrected
        gamma_corrected = rgba_image
    return gamma_corrected, gamma


def check_text_overlap(occupied_area: np.ndarray,
                       position: Tuple[int, int],
                       text_width: int,
                       text_height: int) -> Tuple[int, np.ndarray]:
    """
    Checks for overlap with image boundaries and occupied areas.

    Parameters
    ----------
    occupied_area : np.ndarray
        Binary mask of the already occupied area.

    position : Tuple[int, int]
        Position of the text in the image.
        X, Y coordinates.

    text_width : int
        Width of the text.

    text_height : int
        Height of the text.

    Returns
    -------
    Tuple[int, np.ndarray]
        Tuple of the overlap area and the new occupied area.
    """
    H, W = occupied_area.shape
    col, row = position

    if row < 0 or row + text_height > H or col < 0 or col + text_width > W:
        return float('inf'), occupied_area

    overlap_area = 0

    text_mask = np.zeros_like(occupied_area, dtype=bool)
    text_mask[int(row):int(row + text_height), int(col):int(col + text_width)] = True
    overlap_with_occupied = np.sum(occupied_area[text_mask])
    overlap_area += overlap_with_occupied

    return overlap_area, np.logical_or(occupied_area, text_mask)


def put_text(
        img: VEC_TYPE,
        text: str,
        placement: Optional[str] = "top-center",
        position: Optional[Tuple[int, int]] = None,
        vertical_alignment: str = "top",
        horizontal_alignment: str = "center",
        family: int = DEFAULT,
        size: float = 1,
        thickness: int = 1,
        color: Any = "black",
        margin: int = 5,
        background_color: Any = "white",
        background_stroke: Optional[int] = None,
        background_stroke_color: Any = "black",
        padding: int = 5,
        check_overlap: bool = True,
        overlap_area: List[np.ndarray] = None
) -> np.ndarray:
    """Renders text on an image with a given placement or absolute position and alignment.

    Parameters
    ----------
    img : VEC_TYPE
        Image to render the text on.
        Can be a numpy array or a torch tensor. Should be in the shape H x W x C if numpy or C x H x W if tensor.

    text : str
        The text to render.
        Can be arbitrary text, but will be rendered as a single line.

    placement : Optional[str], optional
        Automatic placement for the text. In format [vertical]-[horizontal], by default "top-center"
        Can be top, center, bottom for vertical and left, center, right for horizontal.
        Takes precedence over position and alignment if specified.

    position : Optional[Tuple[int, int]], optional
        The Text position in image coordinates (x, y) ([0, width), [0, height)), by default None
        If placement is specified, this will be ignored.

    vertical_alignment : str, optional
        Vertical alignment of the Text w.r.t to position, by default "top"
        Can be top, center or bottom.
        If placement is specified, this will be ignored.

    horizontal_alignment : str, optional
        Horizontal alignment of the Text w.r.t to position, by default "center"
        Can be left, center or right.
        If placement is specified, this will be ignored.

    family : int, optional
        One of the opencv supported font families, by default cv.FONT_HERSHEY_DUPLEX

    size : float, optional
        The size / scaling of the text. Is a multiplier of the fonts default size, by default 1

    thickness : int, optional
        Thickness of the font lines, by default 1

    color : Any, optional
        Foreground or text color, by default "black"
        Can be any color supported by matplotlib.
        E.g. Hex, RGB, RGBA, or color names.

    margin : int, optional
        Outside margin when placement is used, by default 5
        Defines the distance from the edge of the image in pixels.

    background_color : Any, optional
        Background color of the text, by default "white"
        If specified, a background rectangle will be drawn behind the text in the specified color.

    background_stroke : Optional[int], optional
        Stroke thickness of the background in pixels, by default None
        If specified, a stroke will be drawn around the background rectangle.

    background_stroke_color : Any, optional
        Color of the stroke around the background rectangle, by default "black"

    padding : int, optional
        Inner padding of the background stroke w.r.t the drawn text, by default 5

    check_overlap : bool, optional
        Whether to check for overlap with the image boundaries and occupied areas, by default True

    overlap_area : List[np.ndarray], optional
        List containing the occupied area of the image, by default None
        First element should be the occupied area as a binary mask.
        Will be updated with the new occupied area and can be reused.

    Returns
    -------
    np.ndarray
        The image with the rendered text. Will be in the shape H x W x C.
        The dtype will be np.uint8.
    """
    from tools.viz.matplotlib import parse_color_rgb, parse_color_rgba
    from matplotlib.pyplot import figure
    import cv2 as cv

    if family == DEFAULT:
        family = cv.FONT_HERSHEY_SIMPLEX

    has_alpha = False
    color_parser = parse_color_rgb
    if img.shape[-1] == 4:
        has_alpha = True
        color_parser = parse_color_rgba

    if background_stroke_color is not None:
        background_stroke_color = (color_parser(
            background_stroke_color) * 255).astype(np.uint8).tolist()
    if background_color is not None:
        background_color = (color_parser(background_color)
                            * 255).astype(np.uint8).tolist()
    color = (color_parser(color) * 255).astype(np.uint8).tolist()
    numpyify_image = ToNumpyImage(output_dtype=np.uint8)
    img = numpyify_image(img)

    if len(img.shape) == 2:
        img = img[..., None]

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=0)

        # img = rgba_to_rgb(img, (255, 255, 255))
        # Assure all colors have alpha channel

    if placement is not None:
        position = position or (img.shape[1] // 2, img.shape[0] // 2)
        vertical_alignment, horizontal_alignment = placement.split("-")
        if vertical_alignment == "top":
            position = (position[0], 0 + margin)
        elif vertical_alignment == "bottom":
            position = (position[0], img.shape[0] - margin)
        elif vertical_alignment == "center":
            position = (position[0], img.shape[0] // 2)
        if horizontal_alignment == "left":
            position = (0 + margin, position[1])
        elif horizontal_alignment == "right":
            position = (img.shape[1] - margin, position[1])
        elif horizontal_alignment == "center":
            position = (img.shape[1] // 2, position[1])

    offset = get_origin(text, vertical_alignment,
                        horizontal_alignment, family, size, thickness)

    position = (position[0] + offset[0], position[1] + offset[1])
    text_width, text_height = cv.getTextSize(text, family, size, thickness)[0]

    if check_overlap and overlap_area is not None:
        ova = overlap_area[0] if len(overlap_area) > 0 else None
        best_pos, ova = find_best_text_position(
            img.shape[0], img.shape[1],
            position, offset, text_width, text_height, ova)
        if best_pos is not None:
            if len(overlap_area) > 0:
                overlap_area[0] = ova
            else:
                overlap_area.append(ova)
            position = best_pos

    if background_color is not None:

        p = np.array(position)
        bl = p + np.array([-padding, padding])
        tr = p + np.array([text_width, -text_height]) + \
            np.array([padding, -padding])
        img = cv.rectangle(img, bl, tr, background_color, -1)
    if background_stroke is not None and background_stroke > 0 and background_stroke_color is not None:
        text_width, text_height = cv.getTextSize(
            text, family, size, thickness)[0]
        p = np.array(position)
        bl = p + np.array([-padding, padding])
        tr = p + np.array([text_width, -text_height]) + \
            np.array([padding, -padding])
        img = cv.rectangle(
            img, bl, tr, background_stroke_color, background_stroke)

    img = cv.putText(
        img,
        text,
        position,
        family,
        size,
        color,
        thickness,
        cv.LINE_AA,
    )
    return img


def find_best_text_position(
    image_height: int,
    image_width: int,
    position: Tuple[int, int],
    offset: Tuple[int, int],
    text_width: int,
    text_height: int,
    occupied_area: Optional[np.ndarray] = None
) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:  # Return tuple of position and occupied_area
    """Finds the best position for text, minimizing overlap."""

    if occupied_area is None:
        occupied_area = np.zeros(
            (image_height, image_width), dtype=bool)  # Initialize if None

    best_position = None
    best_overlap = float('inf')
    # Keep track of the best occupied area
    best_occupied_area = occupied_area.copy()

    search_center = position

    max_search_radius = 20  # Initial maximum search radius
    search_radius = 1  # Start small
    num_search_steps = 8  # Number of search steps in a circle

    while search_radius <= max_search_radius:
        for i in range(num_search_steps):
            angle = 2 * np.pi * i / num_search_steps
            row_offset = int(round(search_radius * np.sin(angle)))
            col_offset = int(round(search_radius * np.cos(angle)))

            candidate_position = (search_center[0] + offset[0] + col_offset,
                                  search_center[1] + offset[1] + row_offset)

            overlap, new_occupied_area = check_text_overlap(
                occupied_area, candidate_position, text_width, text_height)  # Get new occupied area

            if overlap < best_overlap:
                best_overlap = overlap
                best_position = candidate_position
                best_occupied_area = new_occupied_area  # Update best occupied area
            if overlap == 0:
                break

        if best_overlap > 0:  # Increase search radius if still overlapping
            search_radius *= 2  # Or any other factor
        else:  # If a good position is found exit the loop
            break

    if best_position is None:
        return None, occupied_area

    # Return best position and updated occupied area
    return best_position, best_occupied_area


def alpha_compose_with_background_grid(
    images: np.ndarray,
    square_size: int = 10,
    primary_color: Any = (153, 153, 153, 255),
    secondary_color: Any = (102, 102, 102, 255),
) -> np.ndarray:
    """Gets a Batch of images and composes them using the alpha compositing algorithm with a background grid,
    to remove transparency.

    Parameters
    ----------
    images : np.ndarray
        Images with transparency in the shape ([..., B,], H, W, C).
        Where C is 4 (RGBA). With rgba values in the range [0, 1].

    Returns
    -------
    np.ndarray
        Image with the alpha composited images in the shape ([..., B,], H, W, C).
        Where C is 3 (RGB). With rgba values in the range [0, 1].
    """
    from tools.util.numpy import flatten_batch_dims, unflatten_batch_dims
    images, batch_dims = flatten_batch_dims(images, -4)
    grid = torch.tensor(alpha_background_grid(images.shape[-3:-1], square_size=square_size,
                                              primary_color=primary_color, secondary_color=secondary_color)).unsqueeze(0).unsqueeze(0).float() / 255
    image_tensor = torch.tensor(images).unsqueeze(0)
    N, B, H, W, C = image_tensor.shape
    grid = grid.repeat(1, B, 1, 1, 1)
    composition = n_layers_alpha_compositing(
        torch.cat([grid, image_tensor], dim=0), torch.tensor([1, 0]))
    return unflatten_batch_dims(composition.numpy(), batch_dims)


def alpha_background_grid(
    resolution: Tuple[int, int],
    square_size: int = 10,
    primary_color: Any = (153, 153, 153, 255),
    secondary_color: Any = (102, 102, 102, 255),
) -> np.ndarray:
    """Creates a grid pattern with a checkerboard pattern.

    Parameters
    ----------
    resolution : Tuple[int, int]
        Resolution of the grid in pixels.
        Shape is (H, W) where H is the height and W is the width.

    Returns
    -------
    np.ndarray
        The grid pattern in shape H x W x C.
    """
    from tools.viz.matplotlib import parse_color_rgba
    primary_color = (parse_color_rgba(primary_color)
                     * 255).astype(np.uint8).tolist()
    secondary_color = (parse_color_rgba(secondary_color)
                       * 255).astype(np.uint8).tolist()
    img = np.zeros((resolution[0], resolution[1], 4), dtype=np.uint8)
    img[...] = primary_color

    x = np.arange(resolution[1])
    y = np.arange(resolution[0])

    coords = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    # XOR to get the checkerboard pattern
    second_color_mask = ((np.floor(coords / square_size) %
                         2) == 1).sum(axis=-1) == 1
    img.reshape(-1, 4)[second_color_mask] = secondary_color
    return img


def texture_grid(
    resolution: Tuple[int, int],
    color_fnc: Callable[[np.ndarray], np.ndarray] = DEFAULT,
    square_size: int = 10,
) -> np.ndarray:
    """
    Creates a pattern depending on the square size and resolution.

    Parameters
    ----------
    resolution : Tuple[int, int]
        Resolution of the grid in pixels.
        Shape is (H, W) where H is the height and W is the width.

    color_fnc : Callable[[np.ndarray], np.ndarray], optional
        Function to generate the color of the grid. The default is DEFAULT.
        The function should take a 2D array of shape (N, 2) and return a 2D array of shape (N, 4) with RGBA values in np.uint8.

    square_size : int, optional
        Size of the squares in the grid. The default is 10.
    Returns
    -------
    np.ndarray
        The grid pattern in shape H x W x C.
    """
    from tools.viz.matplotlib import parse_color_rgba
    if color_fnc is DEFAULT:
        def color_fnc(x: np.ndarray) -> np.ndarray:
            import matplotlib.pyplot as plt
            max_row = np.max(x[:, 0])
            max_col = np.max(x[:, 1])
            row_cmap = plt.get_cmap("rainbow")
            col_cmap = plt.get_cmap("Greys")
            row_vals = (x[:, 0] / max_row) * row_cmap.N
            cols_vals = (x[:, 1] / max_col) * col_cmap.N

            row_cols = row_cmap(row_vals.round().astype(int))
            col_cols = col_cmap(cols_vals.round().astype(int))

            colors = 0.5 * (row_cols + col_cols)
            colors *= 255
            colors = colors.astype(np.uint8)
            return colors

    img = np.zeros((resolution[0], resolution[1], 4), dtype=np.uint8)
    x = np.arange(resolution[1])
    y = np.arange(resolution[0])

    coords = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)

    # XOR to get the checkerboard pattern
    second_color_mask = ((np.floor(coords / square_size) %
                         2) == 1).sum(axis=-1) == 1

    img.reshape(-1,
                4)[second_color_mask] = color_fnc(coords[second_color_mask])
    return img


@as_tensors()
@torch.jit.script
def n_layers_alpha_compositing(
        images: torch.Tensor, zbuffer: torch.Tensor) -> torch.Tensor:
    """
    Applies N layers alpha compositing to the input images,
    bases on the z-buffer values.

    Parameters
    ----------
    images: torch.Tensor
        The input images with shape (N, [..., B], C).
        C must be 4 (RGBA).
        Image values are expected to be in the range [0, 1].

    zbuffer: torch.Tensor
        The z-buffer values for the images with shape (N).
        Images are sorted based on the z-buffer values in ascending order.
        E.g smaller z-buffer values are closer to the camera and will be rendered on top.

    Returns
    -------
    torch.Tensor
        The alpha composited image with shape ([..., B], C).
    """
    N = images.shape[0]
    flattened_shape = images.shape[1:-1]
    C = images.shape[-1]
    if C != 4:
        raise ValueError(
            "The last dimension of the input images must be 4 (RGBA).")

    B = torch.prod(torch.tensor(flattened_shape)).item()
    images = images.reshape(N, B, C)  # (N, B, C)

    order = torch.argsort(zbuffer).unsqueeze(1)
    colors = images[:, :, :3]  # (N, B, 3)
    alphas = images[:, :, 3]
    inv_alphas = (1 - alphas)

    # Apply N object alpha matting. This is done by multiplying the alpha of the object with the inverse of the alphas of the objects before it.
    # We do it by calculating 1-alpha for each object, and
    bidx = torch.arange(B, device=inv_alphas.device).unsqueeze(
        0).repeat(N, 1)  # (N, B)
    sorted_inv_alphas = inv_alphas[order, bidx]
    sorted_alphas = alphas[order, bidx]  # (N, B, T, 1)
    sorted_colors = colors[order, bidx]

    rolled_inv_alpha = torch.roll(sorted_inv_alphas, 1, dims=0)
    rolled_inv_alpha[0] = 1.

    alpha_chain = torch.cumprod(rolled_inv_alpha, dim=0)
    sorted_per_layer_alphas = alpha_chain * sorted_alphas
    fused_color = (sorted_per_layer_alphas.unsqueeze(-1).repeat(1,
                   1, 3) * sorted_colors).sum(dim=0)  # (B, 3)
    out_image = torch.zeros(B, 4)
    out_image[:, :3] = fused_color  # (3, H, W)
    out_image[:, 3] = sorted_per_layer_alphas.sum(dim=0)
    return out_image.reshape(flattened_shape + (4,))


def n_layers_alpha_compositing_numpy(
        images: np.ndarray, zbuffer: np.ndarray) -> np.ndarray:
    """
    Applies N layers alpha compositing to the input images,
    bases on the z-buffer values.

    Parameters
    ----------
    images: np.ndarray
        The input images with shape (N, [..., B], C).
        C must be 4 (RGBA).
        Image values are expected to be in the range [0, 1].

    zbuffer: np.ndarray
        The z-buffer values for the images with shape (N).
        Images are sorted based on the z-buffer values in ascending order.
        E.g smaller z-buffer values are closer to the camera and will be rendered on top.

    Returns
    -------
    np.ndarray
        The alpha composited image with shape ([..., B], C).
    """
    N = images.shape[0]
    flattened_shape = images.shape[1:-1]
    C = images.shape[-1]
    if C != 4:
        raise ValueError(
            "The last dimension of the input images must be 4 (RGBA).")

    B = np.prod(np.array(flattened_shape)).item()
    images = images.reshape(N, B, C)  # (N, B, C)

    order = np.argsort(zbuffer)[:, np.newaxis]
    colors = images[:, :, :3]  # (N, B, 3)
    alphas = images[:, :, 3]
    inv_alphas = (1 - alphas)

    # Apply N object alpha matting. This is done by multiplying the alpha of the object with the inverse of the alphas of the objects before it.
    # We do it by calculating 1-alpha for each object, and
    bidx = np.arange(B)[np.newaxis].repeat(N, axis=0)  # (N, B)
    sorted_inv_alphas = inv_alphas[order, bidx]
    sorted_alphas = alphas[order, bidx]  # (N, B, T, 1)
    sorted_colors = colors[order, bidx]

    rolled_inv_alpha = np.roll(sorted_inv_alphas, 1, axis=0)
    rolled_inv_alpha[0] = 1.

    alpha_chain = np.cumprod(rolled_inv_alpha, axis=0)
    sorted_per_layer_alphas = alpha_chain * sorted_alphas
    fused_color = (sorted_per_layer_alphas[..., np.newaxis].repeat(
        3, axis=-1) * sorted_colors).sum(axis=0)  # (B, 3)
    out_image = np.zeros((B, 4))
    out_image[:, :3] = fused_color  # (3, H, W)
    out_image[:, 3] = sorted_per_layer_alphas.sum(axis=0)
    return out_image.reshape(flattened_shape + (4,))


def smoothing_function(x: torch.Tensor,
                       slope: float,
                       domain: Tuple[float, float],
                       shift: float = 0.0,
                       ) -> torch.Tensor:
    from tools.util.torch import tensorify
    from tools.transforms.min_max import MinMax
    val = torch.tensor(1.5)
    mm = MinMax(new_min=-val, new_max=val)
    mm.min = tensorify(domain[0], dtype=torch.float32)
    mm.max = tensorify(domain[1], dtype=torch.float32)
    mm.fitted = True
    tanh = torch.tanh(((mm(x) + shift) * slope))
    omin = torch.tanh(((-val + shift) * slope))
    omax = torch.tanh(((val + shift) * slope))
    return (tanh - omin) / (omax - omin) * (domain[1] - domain[0]) + domain[0]


def linear_segmented_smoothing(
    x: torch.Tensor,
    thresholds: Tuple[float, float],
    domain: Tuple[float, float],
    slope: float = 1.0,
    shift: float = 0.0,
) -> torch.Tensor:
    """Linear + tanh segmented smoothing function.

    This function is used to smooth the transition between two thresholds, using primarly a tanh function.

    Parameters
    ----------
    x : torch.Tensor
        Input values to be smoothed.
    thresholds : Tuple[float, float]
        The thresholds for the nonlinear area.
    domain : Tuple[float, float]
        The domain of the nonlinear area.
    slope : float, optional
        Slope of the tanh, by default 1.0
    shift : float, optional
        shift of the tanh, by default 0.0

    Returns
    -------
    torch.Tensor
        Smoothed ouput values.
    """
    from functools import partial
    from tools.util.torch import tensorify, grad_at
    y = torch.zeros_like(x, dtype=torch.float32)

    domain = tensorify(domain, dtype=torch.float32)
    thresholds = tensorify(thresholds, dtype=torch.float32)

    in_domain = (domain[0] <= x) & (x < domain[1])

    threshold_lower, threshold_upper = thresholds
    partial_smoothing_function = partial(
        smoothing_function, slope=slope, domain=domain, shift=shift)

    # Nonlinear area
    mask_nonlinear = (x > threshold_lower) & (x < threshold_upper) & in_domain
    y[mask_nonlinear] = partial_smoothing_function(x[mask_nonlinear])

    # Linearer Bereich für x <= threshold_lower
    val_lower = partial_smoothing_function(torch.tensor([threshold_lower]))
    val_upper = partial_smoothing_function(torch.tensor([threshold_upper]))

    mask_linear_lower = (x <= threshold_lower) & in_domain
    slope_lower = grad_at(partial_smoothing_function, threshold_lower).detach()

    y[mask_linear_lower] = slope_lower * \
        (x[mask_linear_lower] - (threshold_lower)) + val_lower

    # Lin area x >= threshold_upper
    mask_linear_upper = (x >= threshold_upper) & in_domain
    slope_upper = grad_at(partial_smoothing_function, threshold_upper).detach()

    y[mask_linear_upper] = slope_upper * \
        (x[mask_linear_upper] - threshold_upper) + val_upper

    # Set x if outside domain
    y[~in_domain] = x[~in_domain]
    return torch.clamp(y, 0, 1).detach()
