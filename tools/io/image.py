from typing import Any, Dict, List, Optional, Literal, Tuple, Union

import torch
from tools.util.format import parse_enum
from tools.util.reflection import class_name
from tools.util.torch import tensorify_image
from tools.util.typing import VEC_TYPE
import numpy as np
from tools.transforms.to_numpy import numpyify
from tools.transforms.to_numpy_image import numpyify_image
from tools.serialization.json_convertible import JsonConvertible
from tools.logger.logging import logger
from PIL.Image import Exif, Image
from PIL.ExifTags import Base as ExifTagsBase
from tools.util.format import parse_format_string
from tqdm.auto import tqdm
from tools.util.path_tools import numerated_file_name, read_directory
import os
from torch.nn.functional import grid_sample


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
    if make is not None and make == class_name(create_image_exif):
        metadata_json = exif.get(ExifTagsBase.MakerNote, None)
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
        Is exclusive with max_size.

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


def save_image(image: VEC_TYPE,
               path: str,
               override: bool = False,
               mkdirs: bool = True,
               metadata: Optional[Dict[Union[str, ExifTagsBase], Any]] = None) -> str:
    """Saves numpy array or torch tensor as an image.

    Parameters
    ----------
    data : VEC_TYPE
        Data to be saved as an image. Should be in the shape (H, W, C)
        for numpy arrays or (C, H, W) torch tensors.
    path : str
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
    str
        The path where the image was saved.
    """
    from PIL import Image
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


def compute_new_size(image_shape: Tuple[int, int], max_size: int) -> Tuple[int, int]:
    aspect = image_shape[1] / image_shape[0]
    # Get if in landscape or portrait
    if image_shape[1] > image_shape[0]:
        # Landscape
        new_size = (int(max_size / aspect), max_size)
    else:
        # Portrait
        new_size = (max_size, int(max_size * aspect))
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
        image: np.ndarray,
        max_size: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None
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
        Is exclusive with max_size.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    from torchvision.transforms import Compose, Resize, ToTensor
    from tools.transforms.to_numpy_image import ToNumpyImage
    if len(image.shape) == 4:
        B, H, W, C = image.shape
    elif len(image.shape) == 3:
        H, W, C = image.shape
        B = 1
    else:
        raise ValueError("Image should have shape B x H x W x C or H x W x C")
    if max_size is None and size is None:
        raise ValueError("Either max_size or size should be provided.")
    if max_size is not None and size is not None:
        raise ValueError(
            "Either max_size or size should be provided, not both.")
    if max_size is not None:
        new_size = compute_new_size((H, W), max_size)
        transforms = Compose([ToTensor()] + ([Resize(new_size)]
                             if (H > max_size or W > max_size) else []) + [ToNumpyImage()])
    else:
        transforms = Compose([ToTensor(), Resize(size), ToNumpyImage()])
    return transforms(image)


def load_image_stack(
        path: str,
        filename_format: str = r"(?P<index>[0-9]+).png",
        max_size: Optional[int] = None
) -> np.ndarray:
    """Helper function to load images for a given path and filename format.

    Parameters
    ----------
    path : str
        Path to the images.

    filename_format : str, optional
        Filename format regex, by default r"(?P<index>[0-9]+).png"
        Images will be sorted by index.

    max_size : Optional[int], optional
        If the image should be resized to a maximum size, by default None
        The image will be resized to the maximum size while maintaining aspect ratio.

    Returns
    -------
    np.ndarray
        Image stack in shape B x H x W x C.
        B order is index order ascending.
    """
    image_paths = read_directory(
        path, filename_format, parser=dict(index=int), path_key="path")
    sorted_index = sorted(image_paths, key=lambda x: x["index"])

    img = load_image(sorted_index[0]["path"], max_size=max_size)
    images = np.zeros((len(sorted_index), *img.shape), dtype=img.dtype)
    images[0] = img

    it = tqdm(total=len(sorted_index), desc="Loading images")
    it.update(1)
    for i in range(1, len(sorted_index)):
        images[i] = load_image(sorted_index[i]["path"], max_size=max_size)
        it.update(1)
    return images


def save_image_stack(images: VEC_TYPE,
                     format_string_path: str,
                     override: bool = False, mkdirs: bool = True,
                     metadata: Optional[Dict[Union[str,
                                                   ExifTagsBase], Any]] = None,
                     progress_bar: bool = False,
                     additional_filename_variables: Optional[Dict[str, Any]] = None
                     ) -> List[str]:
    """Saves the given image stack.

    Expects to get a stack of images in the shape B x H x W x C for numpy or B x C x H x W if tensor.

    Parameters
    ----------
    images : VEC_TYPE
        Image stack to save.
        Should be in the shape B x H x W x C for numpy or B x C x H x W if tensor.

    format_string_path : str
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

    filenames = parse_format_string(format_string_path, [dict(index=i) for i in range(
        images.shape[0])], additional_variables=additional_filename_variables)
    if len(set(filenames)) != images.shape[0]:
        raise ValueError(
            f"Number of filenames {len(filenames)} does not match number of masks {images.shape[0]} if you specified an index template?")
    it = enumerate(filenames)
    if progress_bar:
        it = tqdm(it, total=images.shape[0], desc="Saving images")
    for i, f in it:
        sf = save_image(images[i], f, override=override,
                        mkdirs=False, metadata=metadata)
        saved_files.append(sf)
    return saved_files
