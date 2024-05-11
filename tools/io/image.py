from typing import Any, Dict, Optional, Literal, Tuple, Union
from tools.util.format import parse_enum
from tools.util.reflection import class_name
from tools.util.typing import VEC_TYPE
import numpy as np
from tools.util.numpy import numpyify, numpyify_image
from tools.serialization.json_convertible import JsonConvertible
from tools.logger.logging import logger
from PIL.Image import Exif, Image
from PIL.ExifTags import Base as ExifTagsBase
from tools.util.format import parse_format_string
from tqdm.auto import tqdm
from tools.util.path_tools import numerated_file_name, read_directory
import os


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
    safe_metadata_load: bool = True
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

    Returns
    -------
    np.ndarray
        Numpy array of the mask.

    OR:

    Tuple[np.ndarray, Dict[Union[str, ExifTagsBase], Any]]
        Tuple of the image and the metadata.
    """
    from PIL import Image
    mask_pil = Image.open(path)
    image = np.array(mask_pil)
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
    if len(metadata) > 0:
        args['exif'] = create_image_exif(metadata)

    Image.fromarray(img).save(path, **args)
    return path
