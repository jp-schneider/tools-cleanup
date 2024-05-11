from typing import Any, Dict, Optional, Literal, Union
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
from tools.util.path_tools import read_directory
import os
from tools.io.image import create_image_exif, load_image_exif


def channel_masks_to_value_mask(masks: VEC_TYPE,
                                object_values: Optional[VEC_TYPE] = None,
                                handle_overlap: Literal['raise', 'ignore',
                                                        'warning', 'warning+exclude'] = 'warning',
                                base_value: Any = 0.
                                ) -> np.ndarray:
    """Converts a list of channel masks to a single mask with a new value per mask object.

    Parameters
    ----------
    masks : VEC_TYPE
        List of channel masks of shape C x H x W or H x W x C (resp. torch.Tensor or np.ndarray)

    object_values : Optional[VEC_TYPE], optional
        The object values to assign to the mask, by default 1, 2, 3, ...
        These values will be used within the mask to identify the object.
        Should be of shape (C, ) where C is the number of masks.

    Returns
    -------
    VEC_TYPE
        Single mask with multiple channels, where each number represents a different object.
        Shape is H x W
    """
    masks = numpyify_image(masks)
    object_values = numpyify(object_values)

    if object_values is None:
        object_values = np.arange(1, masks.shape[-1] + 1)
    else:
        if object_values.shape != (masks.shape[-1],):
            raise ValueError(
                f"Object values shape {object_values.shape} does not match number of masks {masks.shape[-1]}")
        if np.unique(object_values).shape != object_values.shape:
            raise ValueError(
                f"Object values must be unique, got {object_values}")

    mask = np.zeros(masks.shape[:-1], dtype=masks.dtype)
    mask.fill(base_value)
    for i in range(masks.shape[-1]):
        fill = masks[..., i] > 0

        if mask[fill].sum() != 0:
            # Overlap in classes.
            if handle_overlap == 'ignore':
                pass
            else:
                overlap_classes = ', '.join(
                    [str(x) for x in np.unique(mask[fill]).astype(int).tolist() if x != 0])
                if handle_overlap == 'raise':
                    raise ValueError(
                        f"Overlap in classes detected, class {object_values[i]} overlaps with class(es) {overlap_classes}")
                elif handle_overlap == 'warning':
                    logger.warning(
                        f"Overlap in classes detected, class {object_values[i]} overlaps with class(es) {overlap_classes}")
                elif handle_overlap == 'warning+exclude':
                    logger.warning(
                        f"Overlap in classes detected, class {object_values[i]} overlaps with class(es) {overlap_classes}, excluding it")
                    duplicate_class = (mask != 0) & fill
                    fill = fill & ~duplicate_class
                    mask[duplicate_class] = 0.
                    logger.warning(f"Excluded {duplicate_class.sum()} pixels")
                else:
                    raise ValueError(
                        f"Unknown overlap handling {handle_overlap}")
        mask = np.where(fill, object_values[i], mask)
    return mask


def load_mask(
    path: str,
    is_stack: bool = False,
    read_directory_kwargs: Optional[Dict[str, Any]] = None,
    progress_bar: bool = False
) -> np.ndarray:
    """Loads a mask from the given path.
    Will reverse the spread if it was applied.

    Parameters
    ----------
    path : str
        Path to the mask.

    Returns
    -------
    np.ndarray
        Numpy array of the mask.
    """
    from PIL import Image

    def _read_path(path):
        mask_pil = Image.open(path)
        # Load metadata
        metadata = load_image_exif(mask_pil, safe_load=True)
        spread = metadata.get('spread', None)
        mask = np.array(mask_pil)
        # If spread was applied, reverse it
        if spread is not None:
            msk_copy = np.zeros_like(mask, dtype=mask.dtype)
            for k, v in spread.items():
                msk_copy[mask == int(k)] = int(v)
            mask = msk_copy
        return mask
    if not is_stack:
        return _read_path(path)
    # Parse filename which should be a format string
    rdkw = dict() if read_directory_kwargs is None else read_directory_kwargs
    file_name_split = path.split("/")
    file_name_pattern = file_name_split[-1]
    dirname = "/".join(file_name_split[:-1])
    files = [x["path"]
             for x in read_directory(dirname, file_name_pattern, **rdkw)]
    masks = []
    if progress_bar:
        files = tqdm(files, desc="Loading masks")
    for f in files:
        m = _read_path(f)
        masks.append(m)
    return np.stack(masks, axis=0)


def save_mask(mask: VEC_TYPE,
              path: str,
              spread: bool = False,
              metadata: Optional[dict] = None,
              progress_bar: bool = False
              ) -> None:
    """Saves the given (value) based mask to the given path.

    Can handle batches, if the mask is of shape B x C x H x W, it will save each mask in the batch to a separate file.
    Paths can be formatted with the index of the mask in the batch.

    Parameters
    ----------
    mask : VEC_TYPE
        Mask to save, should be of shape B x C x H x W or H x W x 1(resp. torch.Tensor or np.ndarray)
        Can have vales in range 0 - 255. If values are floats, an error is raised.
    path : str
        Path to save the mask to.

    spread: bool, optional
        Whether to spread the values of the mask to the full range 0 - 255 to make them visible with inspecting masks.
        Id mappings are save in the exif tag.

    metadata : Optional[dict], optional
        Optional metadata to save with the mask. Will be saved as MakerNote in the exif tag.

    Raises
    ------
    ValueError
        If the mask contains floats or values outside the range 0 - 255.
    """
    from PIL import Image
    if metadata is None:
        metadata = dict()
    metadata_json = None

    mask = numpyify_image(mask)

    # Check if all values are ints
    if not np.all((mask.astype(int) == mask)):
        raise ValueError("Mask must be integer values")
    # Check if all values are in range 0 - 255
    if not np.all((mask >= 0) & (mask <= 255)):
        raise ValueError("Mask must be in range 0 - 255")
    # Cast to uint8
    mask = mask.astype(np.uint8)

    if spread:
        unique = np.unique(mask)
        maps = np.arange(0, 256, np.floor(
            255 // (unique.shape[0] - 1)), dtype=np.uint8)
        metadata['spread'] = {x[0].item(): x[1].item()
                              for x in zip(maps, unique)}
        msk_copy = np.zeros_like(mask, dtype=mask.dtype)
        for i in range(unique.shape[0]):
            msk_copy[mask == unique[i]] = maps[i]
        mask = msk_copy

    args = dict()
    if len(metadata) > 0:
        args['exif'] = create_image_exif(metadata)

    if len(mask.shape) == 4:
        filenames = parse_format_string(
            path, [dict(index=i) for i in range(mask.shape[0])])
        if len(set(filenames)) != mask.shape[0]:
            raise ValueError(
                f"Number of filenames {len(filenames)} does not match number of masks {mask.shape[0]} if you specified an index template?")
        it = enumerate(filenames)
        if progress_bar:
            it = tqdm(it, total=mask.shape[0], desc="Saving masks")
        for i, f in it:
            img = Image.fromarray(mask[i].squeeze())
            img.save(f, **args)
    else:
        img = Image.fromarray(mask)
        img.save(path, **args)


def convert_batch_instance_dict(frames: Dict[int, Dict[int, np.ndarray]], offset: 0) -> np.ndarray:
    keys = frames.keys()
    mask_shape = next(iter(next(iter(frames.values())).values())).shape
    items = []

    instance_ids = set()
    for k in keys:
        local_keys = frames[k].keys()
        instance_ids = instance_ids.union(set(local_keys))

    instance_ids = sorted(instance_ids)

    for i, k in enumerate(sorted(keys)):
        instance_mask = np.zeros(
            (*mask_shape, len(instance_ids)), dtype=np.uint8)
        for j in instance_ids:
            v = frames[k].get(j, None)
            if v is not None:
                instance_mask[..., j - offset] = v
        items.append(instance_mask)
    return np.stack(items, axis=0)
