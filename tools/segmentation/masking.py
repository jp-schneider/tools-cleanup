from typing import Any, Optional, Literal
from tools.util.reflection import class_name
from tools.util.typing import VEC_TYPE
import numpy as np
from tools.util.numpy import numpyify, numpyify_image
from tools.serialization.json_convertible import JsonConvertible
from tools.logger.logging import logger


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


def save_mask(mask: VEC_TYPE,
              path: str,
              spread: bool = False,
              metadata: Optional[dict] = None
              ) -> None:
    """Saves the given (value) based mask to the given path.

    Parameters
    ----------
    mask : VEC_TYPE
        Mask to save, should be of shape C x H x W or H x W (resp. torch.Tensor or np.ndarray)
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

    if len(metadata) > 0:
        # Serialize to Json
        metadata_json = JsonConvertible.convert_to_json_str(metadata,
                                                            handle_unmatched="raise",
                                                            indent=0)

    img = Image.fromarray(mask)
    args = dict()
    if metadata_json is not None:
        from PIL.ExifTags import TAGS, Base
        from PIL.Image import Exif
        make = class_name(save_mask)
        exif = Exif()
        exif[Base.Make] = make
        exif[Base.MakerNote] = metadata_json
        args['exif'] = exif

    img.save(path, **args)
