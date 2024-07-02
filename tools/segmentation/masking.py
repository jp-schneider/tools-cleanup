from typing import Any, Dict, List, Optional, Literal, Tuple, Union
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
from tools.transforms.to_numpy_image import ToNumpyImage


def split_overlap_channel_masks(masks: VEC_TYPE) -> List[np.ndarray]:
    """Splits a list of channel masks into multiple masks, where each mask has no overlap with the other masks.

    Parameters
    ----------
    masks : VEC_TYPE
        Stack of channel masks of shape C x H x W or H x W x C (resp. torch.Tensor or np.ndarray)

    Returns
    -------
    Tuple[List[np.ndarray], List[List[int]]]
        1. List of masks, where each mask (H x W x C) has no overlap with the other masks.
         E.g list items are disjunct to each other.
        2. List of indices, where each index corresponds to the index of the mask in the original mask stack.
    """

    masks = numpyify_image(masks)
    if len(masks.shape) == 2:
        masks = masks[..., None]

    occupancy_masks = [masks[..., 0]]
    output_masks = [[masks[..., 0]]]
    indices = [[0]]

    for i in range(1, masks.shape[-1]):
        # Get occupancy mask
        current_mask = masks[..., i]
        mask_assigned = False

        for o in range(len(occupancy_masks)):
            o_mask = occupancy_masks[o]
            # Check if the mask area is already occupied
            if np.any(o_mask & current_mask):
                continue
            else:
                occupancy_masks[o] = o_mask | current_mask
                output_masks[o].append(current_mask)
                indices[o].append(i)
                mask_assigned = True
                break

        # If mask was not assigned, create a new occupancy mask for it
        if not mask_assigned:
            occupancy_masks.append(current_mask)
            output_masks.append([current_mask])
            indices.append([i])
    # Stack the masks
    return [np.stack(x, axis=-1) for x in output_masks], indices


def channel_masks_to_value_mask(masks: VEC_TYPE,
                                object_values: Optional[VEC_TYPE] = None,
                                handle_overlap: Literal['raise', 'ignore',
                                                        'warning', 'warning+exclude'
                                                        'multi_mask',
                                                        ] = 'warning',
                                base_value: Any = 0.
                                ) -> Union[np.ndarray, Tuple[List[np.ndarray], List[List[int]]]]:
    """Converts a list of channel masks to a single mask with a new value per mask object.

    Parameters
    ----------
    masks : VEC_TYPE
        List of channel masks of shape C x H x W or H x W x C (resp. torch.Tensor or np.ndarray)

    object_values : Optional[VEC_TYPE], optional
        The object values to assign to the mask, by default 1, 2, 3, ...
        These values will be used within the mask to identify the object.
        Should be of shape (C, ) where C is the number of masks.

    handle_overlap : Literal['raise', 'ignore', 'warning', 'warning+exclude', 'multi_mask'], optional
        How to handle overlap in the masks.
        - 'raise': Raises an error if overlap is detected.
        - 'ignore': Ignores overlap, the last mask will be used.
        - 'warning': Logs a warning if overlap is detected. Then proceeds as 'ignore'.
        - 'warning+exclude': Logs a warning if overlap is detected and excludes the overlapping parts, first mask is dominant.
        - 'multi_mask': Returns a list of masks, where each mask stack has no overlap with the other masks.

    base_value : Any, optional
        The base value to fill the mask with, by default 0.


    Returns
    -------

    Union[np.ndarray, Tuple[List[np.ndarray], List[List[int]]]
        if handle_overlap != 'multi_mask'
            np.ndarray
                Single mask with multiple channels, where each number represents a different object.
                Shape is H x W
        else
            Tuple[List[np.ndarray], List[List[int]]]
                1. List of masks, where each mask (H x W x C) has no overlap with the other masks.
                2. List of indices, where each index corresponds to the index of the mask in the original mask stack.
    """
    masks = numpyify_image(masks)
    object_values = numpyify(
        object_values) if object_values is not None else None

    if object_values is None:
        object_values = np.arange(1, masks.shape[-1] + 1)
    else:
        if object_values.shape != (masks.shape[-1],):
            raise ValueError(
                f"Object values shape {object_values.shape} does not match number of masks {masks.shape[-1]}")
        if np.unique(object_values).shape != object_values.shape:
            raise ValueError(
                f"Object values must be unique, got {object_values}")

    def _convert_masks(masks, object_values):
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
                        logger.warning(
                            f"Excluded {duplicate_class.sum()} pixels")
                    else:
                        raise ValueError(
                            f"Unknown overlap handling {handle_overlap}")
            mask = np.where(fill, object_values[i], mask)
        return mask

    if not handle_overlap == "multi_mask":
        return _convert_masks(masks, object_values)
    else:
        # For multi_mask, we need to split the masks
        masks, indices = split_overlap_channel_masks(masks)
        ovals = []
        for sub_idx in indices:
            vals = np.zeros(len(sub_idx))
            for i, idx in enumerate(sub_idx):
                vals[i] = object_values[idx]
            ovals.append(vals)
        return [_convert_masks(m, o) for m, o in zip(masks, ovals)], indices
    return mask


def value_mask_to_channel_masks(
    mask: VEC_TYPE,
    ignore_value: Optional[Union[int, List[int]]] = None,
    background_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a value mask, where objects are identified as different values, to a channel mask, where each channel represents a different object.

    Parameters
    ----------
    mask : VEC_TYPE
        The mask as a value mask, e.g. where each value represents a different object.
        Should be of shape B x H x W
    ignore_value : Optional[Union[int, List[int]]], optional
        Values to ignore when creating the mask, by default None
    background_value : int, optional
        Value which is treated as the background value, by default 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        1. The channel mask of shape H x W x C
        2. The object values of shape (C, ) (as they where in mask) corresponding to the channel mask index
    """
    mask = ToNumpyImage()(mask)
    mask = mask.squeeze()
    if len(mask.shape) not in [2, 3]:
        raise ValueError(f"Value-Mask should be 2D or 3D, got {mask.shape}")
    invalid_values = set([background_value])
    if ignore_value is not None:
        if isinstance(ignore_value, int):
            invalid_values.add(ignore_value)
        else:
            invalid_values.update(ignore_value)
    vals = np.unique(mask)
    _valid_classes = np.stack([x for x in vals if x not in invalid_values])
    channel_mask = np.zeros(mask.shape + (len(_valid_classes),))
    for i, c in enumerate(_valid_classes):
        channel_mask[..., i] = (mask == c)
    return channel_mask, _valid_classes


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


def masks_to_inpaint_video(
        images: VEC_TYPE,
        masks: VEC_TYPE,
        video_path: Optional[str] = None,
        mask_alpha: float = 0.5,
        cmap: Optional[str] = None) -> str:
    """Creates a video from images and masks, which will be inpainted.

    Parameters
    ----------
    images : VEC_TYPE
        Stack of images of shape B x C x H x W x C or B x H x W x C (resp. torch.Tensor or np.ndarray)
    masks : VEC_TYPE
        Stack of masks of shape B x C x H x W or B x H x W x C (resp. torch.Tensor or np.ndarray)
    video_path : Optional[str], optional
        Path to the output video, by default None
    mask_alpha : float, optional
        Alpha value of the mask, by default 0.5
    cmap : Optional[str], optional
        Colormap to draw mask colors from, by default None

    Returns
    -------
    str
        Path to the video
    """

    to_numpy = ToNumpyImage()
    images = to_numpy(images)
    masks = to_numpy(masks)

    if len(images.shape) != 4:
        raise ValueError(
            "Images must be of shape B x H x W x C or B x C x H x W x C")

    if len(masks.shape) != 4:
        raise ValueError(
            "Masks must be of shape B x H x W x C or B x C x H x W x C")

    if images.shape[0] != masks.shape[0]:
        raise ValueError(
            "Number of images and masks must match in the batch dimension.")

    if video_path is None:
        video_path = os.path.join("temp", "inpainted.mp4")

    if cmap is None:
        cmap = plt.get_cmap("tab10" if masks.shape[-1] <= 10 else "tab20")

    inpainted_images = []

    def _get_mask_color(i, alpha=0.5):
        col = np.array(to_rgba(cmap(i)))
        col[-1] = alpha
        return col

    mask_colors = {i: _get_mask_color(i, mask_alpha)
                   for i in range(mask_stack.shape[-1])}
    contour_colors = {i: _get_mask_color(
        i, mask_alpha) for i in range(mask_stack.shape[-1])}
    contour_width = 0

    num_frames = mask_stack.shape[0]
    # num_frames = 10
    for i in tqdm(range(0, num_frames), total=num_frames):
        image = images[i]
        inpainted_image = None
        for j in range(0, mask_stack.shape[-1]):
            mask = mask_stack[i, ..., j]
            if mask.sum() == 0:
                continue
            inpainted_image = inpaint_mask_image(inpainted_image, mask,
                                                 mask_color=mask_colors[j],
                                                 contour_width=contour_width,
                                                 contour_color=contour_colors[j]
                                                 )
        inpainted_images.append(inpainted_image)
    inpainted_images = np.stack(inpainted_images, axis=0)

    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
    write_mp4(inpainted_images, video_path, fps=24, progress_bar=True)
    return video_path
