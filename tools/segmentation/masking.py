from matplotlib.colors import to_rgba
from typing import Any, Dict, List, Optional, Literal, Tuple, Union
from tools.util.format import parse_enum
from tools.util.progress_factory import ProgressFactory
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
from tools.io.image import create_image_exif, load_image_exif, compute_new_size
from tools.transforms.to_numpy_image import ToNumpyImage
import cv2
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
from multiprocessing import Pool


def split_overlap_channel_masks(masks: VEC_TYPE) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Splits a list of channel masks into multiple masks, where each mask has no overlap with the other masks.

    Parameters
    ----------
    masks : VEC_TYPE
        Stack of channel masks of shape [..., B x] C x H x W or [..., B x] H x W x C (resp. torch.Tensor or np.ndarray)

    Returns
    -------
    Tuple[List[np.ndarray], List[List[int]]]
        1. List of masks, where each mask ([..., B x] H x W x C) has no overlap with the other masks.
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
        if (object_values == base_value).any():
            raise ValueError(
                f"Object values must not contain the base value {base_value}, as this would remove the mask!")

        if object_values.shape != (masks.shape[-1],):
            raise ValueError(
                f"Object values shape {object_values.shape} does not match number of masks {masks.shape[-1]}")
        if np.unique(object_values).shape != object_values.shape:
            raise ValueError(
                f"Object values must be unique, got {object_values}")

    def _convert_masks(masks, object_values):
        mask = np.zeros(masks.shape[:-1], dtype=np.int32)
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
    channel_mask = np.zeros(mask.shape + (len(_valid_classes),), dtype=bool)
    for i, c in enumerate(_valid_classes):
        channel_mask[..., i] = (mask == c)
    return channel_mask, _valid_classes


def load_mask(
    path: str,
    is_stack: bool = False,
    read_directory_kwargs: Optional[Dict[str, Any]] = None,
    size: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None,
    progress_bar: bool = False
) -> np.ndarray:
    """Loads a mask from the given path.
    Will reverse the spread if it was applied.

    Parameters
    ----------
    path : str
        Path to the mask.

    is_stack : bool, optional
        Whether the mask is a stack of masks, by default False

    read_directory_kwargs : Optional[Dict[str, Any]], optional
        Optional keyword arguments to pass to the read_directory function, by default None

    size : Optional[Tuple[int, int]], optional
        Size to resize the mask to, by default None
        Should be (H, W)

    Returns
    -------
    np.ndarray
        Numpy array of the mask.
    """
    from PIL import Image
    from PIL.Image import Resampling

    def _read_path(path):
        nonlocal size
        mask_pil = Image.open(path)
        if size is not None or max_size is not None:
            if size is None:
                size = compute_new_size(mask_pil.size, max_size)[::-1]
            mask_pil = mask_pil.resize(
                (size[1], size[0]), resample=Resampling.BILINEAR)
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
              progress_bar: bool = False,
              additional_filename_variables: Optional[Dict[str, Any]] = None,
              index_offset: int = 0,
              ) -> List[str]:
    """Saves the given (value) based mask to the given path.

    Can handle batches, if the mask is of shape B x C x H x W, it will save each mask in the batch to a separate file.
    Paths can be formatted with the index of the mask in the batch.

    Parameters
    ----------
    mask : VEC_TYPE
        Mask to save, should be of shape B x C x H x W or H x W x 1(resp. torch.Tensor or np.ndarray)
        Can have vales in range 0 - 255. If values are floats, an error is raised.
        Expects a value based mask, where each value represents a different object.

    path : str
        Path to save the mask to.
        If the mask is a batch, the path can be a format string, where the index of the mask in the batch can be used.

    spread: bool, optional
        Whether to spread the values of the mask to the full range 0 - 255 to make them visible with inspecting masks.
        Id mappings are save in the exif tag.

    metadata : Optional[dict], optional
        Optional metadata to save with the mask. Will be saved as MakerNote in the exif tag.

    progress_bar : bool, optional
        Whether to show a progress bar when saving a batch of masks, by default False

    additional_filename_variables : Optional[Dict[str, Any]], optional
        Additional variables to use in the filename format string, by default None

    index_offset : int, optional
        Offset to add to the index of the mask in the batch, by default 0
        Can be used if masks are saved with a format string and the index should start at a different value.

    Returns
    -------
    Union[str, List[str]]
    Either:
        str
            Path to the saved file if the mask is not a batch.
        List[str]
            List of saved filenames.

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
    saved_files = []

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
        filenames = parse_format_string(path, [dict(index=i) for i in range(
            mask.shape[0])], additional_variables=additional_filename_variables, index_offset=index_offset)
        if len(set(filenames)) != mask.shape[0]:
            raise ValueError(
                f"Number of filenames {len(filenames)} does not match number of masks {mask.shape[0]} if you specified an index template?")
        it = enumerate(filenames)
        if progress_bar:
            it = tqdm(it, total=mask.shape[0], desc="Saving masks")
        for i, f in it:
            img = Image.fromarray(mask[i].squeeze())
            img.save(f, **args)
            saved_files.append(f)
    else:
        img = Image.fromarray(mask)
        img.save(path, **args)
        return path
    return saved_files


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
        mask_alpha: float = 0.7,
        cmap: Optional[str] = None,
        fps: float = 24.0
) -> str:
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
    from tools.video.utils import write_mp4
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

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
                   for i in range(masks.shape[-1])}
    contour_colors = {i: _get_mask_color(
        i, mask_alpha) for i in range(masks.shape[-1])}
    contour_width = 0

    num_frames = masks.shape[0]
    # num_frames = 10
    for i in tqdm(range(0, num_frames), total=num_frames):
        inpainted_image = images[i].copy()
        for j in range(0, masks.shape[-1]):
            mask = masks[i, ..., j]
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
    write_mp4(inpainted_images, video_path, fps=fps, progress_bar=True)
    return video_path


def inpaint_mask_image(
        input_image: np.ndarray,
        input_mask: np.ndarray,
        mask_color: Any = "tab:blue",
        contour_color: np.ndarray = "black",
        contour_width: int = 5):
    from matplotlib.colors import to_rgba
    assert input_image.shape[:2] == input_mask.shape, 'different shape between image and mask'
    # 0: background, 1: foreground
    mask = np.clip(input_mask, 0, 1)

    # paint mask
    painted_image = inpaint_mask(
        input_image, mask, np.array(to_rgba(mask_color)))
    # paint contour
    if contour_width > 0:
        contour_radius = (contour_width - 1) // 2
        dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist_transform_back = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)

        contour_radius += 2
        contour_mask = np.abs(
            np.clip(dist_map, -contour_radius, contour_radius))
        contour_mask = contour_mask / np.max(contour_mask)
        contour_mask[contour_mask > 0.5] = 1.
        painted_image = inpaint_mask(
            painted_image, 1-contour_mask, np.array(to_rgba(contour_color)))
    return painted_image


def inpaint_mask(image: np.ndarray,
                 mask: np.ndarray,
                 color: np.ndarray,
                 ) -> np.ndarray:
    """Inpaint the mask on the input image.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    mask : np.ndarray
        Mask to be inpainted. Should be in the format (h, w).
        Will be thresholded at 0.5.
    color : np.ndarray
        Color to be inpainted. Should be in the format (4,) in range [0, 1].
        Where the first 3 values are the RGB values and the last value is the alpha value.

    Returns
    -------
    np.ndarray
        The inpainted image.
    """
    # If image not in uint8, convert to uint8
    input_dtype = image.dtype
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.
    mask = mask > 0.5
    image = image.copy()
    alpha = color[3]
    color = color[:3]
    image[mask] = image[mask] * (1-alpha) + color * alpha
    # If image was uint8, convert back to uint8
    if input_dtype == np.uint8:
        image = (image * 255).astype(np.uint8)
    return image


def save_channel_masks(
        masks: np.ndarray,
        mask_directory: str,
        oids: Optional[np.ndarray] = None,
        filename_pattern: str = "img_{index:02d}_ov_{ov_index}.png",
        spread: bool = True,
        index_offset: int = 0,
        additional_filename_variables: Optional[Dict[str, Any]] = None,
        return_in_ov_format: bool = False
) -> Union[List[str], Tuple[Dict[int, List[str]], Dict[int, np.ndarray]]]:
    """Saves a list of channel masks to a directory.
    Will split the masks into non-overlapping value masks and save them

    Parameters
    ----------

    masks : np.ndarray
        List of channel masks of shape [..., B x] H x W x C
    mask_directory : str
        Directory to save the masks to.
    oids : Optional[np.ndarray], optional
        Object ids of the masks, by default None
    filename_pattern : str, optional
        Filename pattern to save the masks, by default "img_{index:02d}_ov_{ov_index}.png"
    spread : bool, optional
        Whether to spread the values of the mask to the full range 0 - 255 to make them visible with inspecting masks, by default
        True
    index_offset : int, optional
        Offset to add to the index of the mask in the batch, by default 0
        Can be used if masks are saved with a format string and the index should start at a different value.
    additional_filename_variables : Optional[Dict[str, Any]], optional
        Additional variables to use in the filename format string, by default None
        These variables will be used in the filename format string.
    return_in_ov_format : bool, optional
        If the return values should be as Dictionary of indexed overlapping aware value masks and their paths, by default False

    Returns
    -------
    If return_in_ov_format is False:
        List[str]
            List of saved filenames.
    else:
        Tuple[Dict[int, List[str]], Dict[int, np.ndarray]]
            1. Dictionary of indexed overlapping aware value masks.
                Key is the overlapping index 
                Value is a list of paths to the masks which belong to the same object in various time steps if the values are the same.
            2. Dictionary of indexed overlapping aware object ids.
                Key is the overlapping index
                Value is a numpy array of object ids as supplied in the oids parameter, or a 1-based index if oids was not supplied.
        
    """
    if oids is None:
        oids = np.arange(1, masks.shape[-1] + 1)

    # Check if oids are unique
    if len(np.unique(oids)) != len(oids):
        raise ValueError("Object ids must be unique. Duplicate ids found: " + str({int(k):v for k,v in zip(*np.unique(oids, return_counts=True)) if v > 1}))

    if additional_filename_variables is None:
        additional_filename_variables = dict()

    if "ov_index" in additional_filename_variables:
        raise ValueError(
            "ov_index is a reserved variable name, this will be automatically set by the function.")

    overlap_free_comb, overlap_free_comb_ids = split_overlap_channel_masks(
        masks)

    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)

    path = os.path.join(mask_directory, filename_pattern)

    saved_paths = []
    ov_dict = dict()
    overlap_free_oids = []
    for i in range(len(overlap_free_comb)):
        m_stack = overlap_free_comb[i]
        m_stack_oids = oids[overlap_free_comb_ids[i]]
        value_mask = channel_masks_to_value_mask(
            m_stack, m_stack_oids, handle_overlap='raise')[..., None]
        p = save_mask(value_mask, path, spread=spread,
                      additional_filename_variables=dict(ov_index=i, **additional_filename_variables), index_offset=index_offset)
        if return_in_ov_format:
            ov_dict[i] = p
            overlap_free_oids.append(m_stack_oids)
        else:
            saved_paths.extend(p)

    if return_in_ov_format:
        return ov_dict, {i:np.array(x, dtype=int) for i,x in enumerate(overlap_free_oids)}
    else:
        return saved_paths


def index_value_masks_folder(path: str, filename_pattern: str = r"img_(?P<index>[0-9]+)_ov_(?P<ov_index>[0-9]+).png", return_dict: bool = False) -> Union[Dict[int, List[str]], Dict[int, List[Dict[str, Any]]]]:
    """Returns a dictionary of indexed overlapping aware value masks from a directory.

    Each Key is the overlapping index of a stack of masks, while masks in different overlapping indices can overlap, while masks in the same overlapping index do not overlap.
    This indexing is used to losslessly store masks in a directory, by saving them as value masks if they have no overlap.

    Parameters
    ----------
    path : str
        Path to the directory where the masks are stored.

    filename_pattern : str, optional
        Pattern to search for and load the masks, by default r"img_(?P<index>[0-9]+)_ov_(?P<ov_index>[0-9]+).png"

    return_dict : bool, optional
        If the return should be a list of dictionaries including the parsed values, by default False
        If False, only the paths will be returned.

    Returns
    -------
    Union[Dict[int, List[str]], Dict[int, List[Dict[str, Any]]]]
        Dictionary of indexed overlapping aware value masks.
        Key is the overlapping index, value is a list of paths to the masks which belong to the same object in various time steps if the values are the same.

        If return_dict is True, the values are dictionaries with the parsed values, while the path is stored under the key 'path'.
    """
    paths = read_directory(path, filename_pattern, parser=dict(
        index=int, ov_index=int), path_key="path")
    ovs = set([x.get('ov_index', 0) for x in paths])
    ov_paths = dict()
    for p in paths:
        ov_index = p.get('ov_index', 0)
        if ov_index not in ov_paths:
            ov_paths[ov_index] = []
        ov_paths[ov_index].append(p)
    for k in ov_paths:
        sorted_paths = sorted(ov_paths[k], key=lambda x: x['index'])
        ov_paths[k] = [p["path"]
                       for p in sorted_paths] if not return_dict else list(sorted_paths)
    return ov_paths


def load_channel_masks(
    mask_directory: str,
    filename_pattern: str = r"img_(?P<index>[0-9]+)_ov_(?P<ov_index>[0-9]+).png",
    output_format: Literal['channel', 'value'] = 'channel',
    overlapping_mask_paths: Optional[Dict[int, List[str]]] = None,
    size: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None,
    progress_bar: bool = False,
    progress_factory: Optional[ProgressFactory] = None
) -> Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Loads overlapping value masks from a directory.

    Parameters
    ----------
    mask_directory : str
        Directory where the masks are stored.
    filename_pattern : str, optional
        Filename pattern to scan for, by default r"img_(?P<index>[0-9]+)_ov_(?P<ov_index>[0-9]+).png"

    output_format : Literal['channel', 'value'], optional
        Format of the output, by default 'channel'
        - 'channel': Returns the channel mask
        - 'value': Returns the value masks

    overlapping_mask_paths : Optional[Dict[int, List[str]]]
        Optional dictionary of overlapping mask paths, by default None
        This can be an already indexed dictionary of mask paths, if not provided, the function will index the mask directory.

    size : Optional[Tuple[int, int]], optional
        Size to resize the mask to, by default None
        Should be (H, W)

    max_size : Optional[int], optional
        Maximum size to resize the mask to, by default None
        Set this size as the maximum size, keeping aspect ratio.

    Returns
    -------
    Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    Either:
        List[np.ndarray]
            List of non-overlapping value masks if output_format == 'value'
        Tuple[np.ndarray, np.ndarray]
            1. The channel mask of shape [..., B x ] H x W x C
            2. The object values of shape (C, ) corresponding to the channel mask index
    """
    if progress_bar:
        if progress_factory is None:
            progress_factory = ProgressFactory()

    if overlapping_mask_paths is None:
        overlapping_mask_paths = index_value_masks_folder(
            mask_directory, filename_pattern)

    # Read all masks
    overlapping_masks = {k: [] for k in overlapping_mask_paths}
    bar = None
    if progress_bar:
        bar = progress_factory.bar(total=sum([len(x) for x in overlapping_mask_paths.values()]), desc="Loading masks",
                                   tag="LOADING_MASKS",
                                   is_reusable=True)
    for ov_index, path_dict_list in overlapping_mask_paths.items():
        for p in path_dict_list:
            overlapping_masks[ov_index].append(load_mask(p, size=size, max_size=max_size))
            if progress_bar:
                bar.update(1)
    # Stack masks
    for k in overlapping_masks:
        overlapping_masks[k] = np.stack(overlapping_masks[k], axis=0)
    value_mask = list(overlapping_masks.values())
    if output_format == 'channel':
        return merge_value_masks_to_channel_mask(value_mask)
    else:
        return value_mask


def merge_value_masks_to_channel_mask(
    value_masks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Merges a list of value masks to a single channel mask.

    Parameters
    ----------
    value_masks : List[np.ndarray]
        List of potentially overlapping value masks.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        1. The channel mask of shape [..., B] H x W x C
        2. The object values of shape (C, 2) (y, x) where y is the index in the value_masks list, and x is the object value / index in the mask stack.
            For the subindex in value mask, order the elements when filterd by the object value (y) in a increasing order.
    """
    comb = [value_mask_to_channel_masks(x) for x in value_masks]
    gids = [x[1] for x in comb]
    lgids = []
    for i, x in enumerate(gids):
        z = np.zeros(len(x))
        z.fill(i)
        lgids.append(np.stack([z, x], axis=-1))
    lgids = np.concatenate(lgids, axis=0)

    channel_masks, channel_ids = np.concatenate(
        [x[0] for x in comb], axis=-1), np.concatenate(gids, axis=-1)

    # Reorder masks
    reordered_indices = np.argsort(lgids[:, 1])
    channel_masks = channel_masks[..., reordered_indices]
    channel_ids = lgids[reordered_indices]
    return channel_masks, channel_ids


def _inpaint_mask(image: np.ndarray,
                  mask: np.ndarray,
                  color: np.ndarray,
                  ) -> np.ndarray:
    """Inpaint the mask on the input image by alpha blending the color.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    mask : np.ndarray
        Mask to be inpainted. Should be in the format (h, w).
        Will be thresholded at 0.5.
    color : np.ndarray
        Color to be inpainted. Should be in the format (4,).
        Where the first 3 values are the RGB values and the last value is the alpha value.

    Returns
    -------
    np.ndarray
        The inpainted image.
    """
    mask = mask > 0.5
    image = image.copy()
    alpha = color[3]
    color = color[:3]
    # Color to 255
    color = np.round((color * 255)).astype('uint8')
    image[mask] = image[mask] * (1-alpha) + color * alpha
    return image.astype('uint8')


def inpaint_mask_image(
        input_image: np.ndarray,
        input_mask: np.ndarray,
        mask_color: Any = "tab:blue",
        contour_color: np.ndarray = "black",
        contour_width: int = 5) -> np.ndarray:
    """Inpaint the mask on the input image.

    Parameters
    ----------
    input_image : np.ndarray
        Input image of shape H x W x C in range [0, 255] uint8
    input_mask : np.ndarray
        Mask to be inpainted. Should be in the format H x W. Will be thresholded at 0.5.
    mask_color : Any, optional
        Mask color to be used for inpainting, by default "tab:blue"
    contour_color : np.ndarray, optional
        A contour color for the mask, by default "black"
    contour_width : int, optional
        A contour width for the mask, by default 5

    Returns
    -------
    np.ndarray
        The inpainted image.
    """
    assert input_image.shape[:2] == input_mask.shape, 'different shape between image and mask'
    # 0: background, 1: foreground
    mask = np.clip(input_mask, 0, 1)
    # paint mask
    painted_image = _inpaint_mask(
        input_image, mask, np.array(to_rgba(mask_color)))
    # paint contour
    if contour_width > 0:
        contour_radius = (contour_width - 1) // 2
        dist_transform_fore = cv2.distanceTransform(
            mask.astype(np.uint8), cv2.DIST_L2, 3)
        dist_transform_back = cv2.distanceTransform(
            1-mask.astype(np.uint8), cv2.DIST_L2, 3)
        dist_map = dist_transform_fore - dist_transform_back
        contour_radius += 2
        contour_mask = np.abs(
            np.clip(dist_map, -contour_radius, contour_radius))
        contour_mask = contour_mask / np.max(contour_mask)
        contour_mask[contour_mask > 0.5] = 1.
        painted_image = _inpaint_mask(
            painted_image, 1-contour_mask, np.array(to_rgba(contour_color)))
    return painted_image


def inpaint_masks(image: np.ndarray, masks: np.ndarray,
                  colors: Optional[np.ndarray] = None,
                  contour_colors: Optional[np.ndarray] = None,
                  contour_width: int = 0) -> np.ndarray:
    """Inpaint an image with multiple masks.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape H x W x C in range [0, 255] uint8
    masks : np.ndarray
        Channel masks to be inpainted. Should be in the format H x W x C. Will be thresholded at 0.5.
    colors : np.ndarray
        The colors to be used for inpainting. Should be in the format C x 4.
    contour_colors : np.ndarray
        The contour colors for the masks. Should be in the format C x 4.
    contour_width : int
        The contour width for the masks.

    Returns
    -------
    np.ndarray
        Masked image.
    """
    H, W, C = masks.shape
    image = image.copy()
    if colors is None:
        cmap = plt.get_cmap("tab10" if C <= 10 else "tab20")
        colors = [np.array(to_rgba(cmap(i % cmap.N))) for i in range(C)]
    if contour_colors is None:
        if contour_width == 0:
            contour_colors = ["black"] * C
        else:
            black_alpha = 0.5
            black = np.array(to_rgba("black"))
            contour_colors = [black_alpha * black[:3] +
                              (1 - black_alpha) * c[:3] for c in colors]
    for i in range(C):
        mask = masks[..., i]
        image = inpaint_mask_image(
            image, mask, colors[i], contour_colors[i], contour_width)
    return image


def inpaint_mask_video(
    images: np.ndarray,
    masks: np.ndarray,
    colors: Optional[List[str]] = None,
    contour_colors: Optional[List[str]] = None,
    contour_width: int = 5,
    num_workers: int = 8,
    sort_mask_by_size: bool = True
) -> np.ndarray:
    """Inpaints masks on a video.

    Parameters
    ----------
    images : np.ndarray
        Input images of shape B x H x W x C in range [0, 255] uint8
    masks : np.ndarray
        Masks to be inpainted. Should be in the format B x H x W x C. Will be thresholded at 0.5.
    colors : Optional[List[str]], optional
        Colors to be used for inpainting, by default None
    contour_colors : Optional[List[str]], optional
        Contour colors for the masks, by default None
    contour_width : int, optional
        Contour width for the masks, by default 5
    sort_mask_by_size : bool, optional
        If the masks should be sorted by size, by default True
    Returns
    -------
    np.ndarray
        The inpainted video. In shape B x H x W x C
    """
    N, H, W, C = masks.shape
    if colors is None:
        cmap = plt.get_cmap("tab10" if C <= 10 else "tab20")
        colors = [np.array(to_rgba(cmap(i % cmap.N))) for i in range(C)]
    if contour_colors is None:
        if contour_width == 0:
            contour_colors = ["black"] * C
        else:
            black_alpha = 0.5
            black = np.array(to_rgba("black"))
            contour_colors = [black_alpha * black[:3] +
                              (1 - black_alpha) * c[:3] for c in colors]
    if colors is None:
        colors = ["tab:blue"] * masks.shape[0]
    if contour_colors is None:
        contour_colors = ["black"] * masks.shape[0]

    # Calculate the size of each mask
    mask_size = masks.sum(axis=(-3, -2))
    # Sort mask by
    if sort_mask_by_size:
        # Mean size of the mask
        mean_size = mask_size.mean(axis=0)
        # Sort the masks by size
        mask_indices = np.argsort(mean_size)[::-1]  # Large first
        masks = masks[..., mask_indices]

    if num_workers > 1:
        with Pool(num_workers) as p:
            inpainted_images = p.starmap(inpaint_masks, [
                (images[i], masks[i], colors, contour_colors, contour_width) for i in range(N)])
        return np.stack(inpainted_images, axis=0)
    else:
        inpainted_images = np.zeros_like(images)
        for i in range(N):
            inpainted_images[i] = inpaint_masks(
                images[i], masks[i], colors, contour_colors, contour_width)
        return inpainted_images


def fuse_masks(masks: np.ndarray, fuse_indices: List[List[int]], ignore_indices: Optional[List[int]] = None) -> np.ndarray:
    """Fuses masks together, based on the given indices.

    Parameters
    ----------
    masks : np.ndarray
        Stack of channel masks of shape H x W x C
    fuse_indices : List[List[int]]
        The indices to fuse together. Each list in the list represents a group of masks to fuse.
        Indices which are not in the list will be appended, unless they are in the ignore_indices list.
        Shape is (N, M) where N is the number of groups and M is the number of masks to fuse in each group.

    ignore_indices : Optional[List[int]], optional
        Ignore indices which should be not in the result mask., by default None

    Returns
    -------
    np.ndarray
        H x W x N + ? mask, where N is the number of groups and ? is the number of masks which are not in the groups and also not ignored.
    """
    H, W, C = masks.shape
    # Count the number of masks to fuse
    indices_in_channel = []
    covered_channels = set()
    for indices in fuse_indices:
        covered_channels.update(indices)
        indices_in_channel.append(indices)
    mask_indices = np.arange(C)
    indices_to_ignore = set(
        ignore_indices) if ignore_indices is not None else set()
    indices_to_fuse = set(mask_indices) - covered_channels - indices_to_ignore
    for single_idx in list(indices_to_fuse):
        indices_in_channel.append([single_idx])
    # Create the fused mask
    fused_masks = np.zeros((H, W, len(indices_in_channel)), dtype=np.bool_)
    for i, indices in enumerate(indices_in_channel):
        fused_masks[:, :, i] = np.any(masks[:, :, indices], axis=2)
    return fused_masks, indices_in_channel


def fuse_timed_stack(masks: np.ndarray, fuse_indices: List[List[int]], ignore_indices: Optional[List[int]] = None, ignore_all_others: bool = False) -> np.ndarray:
    """Given a stack of masks, fuses them together based on the given indices.
    Masks can now have a time dimension.

    Parameters
    ----------
    masks : np.ndarray
        Masks to fuse of shape T x H x W x C
    fuse_indices : List[List[int]]
        The indices to fuse together. Each list in the list represents a group of masks to fuse.
    ignore_indices : Optional[List[int]], optional
        Indices which should be ignored, by default None
    ignore_all_others : bool, optional
        If the function should only return the masks fused, by default False

    Returns
    -------
    np.ndarray
        T x H x W x C fused masks
    """
    T, H, W, _ = masks.shape
    t = 0
    init_fuse, indices_in_channel = fuse_masks(
        masks[t], fuse_indices, ignore_indices)
    if ignore_all_others:
        init_fuse = init_fuse[..., :len(fuse_indices)]
        indices_in_channel = indices_in_channel[:len(fuse_indices)]
        used_indices = set(
            [x for sublist in indices_in_channel for x in sublist])
        all_indices = set(range(masks.shape[-1]))
        ignore_indices = list(all_indices - used_indices)
    C = init_fuse.shape[-1]
    ret = np.zeros((T, H, W, C), dtype=np.bool_)
    ret[t] = init_fuse
    for t in range(1, T):
        fused_mask, _ = fuse_masks(masks[t], fuse_indices, ignore_indices)
        ret[t] = fused_mask
    return ret
