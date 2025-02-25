from typing import Generator, Optional, Union, Tuple, Literal, Optional
import numpy as np
import cv2
from tqdm.auto import tqdm
from tools.util.path_tools import filer
from tools.transforms.to_numpy_image import ToNumpyImage
import sys
from tools.util.typing import DEFAULT
import os
from tools.util.format import get_leading_zeros_format_string
from tools.video.inpaint_writer import InpaintWriter
from tools.util.sized_generator import SizedGenerator
from tools.util.progress_factory import ProgressFactory


@filer(default_ext='mp4')
def write_mp4(frames: np.ndarray,
              path: str = 'test.mp4',
              fps: float = 24.0,
              progress_bar: bool = False,
              frame_counter: bool = False,
              codec: str = DEFAULT):
    """Writes the frames to a video file.

    Parameters
    ----------
    frames : np.ndarray
        Frames to write to video in shape BxHxWxC or BxHxW. C is either 1 or 3.
    path : str, optional
        Path to the video file, by default 'test.mp4'
    fps : float, optional
        Fps in the video, by default 24.0
    progress_bar : bool, optional
        Show progress bar, by default False
    codec : str, optional
        Codec to use for the video, by default DEFAULT
        If not set, will use the environment variable VIDEO_CODEC or 'avc1' if not set.

    Raises
    ------
    ValueError
        If wrong number of channels in frames.
    """
    numpyify_image = ToNumpyImage(output_dtype=np.uint8)

    frames = numpyify_image(frames)

    if len(frames.shape) not in [3, 4]:
        raise ValueError(f"Unsupported frame shape: {frames.shape}.")

    if len(frames.shape) == 4:
        num_frames, height, width, channels = frames.shape
    elif len(frames.shape) == 3:
        num_frames, height, width = frames.shape
        channels = 1
    else:
        raise ValueError(f"Unsupported frame shape: {frames.shape}.")

    if frame_counter:
        from tools.io.image import put_text
        fmt = get_leading_zeros_format_string(num_frames)
        for i, frame in enumerate(frames):
            text = fmt.format(i)
            frames[i] = put_text(frame.copy(), text,
                                 placement="top-right", background_stroke=1)

    if codec == DEFAULT:
        codec = os.environ.get('VIDEO_CODEC', 'avc1')

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    bar = None
    if progress_bar:
        bar = tqdm(total=num_frames, desc='Writing video frames')

    if channels in [3, 4]:  # RGB(A) -> BGR
        for frame in frames:
            video.write(frame[:, :, [2, 1, 0]])
            if progress_bar:
                bar.update(1)
    elif channels == 1:  # grayscale
        for frame in frames:
            video.write(frame[:, :, None].repeat(3, 2))
            if progress_bar:
                bar.update(1)
    else:
        raise ValueError(f"Unsupported channel size: {channels}.")

    cv2.destroyAllWindows()
    video.release()
    return path


@filer(default_ext='mp4')
def write_mp4_generator(
        frame_generator: Generator[np.ndarray, None, None],
        path: str = 'test.mp4',
        fps: float = 24.0,
        title: str = None,
        frame_counter: bool = False,
        codec: str = DEFAULT,
        progress_bar: bool = False,
        progress_factory: Optional[ProgressFactory] = None,
        frame_counter_offset: int = 0
):
    """Writes the frames to a video file.

    Parameters
    ----------
    frame_generator : np.ndarray
        Frame generator to write frames to a video in shape HxWxC or HxW. C is either 1 or 3.
    path : str, optional
        Path to the video file, by default 'test.mp4'
    fps : float, optional
        Fps in the video, by default 24.0
    title : str, optional
        Title to show in the video, by default None
        Will be inpainted in the video if set.
    frame_counter : bool, optional
        Show frame counter in the video, by default False
        Will be inpainted in the video if set, in the top right corner.
    codec : str, optional
        Codec to use for the video, by default DEFAULT
        If not set, will use the environment variable VIDEO_CODEC or 'avc1' if not set.
    progress_bar : bool, optional
        Show progress bar, by default False
    frame_counter_offset : int, optional
        Offset for the frame counter, by default 0

    Raises
    ------
    ValueError
        If wrong number of channels in frames.
    """

    size = None
    if isinstance(frame_generator, SizedGenerator):
        size = len(frame_generator)

    fmt = get_leading_zeros_format_string(size if size is not None else 100)

    bar = None
    if progress_bar:
        if progress_factory is None:
            progress_factory = ProgressFactory()
        bar = progress_factory.bar(
            total=size, desc='Writing video frames', is_reusable=True, tag='write_mp4_generator')

    with InpaintWriter(path,
                       fps=fps,
                       inpaint_title=title,
                       inpaint_counter=frame_counter,
                       counter_format=fmt,
                       use_transparency_grid=True,
                       codec=codec,
                       counter_offset=frame_counter_offset
                       ) as writer:
        for frame in frame_generator:
            writer.write(frame)
            if progress_bar:
                bar.update(1)
    return path


@filer(default_ext='mp4')
def write_mask_mp4_generator(
        frame_generator: Generator[np.ndarray, None, None],
        mask_generator: Generator[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], None, None],
        path: str = 'test.mp4',
        fps: float = 24.0,
        title: str = None,
        frame_counter: bool = False,
        darkening_background_alpha: float = 0.5,
        codec: str = DEFAULT,
        progress_bar: bool = False,
        backend: Literal["opencv", "matplotlib"] = "opencv",
        progress_factory: Optional[ProgressFactory] = None):
    """Writes the frames to a video file.

    Parameters
    ----------
    frame_generator : Generator[np.ndarray, None, None]
        Frame generator to write frames to a video in shape HxWxC or HxW. C is either 1 or 3.
    mask_generator : np.ndarray
        Mask generator to write frames to a video in shape HxWxC or HxW. C should be the number of masks.
        Accepts channel masks.
    path : str, optional
        Path to the video file, by default 'test.mp4'
    fps : float, optional
        Fps in the video, by default 24.0
    title : str, optional
        Title to show in the video, by default None
        Will be inpainted in the video if set.
    frame_counter : bool, optional
        Show frame counter in the video, by default False
        Will be inpainted in the video if set, in the top right corner.
    codec : str, optional
        Codec to use for the video, by default DEFAULT
        If not set, will use the environment variable VIDEO_CODEC or 'avc1' if not set.
    progress_bar : bool, optional
        Show progress bar, by default False

    Raises
    ------
    ValueError
        If wrong number of channels in frames.
    """
    from tools.segmentation.masking import inpaint_masks
    size = None
    if isinstance(frame_generator, SizedGenerator):
        size = len(frame_generator)

    fmt = get_leading_zeros_format_string(size if size is not None else 100)

    bar = None
    if progress_bar:
        if progress_factory is None:
            progress_factory = ProgressFactory()
        bar = progress_factory.bar(
            total=size, desc='Writing video frames', is_reusable=True, tag='write_mp4_generator')

    with InpaintWriter(path,
                       fps=fps,
                       inpaint_title=title,
                       inpaint_counter=frame_counter,
                       counter_format=fmt,
                       use_transparency_grid=True,
                       codec=codec) as writer:
        for frame in frame_generator:
            masks = next(mask_generator)
            oids = None
            if isinstance(masks, tuple):
                masks, oids = masks
            if backend == "opencv":
                inpainted_frame = inpaint_masks(frame, masks)
            elif backend == "matplotlib":
                from tools.viz.matplotlib import plot_mask, figure_to_numpy
                import matplotlib.pyplot as plt
                with plt.ioff():
                    fig = plot_mask(frame,
                                    masks,
                                    labels=[
                                        str(x) for x in oids] if oids is not None else None,
                                    inpaint_indices=True,
                                    filled_contours=False,
                                    lined_contours=True,
                                    legend=False, tight=True,
                                    inpaint_title=False,
                                    overlap_area=[],
                                    darkening_background=darkening_background_alpha,
                                    )
                    inpainted_frame = figure_to_numpy(fig)
                    plt.close(fig)
            else:
                raise ValueError(f"Unsupported backend: {backend}.")
            writer.write(inpainted_frame)
            if progress_bar:
                bar.update(1)
    return path
