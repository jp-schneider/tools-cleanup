import numpy as np
import cv2
from tqdm.auto import tqdm
from tools.util.path_tools import filer
from tools.transforms.to_numpy_image import ToNumpyImage
import sys
from tools.util.typing import DEFAULT
import os
from tools.util.format import get_leading_zeros_format_string


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
