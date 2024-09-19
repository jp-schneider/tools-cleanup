import numpy as np
import cv2
from tqdm.auto import tqdm
from tools.util.path_tools import filer

@filer(default_ext='mp4')
def write_mp4(frames: np.ndarray,
              path: str = 'test.mp4',
              fps: float = 24.0,
              progress_bar: bool = False):
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

    Raises
    ------
    ValueError
        If wrong number of channels in frames.
    """
    if len(frames.shape) not in [3, 4]:
        raise ValueError(f"Unsupported frame shape: {frames.shape}.")

    if len(frames.shape) == 4:
        num_frames, height, width, channels = frames.shape
    elif len(frames.shape) == 3:
        num_frames, height, width = frames.shape
        channels = 1
    else:
        raise ValueError(f"Unsupported frame shape: {frames.shape}.")

    frames = frames - frames.min()
    frames = (frames / frames.max() * 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
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
