from typing import Optional, Tuple, Union
import cv2
import numpy as np
from tools.util.typing import DEFAULT, _DEFAULT, VEC_TYPE
import os
from tools.transforms.to_numpy_image import ToNumpyImage


class Writer():
    """Video writer context manager arround openCV VideoWriter class."""

    @classmethod
    def get_fourcc(self, codec: Optional[Union[_DEFAULT, str]] = DEFAULT) -> int:
        """Get the fourcc codec from a string.

        Parameters
        ----------
        codec : Optional[Union[_DEFAULT, str]], optional
            Codec to use for the video, by default DEFAULT
            If not set, will use the environment variable VIDEO_CODEC or 'avc1' if not set.

        Returns
        -------
        int
            Fourcc codec.
        """
        if codec is None or codec == DEFAULT:
            codec = os.environ.get('VIDEO_CODEC', 'avc1')
        return cv2.VideoWriter_fourcc(*codec)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame before writing to the video.

        Parameters
        ----------
        frame : np.ndarray
            Frame to process.
            Shape is (H, W, 3 | 4)
            Dtype is np.uint8.
            Channels are RGB or RGBA.

        Returns
        -------
        np.ndarray
            Processed frame. Shape is (H, W, 3) where C is 3. Dtype is np.uint8. Channels are RGB.
        """
        return frame

    def _write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the video.

        Parameters
        ----------
        frame : np.ndarray
            Frame to write to the video.
            Shape is (H, W, C) where C is 1, 3, 4.
            Dtype is np.uint8.
            If C is 3 expect RGB. If C is 4 expect RGBA, while A is ignored.
        """
        H, W, C = frame.shape
        if self.writer is None:
            self._init_writer(H, W)
        if C == 1:
            frame = frame.repeat(3, 2)
        elif C == 4:
            pass
        elif C == 3:
            pass
        else:
            raise ValueError(f"Unsupported channel size: {C}.")
        frame = self._process_frame(frame)
        self.writer.write(frame[:, :, [2, 1, 0]])
        self.frame_counter += 1

    def _init_writer(self, height: int, width: int) -> None:
        """Initialize the video writer.

        Parameters
        ----------
        resolution : Tuple[int, int]
            Resolution of the video.
        """
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception as e:
                pass
            self.writer = None
        self.writer = cv2.VideoWriter(
            self.filename, self.fourcc, self.fps, (width, height))

    def write(self, frame: VEC_TYPE) -> None:
        """Write a frame to the video.

        Parameters
        ----------
        frames : Union[bytes, bytearray]
            Frame to write to the video.
        """
        frame = self.numpify_image(frame)
        if len(frame.shape) == 4:
            for i in range(frame.shape[0]):
                self._write_frame(frame[i])
        else:
            self._write_frame(frame)

    def __init__(self,
                 filename: str,
                 fps: float = 24.0,
                 codec: str = DEFAULT):
        self.numpify_image = ToNumpyImage(output_dtype=np.uint8)
        self.filename = filename
        self.fourcc = self.get_fourcc(codec)
        self.fps = fps
        self.writer = None
        self.frame_counter = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            cv2.destroyAllWindows()
        return False
