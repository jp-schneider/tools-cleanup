from typing import Generator, Optional, Tuple
import cv2
import numpy as np
import os
from tools.util.sized_generator import sized_generator


class Reader():
    """Video writer context manager arround openCV VideoCapture class."""

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __init__(self,
                 file_or_identifier: str,
                 ):
        self.identifier = file_or_identifier
        self.is_file = os.path.isfile(file_or_identifier)
        self.cap = None

    def __enter__(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.identifier)
        return self

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            cv2.destroyAllWindows()

    def __len__(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def next_index(self) -> Optional[int]:
        if self.cap is None:
            self.__enter__()
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def shape(self) -> Tuple[int, int, int, int]:
        if self.cap is None:
            self.__enter__()
        return len(self), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3

    @sized_generator()
    def generator(self) -> Generator[np.ndarray, None, None]:
        """Generator for reading frames from the video.

        Yields
        -------
        np.ndarray
            Frame in shape (H, W, C) where H is the height, W is the width, C is the number of channels.
        """
        with self:
            yield len(self)
            while True:
                frame = self.read()
                if frame is None:
                    break
                yield frame

    def to_stack(self) -> np.ndarray:
        """Convert the video to a stack of frames.

        Returns
        -------
        np.ndarray
            Stack of frames in shape (T, H, W, C) where T is the number of frames, H is the height, W is the width, C is the number of channels.
        """
        frames = []
        for frame in self.generator():
            frames.append(frame)
        if len(frames) == 0:
            return np.empty((0, 0, 0, 0))
        return np.stack(frames, axis=0)
