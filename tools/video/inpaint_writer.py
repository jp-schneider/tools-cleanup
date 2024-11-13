from typing import Optional

import torch
from tools.util.typing import DEFAULT
from tools.video.writer import Writer
import numpy as np
from tools.io.image import put_text, n_layers_alpha_compositing, alpha_background_grid


class InpaintWriter(Writer):

    def __init__(self,
                 filename: str,
                 inpaint_title: Optional[str] = None,
                 inpaint_counter: bool = True,
                 counter_format: str = "{:03d}",
                 use_transparency_grid: bool = True,
                 fps: float = 24.0,
                 codec: str = DEFAULT):
        super().__init__(filename=filename, fps=fps, codec=codec)
        self.inpaint_title = inpaint_title
        self.inpaint_counter = inpaint_counter
        self.counter_format = counter_format
        self.use_transparency_grid = use_transparency_grid
        self.alpha_grid = None

    def _patch_transparency(self, frame: np.ndarray) -> np.ndarray:
        if self.alpha_grid is None:
            self.alpha_grid = alpha_background_grid(
                (frame.shape[0], frame.shape[1]))
        from tools.util.numpy import flatten_batch_dims, unflatten_batch_dims
        images, batch_dims = flatten_batch_dims(frame, -4)
        image_tensor = torch.tensor(images).unsqueeze(0).float() / 255
        N, B, H, W, C = image_tensor.shape
        grid = torch.tensor(self.alpha_grid).unsqueeze(
            0).unsqueeze(0).float() / 255
        grid = grid.expand(-1, B, -1, -1, -1)
        composition = n_layers_alpha_compositing(
            torch.cat([grid, image_tensor], dim=0), torch.tensor([1, 0]))
        return (unflatten_batch_dims(composition.numpy(), batch_dims) * 255).round().astype(np.uint8)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[2] == 4 and self.use_transparency_grid:
            frame = self._patch_transparency(frame)[..., :3]
        if self.inpaint_counter:
            frame = put_text(frame.copy(), self.counter_format.format(self.frame_counter),
                             placement="top-right", background_stroke=1)
        if self.inpaint_title is not None:
            frame = put_text(frame.copy(), self.inpaint_title,
                             placement="top-center", background_stroke=1)
        return frame
