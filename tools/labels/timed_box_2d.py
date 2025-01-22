from tools.serialization.json_convertible import JsonConvertible
from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from tools.scene.coordinate_system_3d import CoordinateSystem3D
from tools.logger.logging import logger
from tools.util.numpy import numpyify, flatten_batch_dims, unflatten_batch_dims, index_of_first
try:
    import cv2
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"Could not import cv2. Inpainting will not be available. Install opencv-python to enable inpainting.")
    cv2 = None

@dataclass
class TimedBox2D(JsonConvertible):
    """Timed 2D box label. For tracking purposes."""
    
    id: str 
    """Unique identifier for the projected box label."""

    center: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T, 2] representing the projected center of the box in the cameras view frame."""

    width: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T] representing the projected width (x) of the box in the cameras view frame."""

    height: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T] representing the projected height (y) of the box in the cameras view frame."""

    frame_times: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T] representing the frame idx of the box."""

    @classmethod
    def _inpaint_boxes(cls, 
                        centers: np.ndarray,
                        widths: np.ndarray,
                        height: np.ndarray,
                        images: np.ndarray,
                        values: np.ndarray,
                ) -> np.ndarray:
        """
        Inpaints the values in a box defined by center, length, width in the images.

        If C == 4, the values are assumed to be RGBA values and the inpainting is done using alpha compositing.

        Parameters
        ----------
        centers : np.ndarray
            Array of shape [T, 2] representing the center of the box in the image.

        widths : np.ndarray
            Array of shape [T] representing the widths (x) of the box in the image.

        height : np.ndarray
            Array of shape [T] representing the height of the box (y) in the image.

        images : np.ndarray
            Array of shape [T, H, W, C] representing the images.

        values : np.ndarray
            Array of shape [T, H, W, C] representing the values to inpaint.

        Returns
        -------
        np.ndarray
            Array of shape [T, H, W, C] representing the inpainted images.
        """
        from tools.io.image import n_layers_alpha_compositing_numpy
        T, H, W, C = images.shape
        edge_ll = centers - np.stack([widths / 2, height / 2], axis=-1)
        edge_tr = centers + np.stack([widths / 2, height / 2], axis=-1)
        edges = np.stack([edge_ll, edge_tr], axis=1)
        edges = np.round(edges).astype(int)
        edges = np.maximum(edges, 0)
        edges = np.maximum(edges, 0)
        for i in range(2):
            edges[..., i] = np.minimum(edges[..., i], images.shape[1:3][::-1][i] - 1)
        indices = np.mgrid[0:H, 0:W].transpose(1, 2, 0)[np.newaxis, ...][..., ::-1].repeat(T, axis=0)
        mask = ((indices > edges[:, 0, None, None]) & (indices < edges[:, 1, None, None])).all(axis=-1)
        # Add batchidx to C of images
        images = images.copy()
        for i in range(T):
            if C == 4:
                patch_img = np.zeros_like(images[i])
                patch_img[mask[i]] = values[i].astype(images.dtype)
                stacked_img = np.stack([patch_img, images[i]], axis=0)
                if stacked_img.dtype not in [np.float32, np.float64]:
                    old_dtype = stacked_img.dtype
                    stacked_img = stacked_img.astype(np.float32)
                    if old_dtype == np.uint8:
                        stacked_img /= 255
                else:
                    old_dtype = stacked_img.dtype
                cvt_img = n_layers_alpha_compositing_numpy(stacked_img, np.array([0, 1]))
                if old_dtype != cvt_img.dtype:
                    if old_dtype == np.uint8:
                        cvt_img = cvt_img * 255
                    cvt_img = cvt_img.astype(old_dtype)
                images[i] = cvt_img
            else:
                images[i, mask[i]] = values[i].astype(images.dtype)
            
        return images
    
    @classmethod
    def _inpaint_box_frame(cls, 
                        centers: np.ndarray,
                        widths: np.ndarray,
                        heights: np.ndarray,
                        images: np.ndarray,
                        color: np.ndarray,
                        thickness: np.ndarray,
                ) -> np.ndarray:
        """
        Draws the box defined by center, length, width in the images.


        Parameters
        ----------
        centers : np.ndarray
            Array of shape [T, 2] representing the center of the box in the image.

        widths : np.ndarray
            Array of shape [T] representing the width (x) of the box in the image.

        heights : np.ndarray
            Array of shape [T] representing the height (y) of the box in the image.

        images : np.ndarray
            Array of shape [T, H, W, C] representing the images.

        color : np.ndarray
            Array of shape [T, H, W, C] representing the boxes line color.

        Returns
        -------
        np.ndarray
            Array of shape [T, H, W, C] representing the box-frame inpainted images.
        """
        import cv2
        from tools.io.image import n_layers_alpha_compositing_numpy
        T, H, W, C = images.shape
        edge_ll = centers - np.stack([widths / 2, heights / 2], axis=-1)
        edge_tr = centers + np.stack([widths / 2, heights / 2], axis=-1)
        edges = np.stack([edge_ll, edge_tr], axis=1)
        edges = np.round(edges).astype(int)
        edges = np.maximum(edges, 0)
        for i in range(2):
            edges[..., i] = np.minimum(edges[..., i], images.shape[1:3][::-1][i] - 1)
        # Add batchidx to C of images
        images = images.copy()
        for i in range(T):
            images[i] = cv2.rectangle(images[i], tuple(edges[i, 0]), tuple(edges[i, 1]), color[i].astype(int).tolist(), thickness=thickness[i])
        return images


    def inpaint(self,
                images: np.ndarray,
                values: np.ndarray,
                time_steps: Optional[np.ndarray] = None,
            ) -> np.ndarray:
        """
        Inpaints the given values in the by the current box defined pixels in the images.

        If C == 4, the values are assumed to be RGBA values and the inpainting is done using alpha compositing.

        Parameters
        ----------
        images : np.ndarray
            Array of shape [T, H, W, C] representing the images.

        values : np.ndarray
            Array of shape [T, H, W, C] representing the values to inpaint.

        time_steps : np.ndarray
            Array of shape [T] representing the time steps to inpaint.

        Returns
        -------
        np.ndarray
            Array of shape [T, H, W, C] representing the inpainted images.
        """
        time_steps, _ = flatten_batch_dims(numpyify(time_steps), -1)
        indices = index_of_first(self.frame_times, time_steps)
        if (indices < 0).any():
            raise ValueError(f"The time steps {time_steps[indices < 0]} are not in the frame times.")
        centers = self.center[indices]
        heights = self.heights[indices]
        widths = self.width[indices]
        images, shp = flatten_batch_dims(numpyify(images), -4)
        values, _ = flatten_batch_dims(numpyify(values), -2)
        centers, _ = flatten_batch_dims(centers, -2)
        heights, _ = flatten_batch_dims(heights, -1)
        widths, _ = flatten_batch_dims(widths, -1)
        if values.shape[-1] != images.shape[-1]:
            raise ValueError(f"Values shape {values.shape} does not match images shape {images.shape}.")
        if values.shape[0] != images.shape[0]:
            if values.shape[0] == 1:
                values = np.repeat(values, images.shape[0], axis=0)
            else:
                raise ValueError(f"Values batch shape {values.shape[0]} does not match images batch shape {images.shape[0]}.")
        return unflatten_batch_dims(self._inpaint_boxes(centers, widths, heights, images, values), shp)


    def inpaint_box_frame(self,
                images: np.ndarray,
                colors: np.ndarray,
                time_steps: Optional[np.ndarray] = None,
                thickness: Optional[np.ndarray] = None,
            ) -> np.ndarray:
        """
        Inpaints the given values in the by the current box defined pixels in the images.


        Parameters
        ----------
        images : np.ndarray
            Array of shape [T, H, W, C] representing the images.

        values : np.ndarray
            Array of shape [T, H, W, C] representing the values to inpaint.

        time_steps : np.ndarray
            Array of shape [T] representing the time steps to inpaint.

        Returns
        -------
        np.ndarray
            Array of shape [T, H, W, C] representing the inpainted images.
        """
        time_steps, _ = flatten_batch_dims(numpyify(time_steps), -1)
        indices = index_of_first(self.frame_times, time_steps)
        if (indices < 0).any():
            raise ValueError(f"The time steps {time_steps[indices < 0]} are not in the frame times.")
        centers = self.center[indices]
        widths = self.length[indices]
        heights = self.width[indices]
        images, shp = flatten_batch_dims(numpyify(images), -4)
        colors, _ = flatten_batch_dims(numpyify(colors), -2)
        centers, _ = flatten_batch_dims(centers, -2)
        widths, _ = flatten_batch_dims(widths, -1)
        heights, _ = flatten_batch_dims(heights, -1)
        if colors.shape[-1] != images.shape[-1]:
            raise ValueError(f"Colors shape {colors.shape} does not match images shape {images.shape}.")
        if colors.shape[0] != images.shape[0]:
            if colors.shape[0] == 1:
                colors = np.repeat(colors, images.shape[0], axis=0)
            else:
                raise ValueError(f"Values batch shape {colors.shape[0]} does not match images batch shape {images.shape[0]}.")
        if thickness is None:
            thickness = np.ones_like(colors[..., 0]) * 5
        
        return unflatten_batch_dims(self._inpaint_box_frame(centers, widths, heights, images, colors, thickness), shp)
