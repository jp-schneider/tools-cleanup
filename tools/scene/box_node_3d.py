from matplotlib.axes import Axes
from tools.model.discrete_module_scene_node_3d import DiscreteModuleSceneNode3D
from tools.model.visual_node_3d import VisualNode3D
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.util.torch import tensorify
from tools.viz.matplotlib import saveable
from typing import Any, Iterable, List, Literal, Optional, Set, Tuple, Union
import torch
from tools.viz.matplotlib import parse_color_rgba

class BoxNode3D(DiscreteModuleSceneNode3D, VisualNode3D):
    """Pytorch Module class for a 3D box."""

    _size : torch.Tensor
    """Size of the box as (width, height, depth) vector."""

    def __init__(self,
                 name: Optional[str] = None,
                 size: Optional[torch.Tensor] = None,
                 **kwargs
                 ):
        super().__init__(name=name, **kwargs)
        # Test if size is given
        if size is None:
            size = torch.tensor([1., 1., 1.], dtype=self.dtype, device=self._translation.device)
        else:
            size = tensorify(size, dtype=self.dtype, device=self._translation.device)
        if size.dim() != 1 or size.shape[0] != 3:
            raise ValueError("Size must be a 3D vector.")
        self.size = size

    def get_local_corners(self, **kwargs) -> torch.Tensor:
        """Returns the corners of the box in local coordinates.

        When viewed in a right-handed coordinate system x-right, y-forward, z-up (matpotlib)
        from the top, the first 4 corners are the bottom face, the last 4 corners are the top face.
        Starting from the bottom left corner and going anti-clockwise.
        
        Returns
        -------
        torch.Tensor
            Corners of the box in local coordinates.
            Shape: (8, 3)
        """
        size = self.size
        half_size = size / 2
        corners = torch.tensor([
            # Bottom fact
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1],
            # Top face
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1]], dtype=self.dtype, device=self._translation.device)
        corners = corners * half_size
        return corners

    def plot_corners(self, 
                     ax: Axes, 
                     box_color: Any = "yellow",
                     plot_box_edge_markers: bool = False,
                     bottom_start_corner_color: Any = "red",
                     top_start_corner_color: Any = "green",
                     **kwargs):
        local_corners = self.get_local_corners()
        global_corners = self.local_to_global(local_corners)[:, :3]
        box_color = parse_color_rgba(box_color)
        bottom_start_corner_color = parse_color_rgba(bottom_start_corner_color)
        top_start_corner_color = parse_color_rgba(top_start_corner_color)

        lines = [
            (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4), # Top face
            (0, 4), (1, 5), (2, 6), (3, 7) # Connections
        ]
        for line in lines:
            ax.plot(*global_corners[line, :].T, color=box_color)

        if plot_box_edge_markers:
            # Plot start corners
            ax.scatter(*global_corners[0, :3], color=bottom_start_corner_color)
            ax.scatter(*global_corners[4, :3], color=top_start_corner_color)

            # Add arrow pointing to the next corner
            ax.quiver(*global_corners[0, :3], *((global_corners[1, :3] - global_corners[0, :3]) / 5),
                    color=bottom_start_corner_color)
            ax.quiver(*global_corners[4, :3], *((global_corners[5, :3] - global_corners[4, :3]) / 5),
                        color=top_start_corner_color)



    def plot_object(self, ax: Axes, **kwargs):
        fig = super().plot_object(ax, **kwargs)
        self.plot_corners(ax, **kwargs)