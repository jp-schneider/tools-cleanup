from typing import Iterable, List, Literal, Optional, Set, Tuple, Union
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from tools.model.module_scene_node import ModuleSceneNode
from tools.model.visual_node_3d import VisualNode3D
from tools.serialization.json_convertible import JsonConvertible
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.transforms.affine.transforms3d import (assure_affine_matrix,
                                                  assure_affine_vector,
                                                  component_position_matrix,
                                                  component_rotation_matrix, position_quaternion_to_affine_matrix, rotmat_to_unitquat, split_transformation_matrix,
                                                  transformation_matrix)
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.util.torch import tensorify
from tools.viz.matplotlib import saveable


class ModuleSceneNode3D(ModuleSceneNode, VisualNode3D):
    """Pytorch Module class for nodes within a scene."""

    def __init__(self,
                 name: Optional[str] = None,
                 children: Optional[Iterable['AbstractSceneNode']] = None,
                 decoding: bool = False,
                 dtype: torch.dtype = torch.float32,
                 **kwargs
                 ):
        super().__init__(name=name, children=children, decoding=decoding, **kwargs)

    def get_global_position(self, **kwargs) -> torch.Tensor:
        """Return the global position of the scene object, taking into account the position of the parents.

        Returns
        -------
        torch.Tensor
            Matrix describing the global position.
        """
        if self._parent is None:
            return self.get_position(**kwargs)
        else:
            return self.get_parent().get_global_position(**kwargs) @ self.get_position(**kwargs)

    def get_global_position_vector(self, **kwargs) -> torch.Tensor:
        """Return the global position of the scene object as (x, y, z, w) vector,
        taking into account the position of the parents.

        Returns
        -------
        torch.Tensor
            Vector describing the global position.
        """
        return self.get_global_position(**kwargs)[0:4, 3]

    # region Visualization
    def plot_object(self, ax: Axes, **kwargs):
        """Gets a projection 3D axis and is called during plot_coordinates
        function. Can be used to plot arbitrary objects.

        Parameters
        ----------
        ax : Axes
            The axes to plot on.

        **kwargs
            Additional arguments.
        """
        if kwargs.get("plot_coordinate_annotations", False) and self._name is not None:
            position = self.get_global_position_vector()
            ax.text(*position[:3], self._name,
                    horizontalalignment='center', verticalalignment='center')

    # endregion
