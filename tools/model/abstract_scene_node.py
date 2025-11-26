from typing import Any, Iterable, List, Literal, Optional, Set, Tuple, FrozenSet
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from tools.serialization.json_convertible import JsonConvertible
from tools.transforms.geometric.transforms3d import (assure_affine_matrix,
                                                     assure_affine_vector,
                                                     component_position_matrix,
                                                     component_rotation_matrix,
                                                     transformation_matrix)
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.util.torch import tensorify
from abc import abstractmethod


class AbstractSceneNode(JsonConvertible):
    """Abstract class for nodes within a geometrical scene."""

    @abstractmethod
    def get_name(self) -> Optional[str]:
        """Get the name of the node.

        Returns
        -------
        Optional[str]
            The name of the node. If no name is set, returns None.
        """
        ...

    @abstractmethod
    def set_name(self, value: Optional[str]) -> None:
        """Sets the name of the node.

        Parameters
        ----------
        value : Optional[str]
            New name of the node or None to remove the name.
        """
        ...

    @abstractmethod
    def get_position(self, *args, **kwargs) -> VEC_TYPE:
        """Get the local position of the node.

        Returns
        -------
        VEC_TYPE
            The position of the node.
        """
        ...

    @abstractmethod
    def set_position(self, value: VEC_TYPE):
        """Set the position of the node.

        Parameters
        ----------
        value : VEC_TYPE
            The new position.
        """
        ...

    @abstractmethod
    def add_scene_children(self, *children: 'AbstractSceneNode', **kwargs) -> None:
        """
        Add children to scene node.
        Will set the parent of the children to this node, before adding them.

        Parameters
        ----------
        children : AbstractSceneNode
            Children to add.
        """
        ...

    @abstractmethod
    def remove_scene_children(self, *children: 'AbstractSceneNode', **kwargs) -> Set["AbstractSceneNode"]:
        """R
        emove children from scene node.
        Removes the parent of the children before removing them as children.

        Parameters
        ----------
        children : AbstractSceneNode
            Children to remove.

        Returns
        -------
        Set[AbstractSceneNode]
            The set of children that were actually removed.
        """
        ...

    @abstractmethod
    def get_scene_children(self) -> FrozenSet["AbstractSceneNode"]:
        """Get the children of the node.

        Returns
        -------
        FrozenSet[AbstractSceneNode]
            The children of the node.
        """
        ...

    @abstractmethod
    def set_parent(self, parent: Optional['AbstractSceneNode']) -> None:
        """Set the parent of the node.

        Parameters
        ----------
        parent : AbstractSceneNode
            The new parent of the node.
        """
        ...

    @abstractmethod
    def get_parent(self) -> Optional['AbstractSceneNode']:
        """Get the parent of the node.

        Returns
        -------
        Optional[AbstractSceneNode]
            The parent of the node. If the node has no parent, returns None.
        """
        ...

    @abstractmethod
    def get_root(self) -> 'AbstractSceneNode':
        """Get the root of the scene.

        Returns
        -------
        AbstractSceneNode
            The root of the scene.
        """
        ...

    @abstractmethod
    def get_global_position(self, **kwargs) -> VEC_TYPE:
        """
        Gets the global position of the node.

        Returns
        -------
        VEC_TYPE
            The global position of the node.
        """
        ...

    @abstractmethod
    def get_index(self) -> Any:
        """Get the index of the node.

        Returns
        -------
        Any
            The index of the node.
        """
        ...

    def __str__(self):
        return f"{type(self).__name__}(name={self.get_name()})"

    def __repr__(self):
        return f"{type(self).__name__}(name={self.get_name()})"
