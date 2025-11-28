from typing import Iterable, List, Literal, Optional, Set, Tuple
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from tools.mixin.torch_dtype_mixin import TorchDtypeMixin
from tools.model.module_scene_parent import ModuleSceneParent
from tools.model.scene_node import SceneNode
from tools.serialization.json_convertible import JsonConvertible
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.transforms.geometric.transforms3d import (assure_homogeneous_matrix,
                                                     assure_homogeneous_vector,
                                                     component_position_matrix,
                                                     component_rotation_matrix,
                                                     transformation_matrix)
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.util.torch import tensorify
from torch.nn import ModuleList


class ModuleSceneNode(SceneNode, TorchDtypeMixin, torch.nn.Module):
    """Pytorch Module class for nodes within a scene."""

    _scene_children: ModuleList
    """Scene children of the node as ModuleList."""

    def __init__(self,
                 name: Optional[str] = None,
                 children: Optional[Iterable['AbstractSceneNode']] = None,
                 decoding: bool = False,
                 **kwargs
                 ):
        super().__init__(name=name,
                         children=None,
                         decoding=decoding,
                         **kwargs)
        self._scene_children = ModuleList()
        if children is not None:
            self.add_scene_children(*children)

    def set_parent(self, parent: Optional['AbstractSceneNode']) -> None:
        """Set the parent of the node.

        Wraps it

        Parameters
        ----------
        parent : Optional[AbstractSceneNode]
            The parent node. If None, the node is the root of the scene.
        """
        if parent is not None:
            parent = ModuleSceneParent(parent)
        self._parent = parent

    def add_scene_children(self, *children: 'AbstractSceneNode', **kwargs) -> None:
        """
        Add children to scene node.
        Will set the parent of the children to this node, before adding them.

        Parameters
        ----------
        children : AbstractSceneNode
            Children to add.
        """
        for child in children:
            if isinstance(child, ModuleSceneParent):
                child = child._node
            child.set_parent(self)
            self._scene_children.append(child)

    def remove_scene_children(self, *children: 'AbstractSceneNode', **kwargs) -> Set["AbstractSceneNode"]:
        ret = set()
        for child in children:
            if isinstance(child, ModuleSceneParent):
                child = child._node
            if child not in self._scene_children:
                continue
            child.set_parent(None)
            idx = next(
                (k for k, v in self._scene_children._modules.items() if v == child))
            self._scene_children.pop(int(idx))
            ret.add(child)
        return ret
