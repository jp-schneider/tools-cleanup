from typing import Iterable, List, Literal, Optional, Set, Tuple, Union
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from tools.model.module_scene_node import ModuleSceneNode
from tools.model.module_scene_node_3d import ModuleSceneNode3D
from tools.model.visual_node_3d import VisualNode3D
from tools.serialization.json_convertible import JsonConvertible
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.transforms.geometric.transforms3d import (assure_affine_matrix,
                                                     assure_affine_vector,
                                                     component_position_matrix,
                                                     component_rotation_matrix, position_quaternion_to_affine_matrix, rotmat_to_unitquat, split_transformation_matrix,
                                                     transformation_matrix)
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.util.torch import flatten_batch_dims, tensorify, unflatten_batch_dims
from tools.viz.matplotlib import saveable


@torch.jit.script
def global_to_local(
    global_position: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Converts global vectors to local vectors.
    Will return the local vectors for all times steps in global_position.

    Parameters
    ----------
    global_position : torch.Tensor
        The global position of the object. Shape is ([... B,], 4, 4).

    v : torch.Tensor
        Vectors of shape ([... B,], (3 | 4)) to convert.

    Returns
    -------
    torch.Tensor
        Vectors in local coordinates. Shape is ([... B,], 4).

    """
    v, v_batch_shape = flatten_batch_dims(
        v, -2)
    B = v.shape[0]

    glob_mat, _ = flatten_batch_dims(global_position, -3)
    B_N = glob_mat.shape[0]

    if B_N != B:
        if B_N == 1:
            glob_mat = glob_mat.repeat(B, 1, 1)
        else:
            raise ValueError(
                "Batch dimensions of v and global_position must match.")

    # Check if last dim = 3, if 3 add w
    if v.shape[-1] == 3:
        v = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)

    # Invert the global matrix
    glob_mat = torch.inverse(glob_mat)

    # Flatten both batch dimensions
    res = torch.bmm(glob_mat, v.unsqueeze(-1)).squeeze(-1)
    res = res.reshape(B, 4)
    return unflatten_batch_dims(res, v_batch_shape)


@torch.jit.script
def local_to_global(
    global_position: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Converts local vectors to global vectors.

    Parameters
    ----------
    global_position : torch.Tensor
        The global position of the object. Shape is ([... B,], 4, 4).

    v : torch.Tensor
        Vectors of shape ([... B,] [3 | 4]) to convert.

    Returns
    -------
    torch.Tensor
        Vectors in global coordinates. Shape is ([... B,], 4).
    """
    v, v_batch_shape = flatten_batch_dims(
        v, -2)
    B = v.shape[0]

    glob_mat, _ = flatten_batch_dims(global_position, -3)
    B_N = glob_mat.shape[0]

    if B_N != B:
        if B_N == 1:
            glob_mat = glob_mat.repeat(B, 1, 1)
        else:
            raise ValueError(
                "Batch dimensions of v and global_position must match.")

    if v.shape[-1] == 3:
        v = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)

    res = torch.bmm(glob_mat, v.unsqueeze(-1)).squeeze(-1)
    res = res.reshape(B, 4)
    return unflatten_batch_dims(res, v_batch_shape)


class DiscreteModuleSceneNode3D(ModuleSceneNode3D):
    """Pytorch Module class for nodes within a scene which have discrete positions."""

    _translation: torch.Tensor
    """The objects relative translation w.r.t its parent
    as a translation vector (3, ) xyz."""

    _orientation: torch.Tensor
    """The objects relative orientation w.r.t its parent as normalized quaternion (4, ). (x, y, z, w)"""

    def __init__(self,
                 translation: Optional[VEC_TYPE] = None,
                 orientation: Optional[VEC_TYPE] = None,
                 position: Optional[torch.Tensor] = None,
                 name: Optional[str] = None,
                 children: Optional[Iterable['AbstractSceneNode']] = None,
                 decoding: bool = False,
                 dtype: torch.dtype = torch.float32,
                 _translation: Optional[torch.Tensor] = None,
                 _orientation: Optional[torch.Tensor] = None,
                 **kwargs
                 ):
        super().__init__(name=name, children=children,
                         decoding=decoding, dtype=dtype, **kwargs)
        if position is not None:
            if translation is not None or orientation is not None:
                raise ValueError(
                    "Cannot pass position and translation or orientation.")
            position = tensorify(position)
            translation, orientation = self._parse_position(position)
        self._init_position(translation=translation,
                            orientation=orientation,
                            dtype=self.dtype,
                            _translation=_translation,
                            _orientation=_orientation
                            )

    def _init_position(self,
                       translation: Optional[VEC_TYPE],
                       orientation: Optional[VEC_TYPE],
                       dtype: torch.dtype,
                       _translation: Optional[torch.Tensor] = None,
                       _orientation: Optional[torch.Tensor] = None,
                       ):
        if _translation is not None:
            self.register_buffer(
                "_translation", _translation, persistent=False)
            self.register_buffer(
                "_orientation", _orientation, persistent=False)
        else:
            if translation is None:
                translation = self._get_default_translation(dtype)
            if orientation is None:
                orientation = self._get_default_orientation(dtype)
            self.register_buffer(
                "_translation", tensorify(translation, dtype=dtype))
            self.register_buffer(
                "_orientation", tensorify(orientation, dtype=dtype, device=self._translation.device))

    def _get_default_translation(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(3, dtype=dtype)

    def _get_default_orientation(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([0., 0., 0., 1.], dtype=dtype)

    def _parse_position(self, position: VEC_TYPE) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse the position into a translation and quaternion."""
        pos, rot = split_transformation_matrix(position)
        squeeze = False
        if len(rot.shape) == 2:  # If only one matrix is passed
            rot = rot.unsqueeze(0)
            squeeze = True
        quat = rotmat_to_unitquat(rot)
        if squeeze:
            quat = quat.squeeze(0)
        if torch.isnan(quat).any():
            w = torch.argwhere(torch.isnan(quat))
            raise ValueError(f"Invalid quaternions! Indexes {w.numpy().tolist()} are NaN. \n \
                             Invalid quaternions {torch.unique(w[:, 0]).numpy().tolist()}: \n {quat[torch.unique(w[:, 0])]} \n \
                             of position matricies: \n {position[torch.unique(w[:, 0])]}")
        return pos, quat

    def get_translation(self) -> torch.Tensor:
        return self._translation

    def get_orientation(self) -> torch.Tensor:
        return self._orientation

    def get_position(self, *args, **kwargs) -> torch.Tensor:
        return position_quaternion_to_affine_matrix(self.get_translation(), self.get_orientation())

    def set_position(self, value: VEC_TYPE):
        if "_translation" not in self._buffers:
            raise ValueError(
                "Position can not be set when translations and orientations are just references.")
        pos, quat = self._parse_position(value)
        self._translation = pos
        self._orientation = quat

    # region Transformation

    def _transform(self, affine_matrix: torch.Tensor, **kwargs):
        self.set_position(affine_matrix @ self.get_position(**kwargs))

    def after_checkpoint_loaded(self, **kwargs):
        """Method to be called after the checkpoint is loaded.

        This method is called after the checkpoint is loaded. It can be used to perform any operations.
        """
        pass

    def _after_checkpoint_loaded(self, **kwargs):
        """
        Method to be called after the checkpoint is loaded.
        """
        self.after_checkpoint_loaded(**kwargs)
        if self._scene_children is not None:
            for child in self._scene_children:
                child._after_checkpoint_loaded(**kwargs)

    def translate(self, translation_vector: VEC_TYPE):
        """Translate the object by moving its position.

        Parameters
        ----------
        translation_vector : VEC_TYPE
            The 3 / 4 element (x, y, z[, w]) vector to perform a position translation.
        """
        mat = transformation_matrix(assure_affine_vector(translation_vector,
                                                         dtype=self._translation.dtype,
                                                         device=self._translation.device,
                                                         requires_grad=False),
                                    dtype=self._translation.dtype,
                                    device=self._translation.device)
        self._transform(mat)

    def transform(self, rotation_matrix: VEC_TYPE):
        """Rotates the position by a given affine rotation matrix of
        shape (3, 3) or (4, 4)

        Parameters
        ----------
        rotation_matrix : torch.Tensor
            The rotation matrix.
        """
        rot_mat = assure_affine_matrix(rotation_matrix, dtype=self._translation.dtype,
                                       device=self._translation.device,
                                       requires_grad=self._position.requires_grad)
        self._transform(rot_mat)

    def yaw(self, angle: Optional[NUMERICAL_TYPE] = None, mode: Literal["deg", "rad"] = 'rad') -> None:
        """Perform a yaw, rotation around Y-axis by an angle.

        Parameters
        ----------
        angle : Optional[NUMERICAL_TYPE], optional
            The angle to rotate for, by default None
        mode : Literal[&quot;deg&quot;, &quot;rad&quot;], optional
            The mode if deg. or radians are passed, by default 'rad'
        """
        if angle is None or angle == 0.:
            return
        self._transform(component_rotation_matrix(angle_y=angle, mode=mode,
                                                  dtype=self._translation.dtype,
                                                  device=self._translation.device,
                                                  requires_grad=self._position.requires_grad))

    def pitch(self, angle: Optional[NUMERICAL_TYPE] = None, mode: Literal["deg", "rad"] = 'rad') -> None:
        """Perform a pitch, rotation around Z-axis by an angle.

        Parameters
        ----------
        angle : Optional[NUMERICAL_TYPE], optional
            The angle to rotate for, by default None
        mode : Literal[&quot;deg&quot;, &quot;rad&quot;], optional
            The mode if deg. or radians are passed, by default 'rad'
        """
        if angle is None or angle == 0.:
            return
        self._transform(component_rotation_matrix(angle_z=angle, mode=mode,
                                                  dtype=self._translation.dtype,
                                                  device=self._translation.device,
                                                  requires_grad=self._position.requires_grad))

    def roll(self, angle: Optional[NUMERICAL_TYPE] = None, mode: Literal["deg", "rad"] = 'rad') -> None:
        """Perform a roll, rotation around X-axis by an angle.

        Parameters
        ----------
        angle : Optional[NUMERICAL_TYPE], optional
            The angle to rotate for, by default None
        mode : Literal[&quot;deg&quot;, &quot;rad&quot;], optional
            The mode if deg. or radians are passed, by default 'rad'
        """
        if angle is None or angle == 0.:
            return
        self._transform(component_rotation_matrix(angle_x=angle, mode=mode,
                                                  dtype=self._translation.dtype,
                                                  device=self._translation.device,
                                                  requires_grad=self._position.requires_grad))

    def local_to_global(self,
                        v: torch.Tensor,
                        **kwargs) -> torch.Tensor:
        """Converts local vectors to global vectors.

        Parameters
        ----------
        v : torch.Tensor
            Vectors of shape ([... B,] 4) to convert.

        Returns
        -------
        torch.Tensor
            Vectors in global coordinates. Shape is ([... B,], 4).
        """
        glob_mat = self.get_global_position(**kwargs)
        return local_to_global(glob_mat, v)

    def global_to_local(self,
                        v: torch.Tensor,
                        **kwargs) -> torch.Tensor:
        """Converts global vectors to local vectors.

        Parameters
        ----------
        v : torch.Tensor
            Vectors of shape ([... B,], (3 | 4)) to convert.

        Returns
        -------
        torch.Tensor
            Vectors in local coordinates. Shape is ([... B,], 4).
        """
        glob_mat = self.get_global_position(**kwargs)
        return global_to_local(glob_mat, v)

    # endregion
