from typing import Iterable, List, Literal, Optional, Set, Tuple, Union
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from tools.model.module_scene_node import ModuleSceneNode
from tools.serialization.json_convertible import JsonConvertible
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.transforms.affine.transforms3d import (assure_affine_matrix,
                                            assure_affine_vector,
                                            component_position_matrix,
                                            component_rotation_matrix,
                                            transformation_matrix)
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.util.torch import tensorify


class ModuleSceneNode3D(ModuleSceneNode):
    """Pytorch Module class for nodes within a scene."""

    _position: torch.Tensor
    """The objects relative position w.r.t its parent 
    in affine an affine transformation matrix containing is position and rotation."""

    def __init__(self,
                 position: Optional[VEC_TYPE] = None,
                 name: Optional[str] = None,
                 children: Optional[Iterable['AbstractSceneNode']] = None,
                 decoding: bool = False,
                 dtype: torch.dtype = torch.float32,
                 **kwargs
                 ):
        super().__init__(name=name, children=children, decoding=decoding, **kwargs)
        if position is None:
            position = torch.tensor(np.identity(4), dtype=dtype)
        position = assure_affine_matrix(position, dtype=dtype)
        self.register_buffer("_position", position)
 
    def translate(self, translation_vector: VEC_TYPE):
        """Translate the object by moving its position.

        Parameters
        ----------
        translation_vector : VEC_TYPE
            The 3 / 4 element (x, y, z[, w]) vector to perform a position translation.
        """
        mat = transformation_matrix(assure_affine_vector(translation_vector,
                                                         dtype=self._position.dtype,
                                                         device=self._position.device,
                                                         requires_grad=False), 
                                                         dtype=self._position.dtype, 
                                                         device=self._position.device)
        self._transform(mat)

    def __str__(self):
        return f"{type(self).__name__}(name={self.name})"

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"

    def get_position(self) -> VEC_TYPE:
        return self._position
    
    def set_position(self, value: VEC_TYPE):
        self._position = assure_affine_matrix(value)
    
    def get_global_position(self) -> torch.Tensor:
        """Return the global position of the scene object, taking into account the position of the parents.

        Returns
        -------
        torch.Tensor
            Matrix describing the global position.
        """
        if self._parent is None:
            return self._position
        else:
            return self.get_parent().get_global_position() @ self._position


    def get_global_position_vector(self) -> torch.Tensor:
        """Return the global position of the scene object as (x, y, z, w) vector, 
        taking into account the position of the parents.

        Returns
        -------
        torch.Tensor
            Vector describing the global position.
        """
        return self.get_global_position()[0:4, 3]

    def _transform(self, rotation_matrix: torch.Tensor):
        self._position = rotation_matrix @ self._position

    def transform(self, rotation_matrix: VEC_TYPE):
        """Rotates the position by a given affine rotation matrix of
        shape (3, 3) or (4, 4)

        Parameters
        ----------
        rotation_matrix : torch.Tensor
            The rotation matrix.
        """
        rot_mat = assure_affine_matrix(rotation_matrix, dtype=self._position.dtype,
                                       device=self._position.device,
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
                                              dtype=self._position.dtype,
                                              device=self._position.device,
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
                                              dtype=self._position.dtype,
                                              device=self._position.device,
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
                                              dtype=self._position.dtype,
                                              device=self._position.device,
                                              requires_grad=self._position.requires_grad))

    # region Visualization

    def plot_scene(self,
                   plot_coordinate_systems: bool = True,
                   plot_line_to_child: bool = False,
                   plot_coordinate_annotations: bool = True,
                   coordinate_system_indicator_length: float = 0.3,
                   units: Optional[Union[List[str], str]] = None,
                   **kwargs
                   ) -> Figure:
        """Returns a matplotlib 3d plot with the scene.

        Returns
        -------
        AxesImage
            The created axis image.
        """
        from mpl_toolkits.mplot3d import Axes3D
        if units is None:
            units = [None] * 3
        elif isinstance(units, list):
            if len(units) < 3:
                units += [None] * (3 - len(units))
            units = units[:2]
        elif isinstance(units, str):
            units = [units] * 3

        fig = plt.figure()
        ax = plt.subplot(projection='3d')
        ax: Axes3D

        args = dict(plot_coordinate_systems=plot_coordinate_systems,
                    plot_coordinate_annotations=plot_coordinate_annotations,
                    coordinate_system_indicator_length=coordinate_system_indicator_length)
        args.update(kwargs)

        def get_positions(component: AbstractSceneNode) -> List[Tuple[torch.Tensor, List[torch.Tensor], List[str]]]:
            # Get global position
            pos = component.get_global_position()
            origin = pos[:3, 3]

            # 3 Vectors indicating local coordinate system
            x_vec = (pos @ component_position_matrix(x=coordinate_system_indicator_length))[:3, 3]
            y_vec = (pos @ component_position_matrix(y=coordinate_system_indicator_length))[:3, 3]
            z_vec = (pos @ component_position_matrix(z=coordinate_system_indicator_length))[:3, 3]

            vecs = []
            local_vecs = [x_vec, y_vec, z_vec]
            texts = ["x", "y", "z"]
            vecs.append((origin, local_vecs, texts))
            for child in component.get_scene_children():
                child: ModuleSceneNode3D
                target = child.get_global_position()[:3, 3]
                if plot_line_to_child:
                    local_vecs.append(target)
                    texts.append("")
                child.plot_object(ax, **args)
                child_vecs = get_positions(child)
                for cv in child_vecs:
                    vecs.append(cv)
            return vecs

        # Plot self
        self.plot_object(ax, **args)
        vec_positions = get_positions(self)
        start_x = []
        start_y = []
        start_z = []
        end_x = []
        end_y = []
        end_z = []
        colors = []
        colors_c = ["red", "green", "blue"] + (["purple"] if plot_line_to_child else [])

        texts = []

        for start, targets, text in vec_positions:
            for i, target in enumerate(targets):
                start_x.append(start[0])
                start_y.append(start[1])
                start_z.append(start[2])

                end_x.append(target[0])
                end_y.append(target[1])
                end_z.append(target[2])
                colors.append(colors_c[min(2 + (1 if plot_line_to_child else 0), i)])
                texts.append(text[i])

        starts = np.stack([start_x, start_y, start_z], axis=1)
        ends = np.stack([end_x, end_y, end_z], axis=1)
        all_positions = np.concatenate([starts, ends], axis=0)

        if plot_coordinate_systems:
            ax.scatter(*all_positions.swapaxes(0, 1), color=colors * 2)
            for s in range(starts.shape[0]):
                ax.plot([starts[s, 0], ends[s, 0]], [starts[s, 1], ends[s, 1]], [
                        starts[s, 2], ends[s, 2]], color=colors[s])
                mult = 0.5 if texts[s] in ["x", "y", "z"] else 1
                if plot_coordinate_annotations:
                    ax.text((starts[s, 0] + mult * (ends[s, 0] - starts[s, 0])),
                            (starts[s, 1] + mult * (ends[s, 1] - starts[s, 1])),
                            (starts[s, 2] + mult * (ends[s, 2] - starts[s, 2])),
                            texts[s],
                            horizontalalignment='center',
                            verticalalignment='center')

        x_label = "X"
        if units[0] is not None:
            x_label += f" [{units[0]}]"
        ax.set_xlabel(x_label)
        
        y_label = "Y"
        if units[1] is not None:
            y_label += f" [{units[1]}]"
        ax.set_ylabel(y_label)
        
        z_label = "Z"
        if units[2] is not None:
            z_label += f" [{units[2]}]"
        
        ax.set_zlabel(z_label)
        ax.set_aspect("equal")
        ax.set_box_aspect
        return fig

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
            ax.text(*position[:3], self._name, horizontalalignment='center', verticalalignment='center')
   
    # endregion