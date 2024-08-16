from typing import List, Optional, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.model.scene_node import SceneNode
from tools.transforms.affine.transforms3d import component_position_matrix
from tools.viz.matplotlib import saveable
from mpl_toolkits.mplot3d import Axes3D
from tools.transforms.to_numpy import numpyify


class VisualNode3D(SceneNode):
    """Scene node for 3D scenes with visual implementation."""

    # region Visualization
    @saveable()
    def plot_scene(self,
                   plot_coordinate_systems: bool = False,
                   plot_line_to_child: bool = False,
                   plot_coordinate_annotations: bool = False,
                   coordinate_system_indicator_length: float = 0.3,
                   units: Optional[Union[List[str], str]] = None,
                   ax: Optional[Axes3D] = None,
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

        init_ax = False

        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(projection='3d')
            init_ax = True
        else:
            fig = ax.get_figure()

        args = dict(plot_coordinate_systems=plot_coordinate_systems,
                    plot_coordinate_annotations=plot_coordinate_annotations,
                    coordinate_system_indicator_length=coordinate_system_indicator_length)
        args.update(kwargs)

        def get_positions(component: AbstractSceneNode) -> List[Tuple[torch.Tensor, List[torch.Tensor], List[str]]]:
            # Get global position
            pos = component.get_global_position(**kwargs)
            origin = pos[..., :3, 3]

            if len(pos.shape) == 2:
                pos = pos.unsqueeze(0)

            if len(origin.shape) == 1:
                origin = origin.unsqueeze(0)

            # 3 Vectors indicating local coordinate system
            cord_vec_x = component_position_matrix(
                x=coordinate_system_indicator_length, dtype=pos.dtype, device=pos.device).repeat(pos.shape[0], 1, 1)
            cord_vec_y = component_position_matrix(
                y=coordinate_system_indicator_length, dtype=pos.dtype, device=pos.device).repeat(pos.shape[0], 1, 1)
            cord_vec_z = component_position_matrix(
                z=coordinate_system_indicator_length, dtype=pos.dtype, device=pos.device).repeat(pos.shape[0], 1, 1)

            x_vec = torch.bmm(pos, cord_vec_x)[..., :3, 3]
            y_vec = torch.bmm(pos, cord_vec_y)[..., :3, 3]
            z_vec = torch.bmm(pos, cord_vec_z)[..., :3, 3]

            vecs = []
            for b in range(pos.shape[0]):
                local_vecs = [x_vec[b], y_vec[b], z_vec[b]]
                texts = ["x", "y", "z"]
                vecs.append((origin[b], local_vecs, texts))

            for child in component.get_scene_children():
                child: VisualNode3D
                target = child.get_global_position()[..., :3, 3]
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
        colors_c = ["red", "green", "blue"] + \
            (["purple"] if plot_line_to_child else [])

        texts = []

        for start, targets, text in vec_positions:
            start = numpyify(start)
            for i, target in enumerate(targets):
                target = numpyify(target)
                start_x.append(start[0])
                start_y.append(start[1])
                start_z.append(start[2])

                end_x.append(target[0])
                end_y.append(target[1])
                end_z.append(target[2])
                colors.append(
                    colors_c[min(2 + (1 if plot_line_to_child else 0), i)])
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
                            (starts[s, 1] + mult *
                             (ends[s, 1] - starts[s, 1])),
                            (starts[s, 2] + mult *
                             (ends[s, 2] - starts[s, 2])),
                            texts[s],
                            horizontalalignment='center',
                            verticalalignment='center')

        if init_ax:
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

        # Set elevation, azimuth and roll if in kwargs
        elev = kwargs.get("elevation", None)
        azim = kwargs.get("azimuth", None)
        roll = kwargs.get("roll", None)

        if elev or azim or roll:
            ax.view_init(elev=elev, azim=azim, roll=roll)
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
            position = self.get_global_position(**kwargs)[..., :3, 3]
            if len(position.shape) == 1:
                ax.text(*position[:3], self._name,
                        horizontalalignment='center', verticalalignment='center')

    # endregion
