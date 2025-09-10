# Class for functions
# File for useful functions when using matplotlib
import inspect
import io
import re
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Union, Tuple, TYPE_CHECKING

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.image import AxesImage

from tools.segmentation.masking import value_mask_to_channel_masks
from tools.util.format import parse_format_string
from tools.util.numpy import numpyify_image, numpyify
from tools.util.torch import VEC_TYPE
from tools.transforms.numpy.min_max import MinMax
from tools.logger.logging import logger
try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
except (ModuleNotFoundError, ImportError):
    if not TYPE_CHECKING:
        plt = None
        Figure = None
        Axes = None
    pass
import os
from functools import wraps
from tools.util.path_tools import numerated_file_name, open_in_default_program
import numpy as np
import math
import sys
import matplotlib.text as mtext
from tools.util.typing import DEFAULT, _DEFAULT
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines  # For creating proxy Line2D objects

LegendArtist = Union[mpatches.Patch, mlines.Line2D]


def set_default_output_dir(output_dir: Optional[Union[str, Path]] = None):
    """Sets the default output directory for saving figures.

    Parameters
    ----------
    output_dir : str
        The default output directory.
    """
    if output_dir is not None:
        if isinstance(output_dir, Path):
            output_dir = str(output_dir)
        os.environ["PLOT_OUTPUT_DIR"] = str(output_dir)
    else:
        os.environ.pop("PLOT_OUTPUT_DIR", None)


class WrapText(mtext.Text):
    def __init__(self,
                 x: float = 0, y: float = 0, text='',
                 width: Optional[int] = None,
                 **kwargs):
        mtext.Text.__init__(self,
                            x=x, y=y, text=text,
                            wrap=True,
                            **kwargs)
        if width is None:
            width = sys.maxsize
        self.width = width  # in screen pixels. You could do scaling first

    def _get_wrap_line_width(self):
        return self.width


def saveable(
    default_ext: Union[str, List[str]] = "png",
    default_output_dir: Optional[Union[str, _DEFAULT]] = DEFAULT,
    default_name: Optional[Union[str, _DEFAULT]] = DEFAULT,
    default_transparent: bool = False,
    default_dpi: int = 300,
    default_override: bool = False,
    default_tight_layout: bool = False,
    is_animation: bool = False,
    is_figure_collection: bool = False,
    default_fps: int = 24,
):
    """Declares a matplotlib figure producing function as saveable so the functions
    figure output can directly saved to disk by setting the save=True as kwarg.

    Supported params:

    kwargs
    ---------

    save: bool, optional
       Triggers saving of the output, Default False.

    open: bool, optional
        Opens the saved figure in the default program, Default False.

    path: str, optional
        Path where the figure should be saved. Can be a path to a directory, or just a filename.
        If it is a filename, the figure will be saved in a default folder.
        Default default_output_dir + uuid4()

    ext: str, optional
        File extension of the path. If path already contains an extension, this is ignored.
        Otherwise it can be a str or a list to save the figure in different formats, like ["pdf", "png"]
        Default see: default_ext

    transparent: bool, optional
        If the generated plot should be with transparent background. Default see: default_transparent

    dpi: int, optional
        The dpi when saving the figure.
        Default see: default_dpi

    override: bool optional
        If the function should override existing file with the same name.
        Default see: default_override

    tight_layout: bool, optional
        If the function should call tight_layout on the figure before saving.
        Default see: default_tight_layout

    set_interactive_mode: bool, optional
        If the function should set the interactive mode of matplotlib before calling the function.
        Default None
        Meaning it will not change the interactive mode. The interactive mode will be restored after the function call.

    auto_close: bool, optional
        If the function should close the figure after saving.
        Default False

    fps: int, optional
        Frames per second for the animation. Default 24.

    Parameters
    ----------
    default_ext : Union[str, List[str]], optional
        [List of] Extension[s] to save the figure, by default "png"
    default_output_dir : Optional[str], optional
        Output directory of figures when path did not contain directory information,
        If the Environment Variable "PLOT_OUTPUT_DIR" is set, it will be used as destination.
        by default "./temp"
    default_name : Optional[str], optional
        Default name of the figure if no path is specified, by default a uuid4 string
    default_transparent : bool, optional
        If the function should output the figures with transparent background, by default False
    default_dpi : int, optional
        If the function should by default override, by default 300
    default_override : bool, optional
        If the function should by default override, by default False
    default_tight_layout : bool, optional
        If the function should by default call tight_layout on the figure, by default False
    is_animation : bool, optional
        If the function returns an animation, by default False
    default_fps : int, optional
        Frames per second for the animation, by default 24
    """
    from uuid import uuid4

    # type: ignore
    def decorator(function: Callable[[Any], Union[Figure, FuncAnimation]]) -> Callable[[Any], Union[Figure, FuncAnimation]]:
        func_params = dict(inspect.signature(function).parameters)

        @wraps(function)
        def wrapper(*args, **kwargs):
            nonlocal default_output_dir, is_animation
            path = kwargs.pop(
                "path", default_name if default_name != DEFAULT else str(uuid4()))
            save = kwargs.pop("save", False)
            ext = kwargs.pop("ext", default_ext)
            transparent = kwargs.pop("transparent", default_transparent)
            dpi = kwargs.pop("dpi", default_dpi)
            if "dpi" in func_params:
                # Dont consume the dpi if it is a param of the function
                kwargs["dpi"] = dpi

            override = kwargs.pop("override", default_override)
            tight_layout = kwargs.pop("tight_layout", default_tight_layout)
            open = kwargs.pop("open", False)
            set_interactive_mode = kwargs.pop("set_interactive_mode", None)
            close = kwargs.pop("auto_close", False)
            display = kwargs.pop("display", False)
            display_auto_close = kwargs.pop("display_auto_close", True)
            fps = kwargs.pop("fps", default_fps)

            ani = None
            # Get interactive mode.
            is_interactive = mpl.is_interactive()

            if set_interactive_mode is not None:
                mpl.interactive(set_interactive_mode)
            try:
                out = function(*args, **kwargs)
                if is_animation:
                    sa = out[0]
                    ani = out[1]
                    out = sa
            finally:
                mpl.interactive(is_interactive)

            if tight_layout:
                if is_figure_collection:
                    for f in out:
                        f.tight_layout()
                else:
                    out.tight_layout()

            if save or open:
                paths = get_figure_path(
                    path, default_output_dir=default_output_dir, ext=ext)

                paths = save_figure_or_animation(
                    out, paths, is_animation=is_animation, ani=ani, is_figure_collection=is_figure_collection,
                    override=override, dpi=dpi, transparent=transparent, fps=fps)
                if open:
                    try:
                        open_in_default_program(paths[0])
                    except Exception as err:
                        pass
            if display:
                from IPython.display import display
                display(out)
                if display_auto_close:
                    plt.close(out)
            if close and not (display and display_auto_close):
                plt.close(out)
            if is_animation:
                return out, ani
            else:
                return out
        return wrapper
    return decorator


def save_and_open_figure(
        fig: Figure,
        path: Optional[str] = None,
        ext: Union[str, List[str]] = "png",
        dpi: int = 300,
        transparent: bool = False):
    """Saves a figure and opens it in the default program of the operating system.

    Parameters
    ----------
    fig : Figure
        Figure to save.

    path : Optional[str], optional
        Path to save the figure, by default None
        If None, a uuid4 string will be used as filename.

    ext : Union[str, List[str]], optional
        Extension of the file, by default "png"

    dpi : int, optional
        Dots per inch of the figure, by default 300

    transparent : bool, optional
        If the figure should be saved with transparent background, by default False
    """
    if path is None:
        path = get_figure_path(path, ext=ext)
    save_paths = save_figure_or_animation(
        fig, path, dpi=dpi, transparent=transparent)
    open_in_default_program(save_paths[0])


def save_figure_or_animation(
    fig: Any,
    paths: Union[str, List[str]],
    is_animation: bool = False,
    ani: Optional[FuncAnimation] = None,
    is_figure_collection: bool = False,
    override: bool = False,
    dpi: int = 300,
    transparent: bool = False,
    fps: int = 24
) -> List[str]:
    out = fig
    if isinstance(paths, str):
        paths = [paths]

    def save_fig_or_ani(path, fig, ani=None):
        if not override:
            path = numerated_file_name(path)
        if is_animation:
            ani.save(path, fps=fps, dpi=dpi)
        else:
            fig.savefig(path, transparent=transparent, dpi=dpi)
        return path

    save_paths = []
    for p in paths:
        if is_figure_collection:
            # Parse format string#
            sub_p = parse_format_string(p, [x for x in out])
            for i, s in enumerate(sub_p):
                ai = None
                if is_animation:
                    ai = ani[i]
                save_paths.append(save_fig_or_ani(s, out[i], ai))
        else:
            save_paths.append(save_fig_or_ani(p, out, ani))
    return save_paths


def get_figure_path(
        path: Optional[Union[str, os.PathLike]] = None,
        default_output_dir: Union[str, _DEFAULT] = DEFAULT,
        ext: Union[str, List[str]] = "png"
) -> List[str]:
    from uuid import uuid4
    import os
    if path is None:
        path = str(uuid4())
    if not isinstance(path, str):
        path = str(path)
    paths = []
    if any([(x in path) for x in ["/", "\\", os.sep]]):
        # Treat path as abspath
        path = os.path.abspath(path)
    else:
        default_output_dir = os.environ.get(
            "PLOT_OUTPUT_DIR", default_output_dir if default_output_dir != DEFAULT else "./temp")
        path = os.path.join(
            os.path.abspath(default_output_dir), path)
    # Check if path has extension
    _, has_ext = os.path.splitext(path)
    if len(has_ext) == 0:
        if isinstance(ext, str):
            ext = [ext]
        for e in ext:
            paths.append(path + "." + e)
    else:
        paths = [path]

    dirs = set([os.path.dirname(p) for p in paths])
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    return paths


def render_text(text: str,
                ax: Optional[Axes],
                width: Optional[int] = None
                ) -> Figure:
    """Renders a text in a matplotlib figure within an axes.


    Parameters
    ----------
    text : str
        The text to render.
    ax : Optional[Axes]
        Axes to render the text in, by default None
    width : Optional[int], optional
        Wrap width of the should be wrapped, by default None

    Returns
    -------
    Figure
        The figure with the text.
    """
    if ax is None:
        fig, ax = get_mpl_figure(1, 1)
    else:
        fig = ax.figure
    ax.axis('off')
    wtxt = WrapText(0, 1, text, width=width if width != None else 0, ha="left", va='top', clip_on=False,
                    family='monospace')
    # Add artist to the axes
    ax.add_artist(wtxt)
    return fig


def parse_color_rgb(color: Any) -> np.ndarray:
    """Parses a color to RGB values.

    Parameters
    ----------
    color : Any
        Color to parse.
        Can be a string, a tuple, a list or a np.ndarray.
        Strings should be valid color names or hex values.

    Returns
    -------
    np.ndarray
        RGB values of the color in [0, 1] range.
    """
    from matplotlib.colors import to_rgb
    if not isinstance(color, str) and isinstance(color, Iterable):
        color = np.array(color)
        if len(color) == 3:
            if color.max() > 1:
                color = color / 255
            return color
        else:
            trgb = to_rgb(color)
            return np.array(trgb)
    else:
        trgb = to_rgb(color)
        return np.array(trgb)


def parse_color_rgba(color: Any, alpha: float = 1.) -> np.ndarray:
    """Parses a color to RGBA values.

    Parameters
    ----------
    color : Any
        Color to parse.
        Can be a string, a tuple, a list or a np.ndarray.
        Strings should be valid color names or hex values.

    Returns
    -------
    np.ndarray
        RGBA values of the color in [0, 1] range.
    """
    from matplotlib.colors import to_rgba
    if not isinstance(color, str) and isinstance(color, Iterable):
        color = np.array(color)
        if len(color) == 4:
            if color.max() > 1:
                color = color / 255
            return color
        else:
            trgb = to_rgba(color, alpha=alpha)
            return np.array(trgb)
    else:
        trgb = to_rgba(color, alpha=alpha)
        return np.array(trgb)


def compute_ratio(ratio_or_img: Optional[Union[float, np.ndarray]] = None) -> float:
    """Computes the ratio of an image or a given ratio.

    Parameters
    ----------
    ratio_or_img : Optional[Union[float, np.ndarray]], optional
        Ratio of Y w.r.t X  (Height / Width) can also be an Image / np.ndarray which will compute it from the axis,
        by default None

    Returns
    -------
    float
        The computed ratio.
    """
    if ratio_or_img is None:
        ratio_or_img = 1.0
    if "torch" in sys.modules:
        from torch import Tensor
        if isinstance(ratio_or_img, Tensor):
            ratio_or_img = ratio_or_img.detach().cpu()
            if len(ratio_or_img.shape) == 4:
                ratio_or_img = ratio_or_img[0]
            if len(ratio_or_img.shape) == 3:
                ratio_or_img = ratio_or_img.permute(1, 2, 0)
            ratio_or_img = ratio_or_img.numpy()

    if isinstance(ratio_or_img, np.ndarray):
        if len(ratio_or_img.shape) == 4:
            ratio_or_img = ratio_or_img[0]
        elif len(ratio_or_img.shape) == 2:
            ratio_or_img = ratio_or_img.shape[-2] / ratio_or_img.shape[-1]
        elif len(ratio_or_img.shape) == 3:
            ratio_or_img = ratio_or_img.shape[-3] / ratio_or_img.shape[-2]
    return ratio_or_img


def get_mpl_figure(
        rows: int = 1,
        cols: int = 1,
        size: float = 5,
        ratio_or_img: Optional[Union[float, np.ndarray]] = None,
        tight: bool = False,
        subplot_kw: Optional[Dict[str, Any]] = None,
        ax_mode: Literal["1d", "2d"] = "1d",
        frame_on: bool = False,
        gridspec_kw: Optional[Dict[str, Any]] = None
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """Create a eventually tight matplotlib figure with axes.

    Parameters
    ----------
    rows : int, optional
        Number of rows for the figure, by default 1
    cols : int, optional
        Number of columns, by default 1
    size : float, optional
        Size of the axes in inches, by default 5
    ratio_or_img : float | np.ndarray, optional
        Ratio of Y w.r.t X  (Height / Width) can also be an Image / np.ndarray which will compute it from the axis, by default 1.0
    tight : bool, optional
        If the figure should be tight => No axis spacing and borders, by default False
    subplot_kw : Optional[Dict[str, Any]], optional
        Optional kwargs for the subplots, by default None
        Only used if tight is False
    frame_on : bool, optional
        If the frame should be on, by default False
        Only used if tight is True.
    gridspec_kw : Optional[Dict[str, Any]], optional
        Optional kwargs for the gridspec, by default None
        Can be used to adjust spacing etc.
        Further read: https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html#matplotlib.gridspec.GridSpec
    Returns
    -------
    Tuple[Figure, Axes | List[Axes]]
        Figure and axes.
    """
    ratio_x = 1
    ratio_y = compute_ratio(ratio_or_img)
    axes = []

    if tight:
        fig = plt.figure()
        fig.set_size_inches(
            size * ratio_x * cols,
            size * ratio_y * rows,
            forward=False)
        # (left, bottom, width, height)
        rel_width = 1 / cols
        rel_height = 1 / rows
        for i in range(rows * cols):
            col, row = divmod(i, rows)
            ax = plt.Axes(fig, [col * rel_width, row *
                          rel_height, rel_width, rel_height])
            if frame_on:
                ax.set_frame_on(True)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            else:
                ax.set_frame_on(False)
                ax.set_axis_off()
            fig.add_axes(ax)
            axes.append(ax)
    else:
        fig, ax = plt.subplots(rows, cols, figsize=(size * ratio_x * cols,
                                                    size * ratio_y * rows),
                               subplot_kw=subplot_kw,
                               gridspec_kw=gridspec_kw
                               )
        axes.append(ax)
    axes = np.array(axes)
    if ax_mode == "2d" and len(axes.shape) == 1:
        axes = axes[None, ...]

    if ax_mode == "2d" and tight:
        axes = np.reshape(axes, (rows, cols), order="F")[::-1]
    elif ax_mode == "2d" and not tight:
        axes = np.reshape(axes, (rows, cols), order="C")  # [::-1]
    elif ax_mode == "1d" and not tight:
        axes = np.reshape(axes, (rows * cols), order="C")

    if ((ax_mode == "2d" and np.multiply(*axes.shape) == 1)
            or (ax_mode == "1d" and len(axes) == 1)):
        return fig, axes[0]
    return fig, axes


def register_alpha_map(base_name: str = 'binary', renew: bool = False) -> str:
    """Registeres an alpha increasing colormap with matplotlib.
    The colormap will be a copy of the base colormap with increasing alpha values from 0 to 1.


    Parameters
    ----------
    base_name : str, optional
        Name of the colormap, by default 'binary'
    renew : bool, optional
        If the colormap should be reregistered if it exists, by default False

    Returns
    -------
    str
        colormap name
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import colormaps

    name = f'alpha_{base_name}'

    try:
        plt.get_cmap(name)
    except ValueError as err:
        pass
    else:
        if not renew:
            return name  # Already exists
        else:
            from matplotlib import cm
            colormaps.unregister(name)

    # get colormap
    ncolors = 256

    base_map = plt.get_cmap(base_name)
    N = base_map.N
    color_array = base_map(range(N))

    # change alpha values
    color_array[:, -1] = np.linspace(0, 1.0, ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(
        name=name, colors=color_array)

    # register this new colormap with matplotlib
    colormaps.register(cmap=map_object)
    return name


def should_use_logarithm(x: np.ndarray, magnitudes: int = 2, allow_zero: bool = True) -> bool:
    """Checks if the data should be plotted with logarithmic scale.
    Result is calculated based on orders of magnitude of the data.

    Parameters
    ----------
    x : np.ndarray
        Data to be plotted

    magnitudes : int, optional
        Number of magnitudes the data should span to be plotted with logarithmic scale, by default 2

    allow_zero : bool, optional
        If zero values should be allowed, by default True

    Returns
    -------
    bool
        If the data should be plotted with logarithmic scale.
    """
    if not allow_zero:
        if np.any(x <= 0):
            return False
    max_ = np.max(np.abs(x))
    min_ = np.min(np.abs(x))
    if min_ == 0:
        if len(x[x > 0]) == 0:
            min_ = 1
        else:
            min_ = np.min(x[x > 0])

    return (max_ / min_) > math.pow(10, magnitudes)


def preserve_legend(ax: Axes,
                    patches: List[LegendArtist],
                    create_if_not_exists: bool = True,
                    **kwargs):
    """Checks if the axis has a legend and appends the patches to the legend.
    If not, it creates a new legend with the patches if create_if_not_exists is True.

    Parameters
    ----------
    ax : Axes
        The axis to check for the legend.

    patches : List[Patch]
        List of patches to append to the legend.

    create_if_not_exists : bool, optional
        If the legend should be created if it does not exist, by default True

    **kwargs
        Additional kwargs for the legend
    """
    if ax.get_legend() is not None:
        lgd = ax.get_legend()
        # Get existing handles (the drawn artists in the legend)
        current_handles = list(lgd.legendHandles)

        # Get existing labels from the legend's Text objects (most reliable way)
        current_labels = [text_obj.get_text() for text_obj in lgd.get_texts()]

        # Combine existing and new handles/labels
        all_handles = current_handles + patches
        # For new handles, assume they have their labels correctly set
        all_labels = current_labels + [h.get_label() for h in patches]

        # Recreate the legend with the combined list
        # Ensure that the old legend is removed before creating the new one
        lgd.remove()
        ax.legend(handles=all_handles, labels=all_labels, **kwargs)
    else:
        if create_if_not_exists:
            ax.legend(handles=patches, **kwargs)


def create_alpha_colormap(
        name: str,
        color: np.ndarray,
        ncolors: int = 256
) -> LinearSegmentedColormap:
    """Creates a linear alpha colormap with matplotlib.
    Colormap has static RGB values and linear alpha values from 0 to 1.
    Meaning 0 is transparent and 1 is opaque.

    Parameters
    ----------
    name : str
        Name of the new colormap.
    color : np.ndarray
        Colorvalues of the colormap. Shape is (3, ).
    ncolors : int, optional
        Number of colors in the map, by default 256

    Returns
    -------
    LinearSegmentedColormap
        The created colormap.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    color_array = np.zeros((ncolors, 4))
    color_array[:, :3] = color

    # change alpha values
    color_array[:, -1] = np.linspace(0, 1.0, ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(
        name=name, colors=color_array)

    return map_object


def get_tab40_colors() -> LinearSegmentedColormap:
    """
    Returns a colormap with the colors of the tab20b + tab20c colormaps,
    effectively creating a colormap with 40 distinct colors.

    Returns
    -------
    LinearSegmentedColormap
        The created colormap.
    """
    tab20b = plt.get_cmap("tab20b")
    tab20c = plt.get_cmap("tab20c")
    colors = np.concatenate([tab20b.colors, tab20c.colors], axis=0)
    return LinearSegmentedColormap.from_list("tab40", colors)


def register_alpha_colormap(color: np.ndarray,
                            name: str,
                            renew: bool = False,
                            ncolors: int = 256) -> str:
    """Registers a linear alpha colormap with matplotlib.
    Colormap has static RGB values and linear alpha values from 0 to 1.
    Meaning 0 is transparent and 1 is opaque.

    Parameters
    ----------
    color : np.ndarray
        Colorvalues of the colormap. Shape is (3, ).
    name : str
        Name of the new colormap.
    renew : bool, optional
        If the map should be recreated when it exists, by default False
    ncolors : int, optional
        Number of colors in the map, by default 256

    Returns
    -------
    str
        Name of the colormap
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    try:
        plt.get_cmap(name)
    except ValueError as err:
        pass
    else:
        if not renew:
            return name  # Already exists
        else:
            from matplotlib import cm
            cm._colormaps.unregister(name)

    map_object = create_alpha_colormap(name, color, ncolors)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    return name


def used_mathjax(x: str) -> Tuple[bool, str]:
    """Checks if a string is a mathjax string.

    Parameters
    ----------
    x : str
        String to check.

    Returns
    -------
    Tuple[bool, str]
        If the string is a mathjax string and the string without $.
    """
    pattern = r"^\$(?P<var_name>.*)\$$"
    match = re.match(pattern, x)
    if match is not None:
        return True, match.group("var_name")
    return False, x


@saveable()
def plot_as_image(data: VEC_TYPE,
                  size: float = 5,
                  variable_name: str = "Image",
                  cscale: Optional[Union[List[str], str]] = None,
                  ticks: bool = True,
                  title: Optional[str] = None,
                  colorbar: bool = False,
                  colorbar_tick_format: str = None,
                  value_legend: bool = False,
                  cmap: Optional[str] = "viridis",
                  phase_cmap: Optional[str] = "twilight",
                  axes: Optional[np.ndarray] = None,
                  interpolation: Optional[str] = None,
                  tight: bool = False,
                  frame_on: bool = False,
                  imshow_kw: Optional[Dict[str, Any]] = None,
                  norm: bool = False,
                  keep_transparency: bool = False,
                  inpaint_title: Union[bool, _DEFAULT] = DEFAULT,
                  inpaint_title_kwargs: Optional[Dict[str, Any]] = None,
                  numbering: bool = True,
                  native_size: bool = False,
                  dpi: int = 300,
                  inset_args: Optional[Union[Dict[str, Any],
                                             Sequence[Dict[str, Any]]]] = None,
                  axis_enabled: Union[bool, _DEFAULT] = DEFAULT,
                  gamma_correction: bool = False,
                  gamma: Union[_DEFAULT, float] = DEFAULT,
                  ) -> AxesImage:
    """Plots a 2D (complex) image with matplotib. Supports numpy arrays and torch tensors.

    Parameters
    ----------
    data : VEC_TYPE
        Data to be plotted
    size : float, optional
        The size of the plot in inches, by default 5
    variable_name : str, optional
        Variable name for the title, by default "Image"
    cscale : Optional[Union[List[str], str]], optional
        Colorscaling can be log, auto, count, by default None
    ticks : bool, optional
        If ticks should be visible, by default True
    title : Optional[str], optional
        Title, by default None
    colorbar : bool, optional
        If a colorbar should be plotted, by default False
    colorbar_tick_format : str, optional
        The colorbar tick format, by default None
    value_legend : bool, optional
        If there should be a legend for individual values within the image, by default False
    cmap : Optional[str], optional
        Colormap to use for plotting, by default None
    axes : Optional[np.ndarray], optional
        Preexisting axis to embed the image into, by default None
    interpolation : Optional[str], optional
        Iterpolation mode for imshow, by default None
    tight : bool, optional
        If the image should be plotted tight, only supported if axes is not provided, by default False
    imshow_kw : Optional[Dict[str, Any]], optional
        Additional kwargs for the imshow function, by default None
    norm : bool, optional
        If the data should be normalized, by default False
        If True, the data will be normalized to [0, 1] using MinMax normalization.
    keep_transparency : bool, optional
        If the transparency of the input image should be kept, by default False
        If the input image is a RGBA image, the transparency will be kept and displayed as white background.
        If False, the transparency will be removed by composing it with a background grid.

    inpaint_title : Union[bool, _DEFAULT], optional
        If the title should be inpainted, by default DEFAULT
        If True, the title will be inpainted, if False it will not be inpainted.
        If DEFAULT, the title will be inpainted if the title is not empty and tight is True.
    inpaint_title_kwargs : Optional[Dict[str, Any]], optional
        Additional kwargs for the inpaint_title function, by default None
    numbering : bool, optional
        If the images should be numbered, by default True
    native_size : bool, optional
        If the image should be plotted in native size, this will ignore the size parameter and instead use the size of the image and the dpi setting to
        reproduce a figure in the original size of the image.
        If the images is saved with the corresponding dpi setting, the image will be saved in the original size.
    dpi : int, optional
        The dpi of the figure, by default 300
        Will be used to calculate the size of the figure in inches when native_size is True.

    inset_args : Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]], optional
        If provided, the function will create an inset axis with the given args.
        If sequence of dicts is provided, the function will create multiple inset axes.
        Supported args are:
        - start_xy: Tuple[float, float]
            Start coordinates of the inset axis. Needs to be specified and in axis coordinates (x, y)
        - end_xy: Tuple[float, float]
            End coordinates of the inset axis. Needs to be specified and in axis coordinates (x, y)
        - loc: str, optional
            Location of the inset axis, by default "upper right"
        - zoom: float, optional
            Zoom factor of the inset axis, by default 1.0
        - connect_corners: Tuple[int, int], optional
            Corners of the respective boxes which should be connected to create the zoom effect, by default 2,4
        - edgecolor: str, optional
            Edgecolor of the inset axis, by default "black"
        - linewidth: int, optional
            Linewidth of the inset axis and its connecting lines, by default 1
    axis_enabled : Union[bool, _DEFAULT], optional
        If the axis should be enabled, by default DEFAULT
        If DEFAULT, the axis will be enabled as long tight is false.
        If False, the axis will be disabled.

    gamma_correction : bool, optional
        If the image should be gamma corrected, by default False
        If True, the image will be gamma corrected with the given gamma value.
    gamma : Union[_DEFAULT, float], optional
        Gamma value for the gamma correction, by default DEFAULT
        If DEFAULT, the gamma value will be estimated based on the observed image intensity.
        Set custom gamma value to use a specific gamma correction.
    Returns
    -------
    AxesImage
        The axes image object.
    """
    import itertools
    from matplotlib.axes import Subplot
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from tools.util.numpy import flatten_batch_dims as np_flatten_batch_dims

    imshow_kw = imshow_kw or dict()

    def flatten_batch(x):
        if len(x.shape) > 4:
            logger.warning(
                "Reshaping data to 2D for plotting. Data has more than 4 dimensions.")
            if "torch" in sys.modules:
                from tools.util.torch import flatten_batch_dims
                if isinstance(x, Tensor):
                    return flatten_batch_dims(x, -4)[0]
            if isinstance(x, np.ndarray):
                from tools.util.numpy import flatten_batch_dims
                return flatten_batch_dims(x, -4)
            else:
                raise ValueError(
                    "Data type not supported. Got: " + str(type(x)))
        return x

    if "torch" in sys.modules:
        from torch import Tensor
        if isinstance(data, Tensor):
            data = data.detach().cpu()
            if len(data.shape) >= 4:
                data = flatten_batch(data)
                data = [data[i] for i in range(data.shape[0])]

    if isinstance(data, np.ndarray):
        if len(data.shape) >= 4:
            data = flatten_batch(data)
            data = [data[i] for i in range(data.shape[0])]

    input_data = []

    if isinstance(data, (List, tuple)):
        for d in data:
            input_data.append(numpyify_image(d))
    else:
        input_data.append(numpyify_image(data))

    rows = 0
    cols = 1

    images = []
    img_title = []
    use_mathjax = []
    cmaps = []

    for i, data in enumerate(input_data):
        title_num_str = ""
        _col_images = []
        _col_titles = []
        _col_cmaps = []
        _col_use_mathjax = []
        v_name = "?"
        if variable_name is None:
            v_name = None
        elif isinstance(variable_name, (List, tuple)):
            if len(variable_name) > i:
                v_name = variable_name[i]
        else:
            v_name = variable_name

        if v_name is not None:
            v_name = str(v_name)
            used_mj, v_name = used_mathjax(v_name)
        else:
            used_mj = False

        if (len(input_data) > 1) and numbering:
            title_num_str = f"{rows + 1}: "
        if 'complex' in str(data.dtype):
            cols = 2
            _col_titles.append(
                f"{title_num_str}|{v_name}|") if v_name is not None else _col_titles.append(None)
            _col_use_mathjax.append(used_mj)
            _col_images.append(np.abs(data))
            _col_cmaps.append(cmap)

            _col_titles.append(
                f"{title_num_str}angle({v_name})") if v_name is not None else _col_titles.append(None)
            _col_use_mathjax.append(used_mj)
            angle = np.angle(data)
            _col_images.append(angle)
            _col_cmaps.append(phase_cmap)
        else:
            _col_titles.append(
                title_num_str + v_name) if v_name is not None else _col_titles.append(None)
            _col_use_mathjax.append(used_mj)
            _col_images.append(data)
            if cmap is None:
                _col_cmaps.append('viridis')
            else:
                if isinstance(cmap, (List, tuple)):
                    if len(cmap) > i:
                        _col_cmaps.append(cmap[i])
                    else:
                        _col_cmaps.append("viridis")
                else:
                    _col_cmaps.append(cmap)

        images.append(_col_images)
        img_title.append(_col_titles)
        cmaps.append(_col_cmaps)
        use_mathjax.append(_col_use_mathjax)

        rows += 1

    images = np.stack(images, axis=0)
    img_title = np.stack(img_title, axis=0)
    cmaps = np.stack(cmaps, axis=0)
    use_mathjax = np.stack(use_mathjax, axis=0)

    if not keep_transparency and images.shape[-1] == 4:
        from tools.io.image import alpha_compose_with_background_grid
        if images.dtype == np.uint8:
            img = images.astype(np.float32) / 255
            images = alpha_compose_with_background_grid(img)[..., :3]
            images = (images * 255).astype(np.uint8)
        else:
            images = alpha_compose_with_background_grid(images)[..., :3]

    if cols == 1:
        # If just one column, and images are not in landscape mode, flip rows and cols
        if len(images) > 1 and images[0][0].shape[0] > images[0][0].shape[1]:
            temp = rows
            rows = cols
            cols = temp
            images = images.swapaxes(0, 1)
            img_title = img_title.swapaxes(0, 1)
            cmaps = cmaps.swapaxes(0, 1)
            use_mathjax = use_mathjax.swapaxes(0, 1)

    if axes is None:
        if native_size:
            shp = np.array(images[0][0].shape[:2])
            cssize = shp / dpi
            size = cssize.max()
        fig, axes = get_mpl_figure(
            rows=rows, cols=cols, size=size, tight=tight, frame_on=frame_on, ratio_or_img=images[0][0], ax_mode="2d")
    else:
        fig = plt.gcf()

    if isinstance(axes, Subplot):
        axes = np.array([axes])

    if len(axes.shape) == 1:
        if rows == 1:
            axes = axes[None, ...]
        else:
            axes = axes[..., None]
    used_idx = 0

    for row in range(rows):
        for col in range(cols):
            ax = axes[row, col]

            _imgs = images[row]
            if col >= len(_imgs):
                ax.set_axis_off()
                continue

            if axis_enabled != DEFAULT and not axis_enabled:
                ax.set_axis_off()

            _image = images[row][col]
            _title = img_title[row][col]
            _use_mathjax = use_mathjax[row][col]

            color_mapping = None

            def op_only_finite(x, fnc):
                return fnc(x[(~np.isnan(x)) & (~np.isinf(x))])

            vmin = op_only_finite(_image, np.min)
            vmax = op_only_finite(_image, np.max)
            _cmap = cmaps[row][col]
            if isinstance(_cmap, str):
                _cmap = plt.get_cmap(_cmap)

            if cscale is not None:
                _cscale = cscale
                if isinstance(cscale, list):
                    _cscale = cscale[i]
                if _cscale == 'auto':
                    _cscale = 'log' if should_use_logarithm(
                        _image, 3) else None
                if _cscale is not None:
                    if _cscale == 'log':
                        zeros = _image == 0
                        non_finite = np.isnan(_image) | np.isinf(_image)
                        negative = _image < 0
                        _image = np.where(zeros, 1., _image)
                        _image = np.where(non_finite, 1., _image)
                        _image = np.where(negative, np.abs(_image), _image)
                        _log = np.log10(_image)
                        _log = np.where(zeros, np.nan, _log)
                        _log = np.where(non_finite, np.nan, _log)
                        _log = np.where(negative, -_log, _log)
                        _image = _log
                        _title = f"log_{{10}}({_title})" if _title is not None else None
                        _use_mathjax = True
                if _cscale == "count":
                    color_mapping = dict()
                    for j, value in enumerate(np.unique(_image)):
                        color_mapping[j] = value
                        _image = np.where(_image == value, j, _image)

            if gamma_correction:
                from tools.io.image import gamma_correction
                from tools.transforms.to_numpy_image import ToNumpyImage
                numpy_image = ToNumpyImage(np.uint8)
                _image = numpy_image(_image)
                _image, _ = gamma_correction(_image, gamma=gamma)

            if isinstance(_cmap, ListedColormap):
                vmax = len(_cmap.colors) - 1
                vmin = 0

            if "vmin" in imshow_kw:
                vmin = imshow_kw.pop("vmin")
            else:
                vmin = op_only_finite(_image, np.min)
            if "vmax" in imshow_kw:
                vmax = imshow_kw.pop("vmax")
            else:
                vmax = op_only_finite(_image, np.max)
            if "cmap" in imshow_kw:
                _cmap = imshow_kw.pop("cmap")
            if "interpolation" in imshow_kw:
                interpolation = imshow_kw.pop("interpolation")

            if norm:
                _norm = MinMax(new_min=0, new_max=1)
                _norm.min = vmin
                _norm.max = vmax
                _norm.fitted = True
                _image = _norm.transform(_image)
                vmin = _norm.new_min
                vmax = _norm.new_max

            # Check if contains bad values
            if np.any(np.isnan(_image)) or np.any(np.isinf(_image)):
                if interpolation is None:
                    interpolation = 'nearest'
                _cmap.set_bad(color='white')

            ax.imshow(_image, vmin=vmin, vmax=vmax, cmap=_cmap,
                      interpolation=interpolation, **imshow_kw)

            if inset_args is not None:
                from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
                from mpl_toolkits.axes_grid1.inset_locator import mark_inset
                from matplotlib.colors import to_rgba
                insa = [inset_args] if isinstance(
                    inset_args, dict) else list(inset_args)
                for inset_args in insa:
                    zoom = inset_args.get("zoom", 2)
                    loc = inset_args.get("loc", 'upper right')
                    connect_corners: Tuple[int, int] = inset_args.get(
                        "connect_corners", (2, 4))
                    edgecolor = to_rgba(inset_args.get("edgecolor", 'black'))
                    linewidth = inset_args.get("linewidth", 1)
                    start_xy = inset_args.get("start_xy", None)
                    end_xy = inset_args.get("end_xy", None)

                    inset_ax = zoomed_inset_axes(ax, zoom=zoom, loc=loc)
                    inset_ax.imshow(_image, vmin=vmin, vmax=vmax, cmap=_cmap,
                                    interpolation=interpolation, **imshow_kw)
                    inset_ax.set_xlim(start_xy[0], end_xy[0])
                    inset_ax.set_ylim(end_xy[1], start_xy[1])
                    inset_ax.get_xaxis().set_ticks([])
                    inset_ax.get_yaxis().set_ticks([])

                    inset_ax.spines['bottom'].set_color(edgecolor)
                    inset_ax.spines['bottom'].set_linewidth(linewidth)
                    inset_ax.spines['top'].set_color(edgecolor)
                    inset_ax.spines['top'].set_linewidth(linewidth)
                    inset_ax.spines['right'].set_color(edgecolor)
                    inset_ax.spines['right'].set_linewidth(linewidth)
                    inset_ax.spines['left'].set_color(edgecolor)
                    inset_ax.spines['left'].set_linewidth(linewidth)

                    def flip(corner):
                        if corner == 2:
                            return 3
                        elif corner == 3:
                            return 2
                        elif corner == 4:
                            return 1
                        elif corner == 1:
                            return 4
                    _patch, pp1, pp2 = mark_inset(
                        ax, inset_ax, loc1=connect_corners[0], loc2=connect_corners[1], fc="none", ec=edgecolor, linewidth=linewidth)
                    # Hacky workaround to get the correct corners
                    pp1.loc1, pp1.loc2 = connect_corners[0], flip(
                        connect_corners[0])
                    # Hacky workaround to get the correct corners
                    pp2.loc1, pp2.loc2 = connect_corners[1], flip(
                        connect_corners[1])

            if inpaint_title == True or (inpaint_title == DEFAULT and tight) and _title is not None:
                _inp_raw = np.zeros(
                    tuple(_image.shape[:2] + (4,)), dtype=np.uint8)
                _inpaint_img = _inpaint_title(
                    _inp_raw, _title, **(inpaint_title_kwargs if inpaint_title_kwargs is not None else dict()))
                ax.imshow(_inpaint_img)
            elif not tight:
                if _title is not None:
                    if _use_mathjax:
                        _title = f"${_title}$"
                    ax.set_title(_title)

            if colorbar:
                _cbar_format = None
                if colorbar_tick_format is not None:
                    cft = ('{:' + colorbar_tick_format + '}')

                    def _cbar_format(x, pos):
                        return cft.format(x)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(ax.get_images()[0], cax=cax,
                             format=_cbar_format, orientation='vertical')

            if not ticks:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

            if value_legend:
                unique_vals = np.unique(_image)
                patches = []
                if isinstance(_cmap, str):
                    _cmap = plt.get_cmap(_cmap)

                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                for i, value in enumerate(unique_vals):
                    c = _cmap(norm(value))
                    if color_mapping is not None:
                        value = color_mapping[value]
                    patches.append(Patch(color=c, label=f"{value:n}"))
                ax.legend(handles=patches)

    if title is not None:
        fig.suptitle(title)
    return fig


def append_colorbar_to_axes(handle: Any, ax: Axes, tick_format: Optional[str] = None):
    """
    Appends a colorbar to a given axes for the given handle.

    Parameters
    ----------
    handle : Any
        Handle to the data to create a colorbar for.

    ax : Axes
        The axes to append the colorbar to.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if tick_format is not None:
        tick_format = ('{:' + tick_format + '}')

        def _tick_format(x, pos):
            return tick_format.format(x)
    else:
        _tick_format = None
    ax.figure.colorbar(handle, cax=cax, format=_tick_format,
                       orientation='vertical')


def assemble_coords_to_image(
        coords: VEC_TYPE,
        data: VEC_TYPE,
        resolution: VEC_TYPE,
        domain: Optional[VEC_TYPE] = None,
) -> np.ndarray:
    """
    Inserts data / image given as per pixel values into an image (stack).
    May have whole if data is not dense.

    Parameters
    ----------
    coords : VEC_TYPE
        Coordinates of the data points.
        Shape should be ([..., B], R, 2)
        Where B is the batch dimension and R is the number of points.
        2 is the x and y coordinate within the domain.
    data : VEC_TYPE
        The data points for the image.
        Shape should be ([..., B], R, D)
    resolution : VEC_TYPE
        Image resolution.
        Given as (width, height)
    domain : VEC_TYPE
        Valid image domain of the data coords.
        Shape should be (2, 2) for min and max values.
        E.g.
        [[x_min, y_min],
         [x_max, y_max]]
        Default is None, which assumes domain is
        [[0, 0], [1, 1]]


    Returns
    -------
    np.ndarray
        The assembled image. Shape is ([..., B], H, W, D)
    """
    from tools.util.numpy import flatten_batch_dims, unflatten_batch_dims
    from tools.transforms.numpy.min_max import MinMax
    coords = numpyify(coords)
    data = numpyify(data)
    resolution = numpyify(resolution)
    if domain is None:
        domain = np.array([[0, 0], [1, 1]])
    else:
        domain = numpyify(domain)

    coords, shp = flatten_batch_dims(coords, -3)  # Shape is (BC, RC, 2)
    data, data_shp = flatten_batch_dims(data, -3)  # Shape is (BD, RD, D)

    BC, RC, CC = coords.shape
    BD, RD, CD = data.shape

    if CC != 2:
        raise ValueError("Coordinates should have 2 dimensions.")

    if RD != RC:
        raise ValueError(
            "Number of points should match in data and coords. But got {} and {}.".format(RD, RC))

    if BD != BC:
        if BC == 1:
            coords = np.repeat(coords, BD, axis=0)
        else:
            raise ValueError(
                "Number of batches should match in data and coords. But got {} and {}.".format(BD, BC))

    img = np.zeros((BD, *resolution[::-1], CD), dtype=data.dtype)

    mmx = MinMax(new_min=0, new_max=resolution[0])
    mmx.min = domain[0, 0]
    mmx.max = domain[1, 0]
    mmx.fitted = True

    mmy = MinMax(new_min=0, new_max=resolution[1])
    mmy.min = domain[0, 1]
    mmy.max = domain[1, 1]
    mmy.fitted = True

    coords = coords.copy()
    coords[..., 0] = mmx.transform(coords[..., 0])
    coords[..., 1] = mmy.transform(coords[..., 1])

    coords = coords[..., ::-1]
    coords = coords.round().astype(int)  # Shape is (BC, RC, 2) (y, x)

    # In domain filter
    in_domain_y = np.logical_and(
        coords[..., 0] >= 0, coords[..., 0] < resolution[1])
    in_domain_x = np.logical_and(
        coords[..., 1] >= 0, coords[..., 1] < resolution[0])
    in_domain = np.logical_and(in_domain_y, in_domain_x)

    # Add batch_idx
    bidx_coords = np.arange(BD)[:, None, None].repeat(RC, axis=1)
    coords = np.concatenate([bidx_coords, coords], axis=-1)

    img[coords[in_domain, 0], coords[in_domain, 1],
        coords[in_domain, 2]] = data[in_domain]
    return unflatten_batch_dims(img, data_shp)


def plot_coords_as_image(
        coords: VEC_TYPE,
        data: VEC_TYPE,
        resolution: VEC_TYPE,
        domain: Optional[VEC_TYPE] = None,
        **kwargs,
) -> Figure:
    """
    Plots data / image given as per pixel values into an image.
    May have whole if data is not dense.

    Parameters
    ----------
    coords : VEC_TYPE
        Coordinates of the data points.
        Shape should be ([..., B], R, 2)
        Where B is the batch dimension and R is the number of points.
        2 is the x and y coordinate within the domain.
    data : VEC_TYPE
        The data points for the image.
        Shape should be ([..., B], R, D)
    resolution : VEC_TYPE
        Image resolution.
        Given as (width, height)
    domain : VEC_TYPE
        Valid image domain of the data coords.
        Shape should be (2, 2) for min and max values.
        E.g.
        [[x_min, y_min],
         [x_max, y_max]]
        Default is None, which assumes domain is
        [[0, 0], [1, 1]]


    Returns
    -------
    Figure
        Matplot lib figure.
    """
    img = assemble_coords_to_image(coords, data, resolution, domain)
    return plot_as_image(img, **kwargs)


def _inpaint_title(image: VEC_TYPE, title: str, **kwargs):
    import torch
    from tools.io.image import put_text, n_layers_alpha_compositing
    from tools.transforms.numpy.min_max import MinMax
    img_shape = image.shape[-3:]

    args = dict(kwargs)
    if "margin" not in args:
        args["margin"] = 20
    if "padding" not in args:
        args["padding"] = 20
    if "size" not in args:
        args["size"] = 1.5
    if "thickness" not in args:
        args["thickness"] = 2
    if "background_stroke" not in args:
        args["background_stroke"] = 3

    title_img = put_text(np.zeros((*img_shape[:2], 4)),
                         title, **args).astype(np.float32) / 255
    mm = MinMax(new_min=0, new_max=1, axis=(-3, -2))
    transformed = mm.fit_transform(image)
    empty_image = (mm.min == mm.max).all()
    not_rgb = transformed.shape[-1] not in [3, 4]
    not_alpha = transformed.shape[-1] != 4
    if not_rgb:
        transformed = transformed.repeat(3, axis=-1)
    if not_alpha:
        transformed = np.concatenate(
            [transformed, np.ones((*img_shape[:2], 1))], axis=-1)
    inpainted = n_layers_alpha_compositing(torch.stack([torch.tensor(
        transformed), torch.tensor(title_img)], dim=0), torch.tensor([1, 0]))
    if not_alpha:
        inpainted = inpainted[..., :3]
    if not_rgb:
        inpainted = inpainted.mean(dim=-1, keepdim=True)
    if not empty_image:
        image = mm.inverse_transform(inpainted)
    else:
        image = inpainted
    return image


@saveable()
def plot_vectors(y: VEC_TYPE,
                 x: Optional[VEC_TYPE] = None,
                 label: Optional[Union[str, List[str]]] = None,
                 mode: Literal["plot", "scatter", "bar"] = "plot",
                 ax: Optional[Axes] = None,
                 bar_width: Optional[float] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 tick_right: bool = False,
                 xticks: Optional[Union[List[str], List[float]]] = None,
                 xticks_rotation: Optional[float] = None,
                 xticks_horizontal_alignment: Optional[str] = None,
                 title: Optional[str] = None,
                 legend: bool = True,
                 yerr: Optional[VEC_TYPE] = None,
                 yerr_kw: Optional[Dict[str, Any]] = None,
                 ) -> Figure:
    """Gets a matplotlib line figure with a plot of vectors.

    Parameters
    ----------
    y : VEC_TYPE
        Data to be plotted. Shape should be ([..., N], D)
        Batch dimensions will be flattened.
        Plots D lines with N points each.

    x : Optional[VEC_TYPE], optional
        X values for the plot. If None, x will be the range of y.shape[0], by default None
        Shape should be (N, )

    label : Optional[Union[str, List[str]]], optional
        Label or each dimension. If None just numerates the dimensions, by default None

    mode : Literal["plot", "scatter", "bar"], optional
        Mode of the plot, by default "plot"
        Plot: Line plot
        Scatter: Scatter plot
        Bar: Bar plot

    ax : Optional[Axes], optional
        Matplotlib axis to plot on, by default None
        If None, will create a new figure.

    bar_width : Optional[float], optional
        Width of the bars in bar mode, by default None
        None will calculate the width based on the x values and number of dimensions.

    xlim : Optional[Tuple[float, float]], optional
        X limits for the plot, by default None

    ylim : Optional[Tuple[float, float]], optional
        Y limits for the plot, by default None

    xlabel : Optional[str], optional
        X label for the plot, by default None

    ylabel : Optional[str], optional
        Y label for the plot, by default None

    tick_right : bool, optional
        If the ticks should be on the right side, by default False

    xticks : Optional[Tuple[List[float], List[str]]], optional
        X ticks for the plot, by default None
        Specifies the
        1. Positions of the ticks
        2. Labels of the ticks

    xticks_rotation : Optional[float], optional
        Rotation of the x ticks, by default None
        If None, will not rotate the ticks.

    xticks_horizontal_alignment : Optional[str], optional
        Horizontal alignment of the x ticks, by default None
        If None, will use "center".

    title : Optional[str], optional
        Title for the plot, by default None

    legend : bool, optional
        If a legend should be shown, by default True

    yerr : Optional[VEC_TYPE], optional
        Y error values for the plot, by default None
        Shape should be ([..., N], D)
        If provided, will add error bars to the plot.
        Batch dimensions will be flattened.

    yerr_kw : Optional[Dict[str, Any]], optional
        Additional kwargs for the error bars, by default None
        If provided, will be passed to the errorbar function.
        E.g. {'ecolor': 'gray', 'capsize': 2, 'elinewidth': 1}

    Returns
    -------
    Figure
        Matplotlib figure with the plot.

    Raises
    ------
    ValueError
        If label does not match the number of dimensions.
    """
    from tools.util.numpy import flatten_batch_dims
    y = numpyify(y)

    if len(y.shape) == 1:
        y = y[:, None]
    y, shape = flatten_batch_dims(y, -2)

    if yerr is not None:
        yerr = numpyify(yerr)
        if len(yerr.shape) == 1:
            yerr = yerr[:, None]
        yerr, _ = flatten_batch_dims(yerr, -2)

    if label is None:
        label = [str(i) for i in range(y.shape[-1])]
    else:
        if not isinstance(label, Iterable):
            label = [label]
        if len(label) != y.shape[-1]:
            raise ValueError(
                "Number of labels should match the last dimension of the input data.")
    x_was_none = x is None
    if x is None:
        x = np.arange(y.shape[0])
    else:
        x = numpyify(x)
        x, _ = flatten_batch_dims(x, -1)
    if ax is None:
        fig, ax = get_mpl_figure(1, 1)
    else:
        fig = ax.figure

    if bar_width is None and mode == "bar":
        bar_width = (np.amin((x[1:] - x[:-1])) if len(x)
                     > 1 else 1) / (1.5 * y.shape[-1])

    handles = []

    for i in range(y.shape[-1]):
        xpos = None
        ypos = None
        if mode == "plot":
            xpos = x
            ypos = y[:, i]
            handle, = ax.plot(xpos, ypos, label=label[i])
        elif mode == "scatter":
            xpos = x
            ypos = y[:, i]
            handle = ax.scatter(xpos, ypos, label=label[i])
        elif mode == "bar":
            position = x + i * bar_width
            # Center the bars
            xpos = position
            position = position - (bar_width * y.shape[-1]) / 2
            ypos = y[:, i]
            handle = ax.bar(
                position, ypos, width=bar_width, label=label[i], align="edge")
        else:
            raise ValueError("Mode should be either plot or scatter.")
        handles.append(handle)
        if yerr is not None:
            if yerr_kw is None:
                yerr_kw = dict()
            if "ecolor" not in yerr_kw:
                yerr_kw["ecolor"] = "gray"
            if "capsize" not in yerr_kw:
                yerr_kw["capsize"] = 2
            if "elinewidth" not in yerr_kw:
                yerr_kw["elinewidth"] = 1
            if "linestyle" not in yerr_kw:
                yerr_kw["linestyle"] = ''
            h = ax.errorbar(xpos, ypos, yerr=yerr[:, i], **yerr_kw)
            handles.append(h)

    if mode == "bar":
        from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    if xticks is not None:
        tpos, tlab = xticks
        if xticks_horizontal_alignment is None:
            xticks_horizontal_alignment = "center"
        ax.set_xticks(ticks=tpos, labels=tlab,
                      rotation=xticks_rotation, ha=xticks_horizontal_alignment)

    if xlim is not None:
        ax.set_xlim(*tuple(xlim))
    if ylim is not None:
        ax.set_ylim(*tuple(ylim))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if tick_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if title is not None:
        ax.set_title(title)
    if legend:
        preserve_legend(ax, handles, create_if_not_exists=True)
    return fig


@saveable()
def plot_histogram(
    x: VEC_TYPE,
    label: Optional[Union[str, List[str]]] = None,
    bins: Any = None,
    filter_nan: bool = False,
    yscale: Optional[str] = None,
    is_counts: bool = False,
    ax: Optional[Axes] = None,
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "bar",
) -> Figure:
    """Gets a matplotlib histogram figure with a plot of vectors.

    Parameters
    ----------
    x : VEC_TYPE
        Data to be plotted. Shape should be ([..., N], D)
        Batch dimensions will be flattened.
        Plots histogram for each dimension D.

    label : Optional[Union[str, List[str]]], optional
        Label or each dimension. If None just numerates the dimensions, by default None

    bins : Any, optional
        Bins for the histogram. If None, will use 'auto' binning, by default None
        Can be a number of bins or a sequence of bin edges.

    filter_nan : bool, optional
        If NaN values should be filtered out, by default False

    yscale : Optional[str], optional
        Y scale for the plot, by default None
        If 'log', will use logarithmic scale for the y-axis.

    is_counts : bool, optional
        If the input data is already counts, by default False
        If True, will not calculate histogram, but use the input data as counts and bins must be provided.

    ax : Optional[Axes], optional
        Matplotlib axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    Figure
        Matplotlib figure with the plot.

    Raises
    ------
    ValueError
        If label does not match the number of dimensions.
    """
    from tools.util.numpy import flatten_batch_dims

    x = numpyify(x)
    x, shape = flatten_batch_dims(x, -2)

    if label is None:
        label = [str(i) for i in range(x.shape[-1])]
    else:
        if not isinstance(label, Iterable):
            label = [label]
        if len(label) != x.shape[-1]:
            raise ValueError(
                "Number of labels should match the last dimension of the input data.")
    if ax is None:
        fig, ax = get_mpl_figure(1, 1)
    else:
        fig = ax.figure

    for i in range(x.shape[-1]):
        l = label[i] if label is not None else None
        v = x[:, i]
        if filter_nan:
            vals = vals[~np.isnan(vals)]
        if not is_counts:
            if bins is None:
                bins = 'auto'
            vals, bins = np.histogram(v, bins=bins)
        else:
            vals = v
        ax.hist(bins[:-1], bins, weights=vals, label=l, histtype=histtype)

    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend()

    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

    return fig


def figure_to_numpy(fig: Figure, dpi: int = 300, transparent: bool = True) -> np.ndarray:
    """Converts a matplotlib figure to a numpy array.

    Parameters
    ----------
    fig : Figure
        The figure to convert

    dpi : int, optional
        Dots per inch, by default 72

    Returns
    -------
    np.ndarray
        The figure as a numpy array
    """

    arr = None
    if fig.dpi != dpi:
        fig.set_dpi(dpi)
    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, format='raw', transparent=transparent, dpi=fig.dpi)
        io_buf.seek(0)
        arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    return arr


def save_as_image_stack(data: VEC_TYPE,
                        folder_path: str,
                        images_filename_format: str = "{index}.png",
                        override: bool = False,
                        mkdirs: bool = True,
                        progress_bar: bool = False) -> List[str]:
    from PIL import Image
    data = numpyify_image(data)
    filenames = parse_format_string(images_filename_format, [
                                    dict(index=i) for i in range(data.shape[0])])
    file_paths = [folder_path + "/" + f for f in filenames]
    if not override:
        file_paths = [numerated_file_name(f) for f in file_paths]
    if mkdirs:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
    bar = None
    if progress_bar:
        from tqdm.auto import tqdm
        bar = tqdm(total=data.shape[0], desc="Saving images")
    for i in range(data.shape[0]):
        minmax = MinMax(new_min=0, new_max=255)
        img = minmax.fit_transform(data[i]).astype(np.uint8)
        Image.fromarray(img).save(file_paths[i])
        if bar is not None:
            bar.update(1)


def save_as_image(data: VEC_TYPE,
                  path: str,
                  override: bool = False,
                  mkdirs: bool = True) -> str:
    """Saves numpy array or torch tensor as an image.

    Parameters
    ----------
    data : VEC_TYPE
        Data to be saved as an image. Should be in the shape (H, W, C)
        for numpy arrays or (C, H, W) torch tensors.
    path : str
        Path to save the image to.
    override : bool, optional
        If an existing image should be overriden, by default False
    mkdirs : bool, optional
        If the directories should be created if they do not exist, by default True
    Returns
    -------
    str
        The path where the image was saved.
    """
    from PIL import Image
    img = numpyify_image(data)
    if not override:
        path = numerated_file_name(path)
    if mkdirs:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def load_as_image(path: str) -> np.ndarray:
    """Loads an image from a path.
    Should give the same result as the data which was saved with save_as_image.

    Parameters
    ----------
    path : str
        Path to the image.

    Returns
    -------
    np.ndarray
        The loaded image.
    """
    from PIL import Image
    return np.array(Image.open(path))


def align_marker(marker: Any, ha: Union[str, float] = 'center', va: Union[str, float] = 'center'):
    """
    create markers with specified alignment.

    Parameters
    ----------

    marker : a valid marker specification.
      See mpl.markers

    ha : string, float {'left', 'center', 'right'}
      Specifies the horizontal alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'center',
      -1 is 'right', 1 is 'left').

    va : string, float {'top', 'middle', 'bottom'}
      Specifies the vertical alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'middle',
      -1 is 'top', 1 is 'bottom').

    Returns
    -------

    marker_array : numpy.ndarray
      A Nx2 array that specifies the marker path relative to the
      plot target point at (0, 0).

    Notes
    -----
    The mark_array can be passed directly to ax.plot and ax.scatter, e.g.::

        ax.plot(1, 1, marker=align_marker('>', 'left'))

    Credit:
    https://stackoverflow.com/questions/26686722/align-matplotlib-scatter-marker-left-and-or-right

    """

    from matplotlib import markers
    from matplotlib.path import Path
    if isinstance(ha, str):
        ha = {'right': -1.,
              'middle': 0.,
              'center': 0.,
              'left': 1.,
              }[ha]

    if isinstance(va, str):
        va = {'top': -1.,
              'middle': 0.,
              'center': 0.,
              'bottom': 1.,
              }[va]

    # Define the base marker
    bm = markers.MarkerStyle(marker)

    # Get the marker path and apply the marker transform to get the
    # actual marker vertices (they should all be in a unit-square
    # centered at (0, 0))
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    # Shift the marker vertices for the specified alignment.
    m_arr[:, 0] += ha / 2
    m_arr[:, 1] += va / 2

    return Path(m_arr, bm.get_path().codes)


def set_axes_equal_3d(ax: Axes3D):
    """Set the aspect ratio of the 3D plot to be equal.

    Parameters
    ----------
    ax : Axes3D
        3D axis to set the aspect ratio for.
    """
    import numpy as np
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


@saveable()
def plot_mask(image: VEC_TYPE,
              mask: VEC_TYPE,
              size: int = 5,
              title: str = None,
              tight: bool = False,
              background_value: int = 0,
              ignore_class: Optional[int] = None,
              _colors: Optional[List[str]] = None,
              color: Union[List[str], str] = "#5999cb",
              contour_linewidths: float = 2,
              mask_mode: Literal['channel_mask',
                                 'value_mask'] = 'channel_mask',
              ax: Optional[Axes] = None,
              darkening_background: float = 0.7,
              labels: Optional[List[str]] = None,
              lined_contours: bool = True,
              filled_contours: bool = False,
              axes_description: bool = False,
              image_cmap: Optional[Any] = None,
              sort: bool = False,
              reverse: bool = True,
              inpaint_indices: bool = False,
              inpaint_title: bool = DEFAULT,
              inpaint_title_kwargs: Optional[Dict[str, Any]] = None,
              legend: bool = DEFAULT,
              overlap_area: Optional[List[np.ndarray]] = None,
              transparent_image: bool = False,
              native_size: bool = False,
              dpi: int = 300,
              correct_gamma: bool = False,
              gamma: Union[_DEFAULT, float] = DEFAULT,
              **kwargs) -> Figure:  # type: ignore
    import matplotlib.patches as mpatches
    from tools.transforms import ToNumpyImage
    from matplotlib.colors import to_rgba, to_rgb
    from collections.abc import Iterable
    from tools.io.image import put_text

    to_numpy = ToNumpyImage()

    if image is None and mask is None:
        raise ValueError("At least one of image or mask should be provided")

    mask = to_numpy(mask) if mask is not None else None
    image = to_numpy(image) if image is not None else None
    if image is None:
        image = np.zeros(mask.shape[:2] + (3,))

    mask = mask.squeeze()
    if len(mask.shape) == 2:
        mask = mask[..., None]
    # Check if mask contains multiple classes

    if ignore_class is not None:
        fill = np.zeros_like(mask)
        fill.fill(background_value)
        mask = np.where(mask == ignore_class, fill, mask)

    channel_mask = None
    if mask_mode == 'value_mask':
        channel_mask, _ = value_mask_to_channel_masks(mask)
    elif mask_mode == 'channel_mask':
        channel_mask = mask
    else:
        raise ValueError("mask_mode should be 'channel_mask' or 'value_mask'")

    multi_class = channel_mask.shape[2] > 1
    any_fg_mask = np.clip(np.sum(np.where(
        mask != background_value, 1, 0), axis=-1), 0, 1)  # True if not background
    background_mask = np.logical_not(any_fg_mask).astype(float)

    cmap_name = 'Blues'
    if sort:
        # Order masks by size, descending
        mask_sizes = np.sum(channel_mask, axis=(0, 1))
        order = np.argsort(mask_sizes)
        if reverse:
            order = order[::-1]
        channel_mask = channel_mask[..., order]
        if labels is not None:
            labels = [labels[i] for i in order]

    if ax is None:
        if native_size:
            shp = np.array(image.shape[:2])
            cssize = shp / dpi
            size = cssize.max()
        fig, ax = get_mpl_figure(
            1, 1, size=size, ratio_or_img=image, tight=tight)
    else:
        fig = ax.figure

    if image is not None and not transparent_image:

        if correct_gamma:
            from tools.io.image import gamma_correction
            np_img = ToNumpyImage(np.uint8)
            image, _ = gamma_correction(image, gamma=gamma)

        ax.imshow(image, cmap=image_cmap)

    cmap = plt.get_cmap("alpha_binary")

    cmap = "tab10" if channel_mask.shape[-1] <= 10 else "tab20"
    if isinstance(color, (list, tuple)) and multi_class:
        colors = color
    else:
        cmap = plt.get_cmap(cmap)
        if not multi_class:
            if isinstance(color, (list)):
                if len(color) == 1:
                    if len(color[0]) not in [3, 4]:
                        raise ValueError("Color should be RGB or RGBA.")
                    else:
                        colors = color
                elif len(color) not in [3, 4]:
                    raise ValueError("Color should be RGB or RGBA.")
                else:
                    colors = [color]
            else:
                colors = [color]
        else:
            colors = cmap(
                [x % cmap.N for x in range(channel_mask.shape[-1])])

    m_inv = np.ones(mask.shape[:-1])

    patches = []

    if overlap_area is None:
        overlap_area = list()

    def _inpaint_indices(mask, image, color):
        if np.any(mask):
            com = np.argwhere(mask).mean(axis=0).round().astype(int)[::-1]
            background_color = "white" if np.mean(
                color[:3]) < 0.5 else "black"
            stroke_color = "black" if np.mean(
                color[:3]) < 0.5 else "white"
            inp = put_text(image, f"{label}",
                           placement=None,
                           position=com,
                           vertical_alignment='center',
                           horizontal_alignment='center',
                           background_color=background_color,
                           background_stroke_color=stroke_color,
                           background_stroke=1,
                           color=color,
                           check_overlap=True,
                           overlap_area=overlap_area
                           )
            return inp
        return image

    for i in range(channel_mask.shape[-1]):
        m = channel_mask[..., i]
        label = None
        if labels is not None:
            if isinstance(labels, Iterable):
                label = labels[i]
            else:
                label = str(labels)

        if lined_contours:
            ax.contour(
                m_inv - m, levels=[0.5], colors=[colors[i]], linewidths=contour_linewidths)
        if filled_contours:
            _color = to_rgba(colors[i][:])
            c_img = np.zeros((*m.shape, 4))
            c_img[:, :, :] = _color
            c_img[:, :, -1] = c_img[:, :, -1] * m
            if inpaint_indices:
                c_img = _inpaint_indices(m, c_img, _color)
            ax.imshow(c_img)
        if not filled_contours and inpaint_indices:
            _color = to_rgba(colors[i][:])
            c_img = np.zeros((*m.shape, 4))
            c_img = _inpaint_indices(m, c_img, _color)
            ax.imshow(c_img, zorder=1000)

        if label is not None:
            patches.append(mpatches.Patch(color=colors[i], label=label))

    ax.imshow(background_mask, cmap='alpha_binary',
              alpha=darkening_background, label='')

    if inpaint_title == True or (inpaint_title == DEFAULT and tight):
        _img = np.zeros_like(
            background_mask)[..., np.newaxis].repeat(4, axis=-1)
        _image = _inpaint_title(
            _img, title, **(inpaint_title_kwargs if inpaint_title_kwargs is not None else dict()))
        ax.imshow(_image)

    if not tight:
        ax.axis('off')
    # plt.legend()
    if title is not None and not tight:
        ax.set_title(title)

    if _colors is not None:
        _colors.clear()
        _colors.extend(colors)

    if legend == DEFAULT:
        legend = len(patches) > 0

    if patches is not None and len(patches) > 0:
        preserve_legend(ax, patches, create_if_not_exists=legend)

    origin_marker_color = kwargs.get('origin_marker_color', None)
    origin_marker_opposite_color = kwargs.get(
        'origin_marker_opposite_color', None)

    if origin_marker_color is not None or origin_marker_opposite_color is not None:
        from matplotlib.colors import get_named_colors_mapping
        # Create markers with imshow
        transparent_nav = np.zeros((*image.shape[:2], 4))
        cmap = get_named_colors_mapping()

        def make_circle(r):
            y, x = np.ogrid[-r:r, -r:r]
            return x**2 + y**2 <= r**2

        origin_marker_size = kwargs.get('origin_marker_size', 24)
        # int(round(math.sqrt((origin_marker_size) / np.pi)))
        marker_radius = origin_marker_size

        if origin_marker_color is not None:
            if isinstance(origin_marker_color, str):
                origin_marker_color = cmap[origin_marker_color]
            origin_marker_color = tuple(to_rgb(origin_marker_color))
            origin_marker_color += (1,)
            # Coloring the origin with 10 pixels
            transparent_nav[:2 * marker_radius, :2 *
                            marker_radius][make_circle(marker_radius)] = origin_marker_color

        if origin_marker_opposite_color is not None:
            if isinstance(origin_marker_opposite_color, str):
                origin_marker_opposite_color = cmap[origin_marker_opposite_color]
            origin_marker_opposite_color = tuple(
                to_rgb(origin_marker_opposite_color))
            origin_marker_opposite_color += (1,)
            # Coloring the origin with 10 pixels
            transparent_nav[-2 * marker_radius:, -2 * marker_radius:][make_circle(
                marker_radius)] = origin_marker_opposite_color

        ax.imshow(transparent_nav)

    if axes_description:
        ax.set_axis_on()
        ax.set_xlabel("Coordinates [x]")
        ax.set_ylabel("Coordinates [y]")
    return fig


@saveable(default_dpi=150)
def plot_mask_overview(image: VEC_TYPE, mask: VEC_TYPE):
    import matplotlib.patches as mpatches
    from tools.transforms import ToNumpyImage
    from matplotlib.colors import to_rgba, to_rgb
    from collections.abc import Iterable

    to_numpy = ToNumpyImage()

    if image is None and mask is None:
        raise ValueError("At least one of image or mask should be provided")

    mask = to_numpy(mask) if mask is not None else None
    image = to_numpy(image) if image is not None else None
    if image is None:
        image = np.zeros(mask.shape[:2] + (3,))

    mask = mask.squeeze()
    if len(mask.shape) == 2:
        mask = mask[..., None]
    H, W, C = mask.shape
    ratio = H / W
    cols = 5
    rows = math.ceil(C / cols)
    size = 3
    fig, ax = get_mpl_figure(
        rows, cols, ratio_or_img=mask[..., 0], size=size, tight=True, ax_mode="2d")
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            if i >= C:
                ax[row, col].axis("off")
                continue
            else:
                plot_mask(image, mask[:, :, i], ax=ax[row, col], labels=[
                          str(i)], lined_contours=False, filled_contours=True)
    return fig


register_alpha_map('binary')
register_alpha_map('Greens')
register_alpha_map('Reds')
register_alpha_map('Blues')
