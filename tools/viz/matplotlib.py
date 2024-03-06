
# Class for functions
# File for useful functions when using matplotlib
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple

from matplotlib.colors import ListedColormap
from matplotlib.image import AxesImage

from tools.util.numpy import numpyify_image
from tools.util.torch import VEC_TYPE

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
except (ModuleNotFoundError, ImportError):
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

def saveable(
    default_ext: Union[str, List[str]] = "png",
    default_output_dir: Optional[str] = "./temp",
    default_transparent: bool = False,
    default_dpi: int = 300,
    default_override: bool = False,
    default_tight_layout: bool = False,
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

    Parameters
    ----------
    default_ext : Union[str, List[str]], optional
        [List of] Extension[s] to save the figure, by default "png"
    default_output_dir : Optional[str], optional
        Output directory of figures when path did not contain directory information, 
        If the Environment Variable "PLOT_OUTPUT_DIR" is set, it will be used as destination.
        by default "./temp"
    default_transparent : bool, optional
        If the function should output the figures with transparent background, by default False
    default_dpi : int, optional
        If the function should by default override, by default 300
    default_override : bool, optional
        If the function should by default override, by default False
    default_tight_layout : bool, optional
        If the function should by default call tight_layout on the figure, by default False
    """
    from uuid import uuid4

    def decorator(function: Callable[[Any], Figure]) -> Callable[[Any], Figure]: # type: ignore
        @wraps(function)
        def wrapper(*args, **kwargs):
            nonlocal default_output_dir
            path = kwargs.pop("path", str(uuid4()))
            save = kwargs.pop("save", False)
            ext = kwargs.pop("ext", default_ext)
            transparent = kwargs.pop("transparent", default_transparent)
            dpi = kwargs.pop("dpi", default_dpi)
            override = kwargs.pop("override", default_override)
            tight_layout = kwargs.pop("tight_layout", default_tight_layout)
            open = kwargs.pop("open", False)
            set_interactive_mode = kwargs.pop("set_interactive_mode", None)
            close = kwargs.pop("auto_close", False)
            display = kwargs.pop("display", False)
            display_auto_close = kwargs.pop("display_auto_close", True)

            # Get interactive mode.
            is_interactive = mpl.is_interactive()

            if set_interactive_mode is not None:
                mpl.interactive(set_interactive_mode)
            try:
                out = function(*args, **kwargs)
            finally:
                mpl.interactive(is_interactive)

            if tight_layout:
                out.tight_layout()

            paths = []
            if save or open:
                if any([(x in path) for x in ["/", "\\", os.sep]]):
                    # Treat path as abspath
                    path = os.path.abspath(path)
                else:
                    default_output_dir = os.environ.get("PLOT_OUTPUT_DIR", default_output_dir)
                    path = os.path.join(os.path.abspath(default_output_dir), path)
                # Check if path has extension
                _, has_ext = os.path.splitext(path)
                if len(has_ext) == 0:
                    if isinstance(ext, str):
                        ext = [ext]
                    for e in ext:
                        paths.append(path + "." + e)
                else:
                    paths = [path]

                # Create parent dirs
                dirs = set([os.path.dirname(p) for p in paths])
                for d in dirs:
                    os.makedirs(d, exist_ok=True)

                for p in paths:
                    if not override:
                        p = numerated_file_name(p)
                    out.savefig(p, transparent=transparent, dpi=dpi)
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
            return out
        return wrapper
    return decorator


def get_mpl_figure(
        rows: int = 1, 
        cols: int = 1, 
        size: float = 5,
        ratio_or_img: Union[float,np.ndarray]= 1.0,
        tight: bool = False,
        subplot_kw: Optional[Dict[str, Any]] = None,
        ax_mode: Literal["1d", "2d"] = "1d",
        ) -> Tuple[Figure, Union[Axes, List[Axes]]]:# type: ignore
    """Create a eventually tight matplotlib figure with axes.

    Parameters
    ----------
    rows : int, optional
        Number of rows for the figure, by default 1
    cols : int, optional
        Nombuer of columns, by default 1
    size : float, optional
        Size of the axes in inches, by default 5
    ratio_or_img : float | np.ndarray, optional
        Ratio of Y w.r.t X can also be an Image / np.ndarray which will compute it from the axis, by default 1.0
    tight : bool, optional
        If the figure should be tight => No axis spacing and borders, by default False
    subplot_kw : Optional[Dict[str, Any]], optional
        Optional kwargs for the subplots, by default None
        Only used if tight is False

    Returns
    -------
    Tuple[Figure, Axes | List[Axes]]
        Figure and axes.
    """
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

    ratio_x = 1
    ratio_y = ratio_or_img
    dpi = 300
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
            ax = plt.Axes(fig, [col * rel_width, row * rel_height, rel_width, rel_height])
            ax.set_axis_off()
            fig.add_axes(ax)
            axes.append(ax)
    else:
        fig, ax = plt.subplots(rows, cols, figsize=(size * ratio_x * cols, 
            size * ratio_y * rows), subplot_kw=subplot_kw)
        axes.append(ax)

    if ax_mode == "2d" and tight:
        axes = np.reshape(np.array(axes), (rows, cols), order="F")[::-1]
    elif ax_mode == "2d" and not tight:
        axes = np.reshape(np.array(axes), (rows, cols), order="C")#[::-1]
    elif ax_mode == "1d" and not tight:
        axes = np.reshape(np.array(axes), (rows * cols), order="C")

    if len(axes) == 1:
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
            cm._colormaps.unregister(name)

    # get colormap
    ncolors = 256

    base_map = plt.get_cmap(base_name)
    N = base_map.N
    color_array = base_map(range(N))

    # change alpha values
    color_array[:, -1] = np.linspace(0, 1.0, ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=name, colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
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
    return np.max(x) / np.min(x) > math.pow(10, magnitudes)

def preserve_legend(ax: Axes, # type: ignore
                    patches: List[Patch], # type: ignore
                    **kwargs):
    """Checks if the axis has a legend and appends the patches to the legend.
    If not, it creates a new legend with the patches.

    Parameters
    ----------
    ax : Axes
        The axis to check for the legend.
    """    """"""
    if ax.get_legend() is not None:
        lgd = ax.get_legend() 
        handles = list(lgd.legend_handles)
        labels = [x.get_label() for x in lgd.legend_handles]
        handles.extend(patches)
        labels.extend([p.get_label() for p in patches])
        ax.legend(handles=handles, labels=labels, **kwargs)
    else:
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
    map_object = LinearSegmentedColormap.from_list(name=name, colors=color_array)

    return map_object

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
                  cmap: Optional[str] = None,
                  axes: Optional[np.ndarray] = None,
                  interpolation: Optional[str] = None,
                  tight: bool = False,
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

    Returns
    -------
    AxesImage
        The axes image object.
    """
    import itertools
    from matplotlib.axes import Subplot
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    data = numpyify_image(data)

    rows = 1
    cols = 1

    images = []
    _img_title = []
    cmaps = []

    if 'complex' in str(data.dtype):
        cols = 2
        _img_title.append(f"|{variable_name}|")
        images.append(np.abs(data))
        cmaps.append("gray")

        _img_title.append(f"angle({variable_name})")
        images.append(np.angle(data))
        cmaps.append("twilight")
    else:
        _img_title.append(variable_name)
        images.append(data)
        if cmaps is None:
            cmaps.append("gray")
        else:
            cmaps.append(cmap)


    if axes is None:
        fig, axes = get_mpl_figure(rows=rows, cols=cols, size=size, tight=tight, ratio_or_img=images[0])
    else:
        fig = plt.gcf()

    if isinstance(axes, Subplot):
        axes = [axes]

    for i, ax in enumerate(itertools.chain(axes)):
        _image = images[i]
        _title = _img_title[i]
        
        color_mapping = None

        vmin = _image.min()
        vmax = _image.max()
        _cmap = cmaps[i]
        if isinstance(_cmap, str):
            _cmap = plt.get_cmap(_cmap)

        if cscale is not None:
            _cscale = cscale
            if isinstance(cscale, list):
                _cscale = cscale[i]
            if _cscale == 'auto':
                _cscale = 'log' if should_use_logarithm(_image.numpy()) else None
            if _cscale is not None:
                if _cscale == 'log':
                    _image = np.log(_image)
                    _title = f"log({_title})"
            if _cscale == "count":
                color_mapping = dict()
                for j, value in enumerate(np.unique(_image)):
                    color_mapping[j] = value
                    _image = np.where(_image == value, j, _image)

        if isinstance(_cmap, ListedColormap):
            vmax = len(_cmap.colors) - 1
            vmin = 0
        
        ax.imshow(_image, vmin=vmin, vmax=vmax, cmap=_cmap, interpolation=interpolation)
        
        if not tight:
            ax.set_title(_title)
        if colorbar:
            _cbar_format = None
            if colorbar_tick_format is not None:
                cft = ('{:' + colorbar_tick_format + '}')

                def _cbar_format(x, pos):
                    return cft.format(x)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(ax.get_images()[0], cax=cax, format=_cbar_format, orientation='vertical')
        
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