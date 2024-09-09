
# Class for functions
# File for useful functions when using matplotlib
import io
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple

from matplotlib.colors import ListedColormap
from matplotlib.image import AxesImage

from tools.segmentation.masking import value_mask_to_channel_masks
from tools.util.format import parse_format_string
from tools.util.numpy import numpyify_image
from tools.util.torch import VEC_TYPE
from tools.transforms.numpy.min_max import MinMax

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
import matplotlib.text as mtext


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

    # type: ignore
    def decorator(function: Callable[[Any], Figure]) -> Callable[[Any], Figure]:
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
                    default_output_dir = os.environ.get(
                        "PLOT_OUTPUT_DIR", default_output_dir)
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


def get_mpl_figure(
        rows: int = 1,
        cols: int = 1,
        size: float = 5,
        ratio_or_img: Optional[Union[float, np.ndarray]] = None,
        tight: bool = False,
        subplot_kw: Optional[Dict[str, Any]] = None,
        ax_mode: Literal["1d", "2d"] = "1d",
) -> Tuple[Figure, Union[Axes, List[Axes]]]:  # type: ignore
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
        Ratio of Y w.r.t X  (Height / Width) can also be an Image / np.ndarray which will compute it from the axis, by default 1.0
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
            ax = plt.Axes(fig, [col * rel_width, row *
                          rel_height, rel_width, rel_height])
            ax.set_axis_off()
            fig.add_axes(ax)
            axes.append(ax)
    else:
        fig, ax = plt.subplots(rows, cols, figsize=(size * ratio_x * cols,
                                                    size * ratio_y * rows), subplot_kw=subplot_kw)
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
    return np.max(x) / np.min(x) > math.pow(10, magnitudes)


def preserve_legend(ax: Axes,  # type: ignore
                    patches: List[Patch],  # type: ignore
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
    map_object = LinearSegmentedColormap.from_list(
        name=name, colors=color_array)

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
                  imshow_kw: Optional[Dict[str, Any]] = None,
                  norm: bool = False,
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

    Returns
    -------
    AxesImage
        The axes image object.
    """
    import itertools
    from matplotlib.axes import Subplot
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    imshow_kw = imshow_kw or dict()

    if "torch" in sys.modules:
        from torch import Tensor
        if isinstance(data, Tensor):
            data = data.detach().cpu()
            if len(data.shape) == 4:
                data = [data[i] for i in range(data.shape[0])]
    if isinstance(data, np.ndarray):
        if len(data.shape) == 4:
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
    cmaps = []

    for data in input_data:
        title_num_str = ""
        _col_images = []
        _col_titles = []
        _col_cmaps = []

        if len(input_data) > 1:
            title_num_str = f"{rows + 1}: "
        if 'complex' in str(data.dtype):
            cols = 2
            _col_titles.append(f"{title_num_str}|{variable_name}|")
            _col_images.append(np.abs(data))
            _col_cmaps.append("gray")

            _col_titles.append(f"{title_num_str}angle({variable_name})")
            _col_images.append(np.angle(data))
            _col_cmaps.append("twilight")
        else:
            _col_titles.append(title_num_str + variable_name)
            _col_images.append(data)
            if cmap is None:
                _col_cmaps.append("gray")
            else:
                _col_cmaps.append(cmap)

            images.append(_col_images)
            img_title.append(_col_titles)
            cmaps.append(_col_cmaps)

        rows += 1

    if axes is None:
        fig, axes = get_mpl_figure(
            rows=rows, cols=cols, size=size, tight=tight, ratio_or_img=images[0][0], ax_mode="2d")
    else:
        fig = plt.gcf()

    if isinstance(axes, Subplot):
        axes = np.array([axes])

    if len(axes.shape) == 1:
        axes = axes[None, ...]
    used_idx = 0

    for row in range(rows):
        for col in range(cols):
            ax = axes[row, col]
            _imgs = images[row]
            if col >= len(_imgs):
                ax.set_axis_off()
                continue

            _image = images[row][col]
            _title = img_title[row][col]

            color_mapping = None

            vmin = _image.min()
            vmax = _image.max()
            _cmap = cmaps[row][col]
            if isinstance(_cmap, str):
                _cmap = plt.get_cmap(_cmap)

            if cscale is not None:
                _cscale = cscale
                if isinstance(cscale, list):
                    _cscale = cscale[i]
                if _cscale == 'auto':
                    _cscale = 'log' if should_use_logarithm(
                        _image.numpy()) else None
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

            if "vmin" in imshow_kw:
                vmin = imshow_kw.pop("vmin")
            else:
                vmin = _image.min()
            if "vmax" in imshow_kw:
                vmax = imshow_kw.pop("vmax")
            else:
                vmax = _image.max()
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

            ax.imshow(_image, vmin=vmin, vmax=vmax, cmap=_cmap,
                      interpolation=interpolation, **imshow_kw)

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


@saveable()
def plot_mask(image: VEC_TYPE,
              mask: VEC_TYPE,
              size: int = 5,
              title: str = None,
              tight: bool = False,
              background_value: int = 0,
              ignore_class: Optional[int] = None,
              _colors: Optional[List[str]] = None,
              color: str = "#5999cb",
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
              **kwargs) -> Figure:  # type: ignore
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

    if ax is None:
        fig, ax = get_mpl_figure(
            1, 1, size=size, ratio_or_img=image, tight=tight)
    else:
        fig = ax.figure

    if image is not None:
        ax.imshow(image, cmap=image_cmap)

    cmap = plt.get_cmap("alpha_binary")

    cmap = "tab10" if channel_mask.shape[-1] <= 10 else "tab20"
    if isinstance(color, (list, tuple)) and multi_class:
        colors = color
    else:
        colors = [color] if not multi_class else plt.get_cmap(
            cmap)(range(channel_mask.shape[-1]))

    m_inv = np.ones(mask.shape[:-1])

    patches = []
    for i in range(channel_mask.shape[-1]):
        m = channel_mask[..., i]
        label = None
        if labels is not None:
            if isinstance(labels, Iterable):
                label = labels[i]
            label = str(labels)

        if lined_contours:
            ax.contour(
                m_inv - m, levels=[0.5], colors=[colors[i]], linewidths=contour_linewidths)
        if filled_contours:
            _color = to_rgba(colors[i][:])
            c_img = np.zeros((*m.shape, 4))
            c_img[:, :, :] = _color
            c_img[:, :, -1] = c_img[:, :, -1] * m
            ax.imshow(c_img)
        if label is not None:
            patches.append(mpatches.Patch(color=colors[i], label=label))

    ax.imshow(background_mask, cmap='alpha_binary',
              alpha=darkening_background, label='')

    if not tight:
        ax.axis('off')
    # plt.legend()
    if title is not None:
        fig.suptitle(title)

    if _colors is not None:
        _colors.clear()
        _colors.extend(colors)

    if patches is not None and len(patches) > 0:
        preserve_legend(ax, patches)

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


register_alpha_map('binary')
register_alpha_map('Greens')
register_alpha_map('Reds')
register_alpha_map('Blues')
