from typing import Union
import numpy as np
import torch
from tools.transforms.numpy.min_max import MinMax
from tools.transforms.to_tensor_image import ToTensorImage
from tools.transforms.to_tensor import tensorify
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims
from tools.util.typing import VEC_TYPE, DEFAULT, _DEFAULT
from tools.transforms.to_numpy import numpyify
from tools.util.format import destinctive_number_float_format
import pandas as pd
from typing import Any, Dict, Optional
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import math


def make_colorwheel() -> np.ndarray:
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
    -------
    colorwheel : np.ndarray
        RGB Color wheel.

    Credits
    -------
    Author: Tom Runia
    Date Created: 2018-08-03

    https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis/flow_vis.py

    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def get_wheel_image(image_size: int = 256) -> np.ndarray:
    """
    Create a color wheel image, interpolated to match the image size.

    Parameters
    ----------
    image_size : int, optional
        Size of the image, by default 256

    Returns
    -------
    np.ndarray
        Image of the color wheel.
        Shape: (image_size, image_size, 4) RGBA image in np.uint8.
    """
    center_x, center_y = image_size // 2, image_size // 2
    # Create a grid of x, y coordinates
    x = np.linspace(-center_x, center_x, image_size)
    y = np.linspace(-center_y, center_y, image_size)

    # Create a meshgrid from x, y
    xx, yy = np.meshgrid(x, y)

    # Create a circle of radius half the image size
    circle = (xx ** 2 + yy ** 2) < (center_x ** 2)
    wheel = make_colorwheel()

    xx_circ = xx[circle]
    yy_circ = yy[circle]

    dist = np.sqrt(xx_circ ** 2 + yy_circ ** 2)
    # Compute the angle for each pixel
    angles = (np.arctan2(yy_circ, xx_circ) + np.pi) % (2 *
                                                       np.pi)  # Normalize to [0, 2 * pi]

    # Interpolate the wheel colors
    base_color = np.ones((1, 3))  # White

    # Supersample the wheel
    xp = np.linspace(0., 2 * np.pi, len(wheel))
    wheel_exact_r = np.interp(angles, xp, wheel[:, 0], period=2 * np.pi)
    wheel_exact_g = np.interp(angles, xp, wheel[:, 1], period=2 * np.pi)
    wheel_exact_b = np.interp(angles, xp, wheel[:, 2], period=2 * np.pi)
    wheel_exact = np.stack(
        [wheel_exact_r, wheel_exact_g, wheel_exact_b], axis=1) / 255.

    # Interpolate the wheel colors with the distance from the center / base color
    angles_colors = wheel_exact
    color_frac = dist / (image_size // 2)
    base_frac = 1 - color_frac

    pixel_colors = base_frac[:, None] * base_color + \
        color_frac[:, None] * angles_colors

    wheel_img = np.zeros((image_size, image_size, 4))
    wheel_img[circle, :3] = pixel_colors
    wheel_img[circle, 3] = 1.
    return wheel_img


def get_wheel_figure(
    size: float = 3,
    marker_size: int = 12,
    axis: bool = True,
    labels: bool = True,
    ticks: bool = True,
    ticks_circle: Optional[Union[VEC_TYPE, _DEFAULT]] = DEFAULT,
    uv_min: Optional[VEC_TYPE] = None,
    uv_max: Optional[VEC_TYPE] = None,
) -> Figure:
    """
    Create a figure with a color wheel in the center.

    Parameters
    ----------
    size : float, optional
        Size of the figure in inches, by default 3

    marker_size : int, optional
        Size of the marker, by default 12

    axis : bool, optional
        Show axis, by default True

    labels : bool, optional
        Show labels, by default True
        These are u and v labels for the respective axis.

    ticks : bool, optional
        Show ticks, by default True

    ticks_circle : Optional[Union[VEC_TYPE, _DEFAULT]], optional
        Render circles at the given ticks, by default DEFAULT
        DEFAULT will render a circle at half the maximum value of the axis.
        If None, no circles will be rendered.
        Shape should be (n, 2) u,v where n is the number of circles and values are in range uv_min and uv_max.

    uv_min : Optional[VEC_TYPE], optional
        Minimum value of the axis, by default None
        If None, it will be set to [-1, -1]
        Will specify the minimum value of the axis. Shape should be (2,) u,v

    uv_max : Optional[VEC_TYPE], optional
        Maximum value of the axis, by default None
        If None, it will be set to [1, 1]

    Returns
    -------
    Figure
        Matplotlib figure with the color wheel in the center.

    """
    from tools.viz.matplotlib import align_marker
    from mpl_toolkits.axisartist.axislines import SubplotZero, AxesZero
    from matplotlib import patches

    if uv_min is None:
        uv_min = np.array([-1., -1.])
    else:
        uv_min = numpyify(uv_min)
    if uv_max is None:
        uv_max = np.array([1., 1.])
    else:
        uv_max = numpyify(uv_max)

    # Check if uv_max and uv_min have the same shape and the center is at 0
    assert uv_min.shape == uv_max.shape, "uv_min and uv_max must have the same shape."
    assert np.all(uv_min + uv_max ==
                  0), "uv_min and uv_max must have the same magnitude but opposite sign."

    fig = plt.figure(figsize=(size, size))
    fig.set_size_inches(
        size,
        size,
        forward=False)
    ax = AxesZero(fig, [0, 0, 1, 1])
    # ax.set_axis_off()
    fig.add_axes(ax)

    # fig.add_subplot(ax)

    pix = round(math.ceil(size * fig.dpi))

    wheel_img = get_wheel_image(pix)

    ax.imshow(wheel_img, extent=(-1., 1., -1., 1.))

    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        # ax.axis[direction].set_axisline_style("-|>")
        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(axis)
        ax.axis[direction].lim = (-1, 1)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    axis_eps = 0
    ax.set_xlim(-1 - axis_eps, 1. + axis_eps)
    ax.set_ylim(-1 - axis_eps, 1. + axis_eps)

    mm_x = MinMax(new_min=uv_min[0], new_max=uv_max[0], axis=0)
    mm_y = MinMax(new_min=uv_min[1], new_max=uv_max[1], axis=0)

    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        norm_x = mm_x.fit_transform(x_ticks)
        norm_y = mm_y.fit_transform(y_ticks)
        x_labels = norm_x[:]
        y_labels = norm_y[:]
        fmtx = destinctive_number_float_format(pd.Series(norm_x))
        fmty = destinctive_number_float_format(pd.Series(norm_y))
        fmt = fmty
        if len(fmtx) > len(fmty):
            fmt = fmtx
        x_labels = [fmt.format(x) for x in x_labels]
        y_labels = [fmt.format(y) for y in y_labels]
        ax.set_xticks(x_ticks, labels=x_labels)
        ax.set_yticks(y_ticks, labels=y_labels)

    if isinstance(ticks_circle, _DEFAULT):
        ticks_circle = np.array([[0.5, 0.5]])
    elif ticks_circle is not None:
        ticks_circle = numpyify(ticks_circle)
        tc_x = mm_x.inverse_transform(ticks_circle[:, 0])
        tc_y = mm_y.inverse_transform(ticks_circle[:, 1])
        ticks_circle = np.stack([tc_x, tc_y], axis=1)

    if labels:
        ax.text(s=r"$\mathcal{u}$", x=0.5, y=-0.15,
                va="top", ha="center", zorder=100)
        ax.text(s=r"$\mathcal{v}$", y=0.5, x=-0.25,
                ha="right", va="center", zorder=100)

    if axis:
        ax.plot((1), (0), ls="", marker=align_marker('>', ha='right'), ms=marker_size, color="k",
                transform=ax.get_yaxis_transform(), clip_on=True)
        ax.plot((0), (1), ls="", marker=align_marker('^', ha='center', va="top"), ms=marker_size, color="k",
                transform=ax.get_xaxis_transform(), clip_on=True)

    # Circle marking end of legend
    circle1 = patches.Circle((0, 0), 1, color='k', fill=False)
    ax.add_patch(circle1)

    if ticks_circle is not None:
        for i in range(ticks_circle.shape[0]):
            circle2 = patches.Ellipse((0, 0),
                                      ticks_circle[i, 0] * 2.,
                                      ticks_circle[i, 1] * 2.,
                                      alpha=0.5,
                                      ls="--",
                                      color='k', fill=False,
                                      zorder=0
                                      )
            ax.add_patch(circle2)

    return fig


def _flow_uv_to_color(flow: VEC_TYPE) -> torch.Tensor:
    """
    Convert a flow image to a color image by sampling the color wheel.

    Parameters
    ----------
    flow : VEC_TYPE
        Flow image of shape ([..., B],2, H, W) if is tensor, or ([..., B], H, W, 2) (u, v) if it is a numpy array.
        Values must be normalized to [-1, 1].

    Returns
    -------
    torch.Tensor
        Color image of shape ([..., B], 3, H, W) where colors where bilinearly sampled from the color wheel.
        Values are in range [0, 1].
    """
    tensorify_image = ToTensorImage(torch.float32)

    flow = tensorify_image(flow)
    wheel_polar = torch.tensor(
        make_colorwheel()).float() / 255  # Shape: (55, 3)
    wheel_polar = wheel_polar.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 55, 3)

    # Shape: (1, 1, 56, 3) # -1 should be same as 1
    wheel_polar = torch.cat([wheel_polar, wheel_polar[:, :, :1, :]], dim=-2)

    # Add white as base color
    # Shape: (1, 2, 56, 3)
    wheel_polar = torch.cat([torch.ones_like(wheel_polar), wheel_polar], dim=1)

    wheel_polar = wheel_polar.flip(-2)

    # Magnitude is in Y of the color wheel
    # Angle is in X of the color wheel

    flow, shape = flatten_batch_dims(flow, -4)

    B, H, W, _ = flow.shape

    if (flow.abs().amax(dim=(-2, -1)) > 1).any():
        raise ValueError("Flow values must be normalized to [-1, 1].")

    flow = flow.permute(0, 2, 3, 1)  # Shape: (B, H, W, 2)
    # Downscale the flow values which have a norm of larger than one so that they are in the range of the color wheel.
    # norm = flow.norm(dim=-1, keepdim=True)
    # norm_flow = torch.where((norm > 1).expand(-1, -1, -1, 2), flow / norm, flow)

    angle = torch.atan2(flow[..., 1], flow[..., 0])  # Shape: (B, H, W)
    magnitude = flow.norm(dim=-1)  # Shape: (B, H, W)

    # Normalize the angle to -1, 1
    angle = angle / math.pi

    # Magnitude is usually in range [0, 1] so we scale it to [-1, 1]
    magnitude = (magnitude - 0.5) * 2

    polar_flow = torch.stack([angle, magnitude], dim=-1)  # Shape: (B, H, W, 2)

    output_values = torch.nn.functional.grid_sample(wheel_polar.permute(
        0, 3, 1, 2), polar_flow, align_corners=True, mode='bilinear', padding_mode='border')
    output_values = unflatten_batch_dims(output_values, shape)
    return output_values


def flow_to_color(flow_uv: VEC_TYPE, uv_max: Optional[VEC_TYPE] = None) -> torch.Tensor:
    """
    Expects a two dimensional flow image of shape.

    Parameters
    ----------
    flow_uv : VEC_TYPE
        Flow image of shape ([..., B], 2, H, W) if is tensor, or ([..., B], H, W, 2) (u, v) if it is a numpy array.

    uv_max : Optional[VEC_TYPE], optional
        Maximum value of the axis, by default None
        Shape should be (2,) u,v. It is recommended to set u == v.
        If None, uv_max will be determined from the flow values, and setting the maximum value to the maximum value
        of the flow disregarding the axis to keep a circular color wheel.

    Returns
    -------
    torch.Tensor
        Flow image converted to a color image.
        Shape: ([..., B], 3, H, W) values are in range [0, 1].
    """
    tensorify_image = ToTensorImage(torch.float32)
    flow_uv = tensorify_image(flow_uv)
    flow_uv, shape = flatten_batch_dims(flow_uv, -4)
    if uv_max is None:
        uv_max = flow_uv.abs().amax(dim=(-4, -3, -2, -1)).unsqueeze(0)
    else:
        uv_max = tensorify(uv_max)
    if uv_max.shape[-1] == 1:
        uv_max = uv_max.expand(2)
    uv_max = uv_max.squeeze()
    if uv_max.shape[-1] != 2:
        raise ValueError("uv_max must have shape (2,).")
    if uv_max.abs().min() == 0:
        raise ValueError("uv_max must have non zero values.")
    B, C, H, W = flow_uv.shape
    norm_flow = flow_uv / uv_max[None, :, None, None].expand(B, -1, H, W)
    color_flow = _flow_uv_to_color(norm_flow)
    color_flow = unflatten_batch_dims(color_flow, shape)
    return color_flow
