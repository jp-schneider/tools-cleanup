import torch

from tools.util.numpy import numpyify
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims
from typing import Tuple
from tools.logger.logging import logger
try:
    from scipy.spatial import Delaunay as DelaunayScipy
except ImportError:
    DelaunayScipy = None
    pass

try:
    from torch_delaunay.functional import shull2d
    shull2d = None
except ImportError:
    shull2d = None
    logger.warning(
        "torch_delaunay is not installed. For best performance, please install it. Falling back to scipy Delaunay.")


@torch.jit.script
def _interpolate_barycentric(corners_X: torch.Tensor, corners_Y: torch.Tensor, corners_V: torch.Tensor, points_X: torch.Tensor, points_Y: torch.Tensor) -> torch.Tensor:
    """Barycentric interpolation for the given coordinates and values.

    Parameters
    ----------
    corners_X : torch.Tensor
        X coordinates of the corners of the simplices. Shape (B, 3)

    corners_Y : torch.Tensor
        Y coordinates of the corners of the simplices. Shape (B, 3)

    corners_V : torch.Tensor
        Values of the corners of the simplices. Shape (B, 3, C)

    Returns
    -------
    torch.Tensor
        Interpolated values for the given coordinates. Shape (B, C)
    """
    x1, y1 = corners_X[:, 0], corners_Y[:, 0]
    x2, y2 = corners_X[:, 1], corners_Y[:, 1]
    x3, y3 = corners_X[:, 2], corners_Y[:, 2]

    lambda1 = ((y2 - y3) * (points_X - x3) + (x3 - x2) * (points_Y - y3)) / \
        ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))

    lambda2 = ((y3 - y1) * (points_X - x3) + (x1 - x3) * (points_Y - y3)) / \
        ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))

    lambda3 = 1 - lambda1 - lambda2

    out = (lambda1.unsqueeze(-1).expand_as(corners_V[:, 0, :]) * corners_V[:, 0, :]
           + lambda2.unsqueeze(-1).expand_as(corners_V[:, 1, :]) * corners_V[:, 1, :]
           + lambda3.unsqueeze(-1).expand_as(corners_V[:, 2, :]) * corners_V[:, 2, :])
    return out


def get_simplices_scipy(
    coords: torch.Tensor,
    value: torch.Tensor,
    points: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets the simplices for the given coordinates and values.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of the points to be interpolated.
        Shape (N, 2) where N is the number of points.

    value : torch.Tensor
        Values of the points to be interpolated.
        Shape (N, C) where N is the number of points and C is the number of channels.

    points : torch.Tensor
        Points where the values are needed. E.g. the sampling points.
        Shape (B, 2) where B is the number of points.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        1. Corners of the simplices. Shape (B, 3, 2)
        2. Color values of the simplices. Shape (B, 3, C)
    """
    tri = DelaunayScipy(numpyify(coords), furthest_site=False)

    N, C = value.shape
    S, _ = tri.simplices.shape
    # Find the corners of each simplice
    corners = coords[tri.simplices, :]
    corners_F = value[tri.simplices.reshape(-1)].reshape(S, 3, C)
    simplice_id = tri.find_simplex(numpyify(points))

    # Find X,Y,F values of the 3 nearest grid points for each
    # pixel in the original grid
    corners_F_pq = corners_F[simplice_id]
    return corners[simplice_id], corners_F_pq


def is_in_triangle(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """
    Checks if the point is in the triangle defined by p0, p1, p2.

    Parameters
    ----------
    points : torch.Tensor
        Input points to check. Shape (B, 2) where B is the number of points.
    triangles : torch.Tensor
        Triangle vertices, Shape (N, 3, 2) where N is the number of total triangles.

    Returns
    -------
    torch.Tensor
        Index tensor of the triangles that contain the points. Shape (B, ) where B is the number of points.
        Yields -1 if the point is not in any triangle.
    """
    B, _ = points.shape
    N, T, _ = triangles.shape
    if T != 3:
        raise ValueError(f"Triangles should have 3 vertices. Got {T} instead.")

    p0 = triangles[:, 0, :]
    p1 = triangles[:, 1, :]
    p2 = triangles[:, 2, :]

    vp0 = points - p0
    v10 = p1 - p0
    cross0 = vp0[:, 0] * v10[:, 1] - vp0[:, 1] * \
        v10[:, 0]  # is pt to left/right of p0->p1

    vp1 = points - p1
    v21 = p2 - p1
    cross1 = vp1[:, 0] * v21[:, 1] - vp1[:, 1] * \
        v21[:, 0]  # is pt to left/right of p1->p2

    vp2 = points - p2
    v02 = p0 - p2
    cross2 = vp2[:, 0] * v02[:, 1] - vp2[:, 1] * \
        v02[:, 0]  # is pt to left/right of p2->p0

    # pt should be to the left of all triangle edges
    is_left(cross0 < 0) & (cross1 < 0) & (cross2 < 0)

    return None


def get_simplices_torch(
    coords: torch.Tensor,
    value: torch.Tensor,
    points: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N, C = value.shape
    simplices = shull2d(coords)

    S, _ = simplices.shape
    # Find the corners of each simplice
    triangles = coords[simplices, :]
    corners_F = value[simplices.reshape(-1)].reshape(S, 3, C)

    simplice_id = is_in_triangle(points, triangles)


class Linear2DInterpolate():

    def __init__(self, height: int, width: int, mode='bilinear') -> None:
        self.height = height
        self.width = width
        self.mode = mode

        self.X, self.Y = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
            indexing='xy')

        self.X = self.X.reshape(-1)
        self.Y = self.Y.reshape(-1)
        if shull2d is not None:
            self.get_simplices = get_simplices_torch
        elif DelaunayScipy is not None:
            self.get_simplices = get_simplices_scipy
        else:
            raise ImportError(
                "Either torch_delaunay or scipy must be installed to use Linear2DInterpolate.")

    def forward(self, coords: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        coords, shp = flatten_batch_dims(coords, -2)
        value, _ = flatten_batch_dims(value, -2)
        N, C = value.shape
        points = torch.stack([self.X, self.Y], dim=-1)
        vertices, corners_F_pq = self.get_simplices(coords, value, points)
        corners_X_pq = vertices[..., 0]
        corners_Y_pq = vertices[..., 1]
        if self.mode == 'bilinear':
            out = _interpolate_barycentric(
                corners_X_pq, corners_Y_pq, corners_F_pq, points[:, 0], points[:, 1])
        else:
            raise ValueError(
                f"Unknown mode {self.mode}. Only bilinear is supported.")
        out = out.reshape(self.height, self.width, C)
        return out
