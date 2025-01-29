import torch
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims


def are_points_on_lines(
        source_points: torch.Tensor, 
        directions: torch.Tensor, 
        points: torch.Tensor, 
        atol: float = 1e-6):
    """
    Checks if a point is on a line defined by a source point and a direction vector.
    
    E.g. if the point is on the line, the vector from the source point to the point should be parallel to the direction vector.

    Parameters
    ----------
    source_points : torch.Tensor
        The source points of the lines. Shape: ([...B], 3)

    directions : torch.Tensor
        The direction vectors of the lines. Shape: ([...B], 3)

    points : torch.Tensor
        The points to check. Shape: ([...B], 3)

    atol : float, optional
        The absolute tolerance for the comparison. Default is 1e-6.
    
    Returns
    -------
    torch.Tensor
        A boolean tensor indicating if the points are on the lines. Shape: ([...B],)
    
    """

    source_points, shp = flatten_batch_dims(source_points, -2)
    directions, _ = flatten_batch_dims(directions, -2)
    points, _ = flatten_batch_dims(points, -2)

    # Check if shapes match
    assert source_points.shape == directions.shape
    assert source_points.shape == points.shape

    # Calculate the vector from the source point to the point
    vector_to_point = points - source_points

    # Calculate the cross product of the direction vector and the vector to the point
    cross_product = torch.cross(directions, vector_to_point)

    # Check if the cross product is close to zero (within a tolerance)
    return unflatten_batch_dims(torch.isclose(cross_product, torch.zeros_like(cross_product), atol=atol).all(dim=-1), shp)