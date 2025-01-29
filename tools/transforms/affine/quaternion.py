
from typing import List, Union
import torch

from tools.transforms.affine.mappings import rotvec_to_unitquat, unitquat_to_rotvec
from tools.transforms.affine.transforms3d import flatten_batch_dims, unflatten_batch_dims


@torch.jit.script
def quat_conjugation(quat: torch.Tensor) -> torch.Tensor:
    """
    Returns the conjugation of input batch of quaternions.

    Args:
        quat (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    Note:
        Conjugation of a unit quaternion is equal to its inverse.
    """
    inv = quat.clone()
    t = torch.tensor(-1, dtype=quat.dtype, device=quat.device)
    inv[..., :3] *= t
    return inv


@torch.jit.script
def quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    """
    Returns the inverse of a batch of quaternions.

    Args:
        quat (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    Note:
        - Inverse of null quaternion is undefined.
        - For unit quaternions, consider using conjugation instead.
    """
    return quat_conjugation(quat) / torch.sum(quat**2, dim=-1, keepdim=True)


@torch.jit.script
def quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    """
    Returns a normalized, unit norm, copy of a batch of quaternions.

    Args:
        quat (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    """
    return quat / torch.norm(quat, dim=-1, keepdim=True)


@torch.jit.script
def quat_abs(quat: torch.Tensor) -> torch.Tensor:
    """
    Returns the absolute value of a batch of quaternions.

    This will convert the quaternion to a positive scalar part.

    Parameters
    ----------
    quat : torch.Tensor
        The input batch of quaternions. Shape should be (..., 4). XYZW convention.

    Returns
    -------
    torch.Tensor
        The absolute value of the input quaternions. Shape is (..., 4).
    """
    quat, shape = flatten_batch_dims(quat, -2)
    quat = quat.clone()
    t = torch.tensor(-1, dtype=quat.dtype, device=quat.device)
    neg = quat[:, 3] < 0.
    quat[neg] *= t
    return unflatten_batch_dims(quat, shape)

# @torch.jit.script
# def quat_product_scalar(quat: torch.Tensor, s: Union[torch.Tensor, float, int], normalized: bool = True) -> torch.Tensor:
#     """
#     Returns the product of a quaternion and a scalar.
#     Old Implementation, approx 10 % slower than the new one
#     Parameters
#     ----------
#     quat : torch.Tensor
#         The input batch of quaternions. Shape should be (..., 4). XYZW convention.

#     s : torch.Tensor
#         The input batch of scalars. Shape should be (...,).

#     Returns
#     ---------
#     torch.Tensor
#         The product of the quaternion and the scalar. Shape is (..., 4).
#     """
#     if isinstance(s, float):
#         s = torch.tensor(s, dtype=quat.dtype, device=quat.device)
#     elif isinstance(s, int):
#         s = torch.tensor(s, dtype=quat.dtype, device=quat.device)
#     if normalized:
#         # Convert to rotvec
#         rvec = unitquat_to_rotvec(quat, shortest_arc=True)
#         # Scale
#         rvec *= s.unsqueeze(-1)
#         # Convert back to quaternion
#         return rotvec_to_unitquat(rvec)
#     else:
#         return quat * s.unsqueeze(-1)


@torch.jit.script
def quat_product_scalar(quat: torch.Tensor, s: Union[torch.Tensor, float, int], normalized: bool = True) -> torch.Tensor:
    """Multiply a quaternion with a scalar.

    Parameters
    ----------
    q : torch.Tensor
        Quaternion to multiply.
    s : torch.Tensor
        Scalar to multiply.

    normalized : bool
        If True, the quaternion output will be normalized.
        If False, the output will not be normalized, this is much faster.

    Returns
    -------
    torch.Tensor
        Result of the multiplication.
    """
    if isinstance(s, float):
        s = torch.tensor(s, dtype=quat.dtype, device=quat.device)
    elif isinstance(s, int):
        s = torch.tensor(s, dtype=quat.dtype, device=quat.device)
    elif isinstance(s, torch.Tensor):
        s = s.to(quat.dtype)
    if normalized:
        quat = quat_abs(quat)
        quat, shape = flatten_batch_dims(quat, -2)
        # q = quat_abs(q)
        w = quat[..., 3:4]
        xyz = quat[..., :3]

        theta = 2 * torch.atan2(torch.norm(xyz, dim=-1, keepdim=True), w)

        zero_theta = (theta == 0).squeeze(-1)

        omega = torch.zeros_like(xyz)
        omega[~zero_theta] = xyz[~zero_theta] / \
            torch.sin(theta[~zero_theta] / 2)

        theta_ = s * theta
        xyz_ = omega * torch.sin(theta_ / 2)
        w_ = torch.cos(theta_ / 2)
        res = torch.cat([xyz_, w_], dim=-1)

        return unflatten_batch_dims(res, shape)
    else:
        return quat * s.unsqueeze(-1)


@torch.jit.script
def quat_product(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Returns the product of two quaternions.

    Args:
        p, q (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    """
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L153
    # batch_shape = p.shape[:-1]
    # assert q.shape[:-1] == batch_shape, "Incompatible shapes"
    # p = p.reshape(-1, 4)
    # q = q.reshape(-1, 4)
    # product = torch.empty_like(q)
    # product[..., 3] = p[..., 3] * q[..., 3] - torch.sum(p[..., :3] * q[..., :3], axis=-1)
    # product[..., :3] = (p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] +
    #                   torch.cross(p[..., :3], q[..., :3], dim=-1))
    if p.dtype != q.dtype:
        raise ValueError("Incompatible dtypes for p {} and q {}".format(p, q))
    vector = (p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] +
              torch.cross(p[..., :3], q[..., :3], dim=-1))
    last = p[..., 3] * q[..., 3] - torch.sum(p[..., :3] * q[..., :3], dim=-1)
    return torch.cat([vector, last[..., None]], dim=-1)


@torch.jit.script
def quat_subtraction(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Returns the difference between two quaternions.

    Parameters
    ----------
    q1 : torch.Tensor
        The first input batch of quaternions. Shape should be (..., 4). XYZW convention.

    q2 : torch.Tensor
        The second input batch of quaternions. Shape should be (..., 4). XYZW convention.

    Returns
    -------
    torch.Tensor
        The difference between the two input quaternions. Shape is (..., 4).
    """
    inv_q2 = quat_inverse(q2)
    if q1.dtype != inv_q2.dtype:
        raise ValueError(
            "Incompatible dtypes for q1 {}, q2 {}, qinv {}".format(q1, q2, inv_q2))
    return quat_product(q1, inv_q2)


@torch.jit.script
def quat_composition(quat: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Returns the product of a batched of quaternions.
    Sequence S are composed from left to right (q1 * q2 * q3 * ... * qS).

    Parameters
    ----------

    batched_quaternions : torch.Tensor
        Batch of quaternions to multiply. Shape is ([..., B], S, 4) where the last dimension is the quaternion in (x, y, z, w) convention.

    normalize : bool
        Whether to normalize the output quaternion.

    Returns
    -------
    torch.Tensor
        Batch of quaternions resulting from the composition.
        Output shape is ([..., B], 4) where the last dimension is the quaternion in (x, y, z, w) convention.

    """
    quat, batch_shape = flatten_batch_dims(quat, -3)
    B, S, C = quat.shape

    res = torch.zeros((B, 4), dtype=quat.dtype, device=quat.device)
    res[..., :] = quat[..., 0, :]

    for i in range(1, S):
        res = quat_product(res, quat[:, i])
    if normalize:
        res = quat_normalize(res)

    return unflatten_batch_dims(res, batch_shape)


@torch.jit.script
def quat_average(quats: torch.Tensor) -> torch.Tensor:
    """
    Averages multiple unit quaternions using weighted averaging.

    Parameters:
    quats: torch.Tensor ([..., B], N, 4) - Batch of unit quaternions.

    Returns:
    torch.Tensor ([..., B], 4) - Averaged unit quaternion.

    """
    quats, shape = flatten_batch_dims(quats, -3)

    # Calculate the weight for each quaternion
    weights = 1.0 / quats.shape[-2]

    # Scale each quaternion by its weight
    weighted_quats = quat_product_scalar(quats, weights)

    # Compose the weighted quaternions
    average = quat_composition(weighted_quats, normalize=True)

    return unflatten_batch_dims(average, shape)


@torch.jit.script
def quat_action(quat: torch.Tensor, vec: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
    """
    Rotate a 3D vector :math:`v=(x,y,z)` by a rotation represented by a quaternion `q`.

    Based on the action by conjugation :math:`q,v : q v q^{-1}`, considering the pure quaternion :math:`v=xi + yj +zk` by abuse of notation.

    Parameters
    --------
    quat : torch.Tensor
        batch of quaternions (..., 4) tensor, XYZW convention).

    vec : torch.Tensor
        batch of 3D vectors (..., 3) tensor.

    Returns
    --------
        batch of rotated 3D vectors (..., 3) tensor.

    Note
    --------
        One should favor rotation matrix representation to rotate multiple vectors by the same rotation efficiently.
    """
    batch_shape = vec.shape[:-1]
    iquat = quat_conjugation(quat) if is_normalized else quat_inverse(quat)
    pure = torch.cat(
        (vec, torch.zeros(batch_shape + (1,), dtype=quat.dtype, device=quat.device)), dim=-1)
    res = quat_product(quat, quat_product(pure, iquat))
    return res[..., :3]
