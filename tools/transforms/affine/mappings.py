# Some of the code is adapted from the following source:
#  RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
"""
Various mappings between different rotation representations.
"""

import torch
from tools.transforms.affine.transforms3d import flatten_batch_dims, unflatten_batch_dims, rotmat_to_unitquat, unitquat_to_rotmat
import numpy as np

try:
    torch.hypot
    hypot = torch.hypot
except AttributeError:
    # torch.hypot is not available in PyTorch 1.6.
    def hypot(x, y):
        return torch.sqrt(torch.square(x) + torch.square(y))


@torch.jit.script
def rotvec_to_unitquat(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation vector into unit quaternion representation.

    Args:
        rotvec (...x3 tensor): batch of rotation vectors.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    """
    rotvec, batch_shape = flatten_batch_dims(rotvec, end_dim=-2)
    num_rotations, D = rotvec.shape
    assert D == 3, "Input should be a Bx3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L621

    norms = torch.norm(rotvec, dim=-1)
    small_angle = (norms <= 1e-3)
    large_angle = ~small_angle

    scale = torch.empty(
        (num_rotations,), device=rotvec.device, dtype=rotvec.dtype)

    # Small Angle Approximation
    scale[small_angle] = (0.5 - norms[small_angle] ** 2 / 48 +
                          norms[small_angle] ** 4 / 3840)
    scale[large_angle] = (torch.sin(norms[large_angle] / 2) /
                          norms[large_angle])

    quat = torch.empty((num_rotations, 4),
                       device=rotvec.device, dtype=rotvec.dtype)
    quat[:, :3] = scale[:, None] * rotvec
    quat[:, 3] = torch.cos(norms / 2)
    return unflatten_batch_dims(quat, batch_shape)


@torch.jit.script
def unitquat_to_rotvec(quat: torch.Tensor, shortest_arc: bool = True):
    """
    Converts unit quaternion into rotation vector representation.

    Based on the representation of a rotation of angle :math:`{\\theta}` and unit axis :math:`(x,y,z)`
    by the unit quaternions :math:`\\pm [\\sin({\\theta} / 2) (x i + y j + z k) + \\cos({\\theta} / 2)]`.

    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
            No normalization is applied before computation.
        shortest_arc (bool): if True, the function returns the smallest rotation vectors corresponding
            to the input 3D rotations, i.e. rotation vectors with a norm smaller than :math:`\\pi`.
            If False, the function may return rotation vectors of norm larger than :math:`\\pi`, depending on the sign of the input quaternions.
    Returns:
        batch of rotation vectors (...x3 tensor).
    Note:
        Behavior is undefined for inputs ``quat=torch.as_tensor([0.0, 0.0, 0.0, -1.0])`` and ``shortest_arc=False``,
        as any rotation vector of angle :math:`2 \\pi` could be a valid representation in such case.
    """
    quat, batch_shape = flatten_batch_dims(quat, end_dim=-2)
    # We perform a copy to support auto-differentiation.
    quat = quat.clone()
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L1006-L1073
    if shortest_arc:
        # Enforce w > 0 to ensure 0 <= angle <= pi.
        # (Otherwise angle can be arbitrary within ]-2pi, 2pi]).
        quat[quat[:, 3] < 0] *= -1
    half_angle = torch.atan2(torch.norm(quat[:, :3], dim=1), quat[:, 3])
    angle = 2 * half_angle
    small_angle = (torch.abs(angle) <= 1e-3)
    large_angle = ~small_angle

    num_rotations = len(quat)
    scale = torch.empty(num_rotations, dtype=quat.dtype, device=quat.device)
    scale[small_angle] = (2 + angle[small_angle] ** 2 / 12 +
                          7 * angle[small_angle] ** 4 / 2880)
    scale[large_angle] = (angle[large_angle] /
                          torch.sin(half_angle[large_angle]))

    rotvec = scale[:, None] * quat[:, :3]
    return unflatten_batch_dims(rotvec, batch_shape)


def axis_angle_to_unitquat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """

    Given a rotation axis and angle, returns the corresponding unit quaternion.

    Parameters
    ----------
    axis : torch.Tensor
        3D rotation axis. Shape ([..., B], 3).
    angle : torch.Tensor
        Rotation angle. Shape ([..., B],).
        Angle should be in radians.

    Returns
    -------
    torch.Tensor
        Unit quaternion. Shape ([..., B], 4).
        Quaternion is in XYZW convention.
    """
    axis, shp = flatten_batch_dims(axis, -2)
    angle, _ = flatten_batch_dims(angle, -1)

    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    sin_half_angle = torch.sin(angle / 2)
    cos_half_angle = torch.cos(angle / 2)
    xyz = axis * sin_half_angle[..., None].expand_as(axis)
    res = torch.cat([xyz, cos_half_angle.unsqueeze(-1)], dim=-1)
    return unflatten_batch_dims(res, shp)


def rotvec_to_rotmat(rotvec: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    """
    Converts rotation vector to rotation matrix representation.
    Conversion uses Rodrigues formula in general, and a first order approximation for small angles.

    Args:
        rotvec (...x3 tensor): batch of rotation vectors.
        epsilon (float): small angle threshold.
    Returns:
        batch of rotation matrices (...x3x3 tensor).
    """
    rotvec, batch_shape = flatten_batch_dims(rotvec, end_dim=-2)
    batch_size, D = rotvec.shape
    assert (D == 3), "Input should be a Bx3 tensor."

    # Rotation angle
    theta = torch.norm(rotvec, dim=-1)
    is_angle_small = theta < epsilon

    # Rodrigues formula for angles that are not small.
    # Note: we use clamping to avoid non finite values for small angles
    # (torch.where produces nan gradients in such case).
    axis = rotvec / torch.clamp_min(theta[..., None], epsilon)
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    one_minus_cos_theta = 1 - cos_theta
    xs = kx*sin_theta
    ys = ky*sin_theta
    zs = kz*sin_theta
    xyc = kx*ky*one_minus_cos_theta
    xzc = kx*kz*one_minus_cos_theta
    yzc = ky*kz*one_minus_cos_theta
    xxc = kx**2*one_minus_cos_theta
    yyc = ky**2*one_minus_cos_theta
    zzc = kz**2*one_minus_cos_theta
    R_rodrigues = torch.stack([1 - yyc - zzc, xyc - zs, xzc + ys,
                               xyc + zs, 1 - xxc - zzc, -xs + yzc,
                               xzc - ys, xs + yzc, 1 - xxc - yyc], dim=-1).reshape(-1, 3, 3)

    # For small angles, use a first order approximation
    xs, ys, zs = rotvec[:, 0], rotvec[:, 1], rotvec[:, 2]
    one = torch.ones_like(xs)
    R_first_order = torch.stack([one, -zs, ys,
                                 zs, one, -xs,
                                 -ys, xs, one], dim=-1).reshape(-1, 3, 3)
    # Select the appropriate expression
    R = torch.where(is_angle_small[:, None, None], R_first_order, R_rodrigues)
    return unflatten_batch_dims(R, batch_shape)


@torch.jit.script
def rotmat_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrix to rotation vector representation.

    Parameters
    ----------
    R : torch.Tensor
        batch of rotation matrices Shape ([..., B], 3, 3).

    Returns
    ------
    torch.Tensor
        Batch of rotation vectors Shape ([..., B], 3).
    """
    q = rotmat_to_unitquat(R)
    return unitquat_to_rotvec(q)


def quat_xyzw_to_wxyz(xyzw):
    """
    Convert quaternion from XYZW to WXYZ convention.

    Args:
        xyzw (...x4 tensor, XYZW convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, WXYZ convention).
    """
    assert xyzw.shape[-1] == 4
    return torch.cat((xyzw[..., -1, None], xyzw[..., :-1]), dim=-1)


def quat_wxyz_to_xyzw(wxyz):
    """
    Convert quaternion from WXYZ to XYZW convention.

    Args:
        wxyz (...x4 tensor, WXYZ convention): batch of quaternions.
    Returns:
        batch of quaternions (...x4 tensor, XYZW convention).
    """
    assert wxyz.shape[-1] == 4
    return torch.cat((wxyz[..., 1:], wxyz[..., 0, None]), dim=-1)


def euler_to_unitquat(convention: str, angles, degrees=False, normalize=True, dtype=None, device=None):
    """
    Convert Euler angles to unit quaternion representation.

    Args:
        convention (string): string defining a sequence of D rotation axes ('XYZ' or 'xzx' for example).
            The sequence of rotation is expressed either with respect to a global 'extrinsic' coordinate system (in which case axes are denoted in lowercase: 'x', 'y', or 'z'),
            or with respect to an 'intrinsic' coordinates system attached to the object under rotation (in which case axes are denoted in uppercase: 'X', 'Y', 'Z').
            Intrinsic and extrinsic conventions cannot be mixed.
        angles (...xD tensor, or tuple/list of D floats or ... tensors): a list of angles associated to each axis, expressed in radians by default.
        degrees (bool): if True, input angles are assumed to be expressed in degrees.
        normalize (bool): if True, normalize the returned quaternion to compensate potential numerical.

    Returns:
        A batch of unit quaternions (...x4 tensor, XYZW convention).

    Warning:
        Case is important: 'xyz' and 'XYZ' denote different conventions.
    """
    from tools.transforms.affine.quaternion import quat_composition
    if type(angles) == torch.Tensor:
        angles = [t.squeeze(dim=-1) for t in torch.split(angles,
                                                         split_size_or_sections=1, dim=-1)]

    assert len(convention) == len(angles)

    extrinsics = convention.islower()
    if extrinsics:
        # Cast from intrinsics to extrinsics convention
        convention = convention.upper()[::-1]
        angles = angles[::-1]

    unitquats = []
    for axis, angle in zip(convention, angles):
        angle = torch.as_tensor(angle, device=device, dtype=dtype)
        if degrees:
            angle = torch.deg2rad(angle)
        batch_shape = angle.shape
        rotvec = torch.zeros(batch_shape + torch.Size((3,)),
                             device=angle.device, dtype=angle.dtype)
        if axis == 'X':
            rotvec[..., 0] = angle
        elif axis == 'Y':
            rotvec[..., 1] = angle
        elif axis == 'Z':
            rotvec[..., 2] = angle
        else:
            raise ValueError(
                "Invalid convention (expected format: 'xyz', 'zxz', 'XYZ', etc.).")
        q = rotvec_to_unitquat(rotvec)
        unitquats.append(q)
    if len(unitquats) == 1:
        return unitquats[0]
    else:
        return quat_composition(unitquats, normalize=normalize)


def _elementary_basis_index(axis):
    """
    Return the index corresponding to a given axis label.
    """
    if axis == 'x':
        return 0
    elif axis == 'y':
        return 1
    elif axis == 'z':
        return 2
    else:
        raise ValueError("Invalid axis.")


def unitquat_to_euler(convention: str, quat: torch.Tensor, degrees: bool = False, epsilon: torch.Tensor = 1e-7):
    """
    Convert unit quaternion to Euler angles representation.

    Args:
        convention (str): string of 3 characters belonging to {'x', 'y', 'z'} for extrinsic rotations, or {'X', 'Y', 'Z'} for intrinsic rotations.
            Consecutive axes should not be identical.
            Extrinsic and intrinsic conventions cannot be mixed.
            Extrinsic convention will return angles in the order of the axes, intrinsic convention will return angles in reverse order.
        quat (...x4 tensor, XYZW convention): input batch of unit quaternion.
        degrees (bool): if True, angles are returned in degrees.
        epsilon (float): a small value used to detect degenerate configurations.

    Returns:
        A stacked ...x3 tensor corresponding to Euler angles, expressed by default in radians.
        In case of gimbal lock, the third angle is arbitrarily set to 0.
    """
    # Code adapted from scipy.spatial.transform.Rotation.
    # Reference: https://github.com/scipy/scipy/blob/ac6bcaf00411286271f7cc21e495192c73168ae4/scipy/spatial/transform/_rotation.pyx#L325C12-L325C15
    assert len(convention) == 3

    pi = np.pi
    lamb = np.pi/2

    extrinsic = convention.islower()
    if not extrinsic:
        convention = convention.lower()[::-1]

    quat, batch_shape = flatten_batch_dims(quat, end_dim=-2)
    N = quat.shape[0]

    i = _elementary_basis_index(convention[0])
    j = _elementary_basis_index(convention[1])
    k = _elementary_basis_index(convention[2])
    assert i != j and j != k, "Consecutive axes should not be identical."

    symmetric = (i == k)

    if symmetric:
        # Get third axis
        k = 3 - i - j

    # Step 0
    # Check if permutation is even (+1) or odd (-1)
    sign = (i - j) * (j - k) * (k - i) // 2

    # Step 1
    # Permutate quaternion elements
    if symmetric:
        a = quat[:, 3]
        b = quat[:, i]
        c = quat[:, j]
        d = quat[:, k] * sign
    else:
        a = quat[:, 3] - quat[:, j]
        b = quat[:, i] + quat[:, k] * sign
        c = quat[:, j] + quat[:, 3]
        d = quat[:, k] * sign - quat[:, i]

    # intrinsic/extrinsic conversion helpers
    if extrinsic:
        angle_first = 0
        angle_third = 2
    else:
        angle_first = 2
        angle_third = 0

    # Step 2
    # Compute second angle...
    angles = [torch.empty(N, device=quat.device, dtype=quat.dtype)
              for _ in range(3)]

    angles[1] = 2 * torch.atan2(hypot(c, d),
                                hypot(a, b))

    # ... and check if equal to is 0 or pi, causing a singularity
    case1 = torch.abs(angles[1]) <= epsilon
    case2 = torch.abs(angles[1] - pi) <= epsilon
    case1or2 = torch.logical_or(case1, case2)
    # Step 3
    # compute first and third angles, according to case
    half_sum = torch.atan2(b, a)
    half_diff = torch.atan2(d, c)

    # no singularities
    angles[angle_first] = half_sum - half_diff
    angles[angle_third] = half_sum + half_diff

    # any degenerate case
    angles[2][case1or2] = 0
    angles[0][case1] = 2 * half_sum[case1]
    angles[0][case2] = 2 * (-1 if extrinsic else 1) * half_diff[case2]

    # for Tait-Bryan/asymmetric sequences
    if not symmetric:
        angles[angle_third] *= sign
        angles[1] -= lamb

    for idx in range(3):
        foo = angles[idx]
        foo[foo < -pi] += 2 * pi
        foo[foo > pi] -= 2 * pi
        if degrees:
            foo = torch.rad2deg(foo)
        angles[idx] = unflatten_batch_dims(foo, batch_shape)

    return torch.stack(angles, dim=-1)
