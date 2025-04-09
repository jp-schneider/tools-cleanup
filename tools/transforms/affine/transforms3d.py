import torch
from typing import List, Literal, Optional, Tuple
from tools.util.torch import as_tensors, tensorify
import numpy as np

import numpy as np
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
import torch.nn.functional as F
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims

__all__ = [
    "assure_affine_vector",
    "assure_affine_matrix",
    "unit_vector",
    "component_rotation_matrix",
    "component_position_matrix",
    "component_transformation_matrix",
    "transformation_matrix",
    "scale_matrix",
    "split_transformation_matrix",
    "vector_angle"
]


def assure_affine_vector(_input: VEC_TYPE,
                         dtype: Optional[torch.dtype] = None,
                         device: Optional[torch.device] = None,
                         requires_grad: bool = False) -> torch.Tensor:
    """Assuring the _input vector instance is an affine vector.
    Converting it into tensor if nessesary.
    Adds 1 to the vector if its size is 3.

    Parameters
    ----------
    _input : Union[torch.Tensor, np.ndarray]
        Vector of length 3 or 4.
    dtype : Optional[torch.dtype], optional
        The dtype of the tensor, by default None
    device : Optional[torch.device], optional
        Its device, by default None
    requires_grad : bool, optional
        Whether it requires grad and the input was numpy array, by default False

    Returns
    -------
    torch.Tensor
        The affine tensor.

    Raises
    ------
    ValueError
        If shape is wrong.
    """
    batched = False
    _input = tensorify(_input, dtype=dtype, device=device,
                       requires_grad=requires_grad)
    if len(_input.shape) == 2:
        batched = True
    if _input.shape[-1] > 4 or _input.shape[-1] < 3:
        raise ValueError(
            f"assure_affine_vector works only for tensors of length 3 or 4.")
    if _input.shape[-1] == 4:
        return _input  # Assuming it contains already affine property
    else:
        # Length of 3
        if not batched:
            return torch.cat([_input, torch.tensor([1], device=_input.device,
                                                   dtype=_input.dtype,
                                                   requires_grad=_input.requires_grad)])
        else:
            return torch.cat([_input, torch.ones(_input.shape[:-1] + (1,),
                                                 device=_input.device, dtype=_input.dtype,
                                                 requires_grad=_input.requires_grad)], axis=-1)


def assure_affine_matrix(_input: VEC_TYPE,
                         dtype: Optional[torch.dtype] = None,
                         device: Optional[torch.device] = None,
                         requires_grad: bool = False) -> torch.Tensor:
    """Assuring the _input matrix instance is an affine matrix.
    Converting it into tensor if nessesary.
    Adds 1 to the vector if its size is 3.

    Parameters
    ----------
    _input : Union[torch.Tensor, np.ndarray]
        Matrix of x / y shape 3 or 4.
    dtype : Optional[torch.dtype], optional
        The dtype of the tensor, by default None
    device : Optional[torch.device], optional
        Its device, by default None
    requires_grad : bool, optional
        Whether it requires grad and the input was numpy array, by default False

    Returns
    -------
    torch.Tensor
        The affine tensor.

    Raises
    ------
    ValueError
        If shape is wrong.
    """
    _input = tensorify(_input, dtype=dtype, device=device,
                       requires_grad=requires_grad)
    if len(_input.shape) < 2:
        raise ValueError(
            f"assure_affine_matrix works only for dimensions (..., 3, 3) or (..., 4, 4).")
    if _input.shape[-2] > 4 or _input.shape[-2] < 3:
        raise ValueError(
            f"assure_affine_matrix works only for tensors of length 3 or 4.")
    if _input.shape[-2] == 4:
        pass
    else:
        mvec = torch.tensor(
            [[0., 0., 0.] + ([] if _input.shape[-1] == 3 else [1.])],
            device=_input.device, dtype=_input.dtype, requires_grad=_input.requires_grad)
        mvec = mvec.repeat(*_input.shape[:-2], 1, 1)
        _input = torch.cat(
            [_input, mvec],
            axis=-2)
    if _input.shape[-1] > 4 or _input.shape[-1] < 3:
        raise ValueError(
            f"assure_affine_matrix works only for tensors of length 3 or 4.")
    if _input.shape[-1] == 4:
        pass
    else:
        # Length of 3
        mvec = torch.tensor(
            [[0., 0., 0., 1.]], device=_input.device, dtype=_input.dtype, requires_grad=_input.requires_grad).T
        mvec = mvec.repeat(*_input.shape[:-2], 1, 1)
        _input = torch.cat([_input, mvec], axis=-1)
    return _input


@as_tensors()
def affine_rotation_matrix(orientation: VEC_TYPE) -> VEC_TYPE:
    affine_mat = torch.eye(4, dtype=orientation.dtype, device=orientation.device).repeat(
        *orientation.shape[:-2], 1, 1)
    affine_mat[:, :3, :3] = orientation
    return affine_mat


def is_transformation_matrix(_input: VEC_TYPE) -> torch.Tensor:
    """Returns whether a given input is a numpy or torch
    matrix of size (3 x 3) / (4 x 4) which can be used as transformation matricies.

    Parameters
    ----------
    _input : VEC_TYPE
        The input to check.

    Returns
    -------
    torch.Tensor
        Whether the input is a transformation matrix.
    """
    if _input is None:
        return False
    if isinstance(_input, (torch.Tensor, np.ndarray)):
        if tuple(_input.shape) in [(3, 3), (4, 4)]:
            return True
    return False


def is_position_vector(_input: VEC_TYPE) -> torch.Tensor:
    """Returns whether a given input is a numpy or torch
    matrix of size (3, ) / (4, ) which can be used as position vector.

    Parameters
    ----------
    _input : VEC_TYPE
        The input to check.

    Returns
    -------
    torch.Tensor
        Whether the input is a transformation vector.
    """
    if _input is None:
        return False
    if isinstance(_input, (torch.Tensor, np.ndarray)):
        if tuple(_input.shape) in [(3,), (4, )]:
            return True
    return False


@torch.jit.script
def _split_transformation_matrix(_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits a transformation matrix in its position and orientation component.

    Parameters
    ----------
    _input : torch.Tensor
        The input matrix

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The position (3, ) and orientation matrix (3, 3).

    Raises
    ------
    ValueError
        If input shape is of invalid shape.
    """
    position = _input[..., :3, 3]
    orientation = _input[..., :3, :3]
    return position, orientation


@torch.jit.script
def _compose_transformation_matrix(position: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
    """Composes a transformation matrix out of position and orientation.

    Parameters
    ----------
    position : torch.Tensor
        The position vector.
    orientation : torch.Tensor
        The orientation matrix.

    Returns
    -------
    torch.Tensor
        The composed transformation matrix.
    """
    shp = position.shape[:-1]
    if len(shp) == 0:
        mat = torch.eye(4, dtype=orientation.dtype, device=orientation.device)
    else:
        mat = torch.eye(4, dtype=orientation.dtype,
                        device=orientation.device).repeat(shp[0], 1, 1)
    mat[..., :3, :3] = orientation
    mat[..., :3, 3] = position[..., :3]
    return mat


@as_tensors()
def compose_transformation_matrix(position: Optional[VEC_TYPE] = None, orientation: Optional[VEC_TYPE] = None) -> VEC_TYPE:
    """Composes a transformation matrix out of position and orientation.

    Parameters
    ----------
    position : VEC_TYPE
        The position vector.
    orientation : VEC_TYPE
        The orientation matrix.

    Returns
    -------
    VEC_TYPE
        The composed transformation matrix.
    """
    if position is None and orientation is None:
        raise ValueError("At least one of the inputs must be provided.")
    if position is None:
        position = torch.zeros(orientation.shape[:-2] + (3, ),
                               dtype=orientation.dtype,
                               device=orientation.device
                               )
    if orientation is None:
        orientation = torch.eye(3, dtype=position.dtype,
                                device=position.device).repeat(
            position.shape[:-1] + (1, 1)
        )
    return _compose_transformation_matrix(position, orientation)


def split_transformation_matrix(_input: VEC_TYPE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits a transformation matrix in its position and orientation component.

    Parameters
    ----------
    _input : VEC_TYPE
        The input matrix

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The position (3, ) and orientation matrix (3, 3).

    Raises
    ------
    ValueError
        If input shape is of invalid shape.
    """
    if tuple(_input.shape[-2:]) != (4, 4):
        raise ValueError(f"Invalid shape for split: {_input.shape}")
    position = _input[..., :3, 3]
    orientation = _input[..., :3, :3]
    return position, orientation


def unit_vector(_input: torch.Tensor) -> torch.Tensor:
    """Calculates a unit vector out of the input.

    Parameters
    ----------
    _input : torch.Tensor
        The input.

    Returns
    -------
    torch.Tensor
        The normed input vector (unit-vector).
    """
    return _input / torch.norm(_input)


def component_rotation_matrix(angle_x: Optional[NUMERICAL_TYPE] = None,
                              angle_y: Optional[NUMERICAL_TYPE] = None,
                              angle_z: Optional[NUMERICAL_TYPE] = None,
                              mode: Literal["deg", "rad"] = 'rad',
                              dtype: torch.dtype = None,
                              device: torch.device = None,
                              requires_grad: bool = False) -> torch.Tensor:
    """Computes the rotation matrix out of angles.

    Parameters
    ----------
    angle_x : Optional[NUMERICAL_TYPE], optional
        Rotation angle around the X-axis, also known as roll, by default None
    angle_y : Optional[NUMERICAL_TYPE], optional
        Rotation angle around the Y-axis, also known as yaw, by default None
    angle_z : Optional[NUMERICAL_TYPE], optional
        Rotation angle around the Z-axis, also known as pitch, by default None
    mode : Literal[&quot;deg&quot;, &quot;rad&quot;], optional
        If the angles are specified in radians [0, 2*pi), or degrees [0, 360), by default 'rad'
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default None
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device.
    Returns
    -------
    torch.Tensor
        The rotation matrix of shape (3, 3)
    """
    angle_x = tensorify(
        angle_x, dtype=dtype, device=device, requires_grad=requires_grad) if angle_x is not None else torch.tensor(
        0, dtype=dtype, device=device, requires_grad=requires_grad)
    angle_y = tensorify(
        angle_y, dtype=dtype, device=device, requires_grad=requires_grad) if angle_y is not None else torch.tensor(
        0, dtype=dtype, device=device, requires_grad=requires_grad)
    angle_z = tensorify(
        angle_z, dtype=dtype, device=device, requires_grad=requires_grad) if angle_z is not None else torch.tensor(
        0, dtype=dtype, device=device, requires_grad=requires_grad)

    if mode == "deg":
        angle_x = torch.deg2rad(angle_x)
        angle_y = torch.deg2rad(angle_y)
        angle_z = torch.deg2rad(angle_z)

    rot = torch.tensor(np.identity(4), dtype=dtype,
                       device=device, requires_grad=requires_grad)
    if dtype is None:
        rot = rot.to(dtype=torch.float32)  # Default dtype for torch.tensor
    if angle_x != 0.:
        r_x = torch.zeros((4, 4), dtype=rot.dtype,
                          device=rot.device, requires_grad=rot.requires_grad)
        r_x[0, 0] = 1
        r_x[1, 1] = torch.cos(angle_x)
        r_x[1, 2] = -torch.sin(angle_x)
        r_x[2, 1] = torch.sin(angle_x)
        r_x[2, 2] = torch.cos(angle_x)
        r_x[3, 3] = 1
        rot @= r_x
    if angle_y != 0.:
        r_y = torch.zeros((4, 4), dtype=rot.dtype,
                          device=rot.device, requires_grad=rot.requires_grad)
        r_y[0, 0] = torch.cos(angle_y)
        r_y[0, 2] = torch.sin(angle_y)
        r_y[1, 1] = 1
        r_y[2, 0] = -torch.sin(angle_y)
        r_y[2, 2] = torch.cos(angle_y)
        r_y[3, 3] = 1
        rot @= r_y
    if angle_z != 0.:
        r_z = torch.zeros((4, 4), dtype=rot.dtype,
                          device=rot.device, requires_grad=rot.requires_grad)
        r_z[0, 0] = torch.cos(angle_z)
        r_z[0, 1] = -torch.sin(angle_z)
        r_z[1, 0] = torch.sin(angle_z)
        r_z[1, 1] = torch.cos(angle_z)
        r_z[2, 2] = 1
        r_z[3, 3] = 1
        rot @= r_z
    return rot


@torch.jit.script
def _euler_angles_to_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:

    x_rot = torch.eye(3, dtype=angles.dtype, device=angles.device).repeat(
        angles.shape[0], 1, 1)
    y_rot = torch.eye(3, dtype=angles.dtype, device=angles.device).repeat(
        angles.shape[0], 1, 1)
    z_rot = torch.eye(3, dtype=angles.dtype, device=angles.device).repeat(
        angles.shape[0], 1, 1)

    x_rot[:, 1, 1] = torch.cos(angles[:, 0])
    x_rot[:, 1, 2] = -torch.sin(angles[:, 0])
    x_rot[:, 2, 1] = torch.sin(angles[:, 0])
    x_rot[:, 2, 2] = torch.cos(angles[:, 0])

    y_rot[:, 0, 0] = torch.cos(angles[:, 1])
    y_rot[:, 0, 2] = torch.sin(angles[:, 1])
    y_rot[:, 2, 0] = -torch.sin(angles[:, 1])
    y_rot[:, 2, 2] = torch.cos(angles[:, 1])

    z_rot[:, 0, 0] = torch.cos(angles[:, 2])
    z_rot[:, 0, 1] = -torch.sin(angles[:, 2])
    z_rot[:, 1, 0] = torch.sin(angles[:, 2])
    z_rot[:, 1, 1] = torch.cos(angles[:, 2])

    return torch.bmm(torch.bmm(x_rot, y_rot), z_rot)


@as_tensors()
def euler_angles_to_rotation_matrix(angles: VEC_TYPE, mode: Literal["deg", "rad"] = 'rad') -> VEC_TYPE:
    """Computes the rotation matrix out of euler angles.

    Parameters
    ----------
    angles : VEC_TYPE
        The euler angles in shape (..., 3).

    mode : Literal[&quot;deg&quot;, &quot;rad&quot;], optional
        If the angles are specified in radians [0, 2*pi), or degrees [0, 360), by default 'rad'

    Returns
    -------
    VEC_TYPE
        The rotation matrix of shape (..., 3, 3)
    """
    if len(angles.shape) == 1:
        angles = angles.unsqueeze(0)
    if len(angles.shape) != 2 or angles.shape[-1] != 3:
        raise ValueError("Input must be of shape (..., 3)")
    if mode == "deg":
        angles = torch.deg2rad(angles)
    return _euler_angles_to_rotation_matrix(angles)


def component_position_matrix(x: Optional[NUMERICAL_TYPE] = None,
                              y: Optional[NUMERICAL_TYPE] = None,
                              z: Optional[NUMERICAL_TYPE] = None,
                              angle_x: Optional[NUMERICAL_TYPE] = None,
                              angle_y: Optional[NUMERICAL_TYPE] = None,
                              angle_z: Optional[NUMERICAL_TYPE] = None,
                              mode: Literal["deg", "rad"] = 'rad',
                              dtype: torch.dtype = None,
                              device: torch.device = None,
                              requires_grad: bool = False) -> torch.Tensor:
    """Creates a position matrix based on individual components.

    Parameters
    ----------
    x : Optional[NUMERICAL_TYPE], optional
        The x-coordinate of the matrix, by default None
    y : Optional[NUMERICAL_TYPE], optional
        The y-coordinate of the matrix, by default None
    z : Optional[NUMERICAL_TYPE], optional
        The z-coordinate of the matrix, by default None
    angle_x : Optional[NUMERICAL_TYPE], optional
        The angle to rotate around x-axis, by default None
    angle_y : Optional[NUMERICAL_TYPE], optional
        The angle to rotate around y-axis, by default None
    angle_z : Optional[NUMERICAL_TYPE], optional
        The angle to rotate around z-axis, by default None
    mode : Literal[&quot;deg&quot;, &quot;rad&quot;], optional
        The unit of angles, by default 'rad'
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default torch.float64
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device.
    requires_grad: bool, optional
        Whether initialized tensors will require gradient backpropagation, by default False

    Returns
    -------
    torch.Tensor
        The 4 x 4 affine position matrix.
    """
    position = component_transformation_matrix(
        x, y, z, dtype=dtype, device=device, requires_grad=requires_grad)
    rot = component_rotation_matrix(angle_x, angle_y, angle_z, mode=mode, dtype=dtype,
                                    device=device, requires_grad=requires_grad)
    return position @ rot


def component_transformation_matrix(x: Optional[NUMERICAL_TYPE] = None,
                                    y: Optional[NUMERICAL_TYPE] = None,
                                    z: Optional[NUMERICAL_TYPE] = None,
                                    dtype: torch.dtype = None,
                                    device: torch.device = None,
                                    requires_grad: bool = False) -> torch.Tensor:
    """Returng a transformation matrix based on given components.

    Parameters
    ----------
    x : Optional[NUMERICAL_TYPE], optional
        X component for the transformation, by default None
    y : Optional[NUMERICAL_TYPE], optional
        Y component for the transformation, by default None
    z : Optional[NUMERICAL_TYPE], optional
        Z component for the transformation, by default None
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default torch.float64
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device, by default "cpu"
    requires_grad: bool, optional
        Whether initialized tensors will require gradient backpropagation, by default False
    Returns
    -------
    torch.Tensor
        Transformation matrix.
    """
    mat = torch.tensor(np.identity(4), dtype=dtype,
                       device=device, requires_grad=requires_grad)
    if dtype is None:
        mat = mat.to(dtype=torch.float32)  # Default dtype for torch.tensor
    mat[0, 3] = x if x is not None else 0.
    mat[1, 3] = y if y is not None else 0.
    mat[2, 3] = z if z is not None else 0.
    return mat


def transformation_matrix(vector: VEC_TYPE,
                          dtype: torch.dtype = None,
                          device: torch.device = None,
                          requires_grad: bool = False) -> torch.Tensor:
    """Getting the transformation matrix from a vector.

    Parameters
    ----------
    vector : VEC_TYPE
        The vector to transform for.
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default torch.float64
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device, by default "cpu"
    requires_grad: bool, optional
        Whether initialized tensors will require gradient backpropagation, by default False

    Returns
    -------
    torch.Tensor
        The resulting transformation matrix.
    """
    vector = tensorify(vector, dtype=dtype, device=device,
                       requires_grad=requires_grad)
    mat = torch.tensor(np.identity(4), dtype=dtype,
                       device=device, requires_grad=requires_grad)
    if dtype is None:
        mat = mat.to(dtype=torch.float32)  # Default dtype for torch.tensor
    mat[0:3, 3] = vector[0:3]
    return mat


def scale_matrix(vector: VEC_TYPE,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 requires_grad: bool = False) -> torch.Tensor:
    """Getting the scale matrix from a vector.

    Parameters
    ----------
    vector : VEC_TYPE
        The vector to get scale from.
    dtype: torch.dtype, optional
        Torch dtype for init of tensors, by default torch.float64
    device: torch.device, optional
        Torch device for init the tensors directly on a specific device, by default "cpu"
    requires_grad: bool, optional
        Whether initialized tensors will require gradient backpropagation, by default False

    Returns
    -------
    torch.Tensor
        The resulting scale matrix.
    """
    vector = tensorify(vector, dtype=dtype, device=device,
                       requires_grad=requires_grad)
    mat = torch.tensor(np.identity(4), dtype=dtype,
                       device=device, requires_grad=requires_grad)
    mat[0, 0] = vector[0]
    mat[1, 1] = vector[1]
    mat[2, 2] = vector[2]
    return mat


@torch.jit.script
def _vector_angle(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Computes the angle between two vectors.

    Parameters
    ----------
    v1 : torch.Tensor
        The first input vector Shape ([..., B], 3)
    v2 : torch.Tensor
        The second input vector Shape ([..., B], 3)

    Returns
    -------
    torch.Tensor
        Angle between vector. Shape ([..., B])
    """
    return torch.acos((v1 * v2).sum(dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)))


@as_tensors()
def vector_angle(v1: VEC_TYPE, v2: VEC_TYPE, mode: Literal["acos", "tan2"] = "acos") -> torch.Tensor:
    """Computes the angle between vector v1 and v2.

    Parameters
    ----------
    v1 : VEC_TYPE
        The first input vector, shape (..., 3)
    v2 : VEC_TYPE
        The second input vector, shape (..., 3)

    Returns
    -------
    torch.Tensor
        Angle between vector. shape (...,)
    """
    if mode == "acos":
        return torch.acos((v1 * v2).sum(dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)))
    elif mode == "tan2":
        return torch.atan2(torch.cross(v1, v2, dim=-1).norm(p="fro", dim=-1), (v1 * v2).sum(dim=-1))
    else:
        raise ValueError("mode must be either 'acos' or 'tan2'")


@torch.jit.script
def _vector_angle_3d(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Computes the angle between two 3D vectors.

    Parameters
    ----------

    v1 : torch.Tensor
        The first input vector, shape (..., 3)

    v2 : torch.Tensor
        The second input vector, shape (..., 3)

    Returns
    -------
    torch.Tensor
        Angle between vector. Shape (..., 3)
    """
    angle_x = torch.atan2(v2[..., 1], v2[..., 2]) - \
        torch.atan2(v1[..., 1], v1[..., 2])  # YZ => X
    angle_y = torch.atan2(v2[..., 0], v2[..., 2]) - \
        torch.atan2(v1[..., 0], v1[..., 2])  # XZ => Y
    angle_z = torch.atan2(v2[..., 0], v2[..., 1]) - \
        torch.atan2(v1[..., 0], v1[..., 1])  # XY => Z
    return torch.stack([angle_x, angle_y, angle_z], dim=-1)


@as_tensors()
def vector_angle_3d(v1: VEC_TYPE, v2: VEC_TYPE, output_mode: Literal["deg", "rad"] = "rad") -> VEC_TYPE:
    """Computes the angle between vector v1 and v2.

    Parameters
    ----------
    v1 : VEC_TYPE
        The first input vector, shape (..., 3)
    v2 : VEC_TYPE
        The second input vector, shape (..., 3)

    Returns
    -------
    VEC_TYPE
        Angle between vector. Shape (...,)
    """
    # Get t he planes to calculate the angle against
    unsqueezed = False
    if len(v1.shape) == 1:
        v1 = v1.unsqueeze(0)
        unsqueezed = True
    if len(v2.shape) == 1:
        v2 = v2.unsqueeze(0)
        unsqueezed = True

    if v1.shape[-1] != 3 or v2.shape[-1] != 3:
        raise ValueError("Input vectors must have shape (..., 3)")
    if len(v1.shape) != len(v2.shape) or len(v1.shape) != 2:
        raise ValueError("Input vectors must have the same shape")

    ret = _vector_angle_3d(v1, v2)
    if unsqueezed and ret.shape[0] == 1:
        ret = ret.squeeze(0)
    if output_mode == "deg":
        ret = torch.rad2deg(ret)
    return ret


@torch.jit.script
def _rotmat_from_vectors(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    """
    Returns a rotation matricies that rotates vectors a into vectors b.

    Good explanation can be found here:
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677

    If the vectors are parallel, the rotation matrix will be the identity matrix.

    If the vectors are antiparallel, the rotation matrix will be a rotation around the x axis by 180 degrees.
    E.g this will not necessarily produce the expected result, depeding on the context.

    Parameters
    ----------
    a : torch.Tensor
        Source direction vectors. Shape should be (..., 3).

    b : torch.Tensor
        Target direction vectors. Shape should be (..., 3).

    Returns
    -------
    torch.Tensor
        Rotation matricies. Shape is (..., 3, 3).
    """

    a, shp = flatten_batch_dims(a, -2)
    b, bshp = flatten_batch_dims(b, -2)

    if shp != bshp:
        raise ValueError(
            "a and b must have the same shape, got {} and {}".format(shp, bshp))

    b = b.to(a.dtype)

    # normalize
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)

    B, C = a.shape

    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1)  # Dot product
    s = torch.norm(v, dim=-1)

    kmat = torch.zeros((3, 3), device=a.device)[None, ...].repeat(B, 1, 1)
    kmat[:, 0, 1] = -v[..., 2]
    kmat[:, 0, 2] = v[..., 1]
    kmat[:, 1, 0] = v[..., 2]
    kmat[:, 1, 2] = -v[..., 0]
    kmat[:, 2, 0] = -v[..., 1]
    kmat[:, 2, 1] = v[..., 0]

    Id = torch.eye(3, device=a.device, dtype=a.dtype)[
        None, ...].expand(B, -1, -1)

    kProd = torch.bmm(kmat, kmat)
    res = (Id + kmat + kProd) * (((1 - c) / (s ** 2))[:, None, None])

    # Check for parallel vectors
    parallel = torch.isclose(v.norm(p=2, dim=-1), torch.tensor(
        0., device=a.device, dtype=a.dtype), atol=atol)  # Check for parallel vectors

    if torch.any(parallel):
        # Check for antiparallel vectors
        pvec = _vector_angle(a[parallel], b[parallel])
        antiparallel = torch.isclose(pvec, torch.tensor(
            torch.pi, device=a.device, dtype=a.dtype), atol=atol).any(dim=-1)

        real_antiparallel = torch.zeros_like(parallel, dtype=torch.bool)
        real_antiparallel[parallel] = antiparallel

        real_parallel = torch.zeros_like(parallel, dtype=torch.bool)
        real_parallel[parallel] = ~antiparallel

        # If antiparallel rotate around x axis by 180 degrees
        if torch.any(real_antiparallel):
            R = torch.eye(3, device=a.device, dtype=a.dtype)
            R[1, 1] = -1
            R[1, 2] = 0
            R[2, 1] = 0
            R[2, 2] = -1
            res[real_antiparallel] = R
        if torch.any(real_parallel):
            res[real_parallel] = torch.eye(3, device=a.device, dtype=a.dtype)
    return res


@as_tensors()
def rotmat_from_vectors(a: VEC_TYPE, b: VEC_TYPE) -> VEC_TYPE:
    """
    Returns a rotation matricies that rotates vectors a into vectors b.

    Good explanation can be found here:
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677

    Parameters
    ----------
    a : VEC_TYPE
        Source direction vectors. Shape should be (..., 3).

    b : VEC_TYPE
        Target direction vectors. Shape should be (..., 3).

    Returns
    -------
    VEC_TYPE
        Rotation matricies. Shape is (..., 3, 3).
    """
    return _rotmat_from_vectors(a, b)


@torch.jit.script
def _norm_rotation_angles(v1: torch.Tensor) -> torch.Tensor:
    """Normalizes the rotation angles to the range of [-pi, pi).

    Parameters
    ----------
    v1 : torch.Tensor
        The input vector.
        Accepts any shape.

    Returns
    -------
    torch.Tensor
        The normalized vector.
    """
    pi_normed = torch.fmod(v1, 2 * np.pi)
    above_half = pi_normed > np.pi
    return torch.where(above_half, pi_normed - 2 * np.pi, pi_normed)


@as_tensors()
def norm_rotation_angles(v1: VEC_TYPE) -> VEC_TYPE:
    """Normalizes the rotation angles to the range of [-pi, pi).

    Parameters
    ----------
    v1 : VEC_TYPE
        The input vector.
        Accepts any shape.

    Returns
    -------
    VEC_TYPE
        The normalized vector.
    """
    return _norm_rotation_angles(v1)


@torch.jit.script
def unitquat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts unit quaternion into rotation matrix representation.

    Parameters
    ----------
    quat : torch.Tensor
        batch of unit quaternions (B x 4 tensor, XYZW convention).

    Returns
    -------
    torch.Tensor
        batch of rotation matrices (B x 3 x 3 tensor).

    Notes
    ------
    Original implementation from:

    https://github.com/naver/roma

    """
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L912
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = torch.empty(quat.shape[:-1] + (3, 3),
                         dtype=quat.dtype, device=quat.device)

    matrix[..., 0, 0] = x2 - y2 - z2 + w2
    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 2, 0] = 2 * (xz - yw)

    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 1, 1] = - x2 + y2 - z2 + w2
    matrix[..., 2, 1] = 2 * (yz + xw)

    matrix[..., 0, 2] = 2 * (xz + yw)
    matrix[..., 1, 2] = 2 * (yz - xw)
    matrix[..., 2, 2] = - x2 - y2 + z2 + w2
    return matrix


@torch.jit.script
def rotmat_to_unitquat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrix to unit quaternion representation.

    Parameters
    ---------
    matrix : torch.Tensor
        batch of rotation matrices (Bx3x3 tensor).

    Returns
    -------
    torch.Tensor
        batch of unit quaternions (Bx4 tensor, XYZW convention)

    Notes
    ------
    Original implementation from:
    https://github.com/naver/roma
    """
    matrix, batch_shape = flatten_batch_dims(matrix, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert ((D1, D2) == (3, 3)), "Input should be a Bx3x3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/7cb3d751756907238996502b92709dc45e1c6596/scipy/spatial/transform/rotation.py#L480

    decision_matrix = torch.empty(
        (num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
    choices = decision_matrix.argmax(dim=1)

    quat = torch.empty((num_rotations, 4),
                       dtype=matrix.dtype, device=matrix.device)

    ind = torch.nonzero(choices != 3)
    if len(ind) > 0:
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3
        quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
        quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
        quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
        quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3)
    if len(ind) > 0:
        quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
        quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
        quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
        quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]
    return unflatten_batch_dims(quat, batch_shape)


@torch.jit.script
def position_quaternion_to_affine_matrix(position: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """
    Create an affine matrix from a position and a quaternion.

    Parameters
    ----------
    position : torch.Tensor
        The position tensor of shape (..., 3).
    quaternion : torch.Tensor
        The quaternion tensor of shape (..., 4).

    Returns
    -------
    torch.Tensor
        The affine matrix of shape (..., 4, 4).
    """
    rotation_matrix = unitquat_to_rotmat(quaternion)
    return _compose_transformation_matrix(position, rotation_matrix[..., :3, :3])


def _calculate_rotation_matrix(a: torch.Tensor, b: torch.Tensor):
    # Check for opposite directions and flip if necessary
    dot_products = torch.sum(a * b, dim=-1)
    opposite_directions = dot_products < 0
    a[opposite_directions] = -a[opposite_directions]  # Flip vectors in a

    # Calculate the cross-product matrix for each vector pair
    A = torch.cross(a, b, dim=-1)

    # Calculate the dot product for each vector pair
    dot_product = torch.sum(a * b, dim=-1)

    # Calculate the rotation matrix using the Rodrigues formula for each vector pair
    R = torch.eye(3)[None, :, :] + torch.cross(A[:, :, None], torch.eye(3)[None, :, :], dim=-2) + \
        torch.einsum('ij,ik->ijk', A, A) * (1 - dot_product[:, None, None]) / (
            torch.linalg.norm(A, dim=-1)[:, None, None] ** 2)
    return R


def calculate_rotation_matrix(a: torch.Tensor, b: torch.Tensor):
    """
    Calculates the rotation matrix R for each vector pair in a and b to transform a into b.

    :math:`RA = B`
    Where R is the rotation matrix, A is the input vector, and B is the target vector.

    Parameters
    ----------

    a : torch.Tensor
        The input vector of shape ([..., B], 3).

    b : torch.Tensor
        The target vector of shape ([..., B], 3).

    Returns
    -------
    torch.Tensor
        A PyTorch tensor of rotation matrices with shape ([..., B], 3, 3).
    """

    # Ensure vectors have the same shape
    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape.")

    # Flatten the batch dims
    a, batch_shape = flatten_batch_dims(a, end_dim=-2)
    b, _ = flatten_batch_dims(b, end_dim=-2)

    nonequal = torch.all(a != b, dim=-1)
    R = torch.eye(3, device=a.device, dtype=a.dtype).repeat(a.shape[0], 1, 1)
    R[nonequal] = _calculate_rotation_matrix(a[nonequal], b[nonequal])

    return unflatten_batch_dims(R, batch_shape)


def compute_ray_plane_intersections_from_position_matrix(
        plane_position: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        eps: float = 1e-6,
) -> torch.Tensor:
    plane_position, plane_shape = flatten_batch_dims(
        plane_position, end_dim=-3)
    B = plane_position.shape[0]
    normal = torch.eye(4, device=plane_position.device,
                       dtype=plane_position.dtype).unsqueeze(0)
    normal[..., 2, 3] = 1  # z = 1
    normal = normal.repeat(B, 1, 1)  # (B, 4, 4)
    plane_normal_target = torch.bmm(
        plane_position, normal)[..., :3, 3]  # (B, 3)
    plane_normals = plane_normal_target - plane_position[..., :3, 3]
    plane_origin = plane_position[..., :3, 3]
    if len(plane_normals.shape) == 2 and plane_normals.shape[0] == 1:
        RB = ray_origins.shape[0]
        plane_normals = plane_normals.repeat(RB, 1)
        plane_origin = plane_origin.repeat(RB, 1)
    return unflatten_batch_dims(compute_ray_plane_intersections(
        plane_origins=plane_origin,
        plane_normals=plane_normals,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        eps=eps,
    ), plane_shape)


def compute_ray_plane_intersections(
        plane_origins: torch.Tensor,
        plane_normals: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        eps: float = 1e-6,
) -> torch.Tensor:
    # Check if shapes are matching

    plane_origins, plane_shape = flatten_batch_dims(plane_origins, end_dim=-2)
    plane_normals, _ = flatten_batch_dims(plane_normals, end_dim=-2)

    ray_origins, ray_shape = flatten_batch_dims(ray_origins, end_dim=-2)
    ray_directions, _ = flatten_batch_dims(ray_directions, end_dim=-2)

    shapes = [plane_origins.shape, plane_normals.shape,
              ray_origins.shape, ray_directions.shape]
    # All shapes should be the same
    if len(set(shapes)) != 1:
        raise ValueError("All input shapes should be the same. (B, 3)")

    B = plane_origins.shape[0]

    # Plane normals to ray_dims
    plane_n = plane_normals  # (B, 3)
    plane_p = plane_origins  # (B, 3)

    d = (plane_n * plane_p).sum(-1)  # dot product

    # Add N to rays
    ray_p = ray_origins  # (B, 3)
    ray_v = ray_directions  # (B, 3)

    intersection_points = torch.empty(
        (B, 3), device=ray_origins.device, dtype=ray_origins.dtype)
    intersection_points.fill_(float('nan'))

    # Calculate intersection points
    denom = (plane_n * ray_v).sum(-1)  # dot product

    # This should be True for parallel rays Shape (N, B)
    is_not_intersecting = torch.abs(denom) < eps

    dot_n_p = (plane_n * ray_p).sum(-1)  # dot product

    t = ((-(dot_n_p.unsqueeze(-1).repeat(1, 3) - d.unsqueeze(-1).repeat(1, 3))
          )[~is_not_intersecting] / denom[~is_not_intersecting].unsqueeze(-1).repeat(1, 3)).squeeze()
    intersection_points[~is_not_intersecting] = ray_p[~is_not_intersecting] + \
        t * ray_v[~is_not_intersecting]  # Intersection points for unlimited planes

    return unflatten_batch_dims(intersection_points, plane_shape)


def align_rectangles(rect1: torch.Tensor, rect2: torch.Tensor):
    """
    Aligns two rectangles using Procrustes analysis and returns the transformation matrix
    to transform the first rectangle to the second rectangle.

    Parameters
    ----------
    rect1 : torch.Tensor
        The first rectangle. Shape: (B, P, 3)

    rect2 : torch.Tensor
        The second rectangle. Shape: (B, P, 3)

    Returns
    -------
    torch.Tensor
        The transformation matrix. Shape: (B, 4, 4)
    """
    rect1, shp = flatten_batch_dims(rect1, -3)
    rect2, _ = flatten_batch_dims(rect2, -3)

    B, P, _ = rect1.shape
    if rect2.shape != (B, P, 3):
        raise ValueError("The input rectangles must have the same shape.")

    # Center the rectangles
    center1 = rect1.mean(dim=-2)
    center2 = rect2.mean(dim=-2)

    rect1_centered = rect1 - center1.unsqueeze(-2).expand_as(rect1)
    rect2_centered = rect2 - center2.unsqueeze(-2).expand_as(rect1)

    # Calculate the optimal rotation using Procrustes analysis
    H = torch.bmm(
        rect2_centered[:, :3].transpose(-2, -1),
        rect1_centered[:, :3]
    )

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.mH

    R = torch.bmm(U, V.transpose(-2, -1))

    # Calculate the translation
    t = center2 - torch.bmm(R, center1.unsqueeze(-1))[..., 0]

    T = torch.eye(4, dtype=rect1.dtype, device=rect1.device)[
        None, ...].repeat(B, 1, 1)
    T[..., :3, :3] = R  # Rotation matrix
    T[..., :3, 3] = t  # Translation vector
    T[..., 3, 3] = 1

    # Sanity check
    # tf = torch.bmm(T.unsqueeze(1).expand(-1, P, -1, -1).reshape(B*P, 4, 4), torch.cat([rect1.reshape(B*P, 3).unsqueeze(-1), torch.ones((B * P, 1, 1))], dim=-2))[:, :3, 0].reshape(B, P, 3)
    # torch.allclose(rect2, tf, atol=1e-6)
    return unflatten_batch_dims(T, shp)


def find_plane(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the best-fit plane for a set of points using least squares.



    Effectively, this function computes the plane equation Ax + By + Cz + D = 0

    Parameters
    ----------
    points : torch.Tensor
        The input points. Shape: (B, P, 3)
        P should be at least 3 to solve the plane equation.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing the centeroid of the found plane and its normal vector.


    """
    points, shp = flatten_batch_dims(points, -3)
    BS, P, _ = points.shape

    # Compute the centroid of the points
    centroid = points.mean(dim=-2)

    # Center the points around the centroid
    centered_points = points - centroid.unsqueeze(-2).expand_as(points)

    A = centered_points.clone()

    has_values = (A == 0.).all(dim=-2)
    valid = ~has_values.any(dim=-1)

    use_dim = -1
    n = torch.zeros(BS, 3, device=points.device, dtype=points.dtype)

    if valid.any():
        A[valid, :, use_dim] = 1.  # Set any coordinate to 1 to solve the plane equation
        B = centered_points[valid, :, use_dim]

        vec = torch.linalg.lstsq(A, B)
        coef = vec.solution

        a = coef[..., 0]
        b = coef[..., 1]
        d = coef[..., 2]

        # Normal vector of the plane
        n[valid] = torch.stack([a, b, -torch.ones_like(b)], dim=-1)
    else:
        # If of one dimension all points are zero, the plane is orthogonal to this dimension
        z = torch.zeros(((~valid).sum(), 3),
                        device=points.device, dtype=points.dtype)
        z[has_values[~valid]] = 1.0
        n[~valid] = z

    n = n / n.norm(dim=-1, keepdim=True)  # Normalize the normal vector
    return unflatten_batch_dims(centroid, shp), unflatten_batch_dims(n, shp)


def plane_eval(p0: torch.Tensor, n: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Evaluate the plane equation at the given points

    Parameters
    ----------
    p0 : torch.Tensor
        The point on the plane (3D vector) Shape: ([... B], 3)

    n : torch.Tensor
        The normal vector of the plane (3D vector) Shape: ([... B], 3)

    values : torch.Tensor
        The values to evaluate at the given (x, y) points (2D vector) Shape: ([... B], 2)

    Returns
    -------
    torch.Tensor
        The evaluated values at the given points (3D vector) Shape: ([... B], 3)
    """
    p0, _ = flatten_batch_dims(p0, -2)
    n, _ = flatten_batch_dims(n, -2)
    values, shp = flatten_batch_dims(values, -2)

    if (n[..., -1] == 0.).any():
        raise ValueError(
            "The plane eval function is not yet implemented for YZ or XZ parallel planes having n[..., -1] == 0.")

    Np0, C = p0.shape
    Nn, C = n.shape
    Nv, C = values.shape

    if Np0 != Nv or Np0 != Nv:
        if Np0 == 1:
            p0 = p0.expand(Nv, -1)
        else:
            raise ValueError("p0, n and values must have the same batch size")
        if Nn == 1:
            n = n.expand(Nv, -1)
        else:
            raise ValueError("p0, n and values must have the same batch size")
    z = (torch.bmm(n.unsqueeze(1), p0.unsqueeze(-1)).squeeze(-1) -
         n[..., 0:1] * values[..., 0:1] - n[..., 1:2] * values[..., 1:2]) / n[..., 2:3]

    zr = torch.cat([values[..., :2], z], dim=-1)
    return unflatten_batch_dims(zr, shp)
