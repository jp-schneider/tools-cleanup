from dataclasses import field, dataclass
from enum import Enum
from tools.util.format import parse_enum
from typing import Union, List, Tuple, Optional
import numpy as np
from tools.util.typing import VEC_TYPE
from tools.util.numpy import flatten_batch_dims, unflatten_batch_dims
from tools.util.numpy import numpyify
from tools.serialization.json_convertible import JsonConvertible


class Handiness(Enum):
    """Handiness of a coordinate system."""

    LEFT = "l"
    """Left-handed coordinate system."""

    RIGHT = "r"
    """Right-handed coordinate system."""

    def __str__(self):
        return self.value


class AxisSpecifier(Enum):
    """Axis specifier for 3D coordinate systems, defines the axis direction from a neutral view."""

    LEFT = "l"
    RIGHT = "r"
    UP = "u"
    DOWN = "d"
    FORWARD = "f"
    BACKWARD = "b"

    def __str__(self):
        return self.name

    def opposide_identifier(self) -> "AxisSpecifier":
        """
        Returns the opposite axis specifier.

        Returns
        -------
        AxisSpecifier
            The opposite axis specifier.
        """
        if self == AxisSpecifier.LEFT:
            return AxisSpecifier.RIGHT
        elif self == AxisSpecifier.RIGHT:
            return AxisSpecifier.LEFT
        elif self == AxisSpecifier.UP:
            return AxisSpecifier.DOWN
        elif self == AxisSpecifier.DOWN:
            return AxisSpecifier.UP
        elif self == AxisSpecifier.FORWARD:
            return AxisSpecifier.BACKWARD
        elif self == AxisSpecifier.BACKWARD:
            return AxisSpecifier.FORWARD
        else:
            raise ValueError("Unknown axis specifier.")

    def is_opposite(self, other: Union[str, "AxisSpecifier"]) -> bool:
        """
        Check if the axis specifier is opposite to another axis specifier.

        Parameters
        ----------
        other : AxisSpecifier
            The other axis specifier.

        Returns
        -------
        bool
            True if the axis specifiers are opposite, False otherwise.
        """
        other = parse_enum(AxisSpecifier, other)
        return self.opposide_identifier() == other

    def get_base_orientation(self) -> np.ndarray:
        """
        Get the base orientation of the axis specifier.

        Underlying assumption is that

        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        v3 = [0, 0, 1]

        Corresponds to a right-handed coordinate system, where v1 x v2 * v3 >= 0.
        From viewers perspective, z would be backward, x would be right and y would be up.

        Returns
        -------
        np.ndarray
            The base orientation of the axis specifier.
        """
        if self == AxisSpecifier.LEFT:
            return np.array([-1, 0, 0])
        elif self == AxisSpecifier.RIGHT:
            return np.array([1, 0, 0])
        elif self == AxisSpecifier.UP:
            return np.array([0, 1, 0])
        elif self == AxisSpecifier.DOWN:
            return np.array([0, -1, 0])
        elif self == AxisSpecifier.FORWARD:
            # Note: This is the negative Z-axis.
            return np.array([0, 0, -1])
        elif self == AxisSpecifier.BACKWARD:
            # Note: This is the positive Z-axis. (opposite to forward)
            return np.array([0, 0, 1])
        else:
            raise ValueError("Unknown axis specifier.")


@dataclass
class CoordinateSystem3D(JsonConvertible):
    """Definition class for a 3D coordinate system."""

    x: AxisSpecifier = AxisSpecifier.RIGHT
    """The X-axis specifier."""

    y: AxisSpecifier = AxisSpecifier.FORWARD
    """The Y-axis specifier."""

    z: AxisSpecifier = AxisSpecifier.UP
    """The Z-axis specifier."""

    def validate(self):
        """Validate the coordinate system.

        Raises
        ------
        ValueError
            If the coordinate system is invalid.
        """
        # Assure that X, Y and Z are all different and not opposite.
        x_grp = {AxisSpecifier.LEFT, AxisSpecifier.RIGHT}
        y_grp = {AxisSpecifier.UP, AxisSpecifier.DOWN}
        z_grp = {AxisSpecifier.FORWARD, AxisSpecifier.BACKWARD}
        if len({self.x, self.y, self.z}) != 3:
            raise ValueError("X, Y and Z axis must be different.")
        # Check if x, y and z are in one of the groups.
        x_grp_hit = self.x in x_grp or self.y in x_grp or self.z in x_grp
        y_grp_hit = self.x in y_grp or self.y in y_grp or self.z in y_grp
        z_grp_hit = self.x in z_grp or self.y in z_grp or self.z in z_grp

        if not (x_grp_hit and y_grp_hit and z_grp_hit):
            raise ValueError("All axes must form a linearly independent base.")

    @property
    def handiness(self) -> Handiness:
        """
        Compute the handiness of the coordinate system.
        E.g. if the coordinate system is left-handed or right-handed.

        Returns
        -------
        Handiness
            The handiness of the coordinate system.
        """
        # Check if the determinant of the matrix is positive.
        x_vec = self.x.get_base_orientation()
        y_vec = self.y.get_base_orientation()
        z_vec = self.z.get_base_orientation()

        # Test if (v1 x v2) * v3 is positive.
        det = np.dot(np.cross(x_vec, y_vec), z_vec)
        if det > 0:
            return Handiness.RIGHT
        elif det < 0:
            return Handiness.LEFT
        else:
            raise ValueError("Coordinate system is degenerate.")

    def get_permutation_matrix(self) -> np.ndarray:
        """
        Get the permutation matrix of the coordinate system,
        which is defined as the permutation matrix which transforms
        a native right-handed coordinate system (x-right, y-up, z-backward)
        to the current coordinate system.

        Returns
        -------
        np.ndarray
            The permutation matrix.
        """
        # Create a permutation matrix.
        matrix = np.zeros((3, 3), dtype=np.int32)
        matrix[0, :] = self.x.get_base_orientation()
        matrix[1, :] = self.y.get_base_orientation()
        matrix[2, :] = self.z.get_base_orientation()
        return matrix

    def __str__(self):
        return f"{self.x.value}{self.y.value}{self.z.value}"

    @classmethod
    def from_string(cls, value: str, validate: bool = True) -> "CoordinateSystem3D":
        """
        Parse a coordinate system from a string.

        Parameters
        ----------
        value : str
            The string to parse.
            Should be in the format "XYZ" where X, Y and Z are axis specifiers e.g. "ruf" for a left-handed coordinate system:
            x-[r]ight, y-[u]p, z-[f]orward.

            All valid axis specifiers are:
            - [l]eft
            - [r]ight
            - [u]p
            - [d]own
            - [f]orward
            - [b]ackward

        validate : bool, optional
            If the coordinate system should be validated, by default True.

        Returns
        -------
        CoordinateSystem3D
            The parsed coordinate system.
        """
        if isinstance(value, CoordinateSystem3D):
            if validate:
                value.validate()
            return value

        value = value.lower()
        if len(value) != 3:
            raise ValueError("Invalid coordinate system string length.")
        system = cls(
            x=parse_enum(AxisSpecifier, value[0]),
            y=parse_enum(AxisSpecifier, value[1]),
            z=parse_enum(AxisSpecifier, value[2])
        )
        if validate:
            system.validate()
        return system

    def convert_vector(self,
                       other: Union[str, "CoordinateSystem3D"],
                       vector: VEC_TYPE) -> np.ndarray:
        """
        Convert a vector from this coordinate system to another coordinate system.

        Parameters
        ----------
        other : CoordinateSystem3D
            The other coordinate system to convert to.

        vector : np.ndarray
            The vector to convert. Shape: ([..., B] 3)

        Returns
        -------
        np.ndarray
            The converted vector. Shape: ([..., B] 3)
        """
        vector, shp = flatten_batch_dims(numpyify(vector), -2)

        B, _ = vector.shape
        current_to_native = np.linalg.inv(self.get_permutation_matrix())
        native_to_other = other.get_permutation_matrix()
        current_to_other = native_to_other @ current_to_native
        current_to_other = current_to_other[np.newaxis].repeat(B, axis=0)
        new_vector = current_to_other @ vector[..., np.newaxis]
        new_vector = new_vector[..., 0]

        return unflatten_batch_dims(new_vector, shp)

    def convert(self,
                other: Union[str, "CoordinateSystem3D"],
                matrix: VEC_TYPE) -> np.ndarray:
        """
        Convert a vector from this coordinate system to another coordinate system.

        Parameters
        ----------
        other : CoordinateSystem3D
            The other coordinate system to convert to.

        matrix : np.ndarray
            A affine transformation matrix of shape ([..., B] 4, 4) which should be converted to the other coordinate system.

        Returns
        -------
        np.ndarray
            The matrix.
        """
        if isinstance(other, str):
            other = CoordinateSystem3D.from_string(other, validate=True)

        matrix, shp = flatten_batch_dims(numpyify(matrix), -3)

        B, _, _ = matrix.shape

        rots = matrix[..., :3, :3]
        positions = matrix[..., :3, 3]

        current_to_native = np.linalg.inv(self.get_permutation_matrix())
        native_to_other = other.get_permutation_matrix()
        current_to_other = native_to_other @ current_to_native

        A = np.eye(4, dtype=matrix.dtype)
        A[:3, :3] = current_to_other
        A = A[np.newaxis].repeat(B, axis=0)

        converted_matrix = A @ matrix @ np.linalg.inv(A)
        return unflatten_batch_dims(converted_matrix, shp)
