from tools.serialization.json_convertible import JsonConvertible
from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from tools.scene.coordinate_system_3d import CoordinateSystem3D
from tools.labels.timed_box_2d import TimedBox2D

@dataclass
class TimedBox3D(JsonConvertible):
    """Timed 3D box label. For tracking purposes."""

    id: str 
    """Unique identifier for the box label."""

    center: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T, 3] representing the center of the box."""

    depth: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T] representing the length of the box."""

    width: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T] representing the width of the box."""

    height: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T] representing the height of the box."""

    heading: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T, 3] representing the heading as x, y, z rotation angle of the box."""

    corners: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T, 8, 3] representing the corners of the box."""
    
    frame_times: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of shape [T] representing the frame idx of the box."""

    coordinate_system: CoordinateSystem3D = field(default_factory=lambda: CoordinateSystem3D.from_string("rfu"))
    """Coordinate system of the box coordinates."""
