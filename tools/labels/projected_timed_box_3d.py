from tools.serialization.json_convertible import JsonConvertible
from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from tools.scene.coordinate_system_3d import CoordinateSystem3D
from tools.labels.timed_box_3d import TimedBox3D
from tools.labels.timed_box_2d import TimedBox2D

@dataclass
class ProjectedTimedBox3D(TimedBox3D):
    """Timed 3D box label, including a view projection. For tracking purposes."""

    projected_label: Optional[TimedBox2D] = None
    """2D projection label of the box.""" 

    camera_idx: int = 0
    """Index of the camera that the box is projected onto."""

    object_id: Optional[int] = None
    """Mask mapped object ID."""