
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.agent.util import Tracker
from tools.event import EventArgs



@dataclass
class TrainingStartedEventArgs(EventArgs):
    """Event args for a start on training."""

    model: Any = None
    """The model instance which is trained / evaluated by an agent."""

    model_args: Dict[str, Any] = None
    """The init arguments of the model which is trained."""

    loss_name: str = None
    """The name of the actual loss."""

    tracker: Tracker = None
    """The current tracker of the agent, so its model state and loss information."""

    remaining_iterations: Optional[int] = None
    """The number of remaining iterations if predefined."""

    dataset_config: Dict[str, Any] = None
    """The dataset / dataloader configuration."""


