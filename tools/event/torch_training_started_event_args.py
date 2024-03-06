from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch import Tensor

from tools.agent.util.tracker import Tracker
from tools.event.event_args import EventArgs
from tools.event.training_started_event_args import TrainingStartedEventArgs

@dataclass
class TorchTrainingStartedEventArgs(TrainingStartedEventArgs):
    """Specialized training started event for a torch model."""

    model: torch.nn.Module = None
    """The torch model instance used as predictor."""

    optimizer: torch.optim.Optimizer = None
    """The optimizer which is currently used to train the model."""

    remaining_iterations: Optional[int] = None
    """The number of remaining iterations if predefined."""
