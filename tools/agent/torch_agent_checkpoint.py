import io
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, BinaryIO, Dict, Optional, Set, Type, Union


from tools.util.reflection import class_name
import torch
from tools.agent.agent import Agent

from .base_agent_checkpoint import BaseAgentCheckpoint
from tools.mixin import FastReprMixin
from tools.util.format import parse_type


@dataclass(repr=False)
class TorchAgentCheckpoint(BaseAgentCheckpoint, FastReprMixin):

    model_state_dict: Any = field(default=None)
    """The state dictionary of the pytorch model."""

    optimizer_state_dict: Optional[Any] = field(default=None)
    """The state dictionary of the optimizer"""

    optimizer_type: Type = field(default=None)
    """The optimizer type.It will be used for recreating it on load."""

    optimizer_args: Dict[str, Any] = field(default=None)
    """Optimizer keyword arguments for recreating."""

    agent_type: str = field(default="tools.agent.torch_agent.TorchAgent")
    """Agent Type for recreating."""

    additional_agent_args: Dict[str, Any] = field(default_factory=dict)
    """Additional arguments for the agent, can be used to save state information for agent subclasses."""

    @classmethod
    def ignore_on_repr(cls) -> Set[str]:
        return {"model_state", "model_args", "tracker", "criterion", "dataset_config", "optimizer_state_dict", "model_state_dict"}

    def save(self, f: Union[PathLike[str], BinaryIO]) -> None:
        """Saves the checkpoint to the dedicated stream.

        Args:
            f (PathLike[str]): The file or buffer, where the data should be saved to.
        """
        obj = self._save_prechecks(f)
        torch.save(obj.__dict__, f)

    @classmethod
    def load(cls, f: Union[PathLike[str], BinaryIO]):
        """Loading a checkpoint from the given file / buffer.

        Args:
            f (Union[PathLike[str], BinaryIO]): File / Buffer

        Returns:
            Checkpoint: The loaded checkpoint
        """
        args = dict()
        if not torch.cuda.is_available():
            args['map_location'] = torch.device('cpu')
        dic = torch.load(f, **args)
        dic.pop('__id__', None)
        return cls(**dic)

    def get_additional_checkpoint_information_content(self) -> Dict[str, Any]:
        return dict(optimizer_type=self.optimizer_type.__name__,
                    optimizer_args=self.optimizer_args)

    def get_default_extension(self) -> str:
        return ".tar"

    def to_agent(self) -> Agent:
        from tools.agent.torch_agent import TorchAgent
        obj_type = parse_type(self.agent_type, parent_type=TorchAgent,
                              handle_invalid="set_default", default_value=TorchAgent)
        agent = obj_type.from_acc(self)
        return agent
