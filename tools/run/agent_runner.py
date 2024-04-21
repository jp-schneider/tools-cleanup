import os
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Type, Optional, TypeVar, Union, get_type_hints

from tools.agent.agent import Agent
from tools.config.config import Config
from tools.config.experiment_config import ExperimentConfig
from tools.metric.metric_entry import MetricEntry
from tools.run.abstract_runner import AbstractRunner
from tools.run.config_runner import ConfigRunner
from tools.run.trainable_runner import TrainableRunner
from tools.util.format import to_snake_case
from tools.util.path_tools import numerated_file_name

from tools.dataset import BaseDataset

from tools.error import ArgumentNoneError
from tools.util.reflection import class_name
import random
import numpy as np
import sys
from tools.logger.logging import logger
from tools.util.seed import seed_all

class AgentRunner(TrainableRunner):
    """Agent Runner which uses an agent and a dataloader."""

    agent: Agent
    """Agent which is used to run the experiment."""

    dataloader: BaseDataset
    """Data loader which is used to train the agent."""

    def __init__(self, config: ExperimentConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.set_seed()

    def set_seed(self) -> None:
        """Sets the seed of the random number generators.
        This is used to make the experiments reproducible.
        """        
        seed_all(self.config.seed)

    def store_config(self, path: Optional[str] = None, **kwargs) -> str:
        """Stores the config of the runner to a file in the agent folder.

        Returns
        -------
        str
            The path where the config is stored.
        """
        path = os.path.join(
            self.agent.agent_folder,
        f"init_cfg_{to_snake_case(type(self.config).__name__)}.yaml")
        return super().store_config(path=path, **kwargs)
        
    @abstractmethod
    def patch_agent(self, agent: Agent) -> None:
        """Sets a given agent to the runner by patching in the agent all necessary context
        veriables.

        Parameters
        ----------
        agent : Agent
            The restored agent.
        """        
        pass
