import os
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Type, Optional, TypeVar, Union, get_type_hints

from tools.agent.agent import Agent
from tools.config.config import Config
from tools.config.experiment_config import ExperimentConfig
from tools.metric.metric_entry import MetricEntry
from tools.run.abstract_runner import AbstractRunner
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


class ConfigRunner(AbstractRunner):
    """Config Runner which gets a config and can "run"."""

    config: Config
    """Configuration of the runner."""

    diff_config: Optional[Dict[str, Any]]
    """Dictionary which contains the difference between the base config and the child config of the runner,
    if the runner is part of a multi runner."""

    __saved_config__: Optional[str]
    """String config which was saved."""

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(**kwargs)
        if config is None:
            raise ArgumentNoneError("config")
        self.config = config
        self.diff_config = None
        self.config.used_runner_type = class_name(self)
        self.__saved_config__ = None

    def store_config(self, path: Optional[str] = None, override: bool = False, **kwargs) -> str:
        """Stores the config of the runner to a file in the agent folder.

        Parameters
        ----------
        path : Optional[str], optional
            Path where the config should be stored, by default None
            If None, the config is not stored to a file.

        override : bool, optional
            Whether to override an existing config file. Defaults to False.

        Returns
        -------
        str
            The path where the config is stored, or None if no path was given.
        """
        if path is not None:
            path = os.path.join(
                path, f"init_cfg_{to_snake_case(type(self.config).__name__)}.yaml")
            path = self.config.save_to_file(
                path, no_uuid=True, no_large_data=True, override=override)
            with open(path, 'r') as f:
                self.__saved_config__ = f.read()
            return path
        else:
            self.__saved_config__ = self.config.convert_to_yaml_str(
                self.config, no_uuid=True, no_large_data=True)
            return None

    def log_config(self, path: Optional[str] = None) -> None:
        """Logs the config of the runner.
        """
        if self.__saved_config__ is None:
            self.store_config(path)
        logger.info(f"Using Config:\n{self.__saved_config__}")
