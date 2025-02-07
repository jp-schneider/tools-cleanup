import os
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Type, Optional, TypeVar, Union, get_type_hints

from tools.agent.agent import Agent
from tools.config.config import Config
from tools.config.experiment_config import ExperimentConfig
from tools.metric.metric_entry import MetricEntry
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
from tools.util.typing import MISSING


class AbstractRunner():
    """Abstract Runner, a runner can be setup "builded" and "run"."""

    __hints__: Dict[Type, Dict[str, Type]] = dict()
    """Stores type hints for runner classes"""

    __runner_context__: Dict[str, Any]
    """Context vars which are additionally declared within the runner.
    contain the actual value fields of the runner, while metadata is stored as usually in __dict__.
    These vars are not annotated with type hints and values are autmatically filled within if assigned to instance."""

    parent: Optional['AbstractRunner']
    """Parent runner if the runner is part of a multi runner."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.parent = None
        self.__runner_context__ = dict()

    @classmethod
    def _is_type_hinted_var(cls, name: str) -> bool:
        if cls not in cls.__hints__:
            cls.__hints__[cls] = get_type_hints(cls)
        hints = cls.__hints__[cls]
        return name in hints

    def __setattribute__(self, name: str, value: Any) -> None:
        if type(self)._is_type_hinted_var(name):
            object.__setattribute__(self, name, value)
        else:
            self.__runner_context__[name] = value

    def __getattr__(self, name: str) -> Any:
        if type(self)._is_type_hinted_var(name):
            return object.__getattribute__(self, name)
        else:
            val = self.__runner_context__.get(name, MISSING)
            if val == MISSING:
                raise AttributeError(f"{name} not found in {class_name(self)}")
            return val

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        """Method to build the runner and makes all necessary preparations.
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def finalize(self, *args, **kwargs) -> None:
        """Method to finalize the runner and makes all necessary cleanup.
        """
        for key, value in dict(self.__runner_context__):
            self.__runner_context__.pop(key)
            del value
