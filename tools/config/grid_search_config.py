from dataclasses import dataclass, field

from frozenlist import FrozenList
from tools.config.config import Config
from tools.config.multi_runner_config import MultiRunnerConfig
from typing import Dict, Any, List, Union
from tools.util.multi_dict import MultiDict


class MultiKey(FrozenList):
    """A MultiKey indicates that multiple keys are part of the parameter grid."""

    def __init__(self, keys: List[str]):
        super().__init__(keys)
        self.freeze()


class MultiValue(list):
    """A MultiValue indicates that multiple values are part of the parameter grid."""
    pass


@dataclass
class GridSearchConfig(MultiRunnerConfig):

    base_config: Union[str, Config, List[Config]] = None
    """The base config which will be used for the child runners. If it is a list, the cartesian product will be created for each config in the list."""

    param_grid: MultiDict = field(default_factory=dict)
    """The parameter grid which will be used to create the child configs."""

    name_experiment: str = field(default="GridSearch")

    def __post_init__(self):
        # if not isinstance(self.param_grid, MultiDict):
        #     self.param_grid = MultiDict(self.param_grid)
        pass
