from dataclasses import dataclass, field
from tools.config.config import Config
from tools.config.multi_runner_config import MultiRunnerConfig
from typing import Dict, Any, List, Union


class MultiKey(frozenset):
    """A MultiKey indicates that multiple keys are part of the parameter grid."""
    pass


class MultiValue(dict):
    """A MultiValue indicates that multiple values are part of the parameter grid."""
    pass


@dataclass
class GridSearchConfig(MultiRunnerConfig):

    base_config: Union[str, Config, List[Config]] = None
    """The base config which will be used for the child runners. If it is a list, the cartesian product will be created for each config in the list."""

    param_grid: Dict[Union[str, MultiKey],
                     Union[Any, MultiValue]] = field(default_factory=dict)
    """The parameter grid which will be used to create the child configs."""

    name_experiment: str = field(default="GridSearch")
