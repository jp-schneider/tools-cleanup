from typing import Any, Union
from tools.config.config import Config
from tools.run.multi_runner import MultiRunner
from tools.config.grid_search_config import GridSearchConfig, MultiKey, MultiValue
import itertools
import copy
from tools.util.diff import changes, NOCHANGE


class GridSearchRunner(MultiRunner):
    """Creates multiple child runners by doing a cartesian product of the param_grid of the config."""

    def __init__(self,
                 config: GridSearchConfig,
                 **kwargs) -> None:
        super().__init__(config=config,
                         **kwargs)

    def _set_values(self,
                    config: Config,
                    key: Union[str, MultiKey],
                    value: Union[Any, MultiValue],
                    check_existing: bool = True
                    ) -> None:
        if isinstance(key, MultiKey) and not isinstance(value, MultiValue) or isinstance(value, MultiKey) and not isinstance(key, MultiValue):
            raise ValueError(
                f"Key and value must be of the same type. Key: {key}, Value: {value}")

        if isinstance(key, str) and not isinstance(value, MultiValue):
            if check_existing and not hasattr(config, key):
                raise ValueError(
                    f"Cant set value {value} for atrribute {key} in config type {type(config).__name__}. Attribute does not exist.")
            setattr(config, key, value)

    def build(self, build_children: bool = True, **kwargs) -> None:
        # Build the config for each child runner by doing a cartesian product of the param_grid and insert it in base config
        keys = self.config.param_grid.keys()
        values = self.config.param_grid.values()
        for value_per_key in itertools.product(*values):

            base_configs = self.base_config

            if not isinstance(self.base_config, list):
                base_configs = [self.base_config]
            for base_config in base_configs:
                # Copy base config
                config = copy.deepcopy(base_config)
                # Insert values
                for k, v_ in zip(keys, value_per_key):
                    setattr(config, k, v_)

                # Create child runner
                config.prepare()

                rnr = self.runner_type(config=config)
                # Create magic property diff-config to directly indicate the difference between the base config and the child config
                rnr.diff_config = dict()
                for k, v_ in zip(keys, value_per_key):
                    chg = changes(getattr(base_config, k), v_)
                    if chg != NOCHANGE:
                        rnr.diff_config[k] = chg
                rnr.parent = self
                rnr.config.diff_config = copy.deepcopy(rnr.diff_config)

                self.child_runners.append(rnr)
        # Build children
        if build_children:
            for runner in self.child_runners:
                runner.build()
