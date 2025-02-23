from typing import Any, Literal, Union, Dict
from tools.config.config import Config
from tools.run.multi_runner import MultiRunner
from tools.config.grid_search_config import GridSearchConfig, MultiKey, MultiValue
import itertools
import copy
from tools.util.diff import changes, NOCHANGE
from tools.util.reflection import get_nested_value, set_nested_value
from tools.util.typing import NOTSET
from tools.logger.logging import logger
class GridSearchRunner(MultiRunner):
    """
    Creates multiple child runners by doing a cartesian product of the param_grid of the config.
    Uses as base_config as the base config for the child runners, which will be modified by the param_grid to create the child configs.
    Supports setting nested properties in the config, then keys should be "paths" to the nested properties. E.g. "my_property.sub_property".
    This works for objects and dicts.

    If properties should not be treated as cartesian product, use MultiKey and MultiValue.

    E.g. 
    If 2 properties should be treated as cartesian product:
    ```
    param_grid = dict(
        my_property= [1, 2],
        my_property2= [3, 4]
    )
    ```
    Will create 4 child runners with the following properties: with the combinations of the values:
    ```
    my_property=1, my_property2=3
    my_property=1, my_property2=4
    my_property=2, my_property2=3
    my_property=2, my_property2=4
    ```

    If 2 properties should not be treated as cartesian product:
    ```
    param_grid = dict()
    param_grid[MultiKey(["my_property", "my_property2"])] = [MultiValue([1, 2]), MultiValue([3, 4])]
    ```

    Will create 2 child runners with the following properties:
    ```
    my_property=1, my_property2=3
    my_property=2, my_property2=4
    ```
    """

    def __init__(self,
                 config: GridSearchConfig,
                 **kwargs) -> None:
        super().__init__(config=config,
                         **kwargs)

    def _set_values(self,
                    config: Config,
                    key: Union[str, MultiKey],
                    value: Union[Any, MultiValue],
                    check_existing: bool = True,
                    non_existing_handling: Literal["raise", "warning"] = "raise"
                    ) -> None:
        if isinstance(key, MultiKey) and not isinstance(value, MultiValue) or isinstance(value, MultiKey) and not isinstance(key, MultiValue):
            raise ValueError(
                f"Key and value must be of the same type. Key: {key}, Value: {value}")

        if isinstance(key, str) and not isinstance(value, MultiValue):
            if check_existing and (get_nested_value(config, key) == NOTSET):
                if non_existing_handling == "warning":
                    logger.warning(
                        f"Attribute {key} does not exist in config type {type(config).__name__}.")
                else:
                    raise ValueError(
                        f"Cant set value {value} for atrribute {key} in config type {type(config).__name__}. Attribute does not exist.")
            set_nested_value(config, key, value)

        elif isinstance(key, MultiKey) and isinstance(value, MultiValue):
            if len(key) != len(value):
                raise ValueError(
                    f"Key and value for propert must have the same length. Key: {key} Key-len: {len(key)}, Value-len: {len(value)}")
            for k, v in zip(key, value):
                if check_existing and (get_nested_value(config, k) == NOTSET):
                    if non_existing_handling == "warning":
                        logger.warning(
                            f"Attribute {k} does not exist in config type {type(config).__name__}.")
                    else:
                        raise ValueError(
                            f"Cant set value {v} for attribute {k} in config type {type(config).__name__}. Attribute does not exist.")
                set_nested_value(config, k, v)
        else:
            raise ValueError(
                f"Inconsistent key and value types. Key: {key}, Value: {value}")

    def build(self, 
            build_children: bool = True, 
            config_prepare_kwargs: Dict[str, Any] = None,
            **kwargs) -> None:
        config_prepare_kwargs = config_prepare_kwargs if config_prepare_kwargs is not None else dict()
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
                    self._set_values(config, k, v_, non_existing_handling="warning")

                # Create child runner
                config.prepare(**config_prepare_kwargs)

                rnr = self.runner_type(config=config)
                # Create magic property diff-config to directly indicate the difference between the base config and the child config
                rnr.diff_config = dict()
                for k, v_ in zip(keys, value_per_key):
                    if isinstance(k, MultiKey):
                        for sub_key, sub_value in zip(k, v_):
                            chg = changes(get_nested_value(base_config, sub_key), sub_value)
                            if chg != NOCHANGE:
                                rnr.diff_config[sub_key] = chg
                    else:
                        chg = changes(get_nested_value(base_config, k), v_)
                        if chg != NOCHANGE:
                            rnr.diff_config[k] = chg
                rnr.parent = self
                rnr.config.diff_config = copy.deepcopy(rnr.diff_config)

                self.child_runners.append(rnr)
        # Build children
        if build_children:
            for runner in self.child_runners:
                runner.build()
