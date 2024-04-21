from typing import List
from tools.config.config import Config
from copy import deepcopy
import re
import os
from tools.error.argument_none_error import ArgumentNoneError
from tools.logger.logging import logger
from tools.util.format import parse_format_string

class ConfigTool():
    """Tool to create and save multiple configurations."""

    def __init__(self, base_config: Config) -> None:
        self.base_config = base_config
        self.child_configs = []


    @classmethod
    def from_iterables(cls, base_config: Config, **kwargs) -> 'ConfigTool':
        """Creates a ConfigTool from iterables.

        Parameters
        ----------
        base_config : Config
            The base config which will be used for the child configs.

        **kwargs
            The iterables which should be used to create the child configs.
            The keys are the names of the attributes of the base config which should be changed.
            The values are the iterables which should be used to change the attributes.

        Returns
        -------
        ConfigTool
            The created ConfigTool.
        """
        tool = cls(base_config)
        tool.add_iterables(**kwargs)
        return tool
    
    def set_by_format_string(self, target_variable:str, format_string: str) -> None:
        """Sets the target variable of the base config by a format string and other variables of the child configs.
        Can be used e.g. to set the name composed of other variables.

        Parameters
        ----------
        target_variable : str
            The name of the attribute of the base config which should be changed.
        
        format_string : str
            The format string which should be used to set the target variable.
            For documentation on the format string, see tools.util.format.parse_format_string.

        Raises
        ------
        ArgumentNoneError
            If target_variable or format_string is None.
        ValueError
            If the target_variable is not an attribute of the base config.
        """

        if target_variable is None:
            raise ArgumentNoneError("target_variable")
        if format_string is None:
            raise ArgumentNoneError("format_string")
        if not hasattr(self.base_config, target_variable):
            raise ValueError(f"Config has no attribute {target_variable}")
        if len(self.child_configs) == 0:
            logger.warning("No child configs created yet. Returning.")
            return
        target_values = parse_format_string(format_string, self.child_configs, index_variable="index")
        for i, value in enumerate(target_values):
            setattr(self.child_configs[i], target_variable, value)

    def add_iterables(self, **kwargs) -> None:
        """
        Creates child configs from iterables.
        and stores them in the child_configs list.

        Parameters
        ----------
        **kwargs
            The iterables which should be used to create the child configs.
            The keys are the names of the attributes of the base config which should be changed.
            The values should be iterables which should be used to change the attributes.
        
        Raises
        ------
        ValueError
            If no parameters are provided.
        """
        if len(kwargs) == 0:
            raise ValueError("No parameters provided.")
        if len(self.child_configs) > 0:
            logger.warning("Child configs already exist. Appending to created ones.")
        keys = kwargs.keys()
        values = kwargs.values()

        for args in zip(*values):
            config = deepcopy(self.base_config)
            for i, key in enumerate(keys):
                setattr(config, key, args[i])
            self.child_configs.append(config)

    
    def save(self, 
             path: str, 
             format_string: str = "{index}_{name}.yml",
             **kwargs
             ) -> List[str]:
        """Saves the child configs to files.


        Parameters
        ----------
        path : str
            The path / directory where the files should be stored.

        format_string : str, optional
            A custom format string to create the filenames, by default "{index}_{name}.yml"
            Every variable in the format string will be replaced by the corresponding value of the config.
            Format specified as {variable:formatter} can be used the format the variable with normal string formatting.
            
            Every property of the config can be used as a variable, in addition to "index" which is the index of the config in the list.

        **kwargs
            Additional keyword arguments which will be passed to the save method of the config.

        Returns
        -------
        List[str]
            Pathnames of the saved files.
        """
        file_names = parse_format_string(format_string, self.child_configs, index_variable="index")
        results = []
        for (file_name, config) in zip(file_names, self.child_configs):
            path = os.path.join(path, file_name)
            results.append(path)
            config.save_to_file(path, **kwargs)
        return results
