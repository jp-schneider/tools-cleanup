from abc import abstractmethod
from argparse import ArgumentParser
import os
from dataclasses import dataclass, field

from tools.mixin import ArgparserMixin
from tools.serialization import JsonConvertible
from tools.util.diff import changes, NOCHANGE
from typing import Any, Dict, List, Optional, Set
import logging

from tools.logger.logging import logger
from tools.util.path_tools import format_os_independent, relpath
from tools.util.progress_factory import ProgressFactory


@dataclass
class Config(JsonConvertible, ArgparserMixin):
    """Basic config for a runner."""

    diff_config: str = field(default=None)
    """When config is altered from another, this can be used to propagate diff values."""

    use_progress_bar: bool = field(default=True)
    """If a progressbar should be used."""

    progress_factory: Optional[ProgressFactory] = field(default=None)
    """Factory for the progress bar."""

    used_runner_type: str = field(default=None)
    """Type of the runner which was used to perform the experiment."""

    seed: int = field(default=42)
    """Seed for the initialization of the random number generator. Before large random operations in synthetic setting model."""

    __config_path__: Optional[str] = field(default=None)
    """Path to the config file. This is set automatically when the config is loaded from a file."""

    def __post_init__(self):
        if self.diff_config is not None and isinstance(self.diff_config, str):
            if os.path.exists(self.diff_config):
                self.diff_config = JsonConvertible.load_from_file(
                    self.diff_config)
            else:
                self.diff_config = JsonConvertible.from_json(self.diff_config)

    @classmethod
    def argparser_ignore_fields(cls) -> List[str]:
        fields = []
        fields.extend(super().argparser_ignore_fields())
        fields.append('__config_path__')
        return fields

    def compute_diff(self, other: 'Config') -> Dict[str, Any]:
        """Computes the differences of the current object to another.
        Result will be the changed properties from self to other.

        Parameters
        ----------
        other : Config
            The object to compare with.

        Returns
        -------
        Dict[str, Any]
            Changes. If no changes, dict will be empty
        """
        diff = changes(self, other)
        if diff == NOCHANGE:
            return dict()
        return diff

    def prepare(self) -> None:
        """Performs preparations on the config. Gets typically invoked by the training runners.
        """
        if self.diff_config is not None:
            if isinstance(self.diff_config, str) and os.path.exists(self.diff_config):
                try:
                    self.diff_config = JsonConvertible.load_from_file(
                        self.diff_config)
                except Exception as err:
                    logger.exception(
                        f"Could not load diff config from path {self.diff_config}")
        if self.use_progress_bar:
            if self.progress_factory is None:
                self.progress_factory = ProgressFactory()

    def after_decoding(self):
        super().after_decoding()
        if self.use_progress_bar:
            if self.progress_factory is None:
                self.progress_factory = ProgressFactory()

    @classmethod
    def parse_args(cls,
                   parser: ArgumentParser,
                   add_config_path: bool = True,
                   sep: str = "-",
                   ) -> "Config":
        """Parses the arguments from the command line and returns a config object.

        Parameters
        ----------
        parser : ArgumentParser
            Predefined parser object.
        add_config_path : bool, optional
            If the parse args method should consider a --config_path argument, by default True
        sep : str, optional
            Separator for the config path, by default "-"

        Returns
        -------
        Config
            The parsed config object.
        """
        config = super().parse_args(parser, add_config_path=add_config_path, sep=sep)
        config.prepare()
        return config

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the config object.

        Returns
        -------
        str
            The name of the config object.
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_runs_path(self) -> str:
        """Returns the path where the runs are stored.

        Returns
        -------
        str
            Path to the runs.
        """
        raise NotImplementedError("Method not implemented")

    def convert_relpaths(self, keys: List[str], base_dir: Optional[str] = None):
        """Convert paths specified by keys / properties to a relative path depending on base_dir.

        Parameters
        ----------
        keys : List[str]
            Keys / Properties of config containing path entries to convert.
        base_dir : Optional[str], optional
            Base directory, by default None
            If none, it will use the current cwd.
        """
        if base_dir is None:
            base_dir = os.getcwd()
        for key in keys:
            tp = getattr(self, key)
            if tp is None:
                continue
            if tp is not None:
                tp = format_os_independent(tp)
            has_ext = len(os.path.splitext(tp)[-1]) > 0
            rp = relpath(base_dir, tp, is_from_file=False, is_to_file=has_ext)
            rp = format_os_independent(rp)
            setattr(self, key, rp)

    def __ignore_on_iter__(self) -> Set[str]:
        s = super().__ignore_on_iter__()
        s.add("progress_factory")
        s.add("__config_path__")
        return s

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        del state["progress_factory"]
        del state["__config_path__"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add baz back since it doesn't exist in the pickle
        if self.use_progress_bar:
            if self.progress_factory is None:
                self.progress_factory = ProgressFactory()

    def __str__(self) -> str:
        yaml_str = self.to_yaml(no_uuid=True, no_large_data=True)
        return yaml_str
