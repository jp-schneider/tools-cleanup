
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from tools.config.config import Config
from tools.config.output_config import OutputConfig
from datetime import datetime

from tools.util.path_tools import process_path


@dataclass
class ExperimentOutputConfig(OutputConfig):
    """Config which defines an actual output folder for experiments. This is heavily used when launch mechanisms need control over experiment outputs."""

    experiment_datetime: Optional[datetime] = field(default=None)
    """Datetime of the experiment. Will be set automatically."""

    name: str = field(default="Example")
    """Name of the experiment."""

    output_path: Union[str, Path] = field(
        default="runs/{year}-{month:02d}-{day:02d}/{experiment_datetime_string}_{name}")
    """Output directory for the final output."""

    experiment_logger: Literal["tensorboard",
                               "wandb"] = field(default="tensorboard")
    """Experiment logger to use. Default is tensorboard."""

    experiment_logger_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the experiment logger constructor. Unused for tensorboard."""

    project: str = field(default="sample_project")
    """Project name for the experiment. Used for wandb."""

    def get_experiment_name(self) -> str:
        """Gets the (formatted) experiment name.

        Returns
        -------
        str
            Name of the experiment
        """
        return self.name

    def get_name(self) -> str:
        return self.get_experiment_name()

    @property
    def experiment_datetime_string(self) -> str:
        return self.experiment_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def output_folder(self) -> str:
        return self.output_path

    @output_folder.setter
    def output_folder(self, value: str):
        self.output_path = value

    @property
    def year(self) -> int:
        return self.experiment_datetime.year

    @property
    def month(self) -> int:
        return self.experiment_datetime.month

    @property
    def day(self) -> int:
        return self.experiment_datetime.day

    @property
    def hour(self) -> int:
        return self.experiment_datetime.hour

    @property
    def minute(self) -> int:
        return self.experiment_datetime.minute

    @property
    def second(self) -> int:
        return self.experiment_datetime.second

    def prepare(self,
                create_output_path: bool = True,
                reevaluate_output_path: bool = True,
                ):
        super().prepare()
        if self.experiment_datetime is None:
            self.experiment_datetime = datetime.now().astimezone()

        self.output_path = process_path(self.output_path,
                                        make_exist=create_output_path,
                                        interpolate=True,
                                        interpolate_object=self,
                                        variable_name="output_path",
                                        reevaluate=reevaluate_output_path
                                        )

    def get_experiment_logger_logging_config(self) -> Dict[str, Any]:
        """
        Gets the dictionary which will be logged to the experiment logger.

        Returns
        -------
        Dict[str, Any]
            Logging configuration.
        """
        return self.to_json_dict(handle_unmatched="jsonpickle", no_uuid=True, no_large_data=True, no_context_paths=True)
