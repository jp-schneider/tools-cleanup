
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from tools.config.config import Config
from tools.config.output_config import OutputConfig
from datetime import datetime

from tools.util.path_tools import process_path

@dataclass
class ExperimentOutputConfig(OutputConfig):
    """Config which defines an actual output folder for experiments. This is heavily used when launch mechanisms need control over experiment outputs."""

    experiment_datetime: datetime = field(default_factory=datetime.now)
    """Datetime of the experiment. Will be set automatically."""

    name: str = field(default="Example")
    """Name of the experiment."""

    output_path: Union[Path, str] = field(
        default="runs/{year}-{month:02d}-{day:02d}/{experiment_datetime_string}_{name}")
    """Output directory for the final output."""

    
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
    
    def prepare(self):
        super().prepare()

        if self.experiment_datetime is None:
            self.experiment_datetime = datetime.now()

        self.output_path = process_path(self.output_path, 
                                        make_exist=True, 
                                        interpolate=True,
                                        interpolate_object=self, variable_name="output_path")