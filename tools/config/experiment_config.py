import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from tools.config.config import Config
from tools.config.output_config import OutputConfig
from datetime import datetime


@dataclass
class ExperimentConfig(OutputConfig):
    """Experiment base config for executable learning experiments."""

    name_experiment: str = field(default="Test")
    """Name of the experiment / agent. Agent will create a subdirectory for each experiment. Default is "Test"."""

    runs_path: str = field(default_factory=lambda: os.path.abspath("./runs/"))
    """Base directory where the runs are stored. Agent will create a subdirectory for each run. Default is ./runs/."""

    run_script_path: str = field(default=None)
    """Path to the run script. Saves the executable path of the script where the run was started with."""

    start_datetime: Optional[datetime] = field(default=None)
    """Datetime of the current execution. Will be set automatically."""

    def get_name(self) -> str:
        return self.name_experiment

    def get_runs_path(self) -> str:
        return self.runs_path

    @property
    def start_datetime_string(self) -> str:
        return self.start_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    def prepare(self) -> None:
        super().prepare()
        if self.start_datetime is None:
            self.start_datetime = datetime.now().astimezone()
