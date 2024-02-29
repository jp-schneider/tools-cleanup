import os
from dataclasses import dataclass, field
from typing import Any, Dict
from tools.config.config import Config


@dataclass
class ExperimentConfig(Config):
    """Experiment base config for executable learning experiments."""

    name_experiment: str = field(default="Test")
    """Name of the experiment / agent. Agent will create a subdirectory for each experiment. Default is "Test"."""

    runs_path: str = field(default=os.path.abspath("./runs/"))
    """Base directory where the runs are stored. Agent will create a subdirectory for each run. Default is ./runs/."""

    output_folder: str = field(default=None)
    """Folder where all outputs are stored overrides runs_path. Default is None."""

    run_script_path: str = field(default=None)
    """Path to the run script. Saves the executable path of the script where the run was started with."""
