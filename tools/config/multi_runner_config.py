from dataclasses import dataclass, field
from typing import Optional, Type, Union

from tools.config.experiment_config import ExperimentConfig
from tools.config.output_config import OutputConfig
from tools.util.path_tools import process_path
from datetime import datetime


@dataclass
class MultiRunnerConfig(ExperimentConfig):
    """Configuration for the MultiRunner. This runner is used to run multiple runners using one runner, usually in a sequential or parallel manner."""

    base_config: Optional[Union[str, OutputConfig]] = None
    """The base config which will be used for the child runners."""

    runner_type: str = None
    """The type of the runner which will be used for the child runners."""

    config_directory: str = field(default="./config")
    """The directory where the child configs will be stored."""

    config_filename_format: str = field(default="{name}_{index:02d}.yaml")
    """The format string for the child config filenames."""

    prune_child_experiment_time_on_save: bool = field(default=False)
    """If True, the child experiment time will be pruned when saving the child config."""

    runner_script_path: str = field(default="./run.py")
    """The path to the runner script."""

    create_job_file: bool = field(default=False)
    """If True, a job file will be created."""

    job_file_path: Optional[str] = field(default="temp/jobfiles")
    """The path to the job file. If None, a job file will be created in the config directory."""

    name_experiment: str = field(default="MultiRunnerConfig")
    """Name for the multi runner, this is usually not used."""

    skip_successfull_executed: bool = field(default=False)
    """If True, the runner will skip configs that have already been executed successfully. Will only work if the output directory is set in the config."""

    dry_run: bool = field(default=False)
    """If True, the runner will not execute training."""

    n_parallel: int = field(default=1)
    """Number of parallel executions."""

    child_config_creation_date: Optional[datetime] = field(default=None)
    """The date when the child configs were created. If None, the current date will be used."""

    preset_output_folder: bool = field(default=False)
    """If True, the output folder for child agents will be preset with a --output-folder argument."""

    preset_output_folder_format_string: str = field(
        default="{get_runs_path}/{index:02d}_{get_name}_{year}_{month}_{day}_{hour}_{minute}_{second}")
    """The format string for the preset output folder."""

    name_cli_argument: Optional[str] = field(default="--name")
    """The name of the command line argument for the experiment name."""

    @classmethod
    def multi_config_runner_type(cls) -> Type:
        """Returns the type of the multi config runner to handle the current config."""
        from tools.run.multi_config_runner import MultiConfigRunner
        return MultiConfigRunner

    def prepare(self) -> None:
        super().prepare()
        self.config_directory = process_path(self.config_directory, interpolate=True, make_exist=True,
                                             interpolate_object=self, variable_name="config_directory", allow_interpolation_invocation=True)
