from dataclasses import field

import sys
from typing import List, Type, Optional, Tuple

from matplotlib import pyplot as plt
from tools.config.config import Config
import os
from tools.context.script_execution import ScriptExecution
from tools.error import ArgumentNoneError
from tools.run.trainable_runner import TrainableRunner
from tools.util.format import parse_format_string
from tools.util.reflection import dynamic_import
from tools.config.multi_runner_config import MultiRunnerConfig
from datetime import datetime
from tools.serialization.json_convertible import JsonConvertible
from tools.util.path_tools import format_os_independent, relpath, replace_file_unallowed_chars, replace_unallowed_chars
from tools.run.config_runner import ConfigRunner
from tools.logger.logging import logger
from tools.util.format import parse_type, parse_format_string
import gc
import copy


class MultiRunner(TrainableRunner):
    """A runner which can run multiple child runners. Typically used to find the best hyperparameters."""

    child_runners: List[ConfigRunner]
    """Child runners which will be run / trained."""

    runner_type: Type
    """The type of the runner which will be used for the child runners."""

    base_config: Config
    """The base config which will be used to create the child configs."""

    __jobs__: List[Tuple[str, List[str]]]
    """List of jobs which will be executed."""

    __jobsrefdir__: str
    """Reference directory for the jobs."""

    __date_created__: str
    """Date when the runner was created."""

    def __init__(self,
                 config: MultiRunnerConfig,
                 **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        if config.runner_type is None:
            raise ArgumentNoneError("runner_type")
        rt = parse_type(config.runner_type, ConfigRunner,
                        variable_name="runner_type")
        self.runner_type = rt
        base_config = self.config.base_config
        if base_config is not None:
            if isinstance(base_config, str):
                base_config = JsonConvertible.load_from_file(base_config)
            if not isinstance(base_config, Config):
                raise TypeError("base_config must be a subclass of Config")
        else:
            base_config = self.config
        self.base_config = base_config
        self.child_runners = []
        self.__jobs__ = []
        self.__date_created__ = None
        self.__jobsrefdir__ = None

    @property
    def date_created(self) -> str:
        """Returns the date when the runner was created."""
        if self.__date_created__ is None:
            self.__date_created__ = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
        return self.__date_created__

    @property
    def child_configs(self) -> List[Config]:
        """Returns the configs of the child runners."""
        return [runner.config for runner in self.child_runners]

    def build(self, build_children: bool = True, **kwargs) -> None:
        pass

    def save_child_configs(self,
                           directory: str,
                           prefix: Optional[str] = None,
                           filename_format: Optional[str] = None,
                           purge_datetime: bool = False,
                           use_raw_context_paths: bool = True
                           ) -> List[str]:
        """Saves the configs of the child runners to a directory.

        Parameters
        ----------
        directory : str
            The directory where the configs will be saved.

        prefix : Optional[str], optional
            The prefix which will be added to the config names, by default None

        filename_format : Optional[str], optional
            The format string for the filenames, by default None
            Default: "config_{index}.yaml"
            Can be an interpolation string with support of all config properties.
            Example: "{name}_{index}.yaml"

        purge_datetime : bool, optional
            If the datetime should be purged from the config, by default False

        Returns
        -------
        List[str]
            The paths where the configs are stored.
        """
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        if filename_format is None:
            num_digits = len(str(len(self.child_configs)))
            num_fmt = f"{{index:0{num_digits}d}}"
            filename_format = (f"{prefix}_" if prefix is not None else "") + \
                f"{{name}}_{num_fmt}.yaml"

        # Save configs
        paths = []

        for i, config in enumerate(self.child_configs):
            base_name = parse_format_string(
                filename_format, [config], index_offset=i)[0]
            base_name = replace_file_unallowed_chars(base_name)
            path = os.path.join(directory, base_name)

            if purge_datetime and hasattr(config, "experiment_datetime"):
                config = copy.deepcopy(config)
                config.experiment_datetime = None

            path = config.save_to_file(
                path, no_uuid=True, override=True, use_raw_context_paths=use_raw_context_paths)
            paths.append(path)
        return paths

    def _check_successful_executed(self, name: str, output_folder: Optional[str] = None, log: bool = True) -> bool:
        if output_folder is not None and os.path.exists(output_folder):
            from tools.context.script_execution import load_exit_codes
            exit_codes = load_exit_codes(output_folder)
            if len(exit_codes) > 0 and 0 in exit_codes['code'].values:
                if log:
                    logger.info(
                        f"Skipping job for config {name} as it has already been executed successfully.")
                return True
        return False

    def create_job_file(self) -> str:
        """Creates a job file for slurm cluster.

        Returns
        -------
        str
            Path to the job file.
        """
        if self.config.config_directory is None:
            raise ArgumentNoneError("config_directory")
        config_directory = os.path.abspath(self.config.config_directory)
        preset_output_folder = self.config.preset_output_folder
        job_file_path = self.config.job_file_path

        if self.base_config is not None:
            exp_name_date = f"{self.base_config.get_name()}_{self.date_created}"
        else:
            exp_name_date = f"{self.config.get_name()}_{self.date_created}"

        # Create job file
        if job_file_path is None or (
                len(os.path.basename(job_file_path)) == 0) or (
                '.py' not in os.path.basename(job_file_path)):
            if job_file_path is None:
                job_file_path = config_directory
            job_file_path = os.path.join(
                job_file_path,
                f"JobFile_{exp_name_date}.py")
        if not os.path.exists(os.path.dirname(job_file_path)):
            os.makedirs(os.path.dirname(job_file_path))

        items = [str(x) for x in self.create_jobs(
            preset_output_folder=preset_output_folder)]
        formatted_items = (', ' + "\n" + '\t').join(items)
        content = (f"from typing import List, Tuple" + os.linesep +
                   "JOBS: List[Tuple[str, List[str]]] = [" + "\n\t" +
                   formatted_items + "\n" +
                   "]")
        with open(job_file_path, "w") as f:
            f.write(content)
        return job_file_path

    def create_jobs(self, ref_dir: Optional[str] = None, preset_output_folder: bool = False) -> List[Tuple[str, List[str]]]:
        created_date = datetime.now()
        created_at = created_date.strftime("%y_%m_%d_%H_%M_%S")
        is_from_file = ref_dir is not None

        if ref_dir is None:
            ref_dir = os.getcwd()
        ref_dir = os.path.abspath(ref_dir)
        if ref_dir != self.__jobsrefdir__:
            self.__jobsrefdir__ = ref_dir
            self.__jobs__ = None
        if self.__jobs__ is not None:
            return self.__jobs__
        if self.config.config_directory is None:
            raise ArgumentNoneError("config_directory")
        config_directory = os.path.abspath(self.config.config_directory)
        runner_script_path = os.path.abspath(self.config.runner_script_path)
        if not os.path.exists(runner_script_path):
            raise FileNotFoundError(
                f"Runner script not found at {runner_script_path}")

        exp_name_date = f"{self.base_config.get_experiment_name()}_{self.date_created}"

        paths = self.save_child_configs(config_directory, exp_name_date)
        runner_script_path = os.path.abspath(runner_script_path)

        rel_paths = [relpath(self.__jobsrefdir__, p,
                             is_from_file=is_from_file) for p in paths]

        items = []

        date_args = dict(
            year=created_date.year,
            month=created_date.month,
            day=created_date.day,
            hour=created_date.hour,
            minute=created_date.minute,
            second=created_date.second
        )
        for i, p in enumerate(rel_paths):
            output_folder = None
            experiment_name = f"{self.base_config.name_experiment}_{i}"

            if preset_output_folder:
                if self.child_configs[i].output_folder is not None:
                    output_folder = self.child_configs[i].output_folder
                else:
                    path = parse_format_string(
                        self.config.preset_output_folder_format_string,
                        [self.child_configs[i]],
                        allow_invocation=True,
                        additional_variables=date_args
                    )[0]
                    directory = os.path.dirname(path)
                    base_name = replace_unallowed_chars(
                        os.path.basename(path), allow_dot=False)
                    output_folder = format_os_independent(
                        os.path.join(directory, base_name))
            if self.config.skip_successfull_executed:
                # Check if the output folder already exists and skip if it contains a success file
                of = output_folder if output_folder is not None else self.child_configs[
                    i].output_folder
                if self._check_successful_executed(experiment_name, output_folder=of):
                    continue

            item = self._generate_single_job(
                runner_script_path=runner_script_path,
                is_ref_dir_from_file=is_from_file,
                name=experiment_name,
                config_path=p,
                output_folder=output_folder
            )
            items.append(item)

        self.__jobs__ = items
        return items

    def _generate_single_job(self,
                             runner_script_path: str,
                             is_ref_dir_from_file: bool,
                             name: str,
                             config_path: str,
                             output_folder: Optional[str] = None,
                             name_argument: str = "--name-experiment",
                             ) -> Tuple[str, List[str]]:
        exec_file = relpath(
            self.__jobsrefdir__, runner_script_path, is_from_file=is_ref_dir_from_file)
        args = [
            f"--config-path", format_os_independent(config_path),
            name_argument, name
        ]
        if output_folder is not None:
            args += [
                "--output-folder", format_os_independent(
                    os.path.normpath(output_folder))
            ]
        return (format_os_independent(exec_file), args)

    def child_runner_commands(self) -> List[Tuple[ConfigRunner, str]]:
        """Returns a list of tuples which contain the child runner and the command for the child runner in a seperate process.

        Returns
        -------
        List[Tuple[Runner, str]]
            List of subrunners and command to execute.
        """
        jobs = self.create_jobs()
        return list(zip(self.child_runners, [f'python {x} {" ".join(y)}' for x, y in jobs]))

    def train(self,
              *args,
              start: int = 0,
              end: Optional[int] = None,
              **kwargs):
        runner = self
        config = self.config
        status = dict()

        if end is not None:
            if end > 0:
                if end < start:
                    raise ValueError("End must be greater or equal to start.")
                if end > len(runner.child_runners):
                    raise ValueError(
                        "End must be smaller than the number of child runners.")
            else:
                end = len(runner.child_runners) - end
                if end < start:
                    raise ValueError("End must be greater or equal to start.")
        else:
            end = len(runner.child_runners)

        for i in range(start, end):
            child_runner = runner.child_runners[i]
            try:
                cfg = child_runner.config
                cfg.prepare()
                # Check if the child runner is already executed successfully
                if config.skip_successfull_executed:
                    if self._check_successful_executed(
                            cfg.get_name(),
                            output_folder=cfg.output_folder,
                            log=False):
                        status[i] = "skipped"
                        logger.info(
                            f"Skipping child runner #{i} as it has already been executed successfully.")
                        continue

                with ScriptExecution(cfg), plt.ioff():
                    gc.collect()
                    if "torch" in sys.modules:
                        import torch
                        torch.cuda.empty_cache()
                    plt.close('all')
                    logger.info(f"Building child runner #{i}...")
                    child_runner.build()
                    # Save config and log it
                    cfg_file = child_runner.store_config(cfg.output_folder)
                    child_runner.log_config()
                    logger.info(f"Stored config in: {cfg_file}")
                    logger.info(
                        f"Training with child runner #{i}")

                    child_runner.train()

                    logger.info(
                        f"Training done with child runner #{i}")

                    child_runner.finalize()
                    logger.info(
                        f"Finalized child runner #{i}")
                    status[i] = "success"
            except Exception as err:
                logger.exception(
                    f"Raised {type(err).__name__} in training child runner #{i}")
                status[i] = "Error: " + str(err)
        logger.info("All child runners executed.")
        status_msg = "\n".join([f"Runner #{i}: {status[i]}" for i in status])
        logger.info(f"Child runner status:\n{status_msg}")
        logger.info("MultiRunner finished.")
