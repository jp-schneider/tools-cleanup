from datetime import datetime
from typing import List, Optional, Tuple
from tools.run.multi_runner import MultiRunner
from tools.config.multi_config_config import MultiConfigConfig
from tools.error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
import os
import re
from tools.util.format import parse_format_string
from tools.util.path_tools import format_os_independent, relpath, replace_file_unallowed_chars, replace_unallowed_chars
from tools.logger.logging import logger


class MultiConfigRunner(MultiRunner):
    """Creates multiple runner based on multiple given configs."""

    config: MultiConfigConfig

    def __init__(self,
                 config: MultiConfigConfig,
                 **kwargs) -> None:
        super().__init__(config=config,
                         **kwargs)

    def scan_dir(self, directory, pattern, recursive: bool = False, depth: int = 100) -> List[str]:
        res = []
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"Config directory {directory} does not exist.")
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            if os.path.isfile(path):
                match = pattern.fullmatch(file)
                if match is not None:
                    res.append(path)
            elif os.path.isdir(path):
                if recursive and depth >= 0:
                    results = self.scan_dir(
                        path, pattern, recursive=recursive, depth=depth-1)
                    res.extend(results)
        return res

    def get_config_paths(self) -> List[str]:
        if self.config.mode == 'plain':
            return self.config.config_paths
        elif self.config.mode == 'scan_dir' or self.config.mode == 'scan_dir_recursive':
            ret = []
            directory = self.config.scan_config_directory
            pattern = re.compile(self.config.config_pattern)
            ret = self.scan_dir(directory, pattern, recursive=(
                self.config.mode == "scan_dir_recursive"))
            ret.sort()
            return ret
        else:
            raise ValueError(
                f"mode must be either 'plain' or 'scan_dir' but is {self.config.mode}")

    def save_child_configs(self, directory: str) -> List[str]:
        """Saves the configs of the child runners to a directory.

        Parameters
        ----------
        directory : str
            The directory where the configs will be saved.

        Returns
        -------
        List[str]
            The paths where the configs are stored.
        """
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save configs
        num_digits = len(str(len(self.child_configs)))
        num_fmt = f"{{:0{num_digits}d}}"
        paths = []
        bar = self.config.progress_factory.bar(
            desc="Saving child configs", total=len(self.child_configs), delay=5)
        for i, config in enumerate(self.child_configs):
            fmt = f"{num_fmt}_{config.get_name()}.yaml"
            base_name = fmt.format(i)
            base_name = replace_file_unallowed_chars(base_name)
            path = os.path.join(directory, base_name)
            path = config.save_to_file(
                path, no_uuid=True, no_large_data=True, override=True)
            paths.append(path)
            bar.update(1)
        return paths

    def create_jobs(self, ref_dir: Optional[str] = None, preset_output_folder: bool = False) -> List[Tuple[str, List[str]]]:
        import pandas as pd
        created_date = datetime.now(
        ) if self.config.child_config_creation_date is None else self.config.child_config_creation_date
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

        child_config_paths = self.save_child_configs(config_directory)
        runner_script_path = os.path.abspath(runner_script_path)

        rel_child_config_paths = [relpath(self.__jobsrefdir__, p,
                                          is_from_file=is_from_file) for p in child_config_paths]

        num_digits = len(str(len(self.child_configs)))
        num_fmt = f"{{:0{num_digits}d}}"

        items = []

        date_args = dict(
            year=created_date.year,
            month=created_date.month,
            day=created_date.day,
            hour=created_date.hour,
            minute=created_date.minute,
            second=created_date.second
        )
        successful_context = dict()
        skipped_rows = pd.DataFrame()
        bar = self.config.progress_factory.bar(
            desc="Creating jobs", total=len(self.child_configs), delay=5)
        for i, (_name, config_path) in enumerate(zip([x.get_name() for x in self.child_configs], rel_child_config_paths)):
            output_folder = None
            cfg = self.child_configs[i]
            if preset_output_folder:
                if cfg.output_folder is not None:
                    output_folder = cfg.output_folder
                else:
                    # name_experiment = f"#{num_fmt.format(i)}_{_name}"
                    # path = os.path.join(
                    #    self.child_configs[i].runs_path, name_experiment + "_" + created_at)
                    path = parse_format_string(
                        self.config.preset_output_folder_format_string,
                        [self.child_configs[i]],
                        allow_invocation=True,
                        additional_variables=date_args,
                        index_offset=i
                    )[0]
                    directory = os.path.dirname(path)
                    base_name = replace_unallowed_chars(
                        os.path.basename(path), allow_dot=False)
                    output_folder = format_os_independent(
                        os.path.join(directory, base_name))
            if self.config.skip_successful_executed:
                # Check if the output folder already exists and skip if it contains a success file
                if self._check_successful_executed(_name, output_folder, config=cfg, context=successful_context, log=False):
                    path = successful_context.get(
                        'current_successful_folder', None)
                    skipped_rows = pd.concat([skipped_rows, pd.DataFrame({
                        'name': [_name],
                        'path': [path],
                        'index': [i]
                    })], ignore_index=True)
                    continue

            item = self._generate_single_job(
                runner_script_path=runner_script_path,
                is_ref_dir_from_file=is_from_file,
                name=_name,
                config_path=config_path,
                output_folder=output_folder,
                name_argument=self.config.name_cli_argument
            )
            items.append(item)
            bar.update(1)

        if len(skipped_rows) > 0:
            # Log skipped rows
            logger.warning(
                f"Skipped {len(skipped_rows)} rows as they have already been executed successfully. Success rate: {len(skipped_rows)}/{len(self.child_configs)}")

        self.__jobs__ = items
        return items

    def build(self, build_children: bool = True, **kwargs) -> None:
        configs = self.get_config_paths()
        bar = self.config.progress_factory.bar(
            desc="Loading child runner configs", total=len(configs), delay=5)
        for config_path in configs:
            config = JsonConvertible.load_from_file(config_path)
            if hasattr(config, '__config_path__'):
                config.__config_path__ = config_path
            rnr = self.runner_type(config=config)
            rnr.parent = self
            self.child_runners.append(rnr)
            bar.update(1)

        # Build children
        if build_children:
            for runner in self.child_runners:
                runner.build()
