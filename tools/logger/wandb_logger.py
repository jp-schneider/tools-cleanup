from typing import Any, Dict, List, Optional, Union

import numpy as np
from tools.config.experiment_output_config import ExperimentOutputConfig
from tools.logger.experiment_logger import ExperimentLogger
from tools.logger.logging import logger
from typing import TYPE_CHECKING

import wandb

from wandb.sdk.wandb_run import Run

import os

from tools.util.path import PATH_TYPE


class WandbLogger(ExperimentLogger):

    run: Run
    """The wandb run object."""

    def __init__(self,
                 run: Run,
                 name: str,
                 logging_directory="./runs/",
                 output_path: Optional[str] = None,
                 ):
        super().__init__(name, logging_directory, output_path)
        self.run: Run = run

    def create(
            name: str,
            project: str,
            logging_directory: str = "./runs/",
            output_path: Optional[PATH_TYPE] = None,
            **kwargs) -> "WandbLogger":
        args = dict(kwargs)
        d = args.pop("dir", None)
        if d is not None:
            logger.warning(
                f"dir argument is automatically set to {logging_directory}/{name} and will be ignored.")

        if output_path is None:
            output_path = os.path.join(logging_directory, name)
        else:
            logging_directory = os.path.dirname(str(output_path))

        run = wandb.init(
            project=project,
            name=name,
            dir=str(output_path),
            **args)
        return WandbLogger(run, name, logging_directory, output_path=output_path)

    @classmethod
    def get_init_args_from_experiment_config(cls, config: ExperimentOutputConfig) -> Dict[str, Any]:
        # name = config.get_name()
        output_path = config.output_path
        name = os.path.basename(output_path)

        project = config.project
        args = dict(config.experiment_logger_kwargs)

        if project is None:
            project = os.environ.get("WANDB_PROJECT", None)
        if "project" in args and project is not None:
            new_project = args.pop("project")
            logger.warning(
                f"Project is set in the config or ENV as {project} and also specified in experiment_logger_kwargs. This will override the project to {new_project}."
            )
            project = new_project
        if "save_dir" in args:
            new_save_dir = args.pop("save_dir")
            logger.warning(
                f"Save dir is set in the config (output_path) as {output_path} and also specified in experiment_logger_kwargs. This will override the save dir to {new_save_dir}."
            )
            output_path = new_save_dir
        if "name" in args:
            new_name = args.pop("name")
            logger.warning(
                f"Name is set in the config as {name} and also specified in experiment_logger_kwargs. This will override the name to {new_name}."
            )
            name = new_name

        # Raise error if project is not set
        if project is None:
            raise ValueError(
                "Wandb project not set, neither in config nor in environment variables. Specify it as a project variable in the config or set the WANDB_PROJECT environment variable.")

        return dict(
            project=project,
            name=name,
            output_path=output_path,
            **args
        )

    @classmethod
    def for_experiment_config(
            cls,
        config: ExperimentOutputConfig,
        log_config: bool = True
    ) -> 'ExperimentLogger':

        setup_args = cls.get_init_args_from_experiment_config(config)
        if "reinit" not in setup_args:
            setup_args["reinit"] = True

        l = WandbLogger.create(
            **setup_args)

        if log_config:
            l.log_experiment_config(config)
        return l

    def log(self,
            data: Dict[str, Any],
            step: int,
            **kwargs):
        """Logs a dictionary to the logger.

        Parameters
        ----------
        data : Dict[str, Any]
            The data to log.
            Keys are the tags, values are the values to log.
        tag : str
            The tag to log to.
        step : int
            The step to log to.
        """
        data.update(self._get_default_entry(step, **kwargs))
        self.run.log(data)

    def add_scalar(self, tag: str, value: Union[int, float, complex], step: int, **kwargs):
        """Adds a scalar value to tensorboard.

        Parameters
        ----------
        tag : str
            The tag to log to.
        value : Union[int, float, complex]
            The scalar value to log.
        step : int
            The global step to log to.
        walltime : Optional[float], optional
            The walltime to log to, by default None
        """
        val = {tag: value}
        val.update(self._get_default_entry(step, **kwargs))
        self.run.log(val)

    def add_graph(self, model: Any, input_to_model: Any, **kwargs):
        """Adds a graph to tensorboard.

        Parameters
        ----------
        model : torch.nn.Module
            The model to log.
        input_to_model : torch.Tensor
            The input to the model.
        """
        logger.warning("Wandb graph logging not implemented.")

    def add_text(self, tag: str, text: str, step: int, **kwargs):
        """Adds text to tensorboard.

        Parameters
        ----------
        tag : str
            The tag of the text.
        text : str
            The text to log.
        step : int
            The global step.
        walltime : Optional[float], optional
            The walltime, by default None
        """
        val = {tag: text}
        val.update(self._get_default_entry(step, **kwargs))
        self.run.log(val)

    def add_hparams(self,
                    hparam_dict: Dict[str, Any],
                    metric_dict: Dict[str, Any],
                    hparam_domain_discrete: Optional[Dict[str,
                                                          List[Any]]] = None,
                    run_name: Optional[str] = None,
                    step: Optional[int] = None, **kwargs):
        pass

    def log_experiment_config(self, config: ExperimentOutputConfig, **kwargs):
        cfg = config.get_experiment_logger_logging_config()
        self.run.config.update(cfg)

    def _get_default_entry(
            self,
            step: int,
            epoch: Optional[int] = None,
            batch: Optional[int] = None,
            **kwargs) -> Dict[str, Any]:
        val = dict()
        if epoch is not None:
            val["epoch"] = epoch
        if batch is not None:
            val["batch"] = batch
        return val

    def add_image(self,
                  tag: str,
                  image: np.ndarray,
                  step: int,
                  **kwargs):
        """Adds an image to the logger.

        Parameters
        ----------
        tag : str
            The tag of the image.
        image : np.ndarray
            The image to log.
        step : int
            The global step.

        """
        val = {tag: wandb.Image(image)}
        val.update(self._get_default_entry(step, **kwargs))
        self.run.log(val)

    def add_figure(self,
                   tag: str,
                   figure: Any,
                   step: int,
                   dpi: int = 150,
                   **kwargs):
        """Adds a figure to wandb.

        Parameters
        ----------
        tag : str
            The tag to log to.
        figure : Any
            The figure to log.
        step : int
            The global step.
        """
        from matplotlib.pyplot import Figure
        from tools.viz.matplotlib import figure_to_numpy
        if not isinstance(figure, Figure):
            raise ValueError("Figure must be a matplotlib.pyplot.Figure.")
        image = figure_to_numpy(figure, dpi=dpi)
        self.add_image(tag, image, step)
