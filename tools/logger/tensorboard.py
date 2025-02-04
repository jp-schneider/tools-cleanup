import logging
from ast import Import
import time as pytime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tools.agent.agent import Agent
from tools.config.experiment_output_config import ExperimentOutputConfig
from tools.metric.metric_entry import MetricEntry

try:
    import torch
    from torch.utils.tensorboard import SummaryWriter
except (NameError, ImportError, ModuleNotFoundError) as err:
    torch = None
    pass
import json
import logging
import os
import os.path
from pathlib import Path
import numpy as np

from tools.event import ModelStepEventArgs
from tools.event.torch_model_step_event_args import TorchModelStepEventArgs
from tools.agent.util import Tracker
from tools.agent.util import LearningMode, LearningScope
from tools.agent.torch_agent import TorchAgent
from tools.serialization import ObjectEncoder, JsonConvertible
from tools.logger.experiment_logger import ExperimentLogger


class Tensorboard(ExperimentLogger):
    """Tensorboard logging adapter with predefined methods.
    """

    @staticmethod
    def check_installed():
        try:
            sm = SummaryWriter
        except NameError:
            raise ImportError(
                "SummaryWriter could not be resolved. Is tensorboard installed?")

    summary_writer: SummaryWriter
    """The summary writer instance."""

    def __init__(self,
                 name: str,
                 logging_directory: str = "./runs/",
                 output_path: Optional[str] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 **kwargs) -> None:
        """Creating a tensorboard logger.

        Parameters
        ----------
        name : str
            The name of the logger, typically corrensponding to the agent.
        logging_directory : str, optional
            The directory where the logs will be inserted to, by default "./runs/"
        """
        Tensorboard.check_installed()
        super().__init__(
            name=name,
            logging_directory=logging_directory,
            output_path=output_path,
            **kwargs)
        if summary_writer is not None:
            self.summary_writer = summary_writer
        else:
            path = os.path.normpath(self.output_path)
            Path(path).mkdir(parents=True, exist_ok=True)
            logging.info(f"Tensorboard logger created at: {path}")
            self.summary_writer = SummaryWriter(path)

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
        for tag, value in data.items():
            self.add_scalar(tag, value, step,
                            walltime=kwargs.get("walltime", None))

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
        self.summary_writer.add_scalar(
            tag, value, step, walltime=kwargs.get("walltime", None))

    def add_graph(self, model: Any, input_to_model: Any, **kwargs):
        """Adds a graph to tensorboard.

        Parameters
        ----------
        model : torch.nn.Module
            The model to log.
        input_to_model : torch.Tensor
            The input to the model.
        """
        self.summary_writer.add_graph(model, input_to_model, verbose=kwargs.get(
            "verbose", False), use_strict_trace=kwargs.get("use_strict_trace", True))

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
        self.summary_writer.add_text(
            tag, text, step, walltime=kwargs.get("walltime", None))

    def add_hparams(self,
                    hparam_dict: Dict[str, Any],
                    metric_dict: Dict[str, Any],
                    hparam_domain_discrete: Optional[Dict[str,
                                                          List[Any]]] = None,
                    run_name: Optional[str] = None,
                    step: Optional[int] = None, **kwargs):
        self.summary_writer.add_hparams(
            hparam_dict, metric_dict, hparam_domain_discrete, run_name, global_step=step)

    def add_figure(self, tag: str, figure: Any, step: int, **kwargs):
        """Adds a figure to tensorboard.

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
        if not isinstance(figure, Figure):
            raise ValueError("Figure must be a matplotlib.pyplot.Figure.")
        self.summary_writer.add_figure(tag, figure, step, close=True)

    @classmethod
    def to_tensorboard_tag(cls, metric_tag: str) -> str:
        mode, scope, metric_name = Tracker.split_tag(metric_tag)
        return "/".join([metric_name, scope, mode])

    @classmethod
    def to_tag(cls, metric_tag: str) -> str:
        """Converts a tag to a logger tag.

        Parameters
        ----------
        tag : str
            The tag to convert.

        Returns
        -------
        str
            The converted tag.
        """
        return cls.to_tensorboard_tag(metric_tag)

    def log_experiment_config(self, config: ExperimentOutputConfig, **kwargs):
        """Logs the experiment configuration to tensorboard.

        Parameters
        ----------
        config : ExperimentOutputConfig
            The configuration to log.
        step : int
            The step to log to.
        """
        cfg = config.get_experiment_logger_logging_config()
        c = dict()
        c[type(config).__name__] = cfg
        _str = JsonConvertible.convert_to_yaml_str(
            c, handle_unmatched="jsonpickle", toplevel_wrapping=False, no_large_data=True, no_uuid=True)

        self.add_text("Config", _str, 0)

    def add_image(self,
                  tag: str,
                  image: np.ndarray,
                  step: int,
                  dataformats: str = "HWC",
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
        dataformats : str, optional
            The format of the image, by default "HWC"

        """
        self.summary_writer.add_image(
            tag, image, step, dataformats=dataformats)
