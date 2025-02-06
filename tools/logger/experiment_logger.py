from abc import abstractmethod
import logging
from ast import Import
import time as pytime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tools.agent.agent import Agent
from tools.config.experiment_output_config import ExperimentOutputConfig
from tools.metric.metric_entry import MetricEntry
from tools.util.path import PATH_TYPE
from tools.util.typing import VEC_TYPE

try:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from torch.nn import Module
except (NameError, ImportError, ModuleNotFoundError) as err:
    torch = None
    Module = None
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
from tools.config.experiment_config import ExperimentConfig


class ExperimentLogger():
    """Abstract base class for experiment loggers."""

    def __init__(self,
                 name: str,
                 logging_directory: str = "./runs/",
                 output_path: Optional[PATH_TYPE] = None,
                 **kwargs) -> None:
        """Creating a logger.

        For output driven loggers its expected that there output is saved to [logging_directory]/[name].

        Parameters
        ----------
        name : str
            The name of the logger, typically corrensponding to the agent.
        logging_directory : str, optional
            The directory where the logs will be inserted to, by default "./runs/"
        """
        self.name = name
        self.logging_directory = logging_directory
        if output_path is None:
            output_path = os.path.join(logging_directory, name)
        self.output_path = output_path

    def finish(self):
        """Finishes the logger. May be overriden for cleanups etc."""
        pass

    @classmethod
    def for_torch_agent(cls,
                        agent: TorchAgent,
                        log_loss: bool = True,
                        additional_metrics: List[str] = None,
                        logging_directory: str = "./runs/",
                        log_optimizer: bool = True,
                        log_config: bool = True,
                        log_graph: bool = True,
                        log_config_only_once: bool = True,
                        name: Optional[str] = None,
                        ) -> 'ExperimentLogger':
        if name is None:
            base_name = os.path.basename(agent.agent_folder)
            folder_name = os.path.dirname(agent.agent_folder)
            name = os.path.basename(base_name)
            logging_directory = folder_name
        logger = cls(name=name, logging_directory=logging_directory)
        agent.logger = logger
        if log_loss:
            agent.batch_processed.attach(logger.log_loss)
            agent.epoch_processed.attach(logger.log_loss)
        if additional_metrics is not None:
            for metric in additional_metrics:
                agent.batch_processed.attach(logger.log_metric(metric))
                agent.epoch_processed.attach(logger.log_metric(metric))
        if log_optimizer:
            agent.epoch_processed.attach(logger.log_optimizer)
        if log_config:
            agent.epoch_processed.attach(
                logger.get_log_agent_config_handle(only_once=log_config_only_once))
        if log_graph:
            agent.batch_processed.attach(logger.log_graph())
        return logger

    def apply_to(self,
                 agent: TorchAgent,
                 log_loss: bool = True,
                 additional_metrics: List[str] = None,
                 log_optimizer: bool = True,
                 log_config: bool = True,
                 log_graph: bool = True,
                 log_config_only_once: bool = True,
                 ) -> 'ExperimentLogger':
        agent.logger = self
        if log_loss:
            agent.batch_processed.attach(self.log_loss)
            agent.epoch_processed.attach(self.log_loss)
        if additional_metrics is not None:
            for metric in additional_metrics:
                agent.batch_processed.attach(self.log_metric(metric))
                agent.epoch_processed.attach(self.log_metric(metric))
        if log_optimizer:
            agent.epoch_processed.attach(self.log_optimizer)
        if log_config:
            agent.epoch_processed.attach(
                self.get_log_agent_config_handle(only_once=log_config_only_once))
        if log_graph:
            agent.batch_processed.attach(self.log_graph())
        return self

    @classmethod
    def for_experiment_config(cls, config: ExperimentOutputConfig) -> 'ExperimentLogger':
        name = config.get_name()
        logging_directory = config.output_path
        project = config.project
        return cls(
            name=name,
            project=project,
            logging_directory=logging_directory)

    def log_loss(self, ctx: Dict[str, Any], output_args: ModelStepEventArgs):
        """Handler for logging the primary loss metric.

        Parameters
        ----------
        ctx : Dict[str, Any]
            The context dict
        output_args : ModelStepEventArgs
            The output args if the model step.
        """
        entry = output_args.tracker.get_recent_performance(scope=LearningScope.to_metric_scope(output_args.scope),
                                                           mode=LearningMode.to_metric_mode(output_args.mode))
        if entry is None:
            return
        tag = self.to_tag(entry.tag)
        value = entry.value
        if value is None:
            value = float("inf")
            logging.warning(f"Loss {tag} was None! Setting it to inf.")
        self.log_value(
            tag=tag,
            value=value,
            step=entry.global_step,
            time=output_args.time)

    def log_metric(self, metric_column: str) -> Callable:
        """Getting a logger function for the given metric name.

        Parameters
        ----------
        metric_column : str
            The metric to log.

        Returns
        -------
        Callable
            The callable event handler method.
        """
        notified = False

        def _log_metric(ctx: Dict[str, Any], output_args: ModelStepEventArgs):
            nonlocal self
            nonlocal notified
            entry = output_args.tracker.get_recent_metric(metric_column,
                                                          scope=LearningScope.to_metric_scope(
                                                              output_args.scope),
                                                          mode=LearningMode.to_metric_mode(output_args.mode))
            if entry is None:
                if not notified:
                    logging.warning(
                        f'Column {metric_column} non in metrics! Can not log to tensorboard.')
                    notified = True
                return
            self.log_metric_entry(entry, time=output_args.time)
        return _log_metric

    def log_metric_entry(self, entry: MetricEntry, time: Optional[float] = None):
        """Logs a metric entry to tensorboard.

        Parameters
        ----------
        entry : MetricEntry
            The metric entry to log.
        time : Optional[float], optional
            The time which should be attached., by default None
        """
        if time is None:
            time = pytime.time()
        tag = self.to_tag(entry.tag)
        value = entry.value
        self.log_value(value=value, tag=tag, step=entry.global_step, time=time)

    def log_value(self, value: Union[VEC_TYPE, int, float, complex], tag: str, step: int, time: Optional[float] = None, **kwargs):
        """Logs a numeric value to tensorboard.
        This can also be used for logging numpy arrays or tensors.
        If array or tensor is multi dimensional, then the dimensions will be flattened and logged as scalars.
        It will be assumed that the shape of the array or tensor is constant over time, otherwise overlapping occurs

        If the value is a complex number, then the real and imaginary part will be logged as separate scalars.

        Parameters
        ----------
        value : Union[VEC_TYPE, int, float, complex]
            Value to log. Can be a scalar, tensor or numpy array.
        tag : str
            The tag to log to.
        step : int
            The step to log to.
        time : Optional[float], optional
            Optional logging time, by default None
        """
        if isinstance(value, (torch.Tensor, np.ndarray)):
            self._log_vec_type(value=value, tag=tag, step=step, time=time)
        else:
            self._log_scalar(value=value, tag=tag, step=step, time=time)

    @abstractmethod
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
        pass

    def _log_scalar(self, value, tag: str, step: int, time: float):
        """Log method for simple scalars."""
        if isinstance(value, complex):
            self.add_scalar(
                tag + "-imag",
                value.imag,
                step=step,
                walltime=time)
            value = value.real
        self.add_scalar(
            tag,
            value,
            step=step,
            walltime=time)

    def _log_vec_type(self, value: torch.Tensor, tag: str, step: int, time: float):
        """Multi dimensional log for numpy arrays or tensors"""
        if len(value.shape) > 0:
            # Remove 1 dims
            value = value.squeeze()
        if len(value.shape) == 0:
            # Proceed as scalar
            self._log_scalar(value=value, tag=tag, step=step, time=time)
        else:
            # Multi dimensional tensor should be logged.
            # Dimensions will be flattend and getting individual steps
            flattened = value.flatten()
            for i, v in enumerate(flattened):
                # Assuming that previous steps had also same size,
                fl_step = max((step - 1), 0) * len(flattened) + i
                self._log_scalar(value=v, tag=tag, step=fl_step, time=time)

    def log_optimizer(self, ctx: Dict[str, Any], output_args: TorchModelStepEventArgs):
        if output_args.mode == LearningMode.TRAINING:
            params = self._get_optimizer_parameters(output_args.optimizer)
            for k, value in params.items():
                self.add_scalar(
                    k,
                    value,
                    step=output_args.tracker.global_epochs,
                    walltime=output_args.time)

    def _log_optimizer(self, optimizer, step: int, time: float):
        params = self._get_optimizer_parameters(optimizer)
        for k, value in params.items():
            self.add_scalar(
                k,
                value,
                step=step,
                walltime=time)

    @abstractmethod
    def add_scalar(self, tag: str, value: Union[int, float, complex], step: int, **kwargs):
        """Adds a scalar to tensorboard.

        Parameters
        ----------
        tag : str
            The tag of the scalar.
        value : Union[int, float, complex]
            The scalar value.
        step : int
            The global step.
        """
        pass

    @abstractmethod
    def add_graph(self, model: Any, input_to_model: Any, **kwargs):
        """Adds a graph to he logger.

        Parameters
        ----------
        model : torch.nn.Module
            The model to log.
        input_to_model : torch.Tensor
            The input to the model.
        """
        pass

    @abstractmethod
    def add_text(self, tag: str, text: str, step: int, **kwargs):
        """Adds text to the logger.

        Parameters
        ----------
        tag : str
            The tag of the text.
        text : str
            The text to log.
        step : int
            The global step.
        """
        pass

    def _infer_dtype_device(self, model: torch.nn.Module) -> Tuple[torch.dtype, str]:
        for p in model.parameters():
            if isinstance(p, torch.Tensor):
                return p.dtype, p.device
        raise ValueError(
            f"Could not get dtype and device from model {type(model).__name__} as parameters dont contain a tensor!")

    def log_graph(self) -> Callable[[Dict[str, Any], TorchModelStepEventArgs], None]:
        """Returns an executable which can be added to the epoch_processed event.

        Returns
        -------
        Callable[[Dict[str, Any], TorchModelStepEventArgs], None]
            The executable.
        """
        logged = False

        def _log_graph(ctx: Dict[str, Any], output_args: TorchModelStepEventArgs):
            nonlocal self
            nonlocal logged
            if (not logged):
                if output_args.scope != LearningScope.BATCH:
                    logging.warn(
                        "The model graph can be only logged in batch mode, as the input is needed!")
                    logged = True
                    return
                if output_args.input is None:
                    logging.warn("Input is not availabe, can not log graph!")
                    logged = True
                    return
                _dt, dev = self._infer_dtype_device(output_args.model)
                _input = output_args.input.to(dtype=_dt, device=dev)
                try:
                    self.add_graph(
                        output_args.model,
                        _input
                    )
                except Exception as e:
                    logging.exception(
                        f"Could not log graph for model {output_args.model} due to {e}")
                logged = True
        return _log_graph

    @abstractmethod
    def log_experiment_config(self, config: ExperimentOutputConfig, **kwargs):
        """Logs the experiment config to the logger.

        Parameters
        ----------
        config : ExperimentOutputConfig
            The config to log.
            Can be assumend to be json serializable.
        """
        pass

    def _format_md_json(self, obj) -> str:
        """Encodes an object to json and formats it for markdown display.

        Parameters
        ----------
        obj : Any
            Any Instance which can be encoded with an ObjectEncoder

        Returns
        -------
        str
            A json formatted string.
        """
        json_str = ObjectEncoder(indent=4, json_convertible_kwargs=dict(
            no_large_data=True, handle_unmatched="jsonpickle")).encode(obj)
        return self.json_to_md_format(json_str)

    @classmethod
    def json_to_md_format(cls, json_str: str) -> str:
        """Formats a json string for markdown display."""
        return "".join("\t" + l for l in json_str.splitlines(True))

    def log_hparams(self,
                    hparams: Dict[str, Any],
                    metric: Dict[str, Any],
                    step: int,
                    domain_discrete_values: Optional[Dict[str,
                                                          List[Any]]] = None,
                    run_name: Optional[str] = None,
                    key_prefix: Optional[str] = None
                    ):
        """Logs hyperparameters and metrics to the logger.

        Parameters
        ----------
        hparams : Dict[str, Any]
            The hyperparameters to log.
            Keys are the names of the hyperparameters. Should be unique in the run.
            Values are the values of the hyperparameters.
        metric : Dict[str, Any]
            The metrics to log.
            Keys are the names of the metrics. Should be unique in the run.
            Values are the values of the metrics.
        step : int
            The global step to log to.
        domain_discrete_values : Optional[Dict[str, List[Any]]], optional
            The domain of the discrete values, by default None
            E.g. If discrete values, like enums are used, then the domain can be specified.
        run_name : Optional[str], optional
            The name of the run, by default None
        key_prefix : Optional[str], optional
            A prefix for all keys in hparams, domain_discrete_values and metrics dict, by default None
        """
        if key_prefix is not None:
            hparams = {f"{key_prefix}{k}": v for k, v in hparams.items()}
            metric = {f"{key_prefix}{k}": v for k, v in metric.items()}
            if domain_discrete_values is not None:
                domain_discrete_values = {
                    f"{key_prefix}{k}": v for k, v in domain_discrete_values.items()}
        self.add_hparams(hparam_dict=hparams,
                         metric_dict=metric,
                         run_name=run_name,
                         hparam_domain_discrete=domain_discrete_values,
                         step=step)

    @abstractmethod
    def log_experiment_config(self,
                              config: ExperimentOutputConfig,
                              **kwargs):
        pass

    @abstractmethod
    def add_hparams(self,
                    hparam_dict: Dict[str, Any],
                    metric_dict: Dict[str, Any],
                    hparam_domain_discrete: Optional[Dict[str,
                                                          List[Any]]] = None,
                    run_name: Optional[str] = None,
                    step: Optional[int] = None, **kwargs):
        """Adds hyperparameters to the logger.

        Parameters
        ----------
        hparams : Dict[str, Any]
            The hyperparameters to log.
            Keys are the names of the hyperparameters. Should be unique in the run.
            Values are the values of the hyperparameters.
        metric : Dict[str, Any]
            The metrics to log.
            Keys are the names of the metrics. Should be unique in the run.
            Values are the values of the metrics.
        step : int
            The global step to log to.
        domain_discrete_values : Optional[Dict[str, List[Any]]], optional
            The domain of the discrete values, by default None
            E.g. If discrete values, like enums are used, then the domain can be specified.
        run_name : Optional[str], optional
            The name of the run, by default None
        key_prefix : Optional[str], optional
            A prefix for all keys in hparams, domain_discrete_values and metrics dict, by default None
        """
        pass

    def get_log_agent_config_handle(self, only_on_change: bool = True, only_once: bool = False) -> Callable[[Dict[str, Any], TorchModelStepEventArgs], None]:
        """Returns a executable for a epoch_processed event which will log the agent config.

        Parameters
        ----------
        only_on_change : bool, optional
            If true only logs the agent config if it has changed., by default True

        only_once : bool, optional
            If set, then the config will be logged only once. If true it will terminate imediatly

        Returns
        -------
        Callable[[Dict[str, Any], TorchModelStepEventArgs], None]
            The executable.
        """
        last_logged: str = None

        def _log(ctx: Dict[str, Any], output_args: ModelStepEventArgs):
            nonlocal last_logged
            nonlocal only_on_change
            nonlocal only_once
            if only_once and last_logged is not None:
                return
            if output_args.mode != LearningMode.TRAINING or output_args.scope != LearningScope.EPOCH:
                return
            agent: Agent
            agent = ctx.get('source')
            args = self._get_config(agent, output_args)
            json_str = self._format_md_json(args)
            if (not only_on_change) or (last_logged is None or json_str != last_logged):
                last_logged = json_str
                self.add_text(f"{type(agent).__name__} config", json_str,
                              step=output_args.tracker.global_epochs)
        return _log

    @abstractmethod
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
        pass

    @abstractmethod
    def add_figure(self, tag: str, figure: Any, step: int, **kwargs):
        """Adds a figure to the logger.

        Parameters
        ----------
        tag : str
            The tag of the figure.
        figure : Any
            The figure to log.
        step : int
            The global step.
        """
        pass

    def _get_config(self, agent: Agent, output_args: ModelStepEventArgs) -> Dict[str, Any]:
        """Gets the config for an agent.

        Parameters
        ----------
        agent : Agent
            The agent to get the config for.
        output_args : ModelStepEventArgs
            Output args of the model step.

        Returns
        -------
        Dict[str, Any]
            The config.
        """
        torch_args = {}
        if isinstance(output_args, TorchModelStepEventArgs):
            output_args: TorchModelStepEventArgs
            agent: TorchAgent
            torch_args = dict(
                optimizer=type(output_args.optimizer).__name__,
                optimizer_init_args=agent.optimizer_args,
                optimizer_params=self._get_optimizer_parameters(
                    output_args.optimizer),
                loss_args=agent.loss,
            )
        return dict(
            agent_name=agent.name,
            model=type(output_args.model).__name__,
            model_init_args=output_args.model_args,
            loss=output_args.loss_name,
            dataset_config={k: v for k, v in output_args.dataset_config.items(
            ) if 'indices' not in k.lower()},
            **torch_args
        )

    def _get_optimizer_parameters(self, optimizer: torch.optim.Optimizer, prefix: str = None) -> Dict[str, Any]:
        """Extract the optimizer parameters into a tag-dict which can be logged.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer
        prefix : str, optional
            A prefix tag, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary of optimizer values.
        """
        name = type(optimizer).__name__
        tag = f'{name}/'
        if prefix is not None:
            tag = prefix
        ret = {}
        it = 0
        for group_num, parameter_group in enumerate(optimizer.param_groups):
            t = tag
            if len(optimizer.param_groups) > 1:
                name = parameter_group.get('name', None)
                if name is None or len(name.strip()) == 0:
                    name = f'group_{group_num}'
                else:
                    name = name.strip()
                t = t + name + '-'
            for parameter in parameter_group:
                if parameter == "name":
                    continue
                if parameter == 'params':
                    continue
                if (isinstance(parameter_group[parameter], float) or
                        isinstance(parameter_group[parameter], int)):
                    ret[t + parameter] = parameter_group[parameter]
                elif isinstance(parameter_group[parameter], (tuple, list)):
                    for i, v in enumerate(parameter_group[parameter]):
                        if isinstance(v, float) or isinstance(v, int):
                            ret[t + parameter + '_' + str(i)] = v
        return ret

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
        return metric_tag
