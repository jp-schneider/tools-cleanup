from pathlib import Path
from typing import Any, Mapping, Optional
from tools.agent.util.tracker import Tracker
from tools.logger.logging import logger
try:
    import pytorch_lightning as pl
    from lightning.pytorch.loggers.logger import Logger
    from lightning.pytorch.utilities import rank_zero_only
    from tools.util.path_tools import process_path
except ImportError:
    from tools.util.decorator import placeholder
    logger.warning(
        "Pytorch Lightning not installed. Pytorch Lightning related classes will not be available.")
    pl = None
    Logger = object
    rank_zero_only = placeholder


class TrackerLogger(Logger):

    experiment_directory: str

    tracker: Tracker

    def __init__(self,
                 experiment_directory: str | Path,
                 tracker: Optional[Tracker] = None,
                 ):
        super().__init__()
        self.experiment_directory = process_path(
            experiment_directory, make_exist=True, allow_none=False, variable_name="experiment_directory")
        self.tracker = tracker if tracker is not None else Tracker()

    @property
    def name(self):
        return "TrackerLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for key, value in metrics.items():
            self.tracker.step_metric(key, value, in_training=True, step=step)

    @rank_zero_only
    def save(self):
        self.tracker.save_to_directory(self.experiment_directory)

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
