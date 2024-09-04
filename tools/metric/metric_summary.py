import math
import os
import random
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, field

from tools.util.reflection import class_name
from .metric_entry import MetricEntry
import numpy as np
from tools.metric.metric_mode import MetricMode
from tools.metric.metric_scope import MetricScope
from tools.util.path_tools import replace_unallowed_chars
from tools.serialization.files.data_frame_csv_file_handle import DataFrameCSVFileHandle
import numpy as np
from tools.util.format import snake_to_upper_camel

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


def create_metric_df() -> pd.DataFrame:
    df = pd.DataFrame(
        columns=MetricEntry.df_fields())
    df.set_index("step", inplace=True)
    return df


class DoNotSet():
    pass


DO_NOT_SET = DoNotSet()

LABEL_TEXT_PATTERN = r"(?P<number>[0-9]+). (?P<text>.+)"


def random_circle_point(angle: Optional[int] = None,
                        radius: Optional[int] = None,
                        radius_min: int = 10,
                        radius_max: int = 80) -> Tuple[float, float]:
    if angle is None:
        angle = random.randint(0, 360)
    angle = angle % 360
    rad = angle * (math.pi / 180)
    if radius is None:
        radius = random.randint(radius_min, radius_max)
    return (math.cos(rad)*radius, math.sin(rad)*radius)


@dataclass
class MetricSummary():
    """A metric summary contains the data for one metric."""

    tag: str = field(default=None)
    """Tag for the metric. Should something which describes what the metric is."""

    values: pd.DataFrame = field(default_factory=create_metric_df)
    """The values for the current metric."""

    mode: MetricMode = field(default=MetricMode.VALIDATION)
    """The mode for this summary."""

    scope: MetricScope = field(default=MetricScope.EPOCH)
    """The scope for this summary."""

    is_primary: bool = False
    """Marking if this metric is the main metric for training, so the model will be optimized for this."""

    metric_qualname: str = field(default=None)
    """The fully qualifying name of the metric class / loss function. Will be used to compare metrics."""

    @property
    def metric_name(self) -> str:
        """Gets the metric name by splitting the tag.

        Returns
        -------
        str
            Metric name.
        """
        from tools.agent.util import Tracker
        return Tracker.split_tag(self.tag)[2]

    @property
    def metric_display_name(self) -> str:
        """
        Returns the display name for the metric."""
        return MetricSummary.get_metric_display_name(self.metric_name)

    @classmethod
    def get_metric_display_name(cls, metric_name: str) -> str:
        """Gets the display name for a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric as its used internally.

        Returns
        -------
        str
            A display version of the internal name.
        """
        if '_' in metric_name:
            metric_name = snake_to_upper_camel(metric_name, " ")
        return metric_name

    def process_value(self, value: Any) -> Any:
        if torch is not None:
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.item()
                else:
                    # Convert tensor to numpy array
                    value = value.detach().cpu().numpy()
        if np is not None:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return value.item()
        return value

    def log(self, step: int, value: Any, global_step: int):
        """Logs the given value in the value dataframe for the given metric.
        Key must already be inside.

        Parameters
        ----------
        step : int
            The current step of the metric / index.
        value : Any
            The actual metric value.
        global_step : int
            The global step.
        """
        err_count = 0
        done = False
        err = None
        # Theres is a weird error appearing setting a iterable value to an empty dataframe, which dissapears after a second try
        while (not done and err_count < 2):
            try:
                self.values.at[step, "value"] = self.process_value(value)
                self.values.at[step, "global_step"] = global_step
                done = True
            except ValueError as err:
                err_count += 1
        if not done:
            raise err

    def extend(self, amount: int):
        """Extends the current values by amount number of elements.
        Assumes, there is a consecutive ordering.

        Parameters
        ----------
        amount : int
            Number of elements to add.
        """
        m = - 1
        if len(self.values) > 0:
            m = max(self.values.index)
        appended = pd.concat([self.values, pd.DataFrame(
            index=np.arange(m + 1, m + 1 + amount))], ignore_index=True)
        self.values = appended

    def trim(self):
        """Removes entries whitch have no value associated.
        """
        self.values = self.values.drop(
            self.values[pd.isna(self.values['value'])].index, axis=0)

    def get_metric_entry(self, step: int) -> Optional[MetricEntry]:
        """Gets the metric entry at the given step.

        Parameters
        ----------
        step : int
            The step where the entry should be taken from.

        Returns
        -------
        Optional[MetricEntry]
            The metric entry at this point or none if it does not exists.
        """
        last = None
        if step in self.values.index:
            last = self.values.loc[step]
        if last is None:
            return None
        return MetricEntry.from_series(last,
                                       additional_data=dict(
                                           step=step,
                                           tag=self.tag,
                                           metric_qualname=self.metric_qualname))

    def _save_to_directory(self,
                           directory: str,
                           override: bool = True,
                           make_dirs: bool = True,
                           **kwargs
                           ) -> Dict[str, Any]:
        """Saves the metric to a directory.

        Parameters
        ----------
        directory : str
            The directory where the metric should be saved.
        step : int
            The step of the metric.
        """
        values = dict(vars(self))
        values["__class__"] = class_name(self)
        t_path = replace_unallowed_chars(self.tag) + ".csv"
        path = os.path.join(directory, t_path)
        values["values"] = DataFrameCSVFileHandle.for_object(values["values"], path,
                                                             override=override,
                                                             make_dirs=make_dirs)
        return values
