from enum import Enum


class MetricScope(Enum):
    """Metric Scope Enum Class. Scopes are used to define the granularity of the metric. For example, a metric can be defined for each batch or for each epoch."""

    EPOCH = "epoch"
    """Epoch scope means usually across all samples within a dataset or subset."""

    BATCH = "batch"
    """Batch scope means usally across all samples within one batch."""

    @classmethod
    def display_name(cls, scope: 'MetricScope') -> str:
        """Returns a display name for the mode.

        Parameters
        ----------
        scope : MetricScope
            The scope to get the display name.

        Returns
        -------
        str
            Display name as str.
        """
        if scope == MetricScope.EPOCH:
            return "Epochs"
        if scope == MetricScope.BATCH:
            return "Batches"
        raise NotImplementedError()