
from typing import Any, Optional
from abc import abstractmethod
import torch

from tools.serialization.json_convertible import JsonConvertible

class Metric(JsonConvertible):
    """Metric to calculate the performance of e.g. models."""

    name: Optional[str]
    """Some alternative name for the loss. By default it will be the class name."""

    def __init__(self,
                 name: Optional[str] = None,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.name = name
        """Some alternative name for the loss. By default it will be the class name."""


    @abstractmethod
    def __call__(self, source: Any, target: Any) -> Any:
        """Computes the metric between the source and the target.

        Parameters
        ----------
        source : Any
            The source, typically the prediction.

        target : Any
            The target or the ground truth.

        Returns
        -------
        Any
            The calculated metric.
        """
        raise NotImplementedError()

    def get_name(self) -> str:
        """Returns the name of the metric.

        Returns
        -------
        str
            The name, typically the class.
        """
        if self.name is not None:
            return self.name
        return type(self).__name__
    
    def __str__(self) -> str:
        return self.get_name()
    
    def __repr__(self) -> str:
        return self.get_name()