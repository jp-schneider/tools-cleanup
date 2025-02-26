
from typing import Any, Optional
from abc import abstractmethod
import torch

from tools.serialization.json_convertible import JsonConvertible
from tools.metric.torch.metric import Metric

class Regularizer(Metric):
    """Regularizer to calculate a regularization term for e.g. models.
    This is a subclass of Metric, but it is not a metric in the traditional sense."""

    name: Optional[str]
    """Some alternative name for the loss. By default it will be the class name."""

    def __init__(self,
                 name: Optional[str] = None,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.name = name
        """Some alternative name for the regularizer. By default it will be the class name."""


    @abstractmethod
    def __call__(self, source: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the regularization based on the source.

        Parameters
        ----------
        source : torch.Tensor
            The source, typically the prediction.

        Returns
        -------
        torch.Tensor
            The calculated regularization.
        """
        raise NotImplementedError()
