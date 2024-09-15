
from typing import Optional
from abc import abstractmethod
import torch
from tools.metric.metric import Metric as BaseMetric

class Metric(BaseMetric):
    """Metric for torch tensors."""

    def __init__(self,
                 name: Optional[str] = None,
                 **kwargs
                 ) -> None:
        super().__init__(name=name, **kwargs)

    @abstractmethod
    def __call__(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the metric between the source and the target.

        Parameters
        ----------
        source : torch.Tensor
            The source, typically the prediction.

        target : Union[torch.Tensor, np.ndarray]
            The target or the ground truth.

        Returns
        -------
        torch.Tensor
            The calculated metric.
        """
        raise NotImplementedError()