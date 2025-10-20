
from typing import Optional, Dict, Type, Tuple, Any
from abc import abstractmethod
import torch
from tools.metric.metric import Metric as BaseMetric


METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _register_metric(name: str, _type: Type, kwargs: Dict[str, Any]) -> None:
    # Make sure rule registry is loaded
    global METRIC_REGISTRY
    if name.lower() in METRIC_REGISTRY:
        raise ValueError(f"Metric {name} already registered!")
    # Add to registry
    elem = dict(kwargs)
    elem["__class__"] = _type
    METRIC_REGISTRY[name.lower()] = elem


def get_metric(name: str) -> "Metric":
    """
    Get a registered metric by name.

    Parameters
    ----------
    name : str
        The name of the metric to retrieve.
        Name is case insensitive.

    Returns
    -------
    Metric
        The metric instance.
    """
    if name.lower() not in METRIC_REGISTRY:
        raise ValueError(
            f"Metric {name} not found in registry. Available metrics: {list(METRIC_REGISTRY.keys())}")
    elem = METRIC_REGISTRY[name.lower()]
    _type = elem["__class__"]
    kwargs = {k: v for k, v in elem.items() if k != "__class__"}
    return _type(**kwargs)


def register_metric(*metric_args: Tuple[str, Dict[str, Any]]):
    """
    Register a type for serialization and deserialization.

    Can be used as a decorator for Metric subclasses.

    Parameters
    ----------
    *metric_args : Tuple[str, Dict[str, Any]]
        Tuples of (name, kwargs) where name is the name to register the metric under and kwargs are the arguments to pass to the respective types constructor.
        Name must be unique.
    """
    def decorator(_type: Type) -> Type:
        if not issubclass(_type, Metric):
            raise ValueError("Can only register subclasses of Metric")
        for name, kwargs in metric_args:
            _register_metric(name, _type, kwargs)
        return _type
    return decorator


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
