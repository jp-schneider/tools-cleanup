from enum import Enum


class MetricMode(Enum):
    """Metric Mode Enum Class."""

    TRAINING = "train"
    """Training Mode."""

    VALIDATION = "eval"
    """Validation / Inference Mode."""