import random
import sys
from typing import Any
import numpy as np


def seed_all(seed: int, deterministic: bool = False, warn_only: bool = False) -> None:
    """Seed all random number generators.

    Parameters
    ----------
    seed : int
        The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
