
import torch.nn as nn
import torch

from typing import Literal, Tuple, Union, Dict, Optional
import numpy as np
from tools.transforms.to_numpy_image import numpyify_image
from tools.util.typing import VEC_TYPE
try:
    from sklearn.metrics import jaccard_score
except ImportError:
    jaccard_score = None


class IoU():
    """Calculates the intersection over union (IoU) for a given output and target, and mean over all images. Calculates it only w.r.t one class."""

    def __init__(
            self,
            noneclass: Optional[int] = None,
            noneclass_replacement: Optional[int] = None,
            average: Optional[Literal["binary", "macro",
                                      "micro", "weighted", "samples"]] = "binary",
            dtype: Optional[np.dtype] = np.float64,
    ) -> None:
        self.noneclass = noneclass
        self.noneclass_replacement = noneclass_replacement
        self.average = average
        if self.noneclass is not None and self.noneclass_replacement is None:
            self.noneclass_replacement = 0
        self.dtype = dtype

    def __call__(self, output: VEC_TYPE, target: VEC_TYPE, **kwargs) -> np.generic:
        """Calculates the intersection over union (IOU) for a given output and target.
        Supports only binary classification and shape must be (H, W) or (H, W, C).
        While in case of (H, W, C) the IOU will be considered accross all dimensions, including channel.

        Parameters
        ----------
        output : VEC_TYPE
            First input vector.
        target : VEC_TYPE
            Second input vector.

        Returns
        -------
        np.generic
            IOU value.
        """
        t = numpyify_image(target)
        o = numpyify_image(output)
        # Test if
        if self.noneclass is not None:
            t = np.where(t == self.noneclass, self.noneclass_replacement, t)
            # Changed to ignore noneclass in output also by target
            o = np.where(t == self.noneclass, self.noneclass_replacement, o)
        t = t.squeeze().reshape(np.prod(t.shape))
        o = o.squeeze().reshape(np.prod(t.shape))
        if np.all(t == 0.):
            return np.array(0., dtype=self.dtype)
        iou = jaccard_score(t, o, average=self.average)
        return iou
