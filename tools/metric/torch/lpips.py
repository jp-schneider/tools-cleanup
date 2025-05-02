from tools.logger.logging import logger
try:
    import torchmetrics
except ImportError:
    logger.warning("torchmetrics is not available. Please install torchmetrics to support LPIPS.")
from tools.metric.torch.reducible import Reducible
from typing import Any, Dict, Optional, Tuple, Union, Literal
import torch

class LPIPS(Reducible, torch.nn.Module):
    """Computes the LPIPS (Learned Perceptual Image Patch Similarity) metric.
    See: https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html
    """

    def __init__(self,
                 dim: Optional[Union[int, Tuple[int, ...]]] = None,
                 net_type: Literal["vgg", "alex", "squeeze"] = "alex",
                 normalize: bool = True,
                 reduction: Literal["sum", "mean", "none", "max", "min"] = "none",
                 name=None,
                    **kwargs):
        ok, dim, reduction = self.check_dim(dim, reduction)
        if not ok:
            raise ValueError(f"LPIPS only supports reduction of 4D tensors [B, C, H, W] along channel, height and width e.g. dim=([-4], -3, -2, -1), ([0], 1, 2, 3) or None. Got {dim}.")
        super().__init__(name=name, dim=dim, reduction=reduction, **kwargs)
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type=net_type, reduction="mean", normalize=normalize)

    def check_dim(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, reduction: str = "none") -> Tuple[bool, Tuple[int, ...], str]:
        if dim is None:
            return True, None, reduction
        d = set(dim)
        if -1 in d and -2 in d and -3 in d:
            if -4 in d or 0 in d:
                return True, 0, reduction
            else:
                return True, None, "none"
        elif 1 in d and 2 in d and 3 in d:
            if -4 in d or 0 in d:
                return True, 0, reduction
            else:
                return True, None, "none"
        else:
            return False, None, "none"

    def __call__(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = source.shape
        res = torch.zeros(B, dtype=source.dtype, device=source.device)
        for i in range(B):
            res[i] = self.lpips(source[i].unsqueeze(0), target[i].unsqueeze(0))
        return self.reduce(res)