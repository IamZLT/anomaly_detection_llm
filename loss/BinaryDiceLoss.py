import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   [B, 1, H, W], abnormal probability
        target: [B, 1, H, W], 0/1 mask
        """
        b = target.shape[0]
        pred = pred.view(b, -1)
        target = target.view(b, -1)
        inter = (pred * target).sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()

