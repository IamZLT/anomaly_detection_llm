from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[float] = 0.25, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 2, H, W]
        target: [B, 1, H, W] or [B, H, W], values in {0, 1}
        """
        if target.dim() == 4:
            target = target[:, 0]
        probs = F.softmax(logits, dim=1).clamp(min=self.smooth, max=1.0 - self.smooth)
        target_onehot = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        pt = (probs * target_onehot).sum(dim=1)
        alpha_t = 1.0
        if self.alpha is not None:
            alpha_t = torch.where(target > 0, torch.full_like(pt, self.alpha), torch.full_like(pt, 1.0 - self.alpha))
        loss = -alpha_t * ((1.0 - pt) ** self.gamma) * torch.log(pt)
        return loss.mean()

