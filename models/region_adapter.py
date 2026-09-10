"""Small trainable region-connection module: spatial cell evidence -> region tokens.

Each region candidate is split into 2x2 spatial cells (up to a configurable cap). For
every cell we have three pieces of content — the inspection feature, its matched
normal feature, and their difference — plus geometry and raw-H statistics. The adapter
projects these branches and adds them, so each output token expresses:

    a location's test content + its normal counterpart + the deviation between them.

The output dimension matches the language model's hidden size so the resulting tokens
can replace ``<|region|>`` placeholders in the input sequence. Empty/invalid cells are
masked to a learned ``empty`` embedding rather than fabricated as anomaly evidence.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class RegionAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        intermediate_dim: int = 256,
        geometry_dim: int = 5,
        hstat_dim: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.intermediate_dim = int(intermediate_dim)
        self.geometry_dim = int(geometry_dim)
        self.hstat_dim = int(hstat_dim)

        self.test_proj = nn.Linear(self.feature_dim, self.intermediate_dim)
        self.normal_proj = nn.Linear(self.feature_dim, self.intermediate_dim)
        self.diff_proj = nn.Linear(self.feature_dim, self.intermediate_dim)
        self.geometry_proj = nn.Linear(self.geometry_dim, self.intermediate_dim)
        self.score_proj = nn.Linear(self.hstat_dim, self.intermediate_dim)
        self.output_proj = nn.Linear(self.intermediate_dim, self.hidden_size)
        self.empty = nn.Parameter(torch.zeros(self.hidden_size))
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        nn.init.normal_(self.empty, std=0.02)

    def forward(self, region_raw: Dict[str, torch.Tensor]) -> torch.Tensor:
        """region_raw: {test, ref, geom, hstat, valid}; returns [B, n_cells, hidden]."""
        test = region_raw["test"]
        ref = region_raw["ref"]
        geom = region_raw["geom"]
        hstat = region_raw["hstat"]
        valid = region_raw["valid"]

        content = (
            self.test_proj(test)
            + self.normal_proj(ref)
            + self.diff_proj(test - ref)
        )
        geo = self.geometry_proj(geom)
        score = self.score_proj(hstat)
        token = self.output_proj(self.act(content + geo + score))

        valid_mask = valid.unsqueeze(-1).to(token.dtype)
        empty = self.empty.reshape(1, 1, self.hidden_size).to(token.dtype)
        return valid_mask * token + (1.0 - valid_mask) * empty

    @property
    def config_dict(self) -> dict:
        return dict(
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            intermediate_dim=self.intermediate_dim,
            geometry_dim=self.geometry_dim,
            hstat_dim=self.hstat_dim,
        )

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, hidden_size={self.hidden_size}, "
            f"intermediate_dim={self.intermediate_dim}, geometry_dim={self.geometry_dim}, "
            f"hstat_dim={self.hstat_dim}"
        )


def build_region_adapter(cfg: dict, feature_dim: int, hidden_size: int) -> RegionAdapter:
    """Build a RegionAdapter from ``outcome.region`` config."""
    rc = (cfg.get("outcome", {}) or {}).get("region", {}) or {}
    return RegionAdapter(
        feature_dim=int(feature_dim),
        hidden_size=int(hidden_size),
        intermediate_dim=int(rc.get("intermediate_dim", 256)),
        geometry_dim=int(rc.get("geometry_dim", 5)),
        hstat_dim=int(rc.get("hstat_dim", 2)),
        activation=str(rc.get("activation", "gelu")),
    )
