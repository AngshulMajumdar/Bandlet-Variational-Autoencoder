from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class BandletVAEOutput:
    x_hat: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    coeff_input: torch.Tensor
    coeff_hat: torch.Tensor


@dataclass
class PackedBandletBatch:
    vectors: torch.Tensor
    template_meta: dict
