from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch


@dataclass
class PackedDirectionalCoeffs:
    coeffs: torch.Tensor          # [B, N, K, G, L]
    valid_mask: torch.Tensor      # [K, G, L]
    coeff_mask: torch.Tensor      # [K, G, L]
    line_count: int
    line_len: int
    tight_scale: float


@dataclass
class EncodedSubband:
    level: int
    subband: str
    orig_shape: Tuple[int, int]
    padded_shape: Tuple[int, int]
    num_blocks_h: int
    num_blocks_w: int
    block_size: int
    num_angles: int
    packed: PackedDirectionalCoeffs


@dataclass
class EncodedBandlet:
    approx: torch.Tensor
    detail_bands: List[Tuple[EncodedSubband, EncodedSubband, EncodedSubband]]
    meta: Dict[str, Any]
