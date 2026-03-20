from __future__ import annotations
from typing import Sequence
import math
import torch

from .directional_spec import get_packed_spec
from .types import PackedDirectionalCoeffs


def analyze_blocks(blocks: torch.Tensor, angles: Sequence[float]) -> PackedDirectionalCoeffs:
    if blocks.ndim != 4:
        raise ValueError(f'Expected blocks with shape [B,N,b,b], got {tuple(blocks.shape)}')
    bsz, nblocks, h, w = blocks.shape
    spec = get_packed_spec(h, w, angles, blocks.device, blocks.dtype)
    k = len(tuple(float(a) for a in angles))
    scale = 1.0 / math.sqrt(k)

    flat = blocks.reshape(bsz, nblocks, -1)
    gather_idx = spec.indices.reshape(1, 1, -1).expand(bsz, nblocks, -1)
    vals = flat.gather(2, gather_idx).reshape(bsz, nblocks, k, spec.line_count, spec.line_len)
    vals = vals * spec.valid_mask.view(1, 1, k, spec.line_count, spec.line_len)
    coeffs = torch.einsum('bnkgl,kglm->bnkgm', vals, spec.dct_bank) * scale
    coeffs = coeffs * spec.coeff_mask.view(1, 1, k, spec.line_count, spec.line_len)

    return PackedDirectionalCoeffs(
        coeffs=coeffs,
        valid_mask=spec.valid_mask,
        coeff_mask=spec.coeff_mask,
        line_count=spec.line_count,
        line_len=spec.line_len,
        tight_scale=scale,
    )


def synthesize_blocks(packed: PackedDirectionalCoeffs, block_size: int, angles: Sequence[float]) -> torch.Tensor:
    return synthesize_blocks_with_spec(packed, block_size=block_size, angles=angles)


def synthesize_blocks_with_spec(packed: PackedDirectionalCoeffs, block_size: int, angles: Sequence[float]) -> torch.Tensor:
    coeffs = packed.coeffs
    bsz, nblocks, k, g, ll = coeffs.shape
    spec = get_packed_spec(block_size, block_size, angles, coeffs.device, coeffs.dtype)
    coeffs = coeffs * spec.coeff_mask.view(1, 1, k, g, ll)
    vals = torch.einsum('bnkgm,kgml->bnkgl', coeffs, spec.synth_bank)
    vals = vals * spec.valid_mask.view(1, 1, k, g, ll) * packed.tight_scale

    out = torch.zeros((bsz, nblocks, block_size * block_size), device=coeffs.device, dtype=coeffs.dtype)
    scatter_idx = spec.indices.reshape(1, 1, -1).expand(bsz, nblocks, -1)
    scatter_vals = vals.reshape(bsz, nblocks, -1)
    out.scatter_add_(2, scatter_idx, scatter_vals)
    return out.reshape(bsz, nblocks, block_size, block_size)


def soft_threshold_packed(packed: PackedDirectionalCoeffs, tau, keep_dc: bool = True) -> PackedDirectionalCoeffs:
    coeffs = packed.coeffs.clone()
    if not torch.is_tensor(tau):
        tau_t = torch.tensor(tau, device=coeffs.device, dtype=coeffs.dtype)
    else:
        tau_t = tau.to(device=coeffs.device, dtype=coeffs.dtype)
    while tau_t.ndim < coeffs.ndim:
        tau_t = tau_t.unsqueeze(-1)
    shrink = torch.sign(coeffs) * torch.clamp(coeffs.abs() - tau_t, min=0.0)
    active = packed.coeff_mask.view(1, 1, *packed.coeff_mask.shape)
    if keep_dc:
        dc_mask = torch.zeros_like(packed.coeff_mask)
        dc_mask[..., 0] = packed.coeff_mask[..., 0]
        active = active & (~dc_mask.view(1, 1, *dc_mask.shape))
    coeffs = torch.where(active, shrink, coeffs)
    coeffs = coeffs * packed.coeff_mask.view(1, 1, *packed.coeff_mask.shape)
    return PackedDirectionalCoeffs(
        coeffs=coeffs,
        valid_mask=packed.valid_mask,
        coeff_mask=packed.coeff_mask,
        line_count=packed.line_count,
        line_len=packed.line_len,
        tight_scale=packed.tight_scale,
    )
