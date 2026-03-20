from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch


@dataclass
class HaarLevel:
    lh: torch.Tensor
    hl: torch.Tensor
    hh: torch.Tensor


def _haar_step_last(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = (x[..., 0::2] + x[..., 1::2]) / (2.0 ** 0.5)
    d = (x[..., 0::2] - x[..., 1::2]) / (2.0 ** 0.5)
    return a, d


def _ihaar_step_last(a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    out = torch.empty(a.shape[:-1] + (a.shape[-1] * 2,), dtype=a.dtype, device=a.device)
    out[..., 0::2] = (a + d) / (2.0 ** 0.5)
    out[..., 1::2] = (a - d) / (2.0 ** 0.5)
    return out


def dwt2_haar(x: torch.Tensor, levels: int = 1) -> Tuple[torch.Tensor, List[HaarLevel]]:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f'Expected x with shape [B,1,H,W], got {tuple(x.shape)}')
    cur = x
    coeffs: List[HaarLevel] = []
    for _ in range(levels):
        lo_r, hi_r = _haar_step_last(cur)
        lo = lo_r.transpose(-2, -1)
        hi = hi_r.transpose(-2, -1)
        ll_t, lh_t = _haar_step_last(lo)
        hl_t, hh_t = _haar_step_last(hi)
        ll, lh = ll_t.transpose(-2, -1), lh_t.transpose(-2, -1)
        hl, hh = hl_t.transpose(-2, -1), hh_t.transpose(-2, -1)
        coeffs.append(HaarLevel(lh=lh, hl=hl, hh=hh))
        cur = ll
    return cur, coeffs


def idwt2_haar(approx: torch.Tensor, coeffs: List[HaarLevel]) -> torch.Tensor:
    if approx.ndim != 4 or approx.shape[1] != 1:
        raise ValueError(f'Expected approx with shape [B,1,H,W], got {tuple(approx.shape)}')
    cur = approx
    for level in reversed(coeffs):
        ll_t, lh_t = cur.transpose(-2, -1), level.lh.transpose(-2, -1)
        hl_t, hh_t = level.hl.transpose(-2, -1), level.hh.transpose(-2, -1)
        lo = _ihaar_step_last(ll_t, lh_t).transpose(-2, -1)
        hi = _ihaar_step_last(hl_t, hh_t).transpose(-2, -1)
        cur = _ihaar_step_last(lo, hi)
    return cur
