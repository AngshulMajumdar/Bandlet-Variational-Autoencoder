from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Sequence, Tuple
import math
import torch


@lru_cache(maxsize=512)
def _dct_matrix(n: int) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    mat = torch.cos(math.pi / n * (i + 0.5) * k)
    mat[0] *= (1.0 / n) ** 0.5
    if n > 1:
        mat[1:] *= (2.0 / n) ** 0.5
    return mat


@lru_cache(maxsize=256)
def orientation_groups(n: int, m: int, angle_deg: float) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    theta = math.radians(angle_deg)
    ct, st = math.cos(theta), math.sin(theta)
    center_i = (n - 1) / 2.0
    center_j = (m - 1) / 2.0
    buckets: Dict[int, list[tuple[float, tuple[int, int]]]] = {}
    for i in range(n):
        for j in range(m):
            x = j - center_j
            y = i - center_i
            s = x * ct + y * st
            p = -x * st + y * ct
            key = int(round(p))
            buckets.setdefault(key, []).append((s, (i, j)))
    groups = []
    for key in sorted(buckets.keys()):
        pts = [ij for _, ij in sorted(buckets[key], key=lambda t: t[0])]
        if pts:
            groups.append(tuple(pts))
    return tuple(groups)


@dataclass(frozen=True)
class PackedOrientationSpec:
    indices: torch.Tensor      # [K,G,L]
    valid_mask: torch.Tensor   # [K,G,L]
    coeff_mask: torch.Tensor   # [K,G,L]
    dct_bank: torch.Tensor     # [K,G,L,L]
    synth_bank: torch.Tensor   # [K,G,L,L]
    group_count: torch.Tensor  # [K]
    line_count: int
    line_len: int
    angles: Tuple[float, ...]


@lru_cache(maxsize=128)
def _packed_spec_cpu(n: int, m: int, angles: Tuple[float, ...]) -> PackedOrientationSpec:
    k_angles = len(angles)
    groups_per_angle = [orientation_groups(n, m, float(angle)) for angle in angles]
    gmax = max(len(groups) for groups in groups_per_angle)
    lmax = max(len(group) for groups in groups_per_angle for group in groups)

    indices = torch.zeros((k_angles, gmax, lmax), dtype=torch.long)
    valid_mask = torch.zeros((k_angles, gmax, lmax), dtype=torch.bool)
    coeff_mask = torch.zeros((k_angles, gmax, lmax), dtype=torch.bool)
    dct_bank = torch.zeros((k_angles, gmax, lmax, lmax), dtype=torch.float64)
    synth_bank = torch.zeros((k_angles, gmax, lmax, lmax), dtype=torch.float64)
    group_count = torch.zeros((k_angles,), dtype=torch.long)

    dct_cache: Dict[int, torch.Tensor] = {}
    for k, groups in enumerate(groups_per_angle):
        group_count[k] = len(groups)
        for g, group in enumerate(groups):
            ll = len(group)
            idx = torch.tensor([i * m + j for i, j in group], dtype=torch.long)
            indices[k, g, :ll] = idx
            valid_mask[k, g, :ll] = True
            coeff_mask[k, g, :ll] = True
            if ll not in dct_cache:
                dct_cache[ll] = _dct_matrix(ll)
            d = dct_cache[ll]
            dct_bank[k, g, :ll, :ll] = d.t()
            synth_bank[k, g, :ll, :ll] = d

    return PackedOrientationSpec(
        indices=indices,
        valid_mask=valid_mask,
        coeff_mask=coeff_mask,
        dct_bank=dct_bank,
        synth_bank=synth_bank,
        group_count=group_count,
        line_count=gmax,
        line_len=lmax,
        angles=angles,
    )


_SPEC_DEVICE_CACHE: Dict[tuple[int, int, Tuple[float, ...], str, str], PackedOrientationSpec] = {}


def get_packed_spec(n: int, m: int, angles: Sequence[float], device: torch.device, dtype: torch.dtype) -> PackedOrientationSpec:
    angles_t = tuple(float(a) for a in angles)
    key = (n, m, angles_t, str(device), str(dtype))
    cached = _SPEC_DEVICE_CACHE.get(key)
    if cached is not None:
        return cached
    cpu = _packed_spec_cpu(n, m, angles_t)
    spec = PackedOrientationSpec(
        indices=cpu.indices.to(device=device),
        valid_mask=cpu.valid_mask.to(device=device),
        coeff_mask=cpu.coeff_mask.to(device=device),
        dct_bank=cpu.dct_bank.to(device=device, dtype=dtype),
        synth_bank=cpu.synth_bank.to(device=device, dtype=dtype),
        group_count=cpu.group_count.to(device=device),
        line_count=cpu.line_count,
        line_len=cpu.line_len,
        angles=cpu.angles,
    )
    _SPEC_DEVICE_CACHE[key] = spec
    return spec
