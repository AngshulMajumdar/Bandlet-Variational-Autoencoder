from __future__ import annotations
from typing import Dict
import torch

from .types import EncodedBandlet


def encoded_stats(enc: EncodedBandlet) -> Dict[str, object]:
    total_blocks = 0
    total_coeffs = 0
    nonzero = 0
    num_subbands = 0
    for triplet in enc.detail_bands:
        for sub in triplet:
            coeffs = sub.packed.coeffs
            total_blocks += coeffs.shape[1]
            total_coeffs += coeffs.numel()
            nonzero += int((coeffs != 0).sum().item())
            num_subbands += 1
    approx_coeffs = enc.approx.numel()
    return {
        'levels': len(enc.detail_bands),
        'subbands': num_subbands,
        'total_blocks': total_blocks,
        'detail_coeffs': total_coeffs,
        'approx_coeffs': approx_coeffs,
        'total_coeffs_including_approx': total_coeffs + approx_coeffs,
        'nonzero_detail_coeffs': nonzero,
        'image_shape': tuple(enc.meta.get('orig_image_shape', ())),
        'approx_shape': tuple(enc.approx.shape),
        'device': str(enc.approx.device),
        'dtype': str(enc.approx.dtype).replace('torch.', ''),
        'angles': enc.meta.get('angles'),
        'block_size': enc.meta.get('block_size'),
    }
