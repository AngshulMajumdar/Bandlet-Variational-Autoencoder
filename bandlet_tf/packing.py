from __future__ import annotations
from typing import Any, Dict, List, Tuple
import torch

from .types import EncodedBandlet, EncodedSubband, PackedDirectionalCoeffs


def export_template_meta(enc: EncodedBandlet) -> Dict[str, Any]:
    detail_meta: List[List[Dict[str, Any]]] = []
    for level_triplet in enc.detail_bands:
        level_meta: List[Dict[str, Any]] = []
        for sub in level_triplet:
            level_meta.append({
                'level': sub.level,
                'subband': sub.subband,
                'orig_shape': sub.orig_shape,
                'padded_shape': sub.padded_shape,
                'num_blocks_h': sub.num_blocks_h,
                'num_blocks_w': sub.num_blocks_w,
                'block_size': sub.block_size,
                'num_angles': sub.num_angles,
                'coeff_shape': tuple(sub.packed.coeffs.shape),
                'line_count': sub.packed.line_count,
                'line_len': sub.packed.line_len,
                'tight_scale': float(sub.packed.tight_scale),
            })
        detail_meta.append(level_meta)
    return {
        'approx_shape': tuple(enc.approx.shape),
        'detail_meta': detail_meta,
        'meta': enc.meta,
    }


def pack_encoded(enc: EncodedBandlet) -> torch.Tensor:
    pieces = [enc.approx.reshape(-1)]
    for triplet in enc.detail_bands:
        for sub in triplet:
            pieces.append(sub.packed.coeffs.reshape(-1))
    return torch.cat(pieces, dim=0)


def unpack_encoded(vec: torch.Tensor, template_meta: Dict[str, Any], device=None, dtype=None) -> EncodedBandlet:
    if device is None:
        device = vec.device
    if dtype is None:
        dtype = vec.dtype
    vec = vec.to(device=device, dtype=dtype)
    pos = 0

    approx_shape = tuple(template_meta['approx_shape'])
    approx_numel = int(torch.tensor(approx_shape).prod().item())
    approx = vec[pos:pos + approx_numel].view(*approx_shape)
    pos += approx_numel

    detail_bands = []
    for level_meta in template_meta['detail_meta']:
        triplet = []
        for sm in level_meta:
            coeff_shape = tuple(sm['coeff_shape'])
            coeff_numel = int(torch.tensor(coeff_shape).prod().item())
            coeffs = vec[pos:pos + coeff_numel].view(*coeff_shape)
            pos += coeff_numel
            k, g, ll = coeff_shape[2], coeff_shape[3], coeff_shape[4]
            valid_mask = torch.ones((k, g, ll), device=device, dtype=torch.bool)
            coeff_mask = torch.ones((k, g, ll), device=device, dtype=torch.bool)
            packed = PackedDirectionalCoeffs(
                coeffs=coeffs,
                valid_mask=valid_mask,
                coeff_mask=coeff_mask,
                line_count=sm['line_count'],
                line_len=sm['line_len'],
                tight_scale=float(sm['tight_scale']),
            )
            triplet.append(EncodedSubband(
                level=sm['level'],
                subband=sm['subband'],
                orig_shape=tuple(sm['orig_shape']),
                padded_shape=tuple(sm['padded_shape']),
                num_blocks_h=int(sm['num_blocks_h']),
                num_blocks_w=int(sm['num_blocks_w']),
                block_size=int(sm['block_size']),
                num_angles=int(sm['num_angles']),
                packed=packed,
            ))
        detail_bands.append(tuple(triplet))

    if pos != vec.numel():
        raise ValueError('Packed vector length does not match template metadata.')

    return EncodedBandlet(approx=approx, detail_bands=detail_bands, meta=template_meta.get('meta', {}))
