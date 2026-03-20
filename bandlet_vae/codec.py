from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch

from bandlet_tf import BandletTransform, EncodedBandlet, EncodedSubband, PackedDirectionalCoeffs
from bandlet_tf.config import BandletConfig


class BandletCodec:
    """Batched bandlet analysis/synthesis codec.

    This is the core native interface used by the VAE. It never packs the whole
    batch into one giant vector; instead it preserves a leading batch dimension
    and maps a batch of images to a batch of coefficient vectors [B, F].
    """

    def __init__(self, config: BandletConfig | None = None):
        self.config = config or BandletConfig()
        self.transform = BandletTransform(self.config)

    def encode_structured(self, x: torch.Tensor) -> EncodedBandlet:
        return self.transform.encode(x)

    def decode_structured(self, enc: EncodedBandlet) -> torch.Tensor:
        return self.transform.reconstruct(enc)

    def pack_batch(self, enc: EncodedBandlet) -> Tuple[torch.Tensor, Dict]:
        pieces: List[torch.Tensor] = [enc.approx.reshape(enc.approx.shape[0], -1)]
        detail_meta: List[List[Dict]] = []
        for triplet in enc.detail_bands:
            level_meta: List[Dict] = []
            for sub in triplet:
                coeffs = sub.packed.coeffs
                pieces.append(coeffs.reshape(coeffs.shape[0], -1))
                level_meta.append({
                    'level': sub.level,
                    'subband': sub.subband,
                    'orig_shape': tuple(sub.orig_shape),
                    'padded_shape': tuple(sub.padded_shape),
                    'num_blocks_h': int(sub.num_blocks_h),
                    'num_blocks_w': int(sub.num_blocks_w),
                    'block_size': int(sub.block_size),
                    'num_angles': int(sub.num_angles),
                    'coeff_shape': tuple(sub.packed.coeffs.shape[1:]),
                    'line_count': int(sub.packed.line_count),
                    'line_len': int(sub.packed.line_len),
                    'tight_scale': float(sub.packed.tight_scale),
                })
            detail_meta.append(level_meta)
        vectors = torch.cat(pieces, dim=1)
        template_meta = {
            'approx_shape': tuple(enc.approx.shape[1:]),
            'detail_meta': detail_meta,
            'meta': enc.meta,
        }
        return vectors, template_meta

    def unpack_batch(self, vectors: torch.Tensor, template_meta: Dict) -> EncodedBandlet:
        if vectors.ndim != 2:
            raise ValueError(f'vectors must have shape [B,F], got {tuple(vectors.shape)}')
        batch = vectors.shape[0]
        pos = 0

        approx_shape = tuple(template_meta['approx_shape'])
        approx_numel = int(torch.tensor(approx_shape).prod().item())
        approx = vectors[:, pos:pos + approx_numel].reshape(batch, *approx_shape)
        pos += approx_numel

        detail_bands = []
        for level_meta in template_meta['detail_meta']:
            triplet = []
            for sm in level_meta:
                coeff_shape = tuple(sm['coeff_shape'])
                coeff_numel = int(torch.tensor(coeff_shape).prod().item())
                coeffs = vectors[:, pos:pos + coeff_numel].reshape(batch, *coeff_shape)
                pos += coeff_numel
                from bandlet_tf.directional_spec import get_packed_spec
                spec = get_packed_spec(
                    int(sm['block_size']), int(sm['block_size']),
                    self.config.angles, coeffs.device, coeffs.dtype,
                )
                packed = PackedDirectionalCoeffs(
                    coeffs=coeffs,
                    valid_mask=spec.valid_mask,
                    coeff_mask=spec.coeff_mask,
                    line_count=int(sm['line_count']),
                    line_len=int(sm['line_len']),
                    tight_scale=float(sm['tight_scale']),
                )
                triplet.append(EncodedSubband(
                    level=int(sm['level']),
                    subband=str(sm['subband']),
                    orig_shape=tuple(sm['orig_shape']),
                    padded_shape=tuple(sm['padded_shape']),
                    num_blocks_h=int(sm['num_blocks_h']),
                    num_blocks_w=int(sm['num_blocks_w']),
                    block_size=int(sm['block_size']),
                    num_angles=int(sm['num_angles']),
                    packed=packed,
                ))
            detail_bands.append(tuple(triplet))

        if pos != vectors.shape[1]:
            raise ValueError('Vector length does not match template metadata.')
        return EncodedBandlet(approx=approx, detail_bands=detail_bands, meta=template_meta.get('meta', {}))

    def encode_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        enc = self.encode_structured(x)
        return self.pack_batch(enc)

    def decode_batch(self, vectors: torch.Tensor, template_meta: Dict) -> torch.Tensor:
        enc = self.unpack_batch(vectors, template_meta)
        return self.decode_structured(enc)
