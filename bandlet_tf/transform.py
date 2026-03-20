from __future__ import annotations
from typing import Any, Dict, List, Tuple
import copy
import torch
import torch.nn.functional as F

from .blocks import assemble_blocks_2d, crop_to_shape, extract_blocks_2d, pad_to_multiple
from .config import BandletConfig
from .directional_ops import analyze_blocks, soft_threshold_packed, synthesize_blocks_with_spec
from .haar import HaarLevel, dwt2_haar, idwt2_haar
from .packing import export_template_meta, pack_encoded, unpack_encoded
from .stats import encoded_stats
from .types import EncodedBandlet, EncodedSubband


class BandletTransform:
    def __init__(self, config: BandletConfig | None = None):
        self.config = config or BandletConfig()
        self.device = self._resolve_device(self.config.device)
        self.dtype = getattr(torch, self.config.dtype)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _to_tensor(self, x) -> torch.Tensor:
        src_is_integer = isinstance(x, torch.Tensor) and not torch.is_floating_point(x)
        if isinstance(x, torch.Tensor):
            out = x.to(device=self.device, dtype=self.dtype)
        else:
            out = torch.tensor(x, device=self.device, dtype=self.dtype)
        if out.ndim == 2:
            out = out.unsqueeze(0).unsqueeze(0)
        elif out.ndim == 3:
            out = out.unsqueeze(1)
        if out.ndim != 4 or out.shape[1] != 1:
            raise ValueError(f'Expected image/tensor with shape [H,W], [B,H,W], or [B,1,H,W], got {tuple(out.shape)}')
        if self.config.auto_normalize_uint8 and out.numel() > 0:
            if src_is_integer or (torch.is_floating_point(out) and out.max() > 1.5):
                out = out / 255.0
        return out.contiguous()

    def _pad_image_for_haar(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        mult = 2 ** self.config.levels
        x_pad, _ = pad_to_multiple(x, mult, mult, mode=self.config.pad_mode_image)
        return x_pad, x.shape[-2:]

    def _encode_subband(self, band: torch.Tensor, level: int, subband_name: str) -> EncodedSubband:
        blocks, orig_shape, padded_shape, nh, nw = extract_blocks_2d(band, self.config.block_size)
        packed = analyze_blocks(blocks, self.config.angles)
        return EncodedSubband(
            level=level,
            subband=subband_name,
            orig_shape=orig_shape,
            padded_shape=padded_shape,
            num_blocks_h=nh,
            num_blocks_w=nw,
            block_size=self.config.block_size,
            num_angles=len(self.config.angles),
            packed=packed,
        )

    def _decode_subband(self, encoded: EncodedSubband) -> torch.Tensor:
        rec_blocks = synthesize_blocks_with_spec(encoded.packed, encoded.block_size, self.config.angles)
        band_pad = assemble_blocks_2d(
            rec_blocks,
            padded_shape=encoded.padded_shape,
            num_blocks_hw=(encoded.num_blocks_h, encoded.num_blocks_w),
            block_size=encoded.block_size,
        )
        return crop_to_shape(band_pad, encoded.orig_shape)

    def encode(self, x) -> EncodedBandlet:
        x = self._to_tensor(x)
        x_pad, orig_image_shape = self._pad_image_for_haar(x)
        approx, coeffs = dwt2_haar(x_pad, levels=self.config.levels)
        detail_bands: List[Tuple[EncodedSubband, EncodedSubband, EncodedSubband]] = []
        for i, level in enumerate(coeffs, start=1):
            detail_bands.append((
                self._encode_subband(level.lh, i, 'LH'),
                self._encode_subband(level.hl, i, 'HL'),
                self._encode_subband(level.hh, i, 'HH'),
            ))
        meta: Dict[str, Any] = {
            'orig_image_shape': tuple(orig_image_shape),
            'padded_image_shape': tuple(x_pad.shape[-2:]),
            'angles': tuple(self.config.angles),
            'block_size': self.config.block_size,
            'levels': self.config.levels,
        }
        return EncodedBandlet(approx=approx, detail_bands=detail_bands, meta=meta)

    def reconstruct(self, enc: EncodedBandlet) -> torch.Tensor:
        coeffs: List[HaarLevel] = []
        for lh_enc, hl_enc, hh_enc in enc.detail_bands:
            coeffs.append(HaarLevel(
                lh=self._decode_subband(lh_enc),
                hl=self._decode_subband(hl_enc),
                hh=self._decode_subband(hh_enc),
            ))
        x_pad = idwt2_haar(enc.approx, coeffs)
        return crop_to_shape(x_pad, tuple(enc.meta['orig_image_shape']))

    def threshold(self, enc: EncodedBandlet, tau) -> EncodedBandlet:
        out = copy.deepcopy(enc)
        for li, triplet in enumerate(out.detail_bands):
            new_triplet = []
            for sj, sub in enumerate(triplet):
                packed = soft_threshold_packed(sub.packed, tau=tau, keep_dc=self.config.keep_dc_on_threshold)
                new_triplet.append(EncodedSubband(
                    level=sub.level,
                    subband=sub.subband,
                    orig_shape=sub.orig_shape,
                    padded_shape=sub.padded_shape,
                    num_blocks_h=sub.num_blocks_h,
                    num_blocks_w=sub.num_blocks_w,
                    block_size=sub.block_size,
                    num_angles=sub.num_angles,
                    packed=packed,
                ))
            out.detail_bands[li] = tuple(new_triplet)
        return out

    def stats(self, enc: EncodedBandlet) -> dict:
        return encoded_stats(enc)

    def pack(self, enc: EncodedBandlet) -> torch.Tensor:
        return pack_encoded(enc)

    def export_template_meta(self, enc: EncodedBandlet) -> Dict[str, Any]:
        return export_template_meta(enc)

    def unpack(self, vec: torch.Tensor, template_meta: Dict[str, Any]) -> EncodedBandlet:
        # restore masks exactly from the current transform geometry, not as all-ones placeholders.
        enc = unpack_encoded(vec, template_meta, device=vec.device, dtype=vec.dtype)
        refreshed = []
        for triplet in enc.detail_bands:
            new_triplet = []
            for sub in triplet:
                from .directional_spec import get_packed_spec
                spec = get_packed_spec(sub.block_size, sub.block_size, self.config.angles, sub.packed.coeffs.device, sub.packed.coeffs.dtype)
                new_triplet.append(EncodedSubband(
                    level=sub.level,
                    subband=sub.subband,
                    orig_shape=sub.orig_shape,
                    padded_shape=sub.padded_shape,
                    num_blocks_h=sub.num_blocks_h,
                    num_blocks_w=sub.num_blocks_w,
                    block_size=sub.block_size,
                    num_angles=sub.num_angles,
                    packed=type(sub.packed)(
                        coeffs=sub.packed.coeffs,
                        valid_mask=spec.valid_mask,
                        coeff_mask=spec.coeff_mask,
                        line_count=spec.line_count,
                        line_len=spec.line_len,
                        tight_scale=sub.packed.tight_scale,
                    ),
                ))
            refreshed.append(tuple(new_triplet))
        enc.detail_bands = refreshed
        return enc
