from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, mult_h: int, mult_w: int, mode: str = 'replicate') -> Tuple[torch.Tensor, Tuple[int, int]]:
    if x.ndim != 4:
        raise ValueError(f'Expected x with shape [B,C,H,W], got {tuple(x.shape)}')
    h, w = x.shape[-2:]
    pad_h = (mult_h - h % mult_h) % mult_h
    pad_w = (mult_w - w % mult_w) % mult_w
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x_pad, x_pad.shape[-2:]


def crop_to_shape(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    h, w = shape
    return x[..., :h, :w]


def extract_blocks_2d(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int], int, int]:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f'Expected x with shape [B,1,H,W], got {tuple(x.shape)}')
    orig_shape = x.shape[-2:]
    x_pad, padded_shape = pad_to_multiple(x, block_size, block_size)
    b, _, hp, wp = x_pad.shape
    nh, nw = hp // block_size, wp // block_size
    blocks = x_pad.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    blocks = blocks.contiguous().view(b, nh * nw, block_size, block_size)
    return blocks, orig_shape, padded_shape, nh, nw


def assemble_blocks_2d(blocks: torch.Tensor, padded_shape: Tuple[int, int], num_blocks_hw: Tuple[int, int], block_size: int) -> torch.Tensor:
    if blocks.ndim != 4:
        raise ValueError(f'Expected blocks with shape [B,N,b,b], got {tuple(blocks.shape)}')
    bsz, n, b1, b2 = blocks.shape
    nh, nw = num_blocks_hw
    hp, wp = padded_shape
    if b1 != block_size or b2 != block_size:
        raise ValueError('Block size mismatch.')
    if n != nh * nw:
        raise ValueError('Number of blocks does not match metadata.')
    x = blocks.view(bsz, nh, nw, block_size, block_size)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(bsz, 1, hp, wp)
    return x
