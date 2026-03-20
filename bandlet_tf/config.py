from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

DEFAULT_ANGLES: Tuple[float, ...] = (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)


@dataclass(frozen=True)
class BandletConfig:
    levels: int = 2
    block_size: int = 8
    angles: Tuple[float, ...] = DEFAULT_ANGLES
    device: str = 'auto'
    dtype: str = 'float32'
    pad_mode_image: str = 'replicate'
    pad_mode_block: str = 'replicate'
    keep_dc_on_threshold: bool = True
    auto_normalize_uint8: bool = False
