from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
from bandlet_tf import BandletConfig


@dataclass
class BandletNativeVAEConfig:
    image_size: int = 64
    in_channels: int = 1
    latent_dim: int = 256
    encoder_hidden_dims: Tuple[int, ...] = (1024, 512)
    decoder_hidden_dims: Tuple[int, ...] = (512, 1024)
    beta: float = 1e-3
    coeff_recon_weight: float = 1.0
    bandlet: BandletConfig = field(default_factory=BandletConfig)
    clamp_logvar_min: float = -10.0
    clamp_logvar_max: float = 10.0
    output_activation: str = 'sigmoid'

    def __post_init__(self) -> None:
        if self.in_channels != 1:
            raise ValueError('Current BandletNativeVAE implementation supports in_channels=1 only.')
        if self.image_size <= 0:
            raise ValueError('image_size must be positive.')
        if self.latent_dim <= 0:
            raise ValueError('latent_dim must be positive.')
        if not self.encoder_hidden_dims or not self.decoder_hidden_dims:
            raise ValueError('encoder_hidden_dims and decoder_hidden_dims must be non-empty.')
        if self.output_activation not in {'sigmoid', 'identity', 'tanh'}:
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', 'tanh'.")
