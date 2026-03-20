from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import BandletCodec
from .config import BandletNativeVAEConfig
from .types import BandletVAEOutput


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BandletNativeVAE(nn.Module):
    """Bandlet-native VAE.

    Native means the analysis operator sits at the front of the encoder and the
    synthesis operator sits at the end of the decoder:

        x -> Bandlet analysis -> coefficient vector -> stochastic bottleneck
          -> decoded coefficient vector -> Bandlet synthesis -> x_hat

    The latent variable therefore lives in a space learned from bandlet
    coefficients rather than from raw image pixels.
    """

    def __init__(self, cfg: BandletNativeVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.codec = BandletCodec(cfg.bandlet)

        with torch.no_grad():
            dev = self.codec.transform.device
            dummy = torch.zeros(1, cfg.in_channels, cfg.image_size, cfg.image_size, device=dev, dtype=self.codec.transform.dtype)
            coeffs, template_meta = self.codec.encode_batch(dummy)
        self.coeff_dim = int(coeffs.shape[1])
        self.register_buffer('_template_stub', torch.zeros(1), persistent=False)
        self.template_meta = template_meta

        hidden_enc = tuple(int(x) for x in cfg.encoder_hidden_dims)
        hidden_dec = tuple(int(x) for x in cfg.decoder_hidden_dims)
        self.input_norm = nn.LayerNorm(self.coeff_dim)
        self.encoder_mu = MLP(self.coeff_dim, hidden_enc, cfg.latent_dim)
        self.encoder_logvar = MLP(self.coeff_dim, hidden_enc, cfg.latent_dim)
        self.decoder_coeff = MLP(cfg.latent_dim, hidden_dec, self.coeff_dim)

    def encode_to_coeffs(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        return self.codec.encode_batch(x)

    def decode_from_coeffs(self, coeff_vectors: torch.Tensor, template_meta: Dict | None = None) -> torch.Tensor:
        template_meta = self.template_meta if template_meta is None else template_meta
        x_hat = self.codec.decode_batch(coeff_vectors, template_meta)
        if self.cfg.output_activation == 'sigmoid':
            x_hat = torch.sigmoid(x_hat)
        elif self.cfg.output_activation == 'tanh':
            x_hat = torch.tanh(x_hat)
        return x_hat

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        coeff_vectors, template_meta = self.encode_to_coeffs(x)
        coeff_vectors = self.input_norm(coeff_vectors)
        mu = self.encoder_mu(coeff_vectors)
        logvar = self.encoder_logvar(coeff_vectors)
        logvar = torch.clamp(logvar, min=self.cfg.clamp_logvar_min, max=self.cfg.clamp_logvar_max)
        return mu, logvar, coeff_vectors, template_meta

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_coeff(z)

    def forward(self, x: torch.Tensor) -> BandletVAEOutput:
        mu, logvar, coeff_vectors, template_meta = self.encode(x)
        z = self.reparameterize(mu, logvar)
        coeff_hat = self.decode(z)
        x_hat = self.decode_from_coeffs(coeff_hat, template_meta)
        return BandletVAEOutput(
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            z=z,
            coeff_input=coeff_vectors,
            coeff_hat=coeff_hat,
        )

    def loss_function(self, x: torch.Tensor, out: BandletVAEOutput, beta: float | None = None, coeff_recon_weight: float | None = None) -> Dict[str, torch.Tensor]:
        beta = self.cfg.beta if beta is None else beta
        coeff_recon_weight = self.cfg.coeff_recon_weight if coeff_recon_weight is None else coeff_recon_weight

        # target coefficient vectors are taken from a fresh bandlet analysis of x.
        coeff_target, _ = self.encode_to_coeffs(x)
        recon = F.mse_loss(out.x_hat, x)
        coeff_recon = F.mse_loss(out.coeff_hat, coeff_target)
        kl = 0.5 * (out.mu.pow(2) + out.logvar.exp() - out.logvar - 1.0).mean()
        loss = recon + coeff_recon_weight * coeff_recon + beta * kl
        return {
            'loss': loss,
            'recon_loss': recon.detach(),
            'coeff_recon_loss': coeff_recon.detach(),
            'kl_loss': kl.detach(),
        }
