# Bandlet-native GPU VAE API

This package exposes a bandlet-native variational autoencoder whose encoder begins with bandlet analysis and whose decoder ends with bandlet synthesis.

Public API:
- `from bandlet_vae import BandletVAE, BandletVAEConfig, BandletCodec`
- `from bandlet_tf import BandletTransform, BandletConfig`

The latent model is built around coefficient vectors produced by batched bandlet analysis, not around raw image-space convolutions with an auxiliary bandlet penalty.
