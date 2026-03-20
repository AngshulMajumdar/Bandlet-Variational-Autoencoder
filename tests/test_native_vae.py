import torch
from bandlet_vae import BandletVAE, BandletVAEConfig
from bandlet_tf import BandletConfig


def test_bandlet_native_vae_forward_and_loss():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = BandletVAEConfig(
        image_size=32,
        in_channels=1,
        latent_dim=64,
        encoder_hidden_dims=(128, 64),
        decoder_hidden_dims=(64, 128),
        beta=1e-3,
        coeff_recon_weight=1.0,
        bandlet=BandletConfig(levels=2, block_size=8, auto_normalize_uint8=False, device=device),
        output_activation='identity',
    )
    model = BandletVAE(cfg).to(device)
    x = torch.randn(2, 1, 32, 32, device=device)
    out = model(x)
    assert out.x_hat.shape == x.shape
    assert out.mu.shape == (2, 64)
    assert out.logvar.shape == (2, 64)
    assert out.z.shape == (2, 64)
    losses = model.loss_function(x, out)
    assert set(losses.keys()) == {'loss', 'recon_loss', 'coeff_recon_loss', 'kl_loss'}
    assert torch.isfinite(losses['loss'])
