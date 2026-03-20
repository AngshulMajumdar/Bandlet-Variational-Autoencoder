import torch
from bandlet_vae import BandletVAE, BandletVAEConfig
from bandlet_tf import BandletConfig


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = BandletVAEConfig(
        image_size=64,
        in_channels=1,
        latent_dim=128,
        encoder_hidden_dims=(256, 128),
        decoder_hidden_dims=(128, 256),
        beta=1e-3,
        coeff_recon_weight=1.0,
        bandlet=BandletConfig(levels=2, block_size=8, auto_normalize_uint8=False, device=device),
        output_activation='identity',
    )
    model = BandletVAE(cfg).to(device)
    x = torch.randn(4, 1, 64, 64, device=device)
    out = model(x)
    losses = model.loss_function(x, out)
    print(type(out))
    print('x_hat', out.x_hat.shape, out.x_hat.device)
    print('mu', out.mu.shape, out.mu.device)
    print('coeff_input', out.coeff_input.shape, out.coeff_input.device)
    print({k: float(v.detach().cpu()) for k, v in losses.items()})


if __name__ == '__main__':
    main()
