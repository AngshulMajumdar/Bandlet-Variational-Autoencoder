import torch
from bandlet_tf import BandletConfig, BandletTransform


def test_latent_roundtrip_cuda_or_cpu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bt = BandletTransform(BandletConfig(auto_normalize_uint8=False, device=device))
    z = torch.randn(2, 1, 64, 64, device=device)
    enc = bt.encode(z)
    z_rec = bt.reconstruct(enc)
    assert z_rec.shape == z.shape
    assert (z - z_rec).abs().max().item() < 1e-4
