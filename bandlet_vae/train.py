from __future__ import annotations
from typing import Dict, Iterable
import torch
from .model import BandletNativeVAE


def train_step(model: BandletNativeVAE, optimizer: torch.optim.Optimizer, x: torch.Tensor) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    losses = model.loss_function(x, out)
    losses['loss'].backward()
    optimizer.step()
    return {k: float(v.detach().cpu()) for k, v in losses.items()}


@torch.no_grad()
def eval_step(model: BandletNativeVAE, x: torch.Tensor) -> Dict[str, float]:
    model.eval()
    out = model(x)
    losses = model.loss_function(x, out)
    return {k: float(v.detach().cpu()) for k, v in losses.items()}
