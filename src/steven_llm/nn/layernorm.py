"""Standalone Layer Normalisation primitive for Gate 013.

The forward and backward equations are implemented explicitly.  PyTorch's
LayerNorm implementation is intentionally not used here; it is reserved for
tests as an independent reference oracle.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> Tensor:
        _validate_inputs(x, gamma, beta, eps)

        mean = x.mean(dim=-1, keepdim=True)
        centred = x - mean
        variance = centred.square().mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(variance + eps)
        x_hat = centred * inv_std
        output = x_hat * gamma + beta

        ctx.save_for_backward(x_hat, inv_std, gamma)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_hat, inv_std, gamma = ctx.saved_tensors
        feature_count = x_hat.shape[-1]

        grad_x_hat = grad_output * gamma
        mean_grad = grad_x_hat.mean(dim=-1, keepdim=True)
        mean_grad_xhat = (grad_x_hat * x_hat).mean(dim=-1, keepdim=True)
        grad_x = inv_std * (grad_x_hat - mean_grad - x_hat * mean_grad_xhat)

        if grad_output.ndim == 1:
            grad_gamma = grad_output * x_hat
            grad_beta = grad_output
        else:
            reduce_dims = tuple(range(grad_output.ndim - 1))
            grad_gamma = (grad_output * x_hat).sum(dim=reduce_dims)
            grad_beta = grad_output.sum(dim=reduce_dims)

        assert grad_x.shape[-1] == feature_count
        return grad_x, grad_gamma, grad_beta, None


def _validate_inputs(x: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> None:
    if x.ndim < 1:
        raise ValueError("LayerNorm input must have rank >= 1")
    if not torch.is_floating_point(x):
        raise TypeError("LayerNorm input must be floating point")
    if gamma.ndim != 1 or beta.ndim != 1:
        raise ValueError("gamma and beta must both have shape (D,)")
    if gamma.shape != beta.shape:
        raise ValueError("gamma and beta must have identical shapes")
    if x.shape[-1] != gamma.shape[0]:
        raise ValueError("final input dimension must match gamma/beta size")
    if gamma.numel() < 1:
        raise ValueError("feature dimension D must be >= 1")
    if not torch.is_floating_point(gamma) or not torch.is_floating_point(beta):
        raise TypeError("gamma and beta must be floating point")
    if x.device != gamma.device or x.device != beta.device:
        raise ValueError("input, gamma and beta must be on the same device")
    if x.dtype != gamma.dtype or x.dtype != beta.dtype:
        raise ValueError("input, gamma and beta must have the same dtype")
    if eps <= 0:
        raise ValueError("eps must be > 0")


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Apply LayerNorm over the final feature dimension only."""

    return _LayerNormFunction.apply(x, gamma, beta, eps)


class LayerNorm(nn.Module):
    """Learnable LayerNorm with explicit Gate 013 semantics."""

    def __init__(self, feature_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        self.feature_dim = feature_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.gamma, self.beta, self.eps)
