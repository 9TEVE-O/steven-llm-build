"""Standalone Layer Normalisation primitive for Gate 013.

The forward and backward equations are implemented explicitly. PyTorch's
LayerNorm implementation is intentionally not used here; it is reserved for
tests as an independent reference oracle.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class _LayerNormFunction(torch.autograd.Function):
    """Explicit LayerNorm autograd primitive used by Gate 013."""

    @staticmethod
    def forward(ctx, x: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> Tensor:
        """Compute LayerNorm while accumulating half-precision statistics in float32."""
        _validate_inputs(x, gamma, beta, eps)

        accumulation_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )
        x_acc = x.to(dtype=accumulation_dtype)
        mean = x_acc.mean(dim=-1, keepdim=True)
        centred = x_acc - mean
        variance = centred.square().mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(variance + eps)
        x_hat_acc = centred * inv_std
        x_hat = x_hat_acc.to(dtype=x.dtype)
        output = x_hat * gamma + beta

        ctx.save_for_backward(x_hat_acc, inv_std, gamma)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Compute explicit gradients using the forward statistics accumulation dtype."""
        x_hat_acc, inv_std, gamma = ctx.saved_tensors
        feature_count = x_hat_acc.shape[-1]
        accumulation_dtype = x_hat_acc.dtype

        grad_output_acc = grad_output.to(dtype=accumulation_dtype)
        gamma_acc = gamma.to(dtype=accumulation_dtype)
        grad_x_hat = grad_output_acc * gamma_acc
        mean_grad = grad_x_hat.mean(dim=-1, keepdim=True)
        mean_grad_xhat = (grad_x_hat * x_hat_acc).mean(dim=-1, keepdim=True)
        grad_x_acc = inv_std * (grad_x_hat - mean_grad - x_hat_acc * mean_grad_xhat)

        if grad_output.ndim == 1:
            grad_gamma_acc = grad_output_acc * x_hat_acc
            grad_beta_acc = grad_output_acc
        else:
            reduce_dims = tuple(range(grad_output.ndim - 1))
            grad_gamma_acc = (grad_output_acc * x_hat_acc).sum(dim=reduce_dims)
            grad_beta_acc = grad_output_acc.sum(dim=reduce_dims)

        assert grad_x_acc.shape[-1] == feature_count
        grad_dtype = gamma.dtype
        return (
            grad_x_acc.to(dtype=grad_dtype),
            grad_gamma_acc.to(dtype=grad_dtype),
            grad_beta_acc.to(dtype=grad_dtype),
            None,
        )


def _validate_inputs(x: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> None:
    """Validate the frozen Gate 013 tensor, dtype, device and epsilon contract."""
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
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Apply LayerNorm over the final feature dimension only."""
    return _LayerNormFunction.apply(x, gamma, beta, eps)


class LayerNorm(nn.Module):
    """Learnable LayerNorm with explicit Gate 013 semantics."""

    def __init__(self, feature_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and > 0")

        self.feature_dim = feature_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply LayerNorm using the module's learned affine parameters."""
        return layer_norm(x, self.gamma, self.beta, self.eps)
