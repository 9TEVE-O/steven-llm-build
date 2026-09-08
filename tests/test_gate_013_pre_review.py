"""Internal pre-review attacks for Gate 013 before external review."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from steven_llm.nn.layernorm import layer_norm


EPS = 1e-5


def test_pr01_float32_forward_and_backward_match_pytorch() -> None:
    """Match the independent PyTorch oracle on the practical default float32 path."""
    torch.manual_seed(913)
    x = torch.randn((4, 7), dtype=torch.float32, requires_grad=True)
    gamma = torch.randn(7, dtype=torch.float32, requires_grad=True)
    beta = torch.randn(7, dtype=torch.float32, requires_grad=True)
    upstream = torch.randn_like(x)

    actual = layer_norm(x, gamma, beta, EPS)
    (actual * upstream).sum().backward()
    actual_grads = (x.grad.detach().clone(), gamma.grad.detach().clone(), beta.grad.detach().clone())

    x_ref = x.detach().clone().requires_grad_(True)
    gamma_ref = gamma.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    expected = F.layer_norm(x_ref, (7,), gamma_ref, beta_ref, EPS)
    (expected * upstream).sum().backward()
    expected_grads = (x_ref.grad, gamma_ref.grad, beta_ref.grad)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)


def test_pr02_explicit_backward_passes_independent_gradcheck() -> None:
    """Pass PyTorch gradcheck without delegating LayerNorm computation to PyTorch."""
    torch.manual_seed(914)
    x = torch.randn((2, 3), dtype=torch.float64, requires_grad=True)
    gamma = torch.randn(3, dtype=torch.float64, requires_grad=True)
    beta = torch.randn(3, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda x_value, gamma_value, beta_value: layer_norm(
            x_value, gamma_value, beta_value, EPS
        ),
        (x, gamma, beta),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_pr03_extreme_finite_float16_range_remains_finite() -> None:
    """Avoid overflow at the largest finite signed float16 magnitudes."""
    x = torch.tensor([[-65504.0, 65504.0]], dtype=torch.float16)
    gamma = torch.ones(2, dtype=torch.float16)
    beta = torch.zeros(2, dtype=torch.float16)

    actual = layer_norm(x, gamma, beta, EPS)
    expected = F.layer_norm(x, (2,), gamma, beta, EPS)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
