# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.kda.fused_norm_gate import (
    apply_kda_rms_norm_sigmoid_gate,
    layer_norm_gated_fwd,
)


def _rms_norm_sigmoid_gate_reference(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference for [1, T, H, D] x and [T, H, D] gate (fp32 math, like the kernel)."""
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    expected = x_float * torch.rsqrt(variance + eps)
    expected = expected * weight.float()
    expected = expected * gate.float().sigmoid().unsqueeze(0)
    return expected.to(x.dtype)


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kimi_kda_fused_rms_norm_sigmoid_gate(dtype: torch.dtype):
    torch.manual_seed(20260801)
    tokens, heads, head_dim = 37, 4, 128
    eps = 1e-6
    core_attn_out = torch.randn(
        1,
        tokens,
        heads,
        head_dim,
        dtype=dtype,
        device="npu",
    )
    output_gate = torch.randn(
        tokens,
        heads,
        head_dim,
        dtype=dtype,
        device="npu",
    )
    weight = torch.randn(head_dim, dtype=dtype, device="npu")
    core_attn_out_before = core_attn_out.clone()

    actual = apply_kda_rms_norm_sigmoid_gate(
        core_attn_out,
        output_gate,
        weight,
        eps,
    )

    x_float = core_attn_out_before.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    expected = x_float * torch.rsqrt(variance + eps)
    expected = expected * weight.float()
    expected = expected * output_gate.float().sigmoid().unsqueeze(0)
    expected = expected.to(dtype)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(core_attn_out, core_attn_out_before)


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kimi_kda_fused_rms_norm_sigmoid_gate_zeroes_padding_rows(dtype: torch.dtype):
    """num_valid_rows < tokens: the kernel must zero the graph-padding rows.

    Padding rows are filled with NaN so any failure to zero them inside the
    kernel (the folded _zero_padded_spec_output behavior) surfaces as NaN in
    the output instead of silently matching a garbage reference.
    """
    torch.manual_seed(20260802)
    tokens, heads, head_dim = 37, 4, 128
    num_valid_tokens = 13  # 13*4=52 valid rows of 148; crosses the BT=32 block boundary
    eps = 1e-6
    core_attn_out = torch.randn(
        1,
        tokens,
        heads,
        head_dim,
        dtype=dtype,
        device="npu",
    )
    output_gate = torch.randn(tokens, heads, head_dim, dtype=dtype, device="npu")
    weight = torch.randn(head_dim, dtype=dtype, device="npu")
    core_attn_out[:, num_valid_tokens:] = float("nan")

    num_valid_rows = torch.tensor(num_valid_tokens, dtype=torch.int32, device="npu")
    actual = apply_kda_rms_norm_sigmoid_gate(
        core_attn_out,
        output_gate,
        weight,
        eps,
        num_valid_rows,
    )

    expected = _rms_norm_sigmoid_gate_reference(
        core_attn_out[:, :num_valid_tokens], output_gate[:num_valid_tokens], weight, eps
    )
    torch.testing.assert_close(actual[:, :num_valid_tokens], expected, rtol=2e-3, atol=2e-3)
    assert (actual[:, num_valid_tokens:] == 0).all(), "padding rows must be zeroed by the kernel"


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
@pytest.mark.parametrize("num_valid_tokens", [0, 37])
def test_kimi_kda_fused_rms_norm_sigmoid_gate_threshold_edges(num_valid_tokens: int):
    """Threshold edges: nvr == tokens equals the un-thresholded result, nvr == 0 zeroes all."""
    torch.manual_seed(20260803)
    tokens, heads, head_dim = 37, 4, 128
    eps = 1e-6
    core_attn_out = torch.randn(
        1,
        tokens,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    output_gate = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device="npu")
    if num_valid_tokens == 0:
        core_attn_out[:, :] = float("nan")

    num_valid_rows = torch.tensor(num_valid_tokens, dtype=torch.int64, device="npu")
    actual = apply_kda_rms_norm_sigmoid_gate(
        core_attn_out,
        output_gate,
        weight,
        eps,
        num_valid_rows,
    )

    if num_valid_tokens == tokens:
        expected = _rms_norm_sigmoid_gate_reference(core_attn_out, output_gate, weight, eps)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    else:
        assert (actual == 0).all(), "num_valid_rows == 0 must zero every row"


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layer_norm_gated_fwd_zeroes_padding_rows_wide_feature(dtype: torch.dtype):
    """D > 512 routes to layer_norm_gated_fwd_kernel1; num_valid_rows is in row units here."""
    torch.manual_seed(20260804)
    rows, dim = 7, 1024
    num_valid_rows_value = 3
    eps = 1e-6
    x = torch.randn(rows, dim, dtype=dtype, device="npu")
    gate = torch.randn(rows, dim, dtype=dtype, device="npu")
    weight = torch.randn(dim, dtype=dtype, device="npu")
    x[num_valid_rows_value:] = float("nan")

    y, mean, rstd, _ = layer_norm_gated_fwd(
        x=x,
        g=gate,
        weight=weight,
        bias=None,
        activation="sigmoid",
        eps=eps,
        out_dtype=x.dtype,
        is_rms_norm=True,
        num_valid_rows=torch.tensor(num_valid_rows_value, dtype=torch.int32, device="npu"),
    )

    x_float = x[:num_valid_rows_value].float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    expected = x_float * torch.rsqrt(variance + eps)
    expected = expected * weight.float()
    expected = expected * gate[:num_valid_rows_value].float().sigmoid()
    expected = expected.to(dtype)

    assert mean is None
    assert rstd.shape == (rows,)
    torch.testing.assert_close(y[:num_valid_rows_value], expected, rtol=2e-3, atol=2e-3)
    assert (y[num_valid_rows_value:] == 0).all(), "padding rows must be zeroed by the kernel"
