"""End-to-end tests for Phase-2 multi-output (matmul tap + per-op tap).

Each test builds a cudnn graph with one or more set_output(True) intermediates,
JITs it, runs, and checks every output buffer against a torch reference.
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (installs hook)
import pytest
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.tile_config import by_name


def _mkdata(M: int, N: int, K: int, B: int = 1):
    torch.manual_seed(0)
    a = torch.empty(B, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(B, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    return a, b


def _vp(compiled, a, b, outs, *aux):
    """Variant-pack dict {cuDNN tensor: buffer} from the compiled binding.
    ``outs`` fills chain.outputs slot order (terminal, then taps); ``aux`` the
    epilogue aux tensors in order."""
    bd = compiled.binding
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {bd.a_operands[0]: a, bd.b_operands[0]: b}
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


def test_matmul_tap_only() -> None:
    """Tap the raw matmul output (Phase-1 path). Terminal: relu (BF16).
    Tap: matmul (BF16)."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    Y = g.relu(input=C, name="r")
    Y.set_output(True)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_tap]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    torch.testing.assert_close(c_tap, mm.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, torch.relu(mm).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,ref_fn",
    (
        (cudnn.reduction_mode.ADD, lambda x: x.sum(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.AMAX, lambda x: x.abs().amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MAX, lambda x: x.amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MIN, lambda x: x.amin(dim=(0, 1, 2), keepdim=True)),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_epilogue_reduction_tap_scalar(mode, ref_fn) -> None:
    """Materialize the normal epilogue output plus a scalar reduction tap."""
    M, N, K = 128, 128, 64
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True)
    R = g.reduction(input=Y, mode=mode, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_term = torch.relu(mm)
    torch.testing.assert_close(c_term, ref_term.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_red, ref_fn(ref_term), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,ref_fn",
    (
        (cudnn.reduction_mode.ADD, lambda x: x.sum(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.AMAX, lambda x: x.abs().amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MAX, lambda x: x.amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MIN, lambda x: x.amin(dim=(0, 1, 2), keepdim=True)),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_epilogue_reduction_tap_scalar_int32(mode, ref_fn) -> None:
    """Int32 reduction uses int32 output/compute atomic semantics."""
    M, N, K = 128, 128, 64
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.INT32)
    R = g.reduction(
        input=C,
        mode=mode,
        name="red_i32",
        compute_data_type=cudnn.data_type.INT32,
    )
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.INT32)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.int32, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.int32, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.int32)
    torch.testing.assert_close(c_term, mm, atol=0, rtol=0)
    torch.testing.assert_close(c_red, ref_fn(mm).to(torch.int32), atol=0, rtol=0)


@pytest.mark.parametrize(
    "mode,red_shape,ref_dims",
    (
        (cudnn.reduction_mode.ADD, "per_batch", (1, 2)),
        (cudnn.reduction_mode.ADD, "per_row", (0, 2)),
        (cudnn.reduction_mode.ADD, "per_col", (0, 1)),
        (cudnn.reduction_mode.AMAX, "per_col", (0, 1)),
        (cudnn.reduction_mode.MAX, "per_col", (0, 1)),
        (cudnn.reduction_mode.MIN, "per_col", (0, 1)),
    ),
    ids=(
        "add_per_batch",
        "add_per_row",
        "add_per_col",
        "amax_per_col",
        "max_per_col",
        "min_per_col",
    ),
)
def test_epilogue_reduction_tap_partial(mode, red_shape: str, ref_dims: tuple[int, ...]) -> None:
    """Reduction taps keep rank 3 and set only the reduced dimensions to 1."""
    B, M, N, K = 2, 128, 128, 64
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B_t = g.tensor(name="B", dim=[B, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B_t, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True)
    R = g.reduction(input=Y, mode=mode, name="red")
    red_dims = {
        "per_batch": [B, 1, 1],
        "per_row": [1, M, 1],
        "per_col": [1, 1, N],
    }[red_shape]
    R.set_dim(red_dims).set_stride([red_dims[1] * red_dims[2], red_dims[2], 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K, B)
    c_term = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    c_red = torch.empty(*red_dims, dtype=torch.float32, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_term = torch.relu(mm)
    if mode == cudnn.reduction_mode.AMAX:
        ref_red = ref_term.abs().amax(dim=ref_dims, keepdim=True)
    elif mode == cudnn.reduction_mode.MAX:
        ref_red = ref_term.amax(dim=ref_dims, keepdim=True)
    elif mode == cudnn.reduction_mode.MIN:
        ref_red = ref_term.amin(dim=ref_dims, keepdim=True)
    else:
        ref_red = ref_term.sum(dim=ref_dims, keepdim=True)
    torch.testing.assert_close(c_term, ref_term.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_red, ref_red, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,red_shape,ref_dims",
    (
        (cudnn.reduction_mode.ADD, "per_batch", (1, 2)),
        (cudnn.reduction_mode.ADD, "per_row", (0, 2)),
        (cudnn.reduction_mode.ADD, "per_col", (0, 1)),
        (cudnn.reduction_mode.AMAX, "per_col", (0, 1)),
    ),
    ids=("add_per_batch", "add_per_row", "add_per_col", "amax_per_col"),
)
def test_epilogue_reduction_tap_strided_output(mode, red_shape: str, ref_dims: tuple[int, ...]) -> None:
    """Reduction output atomics honor the runtime output stride."""
    B, M, N, K = 2, 128, 128, 64
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B_t = g.tensor(name="B", dim=[B, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B_t, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True)
    R = g.reduction(input=Y, mode=mode, name="red")
    red_dims, red_stride = {
        "per_batch": ([B, 1, 1], [2, 1, 1]),
        "per_row": ([1, M, 1], [0, 2, 1]),
        "per_col": ([1, 1, N], [0, 0, 2]),
    }[red_shape]
    R.set_dim(red_dims).set_stride(red_stride)
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K, B)
    c_term = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    c_red = torch.empty_strided(tuple(red_dims), tuple(red_stride), dtype=torch.float32, device="cuda")
    assert not c_red.is_contiguous()
    compiled(_vp(compiled, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_term = torch.relu(mm)
    if mode == cudnn.reduction_mode.AMAX:
        ref_red = ref_term.abs().amax(dim=ref_dims, keepdim=True)
    else:
        ref_red = ref_term.sum(dim=ref_dims, keepdim=True)
    torch.testing.assert_close(c_term, ref_term.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_red, ref_red, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,ref_fn",
    (
        (cudnn.reduction_mode.ADD, lambda x: x.sum(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.AMAX, lambda x: x.abs().amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MAX, lambda x: x.amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MIN, lambda x: x.amin(dim=(0, 1, 2), keepdim=True)),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_epilogue_reduction_tap_scalar_big_cgrp_multi_cta(mode, ref_fn) -> None:
    """Reduction taps use global atomics across many CTA contributors."""
    B, M, N, K = 2, 512, 384, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B_t = g.tensor(name="B", dim=[B, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B_t, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    R = g.reduction(input=C, mode=mode, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster4x2")
    compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=2)
    a, b = _mkdata(M, N, K, B)
    c_term = torch.empty(B, M, N, dtype=torch.float32, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    torch.testing.assert_close(c_term, mm, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_red, ref_fn(mm), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "mode,ref_fn",
    (
        (cudnn.reduction_mode.ADD, lambda x: x.sum(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.AMAX, lambda x: x.abs().amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MAX, lambda x: x.amax(dim=(0, 1, 2), keepdim=True)),
        (cudnn.reduction_mode.MIN, lambda x: x.amin(dim=(0, 1, 2), keepdim=True)),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_epilogue_reduction_tap_scalar_int32_big_cgrp_multi_cta(mode, ref_fn) -> None:
    """Int32 reduction atomics work across many CTA contributors."""
    B, M, N, K = 2, 512, 384, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1])
    B_t = g.tensor(name="B", dim=[B, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B_t, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.INT32)
    R = g.reduction(
        input=C,
        mode=mode,
        name="red_i32",
        compute_data_type=cudnn.data_type.INT32,
    )
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.INT32)

    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster4x2")
    compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=2)
    a, b = _mkdata(M, N, K, B)
    c_term = torch.empty(B, M, N, dtype=torch.int32, device="cuda")
    c_red = torch.empty(1, 1, 1, dtype=torch.int32, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_red]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.int32)
    torch.testing.assert_close(c_term, mm, atol=0, rtol=0)
    torch.testing.assert_close(c_red, ref_fn(mm).to(torch.int32), atol=0, rtol=0)


def test_mid_op_tap() -> None:
    """Tap an intermediate fusion-op result (Phase-2 path).

    Chain: matmul -> bias -> relu.
    Tap: after bias (FP32). Terminal: after relu (BF16).
    """
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1])
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="b")
    Cb.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    Y = g.relu(input=Cb, name="r")
    Y.set_output(True)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K)
    bias_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_tap], bias_t))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_after_bias = mm + bias_t.float()
    torch.testing.assert_close(c_tap, ref_after_bias, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, torch.relu(ref_after_bias).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_dag_two_branches_from_matmul() -> None:
    """Phase 3: matmul fan-out to two parallel branches, each ending at its
    own GMEM output.

    Chain: matmul → {relu, gelu}.
    Last set_output is gelu → slot 0 = terminal (BF16); relu → slot 1 tap (BF16).
    """
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y1 = g.relu(input=C, name="r")
    Y1.set_output(True)  # → tap (slot 1)
    Y2 = g.gelu_approx_tanh(input=C, name="g")
    Y2.set_output(True)  # → terminal (slot 0)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_tap]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_gelu = torch.nn.functional.gelu(mm, approximate="tanh").to(torch.bfloat16)
    ref_relu = torch.relu(mm).to(torch.bfloat16)
    torch.testing.assert_close(c_term, ref_gelu, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_tap, ref_relu, atol=1e-1, rtol=1e-2)


def test_dag_two_branches_with_per_branch_ops() -> None:
    """Phase 3: each branch has its own pointwise ops (bias + activation)."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias1 = g.tensor(name="bias1", dim=[1, 1, N], stride=[N, N, 1])
    bias2 = g.tensor(name="bias2", dim=[1, 1, N], stride=[N, N, 1])
    C = g.matmul(A=A, B=B, name="mm")
    # Branch 1: C → bias1 → relu  (set_output first → tap)
    Cb1 = g.bias(input=C, bias=bias1, name="b1")
    Y1 = g.relu(input=Cb1, name="r")
    Y1.set_output(True)
    # Branch 2: C → bias2 → gelu  (set_output last → terminal)
    Cb2 = g.bias(input=C, bias=bias2, name="b2")
    Y2 = g.gelu_approx_tanh(input=Cb2, name="g")
    Y2.set_output(True)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K)
    bias1_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    bias2_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_tap], bias1_t, bias2_t))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref_gelu = torch.nn.functional.gelu(mm + bias2_t.float(), approximate="tanh").to(torch.bfloat16)
    ref_relu = torch.relu(mm + bias1_t.float()).to(torch.bfloat16)
    torch.testing.assert_close(c_term, ref_gelu, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_tap, ref_relu, atol=1e-1, rtol=1e-2)


def test_fan_in_relu_plus_gelu() -> None:
    """Phase 4 fan-in: Y = add(relu(C), gelu(C)).
    Two op-results re-merged into one binary op; Y is the only output.
    """
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    R = g.relu(input=C, name="r")
    G = g.gelu_approx_tanh(input=C, name="g")
    Y = g.add(a=R, b=G, name="add")
    Y.set_output(True)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K)
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    ref = (torch.relu(mm) + torch.nn.functional.gelu(mm, approximate="tanh")).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


def test_fan_in_with_intermediate_tap() -> None:
    """Phase 4 fan-in + intermediate tap: relu and gelu are both consumed by
    `add`, and one of them (relu) is also set_output as a tap."""
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    R = g.relu(input=C, name="r")
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)  # tap relu's output
    G = g.gelu_approx_tanh(input=C, name="g")
    Y = g.mul(a=R, b=G, name="mul")
    Y.set_output(True)  # terminal

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_relu = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_relu]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    relu_ref = torch.relu(mm)
    gelu_ref = torch.nn.functional.gelu(mm, approximate="tanh")
    torch.testing.assert_close(c_relu, relu_ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, (relu_ref * gelu_ref).to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_complex_fusion_dag() -> None:
    """End-to-end functionality test that exercises every Phase 1–4 feature
    in a single graph:

      Phase 1 — matmul output tap (`C` materialized as FP16)
      Phase 2 — intermediate op taps (after bias as BF16, after relu as FP32,
                after the fan-in add as FP16)
      Phase 3 — fan-out from the matmul output into two parallel branches
                (bias-relu vs. scale-gelu)
      Phase 4 — fan-in via `add(R1, G1)` re-merging the two branches' results
      Plus    — three aux tensors with different broadcast modes
                (per-row, per-col, scalar) and a variety of pointwise ops
                (bias/add, mul, relu, gelu_tanh, tanh)

    Topology (every node is a fp32 vec in registers; `*` marks set_output):

                       C = matmul(A, B)              *  (FP16 tap, slot 1)
                        ├─────────────┐
                        ▼             ▼
              T1 = C + bias_row    T2 = C * scale_col     ← fan-out
                        │             │
                  *(BF16 tap,         │
                    slot 2)           │
                        ▼             ▼
                   R1 = relu(T1)   G1 = gelu_tanh(T2)
                        │             │
                  *(FP32 tap,         │
                    slot 3)           │
                        └──── add ────┘                    ← fan-in
                              │
                          S = R1 + G1                     *(FP16 tap, slot 4)
                              │
                              ▼
                       SC = S * alpha (scalar)
                              │
                              ▼
                       Y = tanh(SC)                       * (terminal, BF16, slot 0)

    Expected `chain.outputs` slot order (terminal first, taps in chain order):
       slot 0  terminal  Y     BF16
       slot 1  matmul    C     FP16
       slot 2  op_0      T1    BF16  (bias result)
       slot 3  op_1      R1    FP32  (relu result)
       slot 4  op_4      S     FP16  (fan-in add result)

    Expected `chain.ops` (Kahn topo from analyzer, with recorder-order tie-break):
       op 0: bias    parent=-1     (C → T1)
       op 1: relu    parent=0      (T1 → R1)
       op 2: mul     parent=-1     (C → T2)
       op 3: gelu    parent=2      (T2 → G1)
       op 4: add     parent=1, parent_idx_b=3   (R1+G1 → S)   ← Phase-4 fan-in
       op 5: mul     parent=4      (S → SC)
       op 6: tanh    parent=5      (SC → Y)                   ← terminal
    """
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias_row = g.tensor(name="bias_row", dim=[1, M, 1], stride=[M, 1, 1])  # per-row
    scale_col = g.tensor(name="scale_col", dim=[1, 1, N], stride=[N, N, 1])  # per-col
    alpha = g.tensor(name="alpha", dim=[1, 1, 1], stride=[1, 1, 1])  # scalar

    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.HALF)  # tap

    # Branch 1: bias + relu
    T1 = g.bias(input=C, bias=bias_row, name="b1")
    T1.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)  # tap
    R1 = g.relu(input=T1, name="r")
    R1.set_output(True).set_data_type(cudnn.data_type.FLOAT)  # tap

    # Branch 2: scale + gelu_tanh
    T2 = g.mul(a=C, b=scale_col, name="m1")
    G1 = g.gelu_approx_tanh(input=T2, name="g")

    # Fan-in: re-merge branches via add
    S = g.add(a=R1, b=G1, name="add")
    S.set_output(True).set_data_type(cudnn.data_type.HALF)  # tap

    # Final: scale by scalar and tanh — terminal
    SC = g.mul(a=S, b=alpha, name="m2")
    Y = g.tanh(input=SC, name="t")
    Y.set_output(True)  # terminal (BF16)

    compiled = jit_from_cudnn_graph(g)

    # ---- Validate IR structure ---------------------------------------------
    chain = compiled.chain
    assert len(chain.ops) == 7
    assert [op.op for op in chain.ops] == ["add", "relu", "mul", "gelu_tanh", "add", "mul", "tanh"]
    # parent_idx + parent_idx_b layout
    assert chain.ops[0].parent_idx == -1 and chain.ops[0].aux == "bias_row"  # bias on C
    assert chain.ops[1].parent_idx == 0  # relu on T1
    assert chain.ops[2].parent_idx == -1 and chain.ops[2].aux == "scale_col"  # mul on C
    assert chain.ops[3].parent_idx == 2  # gelu on T2
    assert chain.ops[4].parent_idx == 1 and chain.ops[4].parent_idx_b == 3  # add(R1, G1)  ← fan-in
    assert chain.ops[4].aux is None  # fan-in has no aux
    assert chain.ops[5].parent_idx == 4 and chain.ops[5].aux == "alpha"  # mul by scalar
    assert chain.ops[6].parent_idx == 5  # tanh on SC
    assert chain.resolved_terminal_idx == 6

    # chain.outputs slot order
    outs = chain.outputs
    assert [(o.source, o.dtype) for o in outs] == [
        ("terminal", "bf16"),
        ("matmul", "fp16"),
        ("op_0", "bf16"),
        ("op_1", "fp32"),
        ("op_4", "fp16"),
    ]

    # ---- Run + verify against torch reference ------------------------------
    a, b = _mkdata(M, N, K)
    bias_row_t = torch.randn(1, M, 1, device="cuda", dtype=torch.bfloat16)
    scale_col_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    alpha_t = torch.tensor([[[2.0]]], device="cuda", dtype=torch.bfloat16)

    c_term = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)
    c_mm = torch.empty(1, M, N, device="cuda", dtype=torch.float16)
    c_bias = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)
    c_relu = torch.empty(1, M, N, device="cuda", dtype=torch.float32)
    c_addS = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    compiled(
        _vp(
            compiled,
            a,
            b,
            [c_term, c_mm, c_bias, c_relu, c_addS],
            bias_row_t,
            scale_col_t,
            alpha_t,
        )
    )
    torch.cuda.synchronize()

    # Reference. Compute runs in fp32, but every tensor with a declared (narrow)
    # data_type rounds the running value to that dtype before the next op reads
    # it — matching cuDNN tensor semantics, including for downstream consumers:
    #   C=fp16 (feeds BOTH branches), T1=bf16 (feeds relu), R1=fp32 (feeds add),
    #   T2/G1/SC are virtual fp32 (intermediate_data_type=FLOAT), S=fp16 (feeds
    #   the final mul). The kernel rounds at exactly these points.
    def _round(x, dt):
        return x.to(dt).float()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    c = _round(mm, torch.float16)  # C: fp16
    t1 = _round(c + bias_row_t.float(), torch.bfloat16)  # T1: bf16
    r1 = torch.relu(t1)  # R1: fp32 (no round)
    t2 = c * scale_col_t.float()  # T2: virtual fp32
    g1 = torch.nn.functional.gelu(t2, approximate="tanh")  # G1: virtual fp32
    s = _round(r1 + g1, torch.float16)  # S: fp16
    sc = s * alpha_t.float()  # SC: virtual fp32
    y = torch.tanh(sc)  # terminal

    torch.testing.assert_close(c_mm, mm.to(torch.float16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_bias, t1.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_relu, r1, atol=1e-3, rtol=1e-3)  # fp32 tight
    torch.testing.assert_close(c_addS, s.to(torch.float16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, y.to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_two_taps_matmul_and_mid_op() -> None:
    """Both Phase-1 (matmul) AND Phase-2 (mid-op) taps in the same chain.

    Chain: matmul -> bias -> gelu_tanh.
    Outputs (in slot order): terminal (BF16), matmul tap (FP32), bias tap (BF16).
    """
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    Cb = g.bias(input=C, bias=bias, name="b")
    Cb.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    Y = g.gelu_approx_tanh(input=Cb, name="g")
    Y.set_output(True)

    compiled = jit_from_cudnn_graph(g)
    a, b = _mkdata(M, N, K)
    bias_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    c_term = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    c_tap_mm = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    c_tap_bias = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    compiled(_vp(compiled, a, b, [c_term, c_tap_mm, c_tap_bias], bias_t))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    after_bias = mm + bias_t.float()
    after_gelu = torch.nn.functional.gelu(after_bias, approximate="tanh")

    torch.testing.assert_close(c_tap_mm, mm, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_tap_bias, after_bias.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_term, after_gelu.to(torch.bfloat16), atol=1e-1, rtol=1e-2)


def test_virtual_bf16_intermediate_loses_precision() -> None:
    """Req 3: a pure-virtual intermediate declared bf16 (via a bf16
    intermediate_data_type) rounds the running value, so the kernel matches a
    reference that rounds at that point — and visibly differs from a reference
    that keeps fp32 throughout. Chain: matmul -> mul(scale) -> gelu_tanh, with
    a fractional scale so the post-mul value isn't bf16-exact.
    """
    M, N, K = 256, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.BFLOAT16,  # virtuals are bf16
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    scale = g.tensor(name="scale", dim=[1, 1, N], stride=[N, N, 1])
    C = g.matmul(A=A, B=B, name="mm")  # virtual bf16
    T = g.mul(a=C, b=scale, name="m")  # virtual bf16  ← rounding bites here
    Y = g.gelu_approx_tanh(input=T, name="g")  # terminal
    Y.set_data_type(cudnn.data_type.FLOAT)  # fp32 output so we can compare tightly
    Y.set_output(True)

    chain = jit_from_cudnn_graph(g).chain
    assert chain.matmul.out_dtype == "bf16"  # C rounded
    assert chain.ops[0].out_dtype == "bf16"  # T rounded before gelu
    compiled = jit_from_cudnn_graph(g)

    a, b = _mkdata(M, N, K)
    scale_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)
    c_out = torch.empty(1, M, N, device="cuda", dtype=torch.float32)
    compiled(_vp(compiled, a, b, [c_out], scale_t))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())

    # Reference WITH rounding at the bf16 virtuals (what the kernel should do):
    c_r = mm.to(torch.bfloat16).float()
    t_r = (c_r * scale_t.float()).to(torch.bfloat16).float()
    y_rounded = torch.nn.functional.gelu(t_r, approximate="tanh")

    # Reference WITHOUT any intermediate rounding (the old/wrong behavior):
    y_fp32 = torch.nn.functional.gelu(mm * scale_t.float(), approximate="tanh")

    # Kernel matches the rounded reference...
    torch.testing.assert_close(c_out, y_rounded, atol=1e-1, rtol=1e-2)
    # ...and the rounding actually mattered (otherwise this test proves nothing).
    assert (y_rounded - y_fp32).abs().max().item() > 0.1, "intermediate rounding had no measurable effect — pick data that exercises it"


def test_m_major_op_tap() -> None:
    """M-major op tap + tanh terminal with 2CTA. The BF16 terminal uses the
    TMA path (TMEM-load -> stmatrix -> TMA-store); the FP32 op tap stores by scalar STG."""
    M, N, K = 256, 256, 128
    tap_dt = cudnn.data_type.FLOAT
    tap_tdt = torch.float32
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    R = g.gelu_approx_tanh(input=C, name="r")
    R.set_stride([M * N, 1, M])  # op-tap: M-major
    R.set_output(True).set_data_type(tap_dt)
    Y = g.tanh(input=R, name="t")
    Y.set_stride([M * N, 1, M])  # terminal: M-major
    Y.set_output(True)

    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1")
    compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=2, scheduler="clc")

    a, b = _mkdata(M, N, K)
    c_term = torch.empty(1, N, M, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    c_tap = torch.empty(1, N, M, dtype=tap_tdt, device="cuda").transpose(1, 2)
    compiled(_vp(compiled, a, b, [c_term, c_tap]))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    tap = torch.nn.functional.gelu(mm, approximate="tanh")
    rounded_tap = tap.to(tap_tdt)
    torch.testing.assert_close(c_tap, rounded_tap, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(
        c_term,
        torch.tanh(rounded_tap.float()).to(torch.bfloat16),
        atol=1e-1,
        rtol=1e-2,
    )
