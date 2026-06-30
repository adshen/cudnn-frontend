"""Fused dual MoE grouped matmul + SwiGLU — cuDNN ``Dual_Grouped_Matmul_Swiglu``:

    c0 = moe_grouped_matmul(token, w0, fto)      # moe0
    c1 = moe_grouped_matmul(token, w1, fto)      # moe1 (shares token + fto)
    c0silu = silu(c0)                            # CUDNN_POINTWISE_SWISH_FWD
    mul = c0silu * c1                            # CUDNN_POINTWISE_MUL
    out = mul * scaleFactor                      # CUDNN_POINTWISE_MUL (dequant)

Two grouped matmuls run in parallel sharing the token (A) and the single
``first_token_offset`` (so both have identical routed-group layout); the weights
(B) are distinct. Both feed one pointwise epilogue DAG. Multi-GEMM extension of
the MoE grouped matmul pipeline (1ctamma + 2ctamma). Checked against a torch
group-loop reference at the spec's rel/abs tolerances.
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (installs hook)
import pytest
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.graph_analyzer import analyze
from cudnn.TBD.gemm.tile_config import CATALOG


def _vp_moe_mg(compiled, gemm_pairs, fto, outs, *aux):
    """MoE multi-GEMM variant-pack dict from the binding (dedup (token, weight)
    pairs → distinct A/B slots; + first_token_offset + outputs + aux)."""
    bd = compiled.binding
    a_seen, b_seen = [], []
    for ag, bg in gemm_pairs:
        if not any(ag is x for x in a_seen):
            a_seen.append(ag)
        if not any(bg is x for x in b_seen):
            b_seen.append(bg)
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {bd.first_token_offset: fto}
    vp.update({t: buf for t, buf in zip(bd.a_operands, a_seen)})
    vp.update({t: buf for t, buf in zip(bd.b_operands, b_seen)})
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


# (config name, cta_group): 1-CTA cluster1x1 + 2-CTA cluster2x1 (reference design).
_GEOMETRIES = [
    ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", 1),
    ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2),
]


def _build_graph(
    E,
    S,
    N,
    K,
    num_groups,
    out_dt=cudnn.data_type.BFLOAT16,
    reduction_mode=None,
    reduction_dims=None,
):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
    w0 = g.tensor(name="weight0", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    w1 = g.tensor(name="weight1", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="first_token_offset", dim=[num_groups, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    sf = g.tensor(name="scaleFactor", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    c0 = g.moe_grouped_matmul(tok, w0, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe0")
    c1 = g.moe_grouped_matmul(tok, w1, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe1")
    c0silu = g.swish(input=c0, name="silu0")
    mul = g.mul(a=c0silu, b=c1, name="mul0")
    dq = g.mul(a=mul, b=sf, name="dequant0")
    dq.set_data_type(out_dt).set_output(True)
    if reduction_mode is not None:
        assert reduction_dims is not None
        R = g.reduction(input=dq, mode=reduction_mode, name="red")
        R.set_dim(list(reduction_dims)).set_stride([reduction_dims[1] * reduction_dims[2], reduction_dims[2], 1])
        R.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g


def _ref_f32(token, w0, w1, offsets, scale, S, N, num_experts, num_groups):
    out = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    starts = offsets.tolist()
    for gi in range(num_groups):
        b = starts[gi]
        e = starts[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ex = gi % num_experts
        c0 = token[0, b:e].float() @ w0[ex].float().T
        c1 = token[0, b:e].float() @ w1[ex].float().T
        out[b:e] = torch.nn.functional.silu(c0) * c1 * scale.flatten()[0]
    return out


def _ref(token, w0, w1, offsets, scale, S, N, num_experts, num_groups):
    out = _ref_f32(token, w0, w1, offsets, scale, S, N, num_experts, num_groups)
    return out.to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Analyzer (no GPU needed)
# --------------------------------------------------------------------------- #


def test_analyzer_detects_dual_moe() -> None:
    chain = analyze(_build_graph(9, 2000, 248, 520, 36))
    assert chain.has_moe and chain.is_multi_gemm
    assert chain.num_gemms == 2
    assert chain.num_a_operands == 1 and chain.num_b_operands == 2
    assert chain.gemm_operands == [(0, 0), (0, 1)]
    assert chain.moe.num_experts == 9
    assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (2000, 248, 520)
    assert [o.op for o in chain.ops] == ["swish", "mul", "mul"]
    # single fused output (the terminal)
    assert len(chain.outputs) == 1 and chain.outputs[0].source == "terminal"


def test_analyzer_detects_dual_moe_reduction() -> None:
    chain = analyze(
        _build_graph(
            9,
            2000,
            248,
            520,
            36,
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        )
    )
    assert chain.has_moe and chain.is_multi_gemm
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "add"
    assert [o.source for o in chain.outputs] == ["terminal", "reduction_0"]


# --------------------------------------------------------------------------- #
# End-to-end correctness (GPU)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
def test_dual_moe_swiglu_exact_case(cfg_name, cta_group) -> None:
    """The ``Dual_Grouped_Matmul_Swiglu_KNone_Mode`` spec case: S=2000, N=248,
    K=520, E=9, 36 routed groups (BxE > E) with custom offsets."""
    S, N, K, E = 2000, 248, 520, 9
    offset_values = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        127,
        255,
        383,
        483,
        515,
        643,
        718,
        924,
        1100,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
    ]
    num_groups = len(offset_values)
    cfg = next(c for c in CATALOG if c.name == cfg_name)
    compiled = jit_from_cudnn_graph(_build_graph(E, S, N, K, num_groups), config=cfg, cta_group=cta_group)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    w0 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device="cuda")
    out = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(offset_values, dtype=torch.int32, device="cuda")

    compiled(_vp_moe_mg(compiled, [(token, w0), (token, w1)], offsets, out, scale))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out[0],
        _ref(token, w0, w1, offsets, scale, S, N, E, num_groups),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize(
    "group_sizes",
    [
        [64, 0, 200, 128, 100, 12, 196, 68],  # uneven + one empty group
        [96, 96, 96, 96, 96, 96, 96, 96],  # balanced
    ],
)
def test_dual_moe_swiglu_groups(group_sizes, cfg_name, cta_group) -> None:
    E, N, K = 8, 256, 128
    S = sum(group_sizes)
    num_groups = E
    cfg = next(c for c in CATALOG if c.name == cfg_name)
    compiled = jit_from_cudnn_graph(_build_graph(E, S, N, K, num_groups), config=cfg, cta_group=cta_group)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    w0 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device="cuda")
    out = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    starts, cur = [], 0
    for gs in group_sizes:
        starts.append(cur)
        cur += gs
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")

    compiled(_vp_moe_mg(compiled, [(token, w0), (token, w1)], offsets, out, scale))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out[0],
        _ref(token, w0, w1, offsets, scale, S, N, E, num_groups),
        atol=2e-1,
        rtol=5e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_dual_moe_swiglu_reduction_scalar() -> None:
    E, N, K = 4, 128, 128
    group_sizes = [64, 0, 120, 72]
    S = sum(group_sizes)
    num_groups = E
    cfg = next(c for c in CATALOG if c.name == _GEOMETRIES[0][0])
    compiled = jit_from_cudnn_graph(
        _build_graph(
            E,
            S,
            N,
            K,
            num_groups,
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        ),
        config=cfg,
        cta_group=_GEOMETRIES[0][1],
    )

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    w0 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device="cuda")
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device="cuda")
    red = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    starts, cur = [], 0
    for gs in group_sizes:
        starts.append(cur)
        cur += gs
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")

    compiled(_vp_moe_mg(compiled, [(token, w0), (token, w1)], offsets, [out, red], scale))
    torch.cuda.synchronize()

    ref = _ref_f32(token, w0, w1, offsets, scale, S, N, E, num_groups)
    torch.testing.assert_close(out[0], ref.to(torch.bfloat16), atol=2e-1, rtol=5e-2)
    torch.testing.assert_close(
        red,
        ref.view(1, S, N).sum(dim=(1, 2), keepdim=True),
        atol=1e-1,
        rtol=1e-2,
    )
