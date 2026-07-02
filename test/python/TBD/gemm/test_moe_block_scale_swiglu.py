"""Fused dual MoE grouped **block-scale** matmul + SwiGLU — cuDNN
``NVFP4_Dual_Block_Scale_Moe_Grouped_Matmul_Swiglu_KNone_Mode``:

    tok_d = dequant(token, SFA)                  # block_scale_dequantize
    w0_d  = dequant(weight0, SFB0)
    w1_d  = dequant(weight1, SFB1)
    c0 = moe_grouped_matmul(tok_d, w0_d, fto)    # moe0
    c1 = moe_grouped_matmul(tok_d, w1_d, fto)    # moe1  (shares token + fto)
    out = silu(c0) * c1 * scaleFactor            # swish -> mul -> mul(scale)

Two block-scaled grouped matmuls run in parallel sharing the token (A) + its SFA
and the single ``first_token_offset``; the weights (B) and their SFB are distinct.
Multi-GEMM extension of the MoE grouped block-scale pipeline (sm100 2ctamma),
covering nvfp4 / mxfp4 / mxfp8. Checked vs a torch dequant + group-loop reference.
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (installs hook)
import pytest
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.graph_analyzer import analyze
from cudnn.TBD.gemm.tile_config import CATALOG


class _Plan:
    """Test handle that JIT-compiles a recorded graph with a forced tile config
    via ``jit_from_cudnn_graph`` (sweeps pin a specific config directly rather
    than letting the TBD engine auto-select). Exposes chain / binding / block_scale /
    aux_names and is callable with a variant pack."""

    def __init__(self, graph, config=None, cta_group=2, scheduler="clc"):
        self.g = graph
        kw = dict(cta_group=cta_group, scheduler=scheduler)
        if config is not None:
            kw["config"] = config
        self._compiled = jit_from_cudnn_graph(graph, **kw)
        self.chain = self._compiled.chain
        self.binding = self._compiled.binding
        self.block_scale = self.chain.has_block_scale
        self.aux_names = [t.name for t in self.chain.aux_tensors]

    def __call__(self, variant_pack):
        return self._compiled(variant_pack)


def _plan(graph, config=None, cta_group=2, scheduler="clc"):
    return _Plan(graph, config=config, cta_group=cta_group, scheduler=scheduler)


def _vp_moe_bs_mg(compiled, gemm_pairs, fto, outs, *aux):
    """MoE block-scale multi-GEMM variant-pack dict from the binding. Each pair
    is ``((token, sfa), (weight, sfb))``; dedup by packed-data identity into
    distinct A/B slots, + first_token_offset + outputs + aux."""
    bd = compiled.binding
    a_seen, b_seen, sfa_seen, sfb_seen = [], [], [], []
    for (ag, sfag), (bg, sfbg) in gemm_pairs:
        if not any(ag is x for x in a_seen):
            a_seen.append(ag)
            sfa_seen.append(sfag)
        if not any(bg is x for x in b_seen):
            b_seen.append(bg)
            sfb_seen.append(sfbg)
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {bd.first_token_offset: fto}
    vp.update({t: buf for t, buf in zip(bd.a_operands, a_seen)})
    vp.update({t: buf for t, buf in zip(bd.b_operands, b_seen)})
    vp.update({t: buf for t, buf in zip(bd.sfa_operands, sfa_seen)})
    vp.update({t: buf for t, buf in zip(bd.sfb_operands, sfb_seen)})
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


# cta_tile_n=128 (dual block-scale TMEM fits two accs + SF only at n<=128).
# (config, cta_group): 2-CTA cluster2x1 (reference) + 1-CTA cluster1x1.
_GEOMETRIES = [
    ("CONFIG_sm100_128x128x128_128x128x32_cluster2x1", 2),
    ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1),
]

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# combo -> (block_size, data dtype, SF dtype).
_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_blocked(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    nrb, ncb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _unpack_fp4(u8: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    lo = lut[(u8 & 0xF).long()]
    hi = lut[(u8 >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _rand_e8m0(shape, dev):
    return torch.randint(125, 129, shape, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)


def _build_graph(
    E,
    S,
    N,
    K,
    num_groups,
    combo,
    offset_dt=cudnn.data_type.INT32,
    reduction_mode=None,
    reduction_dims=None,
):
    block_size, a_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    w0 = g.tensor(name="weight0", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
    w1 = g.tensor(name="weight1", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
    SFA = g.tensor(name="SFA", dim=[1, S, sf_k], stride=[S * sf_k, sf_k, 1], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
    SFB0 = g.tensor(name="SFB0", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
    SFB1 = g.tensor(name="SFB1", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
    fto = g.tensor(name="first_token_offset", dim=[num_groups, 1, 1], stride=[1, 1, 1], data_type=offset_dt)
    sf = g.tensor(name="scaleFactor", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size])
    w0_d = g.block_scale_dequantize(input=w0, descale=SFB0, block_size=[block_size, 1])
    w1_d = g.block_scale_dequantize(input=w1, descale=SFB1, block_size=[block_size, 1])
    c0 = g.moe_grouped_matmul(tok_d, w0_d, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe0")
    c1 = g.moe_grouped_matmul(tok_d, w1_d, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe1")
    c0silu = g.swish(input=c0, name="silu0")
    mul = g.mul(a=c0silu, b=c1, name="mul0")
    dq = g.mul(a=mul, b=sf, name="dequant0")
    dq.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    if reduction_mode is not None:
        assert reduction_dims is not None
        R = g.reduction(input=dq, mode=reduction_mode, name="red")
        R.set_dim(list(reduction_dims)).set_stride([reduction_dims[1] * reduction_dims[2], reduction_dims[2], 1])
        R.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g


# --------------------------------------------------------------------------- #
# Analyzer (no GPU)
# --------------------------------------------------------------------------- #


def test_analyzer_detects_dual_moe_block_scale() -> None:
    chain = analyze(_build_graph(2, 1024, 256, 512, 4, "nvfp4"))
    assert chain.has_moe and chain.has_block_scale and chain.is_multi_gemm
    assert chain.num_gemms == 2
    assert chain.num_a_operands == 1 and chain.num_b_operands == 2
    assert chain.block_scale.combo == "nvfp4"
    assert chain.moe.num_experts == 2
    assert [o.op for o in chain.ops] == ["swish", "mul", "mul"]
    assert len(chain.outputs) == 1 and chain.outputs[0].source == "terminal"


def test_analyzer_detects_dual_moe_block_scale_reduction() -> None:
    chain = analyze(
        _build_graph(
            2,
            1024,
            256,
            512,
            4,
            "nvfp4",
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        )
    )
    assert chain.has_moe and chain.has_block_scale and chain.is_multi_gemm
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "add"
    assert [o.source for o in chain.outputs] == ["terminal", "reduction_0"]


# --------------------------------------------------------------------------- #
# End-to-end correctness (GPU)
# --------------------------------------------------------------------------- #


def _mk_operand(combo, batch, rows, K, dev, lut):
    """Return (runtime packed tensor, dequantized float (.., rows, K))."""
    is_fp4 = combo in ("nvfp4", "mxfp4")
    if is_fp4:
        u8 = torch.randint(0, 256, (batch, rows, K // 2), dtype=torch.uint8, device=dev)
        return u8.view(torch.float4_e2m1fn_x2), _unpack_fp4(u8, lut).view(batch, rows, K)
    rt = (torch.randn(batch, rows, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    return rt, rt.float().view(batch, rows, K)


def _mk_sf(combo, shape, dev):
    if combo == "nvfp4":
        return torch.randint(1, 4, shape, device=dev).to(torch.float8_e4m3fn)
    return _rand_e8m0(shape, dev)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
def test_dual_moe_block_scale_swiglu(combo, cfg_name, cta_group) -> None:
    """The ``..._Dual_Block_Scale_Moe_Grouped_Matmul_Swiglu_KNone_Mode`` spec
    case: S=1024, N=256, K=512, E=2, 4 groups (offsets [0,256,384,512])."""
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 1024, 256, 512
    offsets_list = [0, 256, 384, 512]
    num_groups = len(offsets_list)
    block_size = _COMBOS[combo][0]
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    tok_rt, tok_deq = _mk_operand(combo, 1, S, K, dev, lut)
    w0_rt, w0_deq = _mk_operand(combo, E, N, K, dev, lut)
    w1_rt, w1_deq = _mk_operand(combo, E, N, K, dev, lut)
    tok_deq = tok_deq.view(S, K)
    sfa_log = _mk_sf(combo, (S, sf_k), dev)
    sfb0_log = _mk_sf(combo, (E, N, sf_k), dev)
    sfb1_log = _mk_sf(combo, (E, N, sf_k), dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)

    cfg = next(c for c in CATALOG if c.name == cfg_name)
    compiled = _plan(_build_graph(E, S, N, K, num_groups, combo), config=cfg, cta_group=cta_group)
    assert compiled.chain.block_scale.combo == combo

    # SFA padded to 128 rows PER GROUP, then concatenated; SFB per-expert.
    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1] if gi + 1 < num_groups else S]) for gi in range(num_groups)]
    sfa_blk = torch.cat(sfa_parts).view(1, -1, 1)
    sfb0_blk = torch.cat([_to_blocked(sfb0_log[e]) for e in range(E)]).view(E, sf_k, N)
    sfb1_blk = torch.cat([_to_blocked(sfb1_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)

    compiled(
        _vp_moe_bs_mg(
            compiled,
            [((tok_rt, sfa_blk), (w0_rt, sfb0_blk)), ((tok_rt, sfa_blk), (w1_rt, sfb1_blk))],
            offsets,
            output,
            scale,
        )
    )
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w0_s = w0_deq * sfb0_log.float().repeat_interleave(block_size, 2)
    w1_s = w1_deq * sfb1_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ex = gi % E
        c0 = tok_s[b:e] @ w0_s[ex].T
        c1 = tok_s[b:e] @ w1_s[ex].T
        ref[b:e] = torch.nn.functional.silu(c0) * c1 * 0.5
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_dual_moe_block_scale_swiglu_reduction_scalar() -> None:
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 512, 128, 512
    combo = "nvfp4"
    offsets_list = [0, 100, 300]
    num_groups = len(offsets_list)
    block_size = _COMBOS[combo][0]
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    tok_rt, tok_deq = _mk_operand(combo, 1, S, K, dev, lut)
    w0_rt, w0_deq = _mk_operand(combo, E, N, K, dev, lut)
    w1_rt, w1_deq = _mk_operand(combo, E, N, K, dev, lut)
    tok_deq = tok_deq.view(S, K)
    sfa_log = _mk_sf(combo, (S, sf_k), dev)
    sfb0_log = _mk_sf(combo, (E, N, sf_k), dev)
    sfb1_log = _mk_sf(combo, (E, N, sf_k), dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)

    cfg = next(c for c in CATALOG if c.name == _GEOMETRIES[1][0])
    compiled = _plan(
        _build_graph(
            E,
            S,
            N,
            K,
            num_groups,
            combo,
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        ),
        config=cfg,
        cta_group=_GEOMETRIES[1][1],
    )

    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1] if gi + 1 < num_groups else S]) for gi in range(num_groups)]
    sfa_blk = torch.cat(sfa_parts).view(1, -1, 1)
    sfb0_blk = torch.cat([_to_blocked(sfb0_log[e]) for e in range(E)]).view(E, sf_k, N)
    sfb1_blk = torch.cat([_to_blocked(sfb1_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    red = torch.empty(1, 1, 1, dtype=torch.float32, device=dev)

    compiled(
        _vp_moe_bs_mg(
            compiled,
            [((tok_rt, sfa_blk), (w0_rt, sfb0_blk)), ((tok_rt, sfa_blk), (w1_rt, sfb1_blk))],
            offsets,
            [output, red],
            scale,
        )
    )
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w0_s = w0_deq * sfb0_log.float().repeat_interleave(block_size, 2)
    w1_s = w1_deq * sfb1_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ex = gi % E
        c0 = tok_s[b:e] @ w0_s[ex].T
        c1 = tok_s[b:e] @ w1_s[ex].T
        ref[b:e] = torch.nn.functional.silu(c0) * c1 * 0.5
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        red,
        ref.view(1, S, N).sum(dim=(1, 2), keepdim=True),
        atol=1e-1,
        rtol=1e-2,
    )
