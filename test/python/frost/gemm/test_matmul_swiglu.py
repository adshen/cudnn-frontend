"""Fused SwiGLU (cuDNN DualMatmulSiluMulDequant): out = silu(a@b0) * (a@b1) * scale.

A is shared by both GEMMs; b0/b1 distinct. Runs the multi-GEMM path, checked vs
a torch reference.
"""

from __future__ import annotations

import cudnn
import cudnn.frost.gemm  # noqa: F401  (installs hook)
import pytest
import torch

from cudnn.frost.gemm.compiler import jit_from_cudnn_graph
from cudnn.frost.gemm.tile_config import CATALOG

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


class _Plan:
    """JIT-compiles a recorded graph with a forced tile config (bypassing the
    FROST engine's auto-select). Exposes chain / binding / block_scale / aux_names;
    callable with a variant pack."""

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


def _vp_mg(compiled, gemm_pairs, outs, *aux):
    """Multi-GEMM variant-pack dict: dedup pairs by identity → distinct A/B slots,
    + outputs + aux."""
    bd = compiled.binding
    a_seen, b_seen = [], []
    for ag, bg in gemm_pairs:
        if not any(ag is x for x in a_seen):
            a_seen.append(ag)
        if not any(bg is x for x in b_seen):
            b_seen.append(bg)
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {}
    vp.update({t: buf for t, buf in zip(bd.a_operands, a_seen)})
    vp.update({t: buf for t, buf in zip(bd.b_operands, b_seen)})
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


_CUDNN_DT = {"bf16": cudnn.data_type.BFLOAT16, "fp16": cudnn.data_type.HALF}
_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}

# N=128 cta_group=1 geometry — small enough that dual-GEMM (2 acc) and triple fit.
_CFG_N128 = next(c for c in CATALOG if c.cta_tile_m == 128 and c.cta_tile_n == 128 and c.cta_tile_k_bytes == 128 and c.cgrp_size_m == 1 and c.cgrp_size_n == 1)
_CFG_N256 = next(c for c in CATALOG if c.cta_tile_m == 128 and c.cta_tile_n == 256 and c.cta_tile_k_bytes == 128 and c.cgrp_size_m == 1 and c.cgrp_size_n == 1)
# cluster2x1 N=256 geometry for the 2-CTA-MMA templates.
_CFG_N256_C2 = next(
    c for c in CATALOG if c.cta_tile_m == 128 and c.cta_tile_n == 256 and c.cta_tile_k_bytes == 128 and c.cgrp_size_m == 2 and c.cgrp_size_n == 1
)

# Every multi-GEMM-capable plain-matmul strategy: (label, cta_group, scheduler, config).
_STRATEGIES = [
    ("1ctamma-clc", 1, "clc", _CFG_N256),
    ("1ctamma-static", 1, "static", _CFG_N256),
    ("2ctamma-clc", 2, "clc", _CFG_N256_C2),
    ("2ctamma-static", 2, "static", _CFG_N256_C2),
]


def _build_swiglu(B, M, N, K, in_dt, out_dt):
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DT[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    aT = g.tensor(name="aTensor", dim=[B, M, K], stride=[M * K, K, 1])
    b0T = g.tensor(name="b0Tensor", dim=[B, K, N], stride=[K * N, 1, K])
    b1T = g.tensor(name="b1Tensor", dim=[B, K, N], stride=[K * N, 1, K])
    sf = g.tensor(
        name="scaleFactor",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    c0 = g.matmul(A=aT, B=b0T, name="mm0")
    c1 = g.matmul(A=aT, B=b1T, name="mm1")
    c0silu = g.swish(input=c0, name="silu0")
    mul = g.mul(a=c0silu, b=c1, name="mul0")
    dq = g.mul(a=mul, b=sf, name="dequant0")
    dq.set_output(True).set_data_type(_CUDNN_DT[out_dt])
    return g


def _reference(a, b0, b1, scale, out_dt):
    c0 = torch.einsum("bmk,bnk->bmn", a.float(), b0.float())
    c1 = torch.einsum("bmk,bnk->bmn", a.float(), b1.float())
    out = torch.nn.functional.silu(c0) * c1 * scale.flatten()[0]
    return out.to(_TORCH_DT[out_dt])


def _run(B, M, N, K, in_dt, out_dt, cfg, *, seed=0, cta_group=1, scheduler="clc"):
    torch.manual_seed(seed)
    a = torch.randn(B, M, K, device="cuda", dtype=_TORCH_DT[in_dt]) * 0.4
    b0 = torch.randn(B, N, K, device="cuda", dtype=_TORCH_DT[in_dt]) * 0.4
    b1 = torch.randn(B, N, K, device="cuda", dtype=_TORCH_DT[in_dt]) * 0.4
    scale = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)
    out = torch.zeros(B, M, N, device="cuda", dtype=_TORCH_DT[out_dt])
    compiled = _plan(
        _build_swiglu(B, M, N, K, in_dt, out_dt),
        config=cfg,
        cta_group=cta_group,
        scheduler=scheduler,
    )
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], out, scale))
    torch.cuda.synchronize()
    return out, _reference(a, b0, b1, scale, out_dt)


def _nonpacked_inputs(B, M, N, K, in_dt, out_dt, mode):
    td_in, td_out = _TORCH_DT[in_dt], _TORCH_DT[out_dt]
    torch.manual_seed(11)
    if mode == "zero_stride":
        a_base = torch.randn(K, device="cuda", dtype=td_in) * 0.4
        b0_base = torch.randn(K, device="cuda", dtype=td_in) * 0.4
        b1_base = torch.randn(K, device="cuda", dtype=td_in) * 0.4
        a = torch.as_strided(a_base, (B, M, K), (0, 0, 1))
        b0 = torch.as_strided(b0_base, (B, N, K), (0, 0, 1))
        b1 = torch.as_strided(b1_base, (B, N, K), (0, 0, 1))
    else:
        a_store = torch.randn(B, M, K + 16, device="cuda", dtype=td_in) * 0.4
        b0_store = torch.randn(B, N, K + 16, device="cuda", dtype=td_in) * 0.4
        b1_store = torch.randn(B, N, K + 32, device="cuda", dtype=td_in) * 0.4
        a = a_store[:, :, :K]
        b0 = b0_store[:, :, :K]
        b1 = b1_store[:, :, :K]
    out_store = torch.zeros(B, M, N + 16, device="cuda", dtype=td_out)
    scale = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)
    return a, b0, b1, out_store[:, :, :N], scale


@pytest.mark.parametrize("label,cta_group,scheduler,cfg", _STRATEGIES, ids=[s[0] for s in _STRATEGIES])
def test_swiglu_all_templates(label, cta_group, scheduler, cfg) -> None:
    """SwiGLU on every multi-GEMM template: 1ctamma / 2ctamma × clc / static."""
    out, ref = _run(
        512,
        256,
        256,
        128,
        "bf16",
        "bf16",
        cfg,
        cta_group=cta_group,
        scheduler=scheduler,
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize(
    "B,M,N,K",
    [
        (1, 256, 256, 128),
        (1, 512, 256, 256),
        (1, 384, 128, 128),  # M not a tile multiple
    ],
)
def test_swiglu_bf16(B, M, N, K) -> None:
    cfg = _CFG_N256 if N >= 256 else _CFG_N128
    out, ref = _run(B, M, N, K, "bf16", "bf16", cfg)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_swiglu_fp16() -> None:
    out, ref = _run(1, 256, 256, 128, "fp16", "fp16", _CFG_N256)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_swiglu_batched() -> None:
    """B>1: independent same-shape SwiGLU blocks."""
    out, ref = _run(2, 256, 256, 128, "bf16", "bf16", _CFG_N256)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize(
    "label,cta_group,scheduler,cfg,mode",
    [
        ("1ctamma-clc-padded", 1, "clc", _CFG_N256, "padded"),
        ("2ctamma-clc-padded", 2, "clc", _CFG_N256_C2, "padded"),
        ("1ctamma-static-zero", 1, "static", _CFG_N256, "zero_stride"),
        ("2ctamma-static-padded", 2, "static", _CFG_N256_C2, "padded"),
    ],
)
def test_swiglu_nonpacked_tensors(label, cta_group, scheduler, cfg, mode) -> None:
    B, M, N, K = 1, 256, 256, 128
    a, b0, b1, out, scale = _nonpacked_inputs(B, M, N, K, "bf16", "bf16", mode)
    assert not a.is_contiguous() or not b0.is_contiguous() or not b1.is_contiguous()
    assert not out.is_contiguous()
    compiled = _plan(
        _build_swiglu(B, M, N, K, "bf16", "bf16"),
        config=cfg,
        cta_group=cta_group,
        scheduler=scheduler,
    )
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], out, scale))
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _reference(a, b0, b1, scale, "bf16"), rtol=2e-2, atol=2e-1)
