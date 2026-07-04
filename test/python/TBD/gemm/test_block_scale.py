"""Block-scaled matmul (FP4/FP8 + per-block scale factors): analyzer
pattern-match unit tests (no GPU) + end-to-end numerics vs a torch reference."""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (installs recorder)
import pytest
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.graph_analyzer import analyze
import re

from cudnn.TBD.gemm.tile_config import by_name


class _Plan:
    """JIT-compiles a recorded graph with a forced tile config (sweeps pin a
    config directly). Exposes chain/binding/block_scale/aux_names; callable with
    a variant pack."""

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


def _vp_bs(compiled, a, b, outs, sfa, sfb, *aux):
    """Block-scale single-GEMM variant-pack dict from the binding."""
    bd = compiled.binding
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {
        bd.a_operands[0]: a,
        bd.b_operands[0]: b,
        bd.sfa_operands[0]: sfa,
        bd.sfb_operands[0]: sfb,
    }
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


_LEGACY_RE = re.compile(r"^(CONFIG_sm100_\d+x\d+x\d+_\d+x\d+x\d+_cluster\d+x\d+)_([12])ctamma(_static)?$")


def _kw(legacy_name):
    """Legacy config-name (with _Nctamma/_static) -> jit kwargs: pure-geometry
    config + cta_group + scheduler."""
    m = _LEGACY_RE.match(legacy_name)
    assert m, legacy_name
    return dict(
        config=by_name(m.group(1)),
        cta_group=int(m.group(2)),
        scheduler="static" if m.group(3) else "clc",
    )


# Helpers

_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_blocked(x):
    rows, cols = x.shape
    nrb, ncb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _unpack_fp4(u8, lut):
    lo = lut[(u8 & 0xF).long()]
    hi = lut[(u8 >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _build_nvfp4_graph(
    M,
    N,
    K,
    block_size=16,
    sf_dt=cudnn.data_type.FP8_E4M3,
    a_dt=cudnn.data_type.FP4_E2M1,
    b_dt=None,
    a_major="k",
    b_major="k",
    reorder=True,
    out_major="n",
):
    sf_k = K // block_size
    b_dt = b_dt if b_dt is not None else a_dt
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # A: K-major (stride[-1]=1) or M-major (stride[-2]=1).
    a_stride = [M * K, K, 1] if a_major == "k" else [M * K, 1, M]
    # B logical (K, N): K-major (stride[-2]=1) or N-major.
    b_stride = [K * N, 1, K] if b_major == "k" else [K * N, N, 1]
    A = g.tensor(name="A", dim=[1, M, K], stride=a_stride, data_type=a_dt)
    B = g.tensor(name="B", dim=[1, K, N], stride=b_stride, data_type=b_dt)
    sf_kw = dict(reordering_type=cudnn.tensor_reordering.F8_128x4) if reorder else {}
    SFA = g.tensor(
        name="SFA",
        dim=[1, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=sf_dt,
        **sf_kw,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[1, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        **sf_kw,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block_size])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    if out_major == "m":
        C.set_stride([M * N, 1, M])
    C.set_output(True).set_data_type(cudnn.data_type.HALF)
    return g


def _build_block_scale_reduction_graph(
    M,
    N,
    K,
    mode,
    red_dims,
    block_size=16,
    sf_dt=cudnn.data_type.FP8_E4M3,
    a_dt=cudnn.data_type.FP4_E2M1,
    red_stride=None,
    red_dtype=cudnn.data_type.FLOAT,
    red_compute_dtype=None,
):
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=[M * K, K, 1],
        data_type=a_dt,
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=[K * N, 1, K],
        data_type=a_dt,
    )
    SFA = g.tensor(
        name="SFA",
        dim=[1, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[1, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block_size])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    red_kwargs = {}
    if red_compute_dtype is not None:
        red_kwargs["compute_data_type"] = red_compute_dtype
    R = g.reduction(input=C, mode=mode, name="red", **red_kwargs)
    if red_stride is None:
        red_stride = [red_dims[1] * red_dims[2], red_dims[2], 1]
    R.set_dim(red_dims).set_stride(red_stride)
    R.set_output(True).set_data_type(red_dtype)
    return g


def _build_block_scale_quant_graph(
    M,
    N,
    K,
    dequant_block_size=16,
    quant_block_size=32,
    sf_dt=cudnn.data_type.FP8_E8M0,
    a_dt=cudnn.data_type.FP8_E4M3,
    out_dt=cudnn.data_type.FP8_E4M3,
    scale_dt=cudnn.data_type.FP8_E8M0,
    scale_reorder=False,
    scale_dim=None,
    global_scale=False,
):
    sf_k = K // dequant_block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=[M * K, K, 1],
        data_type=a_dt,
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=[K * N, 1, K],
        data_type=a_dt,
    )
    SFA = g.tensor(
        name="SFA",
        dim=[1, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[1, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, dequant_block_size])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[dequant_block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    if global_scale:
        scale = g.tensor(
            name="global_scale",
            dim=[1, 1, 1],
            stride=[1, 1, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        C = g.mul(a=C, b=scale, name="global_scale_mul")
    Q, QS = g.block_scale_quantize(input=C, block_size=quant_block_size, name="q")
    Q.set_output(True).set_data_type(out_dt)
    if scale_dim is not None:
        QS.set_dim(list(scale_dim)).set_stride([scale_dim[1] * scale_dim[2], scale_dim[2], 1])
    QS.set_output(True).set_data_type(scale_dt)
    if scale_reorder:
        QS.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
    return g


def _reduction_ref(x, mode, dims):
    if mode == cudnn.reduction_mode.AMAX:
        return x.abs().amax(dim=dims, keepdim=True)
    if mode == cudnn.reduction_mode.MAX:
        return x.amax(dim=dims, keepdim=True)
    if mode == cudnn.reduction_mode.MIN:
        return x.amin(dim=dims, keepdim=True)
    return x.sum(dim=dims, keepdim=True)


def _block_quant_ref(x, block_size, out_dtype, scale_dtype):
    blocks = x.view(1, x.shape[0], x.shape[1] // block_size, block_size)
    output_max = 448.0 if out_dtype is torch.float8_e4m3fn else 57344.0
    scale_f = blocks.abs().amax(dim=-1) / output_max
    if scale_dtype is torch.float8_e8m0fnu:
        # Reference rounds E8M0 scale factors toward +inf.
        safe = torch.where(scale_f > 0, scale_f, 1.0)
        scale_f = torch.where(scale_f > 0, torch.pow(2.0, torch.ceil(torch.log2(safe))), 0.0)
    scale = scale_f.to(scale_dtype)
    inv = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    q = (blocks * inv.unsqueeze(-1)).clamp(-output_max, output_max)
    q = q.to(out_dtype).view(1, x.shape[0], x.shape[1])
    return q, scale


# Compile-stage support gate: sm100_block_scale_matmul exact per-side cases

_DT_FP4, _DT_E4M3, _DT_E5M2, _DT_E8M0 = (
    cudnn.data_type.FP4_E2M1,
    cudnn.data_type.FP8_E4M3,
    cudnn.data_type.FP8_E5M2,
    cudnn.data_type.FP8_E8M0,
)
# (a_dt, sf_dt, b_dt, block_size) for the 6 supported cases.
_SUPPORTED_BS_CASES = [
    (_DT_FP4, _DT_E4M3, _DT_FP4, 16),  # 1 nvfp4
    (_DT_FP4, _DT_E8M0, _DT_FP4, 32),  # 2 mxfp4
    (_DT_E4M3, _DT_E8M0, _DT_E4M3, 32),  # 3 mxfp8 e4m3×e4m3
    (_DT_E4M3, _DT_E8M0, _DT_E5M2, 32),  # 4 mxfp8 e4m3×e5m2
    (_DT_E5M2, _DT_E8M0, _DT_E4M3, 32),  # 5 mxfp8 e5m2×e4m3
    (_DT_E5M2, _DT_E8M0, _DT_E5M2, 32),  # 6 mxfp8 e5m2×e5m2
]


@pytest.mark.parametrize("a_dt,sf_dt,b_dt,bs", _SUPPORTED_BS_CASES)
def test_block_scale_gate_accepts_supported(a_dt, sf_dt, b_dt, bs):
    from cudnn.TBD.gemm.compiler import _check_block_scale_supported

    chain = analyze(_build_nvfp4_graph(256, 256, 512, block_size=bs, sf_dt=sf_dt, a_dt=a_dt, b_dt=b_dt))
    _check_block_scale_supported(chain)


def test_block_scale_gate_rejects_mismatches():
    from cudnn.TBD.gemm.compiler import _check_block_scale_supported

    # Missing F8_128x4 SF reorder layout.
    with pytest.raises(NotImplementedError, match="does not support"):
        _check_block_scale_supported(analyze(_build_nvfp4_graph(256, 256, 512, block_size=16, sf_dt=_DT_E4M3, reorder=False)))
    # nvfp4 (fp4+e4m3) with block32 — no supported case.
    with pytest.raises(NotImplementedError, match="does not support"):
        _check_block_scale_supported(analyze(_build_nvfp4_graph(256, 256, 512, block_size=32, sf_dt=_DT_E4M3)))
    # mixed FP4 A / FP8 B (cross-family) — unsupported.
    with pytest.raises(NotImplementedError, match="does not support"):
        _check_block_scale_supported(
            analyze(
                _build_nvfp4_graph(
                    256,
                    256,
                    512,
                    block_size=32,
                    sf_dt=_DT_E8M0,
                    a_dt=_DT_FP4,
                    b_dt=_DT_E4M3,
                )
            )
        )


def test_block_scale_gate_rejects_wrong_arch(monkeypatch):
    import cudnn.TBD.gemm.compiler as compiler
    from cudnn.TBD.gemm.compiler import _check_block_scale_supported

    chain = analyze(_build_nvfp4_graph(256, 256, 512, block_size=16))
    monkeypatch.setattr(compiler, "_current_sm", lambda: 90)
    with pytest.raises(NotImplementedError, match="100 <= SM < 120.*sm_90"):
        _check_block_scale_supported(chain)


# Analyzer pattern matching (no GPU)


def test_analyze_detects_nvfp4_block_scale():
    chain = analyze(_build_nvfp4_graph(128, 256, 256, block_size=16))
    assert chain.has_block_scale
    bs = chain.block_scale
    assert bs.combo == "nvfp4"
    assert bs.block_size == 16
    assert bs.sf_dtype == "fp8_e4m3"
    assert bs.mma_block_scale_kind == "MXF4NVF4"
    assert bs.scale_vec_size == "BLOCK16"
    assert bs.sf_scale_format == 0
    # Operands redirected to the packed FP4 data tensors.
    assert chain.matmul.a_dtype == "fp4_e2m1"
    assert chain.matmul.b_dtype == "fp4_e2m1"
    assert chain.matmul.M == 128 and chain.matmul.N == 256 and chain.matmul.K == 256
    # Per-side info: SF tensors are runtime-positional, not stored as TensorRefs.
    assert bs.both_sided
    assert bs.block_size_a == (1, 16) and bs.block_size_b == (16, 1)
    assert bs.sf_dtype_a == "fp8_e4m3" and bs.sf_dtype_b == "fp8_e4m3"


def test_analyze_detects_mxfp8_block_scale():
    chain = analyze(
        _build_nvfp4_graph(
            128,
            256,
            256,
            block_size=32,
            sf_dt=cudnn.data_type.FP8_E8M0,
            a_dt=cudnn.data_type.FP8_E4M3,
        )
    )
    bs = chain.block_scale
    assert bs.combo == "mxfp8"
    assert bs.block_size == 32 and bs.sf_dtype == "fp8_e8m0"
    assert bs.mma_block_scale_kind == "MXF8F6F4"
    assert bs.scale_vec_size == "BLOCK32"
    assert bs.sf_scale_format == 1


def test_analyze_detects_mxfp4_block_scale():
    chain = analyze(_build_nvfp4_graph(128, 256, 256, block_size=32, sf_dt=cudnn.data_type.FP8_E8M0))
    bs = chain.block_scale
    assert bs.combo == "mxfp4"
    assert bs.block_size == 32 and bs.sf_dtype == "fp8_e8m0"
    assert bs.mma_block_scale_kind == "MXF4NVF4"


def test_plain_matmul_has_no_block_scale():
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, 128, 128], stride=[128 * 128, 128, 1])
    B = g.tensor(name="B", dim=[1, 128, 128], stride=[128 * 128, 1, 128])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    assert not analyze(g).has_block_scale


# End-to-end numerics (GPU)

_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _rand_e8m0(shape, dev):
    # E8M0 holds a power-of-2; small exponents around bias 127 keep the FP32 ref in range.
    return torch.randint(125, 129, shape, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)


def _make_block_scale_inputs(combo, M, N, K, dev="cuda"):
    is_fp4 = combo in ("nvfp4", "mxfp4")
    bs = 16 if combo == "nvfp4" else 32
    sf_k = K // bs
    a_dt = cudnn.data_type.FP4_E2M1 if is_fp4 else cudnn.data_type.FP8_E4M3
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0

    if is_fp4:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
        b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
        a_rt = a_u8.view(torch.float4_e2m1fn_x2)
        b_rt = b_u8.view(torch.float4_e2m1fn_x2)
        a_deq = _unpack_fp4(a_u8, lut).view(M, K)
        b_deq = _unpack_fp4(b_u8, lut).view(N, K)
    else:
        a_rt = (torch.randn(1, M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b_rt = (torch.randn(1, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        a_deq = a_rt.float().view(M, K)
        b_deq = b_rt.float().view(N, K)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((M, sf_k), dev)
        sfb_log = _rand_e8m0((N, sf_k), dev)

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 1)
    b_s = b_deq * sfb_log.float().repeat_interleave(bs, 1)
    ref = a_s @ b_s.t()
    return a_rt, b_rt, sfa_log, sfb_log, ref, bs, sf_dt, a_dt


def _run_bs_numeric(combo, config_name, M, N, K, out_major="n"):
    """Block-scale matmul vs a torch dequant-matmul reference."""
    dev = "cuda"
    torch.manual_seed(0)
    is_fp4 = combo in ("nvfp4", "mxfp4")
    bs = 16 if combo == "nvfp4" else 32
    sf_k = K // bs
    a_dt = cudnn.data_type.FP4_E2M1 if is_fp4 else cudnn.data_type.FP8_E4M3
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0

    if is_fp4:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
        b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
        a_rt = a_u8.view(torch.float4_e2m1fn_x2)
        b_rt = b_u8.view(torch.float4_e2m1fn_x2)
        a_deq = _unpack_fp4(a_u8, lut).view(M, K)
        b_deq = _unpack_fp4(b_u8, lut).view(N, K)
    else:
        a_rt = (torch.randn(1, M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b_rt = (torch.randn(1, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        a_deq = a_rt.float().view(M, K)
        b_deq = b_rt.float().view(N, K)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((M, sf_k), dev)
        sfb_log = _rand_e8m0((N, sf_k), dev)

    g = _build_nvfp4_graph(M, N, K, block_size=bs, sf_dt=sf_dt, a_dt=a_dt, out_major=out_major)
    compiled = _plan(g, **_kw(config_name))
    assert compiled.block_scale and compiled.chain.block_scale.combo == combo

    if out_major == "m":
        c = torch.zeros(1, N, M, dtype=torch.float16, device=dev).transpose(1, 2)
    else:
        c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    compiled(
        _vp_bs(
            compiled,
            a_rt,
            b_rt,
            c,
            _to_blocked(sfa_log).view(1, M, sf_k),
            _to_blocked(sfb_log).view(1, N, sf_k),
        )
    )
    torch.cuda.synchronize()

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 1)
    b_s = b_deq * sfb_log.float().repeat_interleave(bs, 1)
    ref = (a_s @ b_s.t()).to(torch.float16)
    # nvfp4 is bit-exact; mx paths carry fp16 rounding.
    torch.testing.assert_close(c[0], ref, atol=2e-1, rtol=2e-2)


def _run_bs_nonpacked_numeric(combo, config_name, M, N, K, mode):
    dev = "cuda"
    torch.manual_seed(0)
    is_fp4 = combo in ("nvfp4", "mxfp4")
    bs = 16 if combo == "nvfp4" else 32
    sf_k = K // bs
    a_dt = cudnn.data_type.FP4_E2M1 if is_fp4 else cudnn.data_type.FP8_E4M3
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0
    c_pad = 16

    if is_fp4:
        assert mode == "padded"
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        pad = 16
        a_storage = torch.randint(0, 256, (1, M, K // 2 + pad), dtype=torch.uint8, device=dev)
        b_storage = torch.randint(0, 256, (1, N, K // 2 + pad), dtype=torch.uint8, device=dev)
        a_u8 = a_storage[:, :, : K // 2]
        b_u8 = b_storage[:, :, : K // 2]
        a_rt = a_u8.view(torch.float4_e2m1fn_x2)
        b_rt = b_u8.view(torch.float4_e2m1fn_x2)
        a_deq = _unpack_fp4(a_u8, lut).view(M, K)
        b_deq = _unpack_fp4(b_u8, lut).view(N, K)
    elif mode == "zero_stride":
        a_base = (torch.randn(K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b_base = (torch.randn(K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        a_rt = torch.as_strided(a_base, (1, M, K), (0, 0, 1))
        b_rt = torch.as_strided(b_base, (1, N, K), (0, 0, 1))
        a_deq = a_rt.float()[0]
        b_deq = b_rt.float()[0]
    else:
        pad = 16
        a_storage = (torch.randn(1, M, K + pad, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b_storage = (torch.randn(1, N, K + pad, device=dev) * 0.5).to(torch.float8_e4m3fn)
        a_rt = a_storage[:, :, :K]
        b_rt = b_storage[:, :, :K]
        a_deq = a_rt.float()[0]
        b_deq = b_rt.float()[0]

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((M, sf_k), dev)
        sfb_log = _rand_e8m0((N, sf_k), dev)

    g = _build_nvfp4_graph(M, N, K, block_size=bs, sf_dt=sf_dt, a_dt=a_dt)
    compiled = _plan(g, **_kw(config_name))
    c_storage = torch.zeros(1, M, N + c_pad, dtype=torch.float16, device=dev)
    c = c_storage[:, :, :N]
    assert not a_rt.is_contiguous() or not b_rt.is_contiguous()
    assert not c.is_contiguous()

    compiled(
        _vp_bs(
            compiled,
            a_rt,
            b_rt,
            c,
            _to_blocked(sfa_log).view(1, M, sf_k),
            _to_blocked(sfb_log).view(1, N, sf_k),
        )
    )
    torch.cuda.synchronize()

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 1)
    b_s = b_deq * sfb_log.float().repeat_interleave(bs, 1)
    ref = (a_s @ b_s.t()).to(torch.float16)
    torch.testing.assert_close(c[0], ref, atol=2e-1, rtol=2e-2)


@_GPU
@pytest.mark.parametrize(
    "combo,config_name,M,N,K",
    [
        # nvfp4 (fp4 + e4m3 scale, block16) — bit-exact (integer-valued operands).
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
            256,
            256,
            512,
        ),  # acc_stages=1
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            128,
            128,
            256,
        ),  # acc_stages=2
        # mxfp4 (fp4 + e8m0 scale, block32).
        (
            "mxfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
            256,
            256,
            512,
        ),
        # mxfp8 (fp8 e4m3 + e8m0 scale, block32) — multi N-block + single N-block.
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
            256,
            512,
            512,
        ),
        (
            "mxfp8",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            128,
            128,
            256,
        ),
        # Multicast cluster configs — validate SF multicast. 512³ keeps every CTA active.
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x4_1ctamma",
            512,
            512,
            512,
        ),  # A-multicast x4
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x2_1ctamma",
            512,
            512,
            512,
        ),  # A+B multicast
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x1_1ctamma",
            512,
            512,
            512,
        ),  # B-multicast x4
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x2_1ctamma",
            512,
            512,
            512,
        ),  # mx + A+B multicast
        # 2-CTA MMA pair: cta_n=128 → non-overlap acc (acc_stages=2); cta_n=256 → acc-overlap.
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
            256,
            128,
            512,
        ),  # non-overlap
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),  # acc-overlap
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),  # mx + overlap
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x1_2ctamma",
            512,
            512,
            512,
        ),  # B-mcast pair
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x2_2ctamma",
            512,
            512,
            512,
        ),  # A+B-mcast pair
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x2_2ctamma",
            512,
            512,
            512,
        ),  # mx + A+B pair
        # Static scheduler (no CLC) — cta_n=128 acc_stages=2, cta_n=256 acc-overlap.
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_static",
            128,
            128,
            256,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma_static",
            256,
            256,
            512,
        ),  # acc-overlap
        (
            "mxfp8",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_static",
            256,
            256,
            256,
        ),
        # 2-CTA pair static (no CLC, 1 tile/pair). cta_n=128 non-overlap + cta_n=256 overlap.
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma_static",
            256,
            128,
            512,
        ),  # non-overlap
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",
            256,
            256,
            512,
        ),  # acc-overlap
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",
            256,
            256,
            512,
        ),  # mx + overlap
    ],
)
def test_block_scale_numerics(combo, config_name, M, N, K):
    _run_bs_numeric(combo, config_name, M, N, K)


def _run_bs_quant_numeric(
    config_name,
    M,
    N,
    K,
    out_dt,
    out_torch_dt,
    scale_dt,
    scale_torch_dt,
    scale_reorder=False,
    combo="nvfp4",
    scale_dim=None,
    global_scale=None,
):
    dev = "cuda"
    torch.manual_seed(0)
    a_rt, b_rt, sfa_log, sfb_log, ref, bs, sf_dt, a_dt = _make_block_scale_inputs(combo, M, N, K, dev)
    global_scale_tensor = None
    if global_scale is not None:
        global_scale_tensor = torch.tensor([[[global_scale]]], dtype=torch.float32, device=dev)
        ref = ref * global_scale
    g = _build_block_scale_quant_graph(
        M,
        N,
        K,
        dequant_block_size=bs,
        quant_block_size=32,
        sf_dt=sf_dt,
        a_dt=a_dt,
        out_dt=out_dt,
        scale_dt=scale_dt,
        scale_reorder=scale_reorder,
        scale_dim=scale_dim,
        global_scale=global_scale is not None,
    )
    compiled = _plan(g, **_kw(config_name))
    assert compiled.block_scale and compiled.chain.block_quant is not None

    q = torch.empty(1, M, N, dtype=out_torch_dt, device=dev)
    q_scale_shape = scale_dim if scale_dim is not None else (1, M, N // 32)
    if scale_reorder:
        q_scale = torch.zeros(*q_scale_shape, dtype=scale_torch_dt, device=dev)
    else:
        q_scale = torch.empty(*q_scale_shape, dtype=scale_torch_dt, device=dev)
    aux = () if global_scale_tensor is None else (global_scale_tensor,)
    sf_k_padded = _ceil_div(K // bs, 4) * 4
    sfa_rows_padded = _ceil_div(M, 128) * 128
    sfb_rows_padded = _ceil_div(N, 128) * 128
    compiled(
        _vp_bs(
            compiled,
            a_rt,
            b_rt,
            [q, q_scale],
            _to_blocked(sfa_log).view(1, sfa_rows_padded, sf_k_padded),
            _to_blocked(sfb_log).view(1, sfb_rows_padded, sf_k_padded),
            *aux,
        )
    )
    torch.cuda.synchronize()

    q_ref, scale_ref = _block_quant_ref(ref, 32, out_torch_dt, scale_torch_dt)
    if scale_reorder:
        scale_ref = _to_blocked(scale_ref[0]).view_as(q_scale)
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


@_GPU
@pytest.mark.parametrize(
    "out_dt,out_torch_dt,scale_dt,scale_torch_dt,scale_reorder",
    [
        (
            cudnn.data_type.FP8_E4M3,
            torch.float8_e4m3fn,
            cudnn.data_type.FP8_E8M0,
            torch.float8_e8m0fnu,
            False,
        ),
        (
            cudnn.data_type.FP8_E5M2,
            torch.float8_e5m2,
            cudnn.data_type.FP8_E8M0,
            torch.float8_e8m0fnu,
            False,
        ),
        (
            cudnn.data_type.FP8_E4M3,
            torch.float8_e4m3fn,
            cudnn.data_type.FP8_E4M3,
            torch.float8_e4m3fn,
            False,
        ),
        (
            cudnn.data_type.FP8_E4M3,
            torch.float8_e4m3fn,
            cudnn.data_type.FP8_E8M0,
            torch.float8_e8m0fnu,
            True,
        ),
    ],
    ids=(
        "e4m3_out_e8m0_scale",
        "e5m2_out_e8m0_scale",
        "e4m3_out_e4m3_scale",
        "e4m3_out_e8m0_scale_f8_128x4",
    ),
)
def test_block_scale_quant_epilogue_1cta(out_dt, out_torch_dt, scale_dt, scale_torch_dt, scale_reorder):
    _run_bs_quant_numeric(
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        128,
        128,
        256,
        out_dt,
        out_torch_dt,
        scale_dt,
        scale_torch_dt,
        scale_reorder=scale_reorder,
    )


@_GPU
def test_block_scale_quant_epilogue_2cta():
    _run_bs_quant_numeric(
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
        256,
        256,
        512,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
    )


@_GPU
def test_block_scale_quant_epilogue_fp4_input_global_scale_padded_f8_scale():
    _run_bs_quant_numeric(
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        144,
        160,
        256,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        scale_reorder=True,
        combo="nvfp4",
        scale_dim=(1, 256, 8),
        global_scale=0.5,
    )


@_GPU
@pytest.mark.parametrize(
    "combo,config_name,M,N,K",
    [
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            256,
            256,
            256,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma_static",
            256,
            256,
            512,
        ),
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",
            256,
            256,
            512,
        ),
    ],
)
def test_block_scale_m_major(combo, config_name, M, N, K):
    """M-major block-scale output across dynamic/static 1-CTA and 2-CTA."""
    _run_bs_numeric(combo, config_name, M, N, K, out_major="m")


@_GPU
@pytest.mark.parametrize(
    "combo,config_name,mode",
    [
        ("nvfp4", "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma", "padded"),
        ("nvfp4", "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma", "padded"),
        (
            "mxfp8",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_static",
            "zero_stride",
        ),
    ],
)
def test_block_scale_nonpacked_tensors(combo, config_name, mode):
    _run_bs_nonpacked_numeric(combo, config_name, 256, 256, 512, mode)


def _assert_block_scale_reduction_close(actual, expected, mode):
    if mode == cudnn.reduction_mode.ADD:
        torch.testing.assert_close(actual, expected, atol=2.0, rtol=1e-4)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)


def _run_bs_reduction_numeric(combo, config_name, M, N, K, mode, red_dims, red_stride, ref_dims):
    dev = "cuda"
    torch.manual_seed(0)
    a_rt, b_rt, sfa_log, sfb_log, ref, bs, sf_dt, a_dt = _make_block_scale_inputs(combo, M, N, K, dev)
    g = _build_block_scale_reduction_graph(
        M,
        N,
        K,
        mode,
        red_dims,
        block_size=bs,
        sf_dt=sf_dt,
        a_dt=a_dt,
        red_stride=red_stride,
    )
    compiled = _plan(g, **_kw(config_name))
    assert compiled.block_scale and compiled.chain.reductions

    c_term = torch.empty(1, M, N, dtype=torch.float32, device=dev)
    if red_stride is None:
        c_red = torch.empty(*red_dims, dtype=torch.float32, device=dev)
    else:
        c_red = torch.empty_strided(tuple(red_dims), tuple(red_stride), dtype=torch.float32, device=dev)
        assert not c_red.is_contiguous()
    compiled(
        _vp_bs(
            compiled,
            a_rt,
            b_rt,
            [c_term, c_red],
            _to_blocked(sfa_log).view(1, M, K // bs),
            _to_blocked(sfb_log).view(1, N, K // bs),
        )
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(c_term[0], ref, atol=2e-1, rtol=2e-2)
    red_ref = _reduction_ref(c_term, mode, ref_dims)
    _assert_block_scale_reduction_close(c_red, red_ref, mode)


@_GPU
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.AMAX,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
    ids=("add", "amax", "max", "min"),
)
@pytest.mark.parametrize(
    "combo,config_name,M,N,K",
    [
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            128,
            128,
            256,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x2_2ctamma",
            512,
            512,
            512,
        ),
    ],
)
def test_block_scale_reduction_scalar(mode, combo, config_name, M, N, K):
    _run_bs_reduction_numeric(
        combo,
        config_name,
        M,
        N,
        K,
        mode,
        red_dims=[1, 1, 1],
        red_stride=None,
        ref_dims=(0, 1, 2),
    )


@_GPU
@pytest.mark.parametrize(
    "config_name,M,N,K",
    [
        (
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_static",
            128,
            128,
            256,
        ),
        (
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",
            256,
            256,
            512,
        ),
    ],
)
def test_block_scale_reduction_static_templates(config_name, M, N, K):
    _run_bs_reduction_numeric(
        "nvfp4",
        config_name,
        M,
        N,
        K,
        cudnn.reduction_mode.ADD,
        red_dims=[1, 1, 1],
        red_stride=None,
        ref_dims=(0, 1, 2),
    )


@_GPU
@pytest.mark.parametrize(
    "mode,red_dims,red_stride,ref_dims",
    [
        (cudnn.reduction_mode.ADD, [1, 1, 256], [0, 0, 2], (0, 1)),
        (cudnn.reduction_mode.AMAX, [1, 256, 1], [0, 2, 1], (0, 2)),
    ],
    ids=("add_per_col_strided_n", "amax_per_row_strided_m"),
)
def test_block_scale_reduction_strided_output(mode, red_dims, red_stride, ref_dims):
    _run_bs_reduction_numeric(
        "nvfp4",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
        256,
        256,
        512,
        mode,
        red_dims,
        red_stride,
        ref_dims,
    )


def test_block_scale_reduction_rejects_int32():
    g = _build_block_scale_reduction_graph(
        128,
        128,
        256,
        cudnn.reduction_mode.ADD,
        [1, 1, 1],
        red_dtype=cudnn.data_type.INT32,
        red_compute_dtype=cudnn.data_type.INT32,
    )
    with pytest.raises(NotImplementedError, match="fp32 compute/output"):
        jit_from_cudnn_graph(g, **_kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))


@_GPU
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_1ctamma",  # M-OOB with cgrp_m > 1
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x2_1ctamma",  # N tile/cluster > N
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",  # 2-CTA pair + acc-overlap
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_static",  # static scheduler + OOB
    ],
)
def test_nvfp4_oob_shape(config_name):
    """nvfp4 -> bf16 on an awkward shape (M=23, N=56, K=736): exercises
    ceil-padded SF descriptors + M/N/K OOB."""
    dev = "cuda"
    M, N, K, bs = 23, 56, 736, 16
    sf_k = K // bs
    Kp = ((sf_k + 3) // 4) * 4
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    torch.manual_seed(0)
    a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
    sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=[M * K, K, 1],
        data_type=cudnn.data_type.FP4_E2M1,
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.FP4_E2M1,
    )
    SFA = g.tensor(
        name="SFA",
        dim=[1, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[1, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, bs])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[bs, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    compiled = _plan(g, **_kw(config_name))

    c = torch.zeros(1, M, N, dtype=torch.bfloat16, device=dev)
    compiled(
        _vp_bs(
            compiled,
            a_u8.view(torch.float4_e2m1fn_x2),
            b_u8.view(torch.float4_e2m1fn_x2),
            c,
            _to_blocked(sfa_log).view(1, 128, Kp),
            _to_blocked(sfb_log).view(1, 128, Kp),
        )
    )
    torch.cuda.synchronize()

    a_deq = _unpack_fp4(a_u8, lut).view(M, K) * sfa_log.float().repeat_interleave(bs, 1)
    b_deq = _unpack_fp4(b_u8, lut).view(N, K) * sfb_log.float().repeat_interleave(bs, 1)
    ref = (a_deq @ b_deq.t()).to(torch.bfloat16)
    torch.testing.assert_close(c[0], ref, atol=2e-1, rtol=2e-2)


# mxfp8 M-major A / N-major B (operand-major layouts). FP4 stays K-major only;
# the SF layout is unchanged — only the packed-data descriptor flips.
@_GPU
@pytest.mark.parametrize(
    "config_name,M,N,K",
    [
        ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma", 128, 128, 256),
        (
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
            256,
            512,
            512,
        ),  # B has 2 N-groups
        (
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),  # 2-CTA + overlap
    ],
)
def test_mxfp8_m_major_a_n_major_b(config_name, M, N, K):
    dev = "cuda"
    torch.manual_seed(0)
    bs = 32
    sf_k = K // bs

    # K-major data re-laid-out so A is M-contiguous, B N-contiguous (same values → same ref).
    a_rt = (torch.randn(1, M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    b_rt = (torch.randn(1, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    a_deq = a_rt.float()[0]
    b_deq = b_rt.float()[0]
    a_rt_m = a_rt.transpose(1, 2).contiguous().transpose(1, 2)  # M contiguous
    b_rt_n = b_rt.transpose(1, 2).contiguous().transpose(1, 2)  # N contiguous
    assert a_rt_m.stride() == (M * K, 1, M) and b_rt_n.stride() == (N * K, 1, N)

    sfa_log = _rand_e8m0((M, sf_k), dev)
    sfb_log = _rand_e8m0((N, sf_k), dev)

    g = _build_nvfp4_graph(
        M,
        N,
        K,
        block_size=bs,
        sf_dt=cudnn.data_type.FP8_E8M0,
        a_dt=cudnn.data_type.FP8_E4M3,
        a_major="m",
        b_major="n",
    )
    chain = analyze(g)
    assert chain.matmul.a_major == "m" and chain.matmul.b_major == "n"
    compiled = _plan(g, **_kw(config_name))

    c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    compiled(
        _vp_bs(
            compiled,
            a_rt_m,
            b_rt_n,
            c,
            _to_blocked(sfa_log).view(1, M, sf_k),
            _to_blocked(sfb_log).view(1, N, sf_k),
        )
    )
    torch.cuda.synchronize()

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 1)
    b_s = b_deq * sfb_log.float().repeat_interleave(bs, 1)
    ref = (a_s @ b_s.t()).to(torch.float16)
    torch.testing.assert_close(c[0], ref, atol=2e-1, rtol=2e-2)


def test_fp4_rejects_non_k_major():
    """FP4 must be K-major — sub-byte packing mis-strides an M/N-major
    descriptor, so the compiler rejects it at JIT time."""
    M = N = K = 256
    for a_major, b_major in (("m", "k"), ("k", "n")):
        g = _build_nvfp4_graph(
            M,
            N,
            K,
            block_size=16,
            sf_dt=cudnn.data_type.FP8_E4M3,
            a_dt=cudnn.data_type.FP4_E2M1,
            a_major=a_major,
            b_major=b_major,
        )
        with pytest.raises(ValueError, match="must be K-major"):
            jit_from_cudnn_graph(g, **_kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))
