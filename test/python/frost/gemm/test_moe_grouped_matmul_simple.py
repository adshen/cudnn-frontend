"""MoE grouped matmul forward (mode=NONE): analyzer detection + end-to-end
correctness vs a torch group-loop reference (uneven + empty groups)."""

from __future__ import annotations

import cudnn
import cudnn.frost.gemm  # noqa: F401  (installs hook)
import pytest
import torch

from cudnn.frost.gemm.compiler import jit_from_cudnn_graph
from cudnn.frost.gemm.graph_analyzer import analyze
from cudnn.frost.gemm.tile_config import CATALOG

pytestmark = pytest.mark.L0


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


def _vp_moe(compiled, token, weight, fto, output):
    """MoE single-GEMM variant-pack dict from the binding."""
    bd = compiled.binding
    outs = list(output) if isinstance(output, (list, tuple)) else [output]
    vp = {
        bd.a_operands[0]: token,
        bd.b_operands[0]: weight,
        bd.first_token_offset: fto,
    }
    vp.update({t: buf for t, buf in zip(bd.outputs, outs)})
    return vp


_CFG = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1"
# (config name, cta_group): 2-CTA cluster2x1 (reference) + 1-CTA cluster1x1.
_GEOMETRIES = [
    ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2),
    ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", 1),
]
_FULL_EXPERT_REDUCE_OFFSETS = [
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

_QUANT_CASES = [
    (
        "e4m3_out_e8m0_scale",
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        [64, 0, 128, 64],
        256,
    ),
    (
        "e5m2_out_e8m0_scale",
        cudnn.data_type.FP8_E5M2,
        torch.float8_e5m2,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        [64, 0, 128, 64],
        256,
    ),
    (
        "e4m3_out_e4m3_scale",
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        False,
        [64, 0, 128, 64],
        256,
    ),
    (
        "e4m3_out_e8m0_scale_f8_128x4",
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        True,
        [100, 0, 120, 80],
        160,
    ),
]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _quant_scale_shape(S: int, N: int, reorder: bool) -> tuple[int, int, int]:
    if reorder:
        return (1, _ceil_div(S, 128) * 128, _ceil_div(N // 32, 4) * 4)
    return (1, S, N // 32)


def _to_blocked(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    nrb, ncb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _build_graph(
    E: int,
    S: int,
    N: int,
    K: int,
    mode=cudnn.moe_grouped_matmul_mode.NONE,
    token_index=None,
    offset_dt=cudnn.data_type.INT32,
    num_groups: int | None = None,
    output_dt=cudnn.data_type.BFLOAT16,
    reduction_mode=None,
    reduction_dims: tuple[int, int, int] | None = None,
    reduction_stride: tuple[int, int, int] | None = None,
    reduction_dt=cudnn.data_type.FLOAT,
    reduction_compute_dt=None,
    reduction_group_offset: bool = False,
    quant: bool = False,
    quant_out_dt=cudnn.data_type.FP8_E4M3,
    quant_scale_dt=cudnn.data_type.FP8_E8M0,
    quant_scale_reorder: bool = False,
    quant_scale_dim: tuple[int, int, int] | None = None,
):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(
        name="token",
        dim=[1, S, K],
        stride=[S * K, K, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    w = g.tensor(
        name="weight",
        dim=[E, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.BFLOAT16,
    )
    fto_groups = E if num_groups is None else num_groups
    fto = g.tensor(
        name="first_token_offset",
        dim=[fto_groups, 1, 1],
        stride=[1, 1, 1],
        data_type=offset_dt,
    )
    kwargs = {} if token_index is None else {"token_index": token_index}
    out = g.moe_grouped_matmul(
        tok,
        w,
        fto,
        mode=mode,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
        **kwargs,
    )
    if reduction_mode is not None:
        red_kwargs = {}
        if reduction_compute_dt is not None:
            red_kwargs["compute_data_type"] = reduction_compute_dt
        if reduction_group_offset:
            red_kwargs["group_offset"] = fto
        R = g.reduction(input=out, mode=reduction_mode, name="red", **red_kwargs)
        assert reduction_dims is not None
        stride = reduction_stride
        if stride is None:
            stride = (
                reduction_dims[1] * reduction_dims[2],
                reduction_dims[2],
                1,
            )
        R.set_dim(list(reduction_dims)).set_stride(list(stride))
        R.set_output(True).set_data_type(reduction_dt)
    if quant:
        q, q_scale = g.block_scale_quantize(input=out, block_size=32, name="q")
        q.set_data_type(quant_out_dt).set_output(True)
        if quant_scale_dim is not None:
            q_scale.set_dim(list(quant_scale_dim)).set_stride([quant_scale_dim[1] * quant_scale_dim[2], quant_scale_dim[2], 1])
        q_scale.set_data_type(quant_scale_dt).set_output(True)
        if quant_scale_reorder:
            q_scale.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
        return g
    out.set_data_type(output_dt).set_output(True)
    return g


# --------------------------------------------------------------------------- #
# Analyzer (no GPU needed)
# --------------------------------------------------------------------------- #


def test_analyzer_detects_moe() -> None:
    E, S, N, K = 8, 768, 256, 128
    chain = analyze(_build_graph(E, S, N, K))
    assert chain.has_moe
    assert chain.moe.num_experts == E
    assert chain.moe.mode == "none"
    assert chain.moe.offset_dtype == "int32"
    assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (S, N, K)
    assert chain.matmul.a_major == "k" and chain.matmul.b_major == "k"
    assert chain.output_dtype == "bf16"


def test_analyzer_offset_dtype_int64() -> None:
    chain = analyze(_build_graph(8, 768, 256, 128, offset_dt=cudnn.data_type.INT64))
    assert chain.moe.offset_dtype == "int64"


def test_analyzer_detects_moe_reduction() -> None:
    chain = analyze(
        _build_graph(
            8,
            768,
            256,
            128,
            reduction_mode=cudnn.reduction_mode.AMAX,
            reduction_dims=(1, 1, 256),
        )
    )
    assert chain.has_moe
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "amax"
    assert chain.reductions[0].source_ref < 0
    assert not chain.reductions[0].grouped_by_moe
    assert [o.source for o in chain.outputs] == ["terminal", "reduction_0"]


def test_analyzer_detects_moe_group_reduction() -> None:
    chain = analyze(
        _build_graph(
            8,
            768,
            256,
            128,
            reduction_mode=cudnn.reduction_mode.AMAX,
            reduction_dims=(8, 1, 1),
            reduction_group_offset=True,
        )
    )
    assert chain.has_moe
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "amax"
    assert chain.reductions[0].dim == (8, 1, 1)
    assert chain.reductions[0].grouped_by_moe


def test_analyzer_rejects_moe_group_reduction_without_group_offset() -> None:
    with pytest.raises(ValueError, match="axis 0"):
        analyze(
            _build_graph(
                8,
                768,
                256,
                128,
                reduction_mode=cudnn.reduction_mode.AMAX,
                reduction_dims=(8, 1, 1),
            )
        )


def test_analyzer_rejects_moe_group_reduction_wrong_offset_dim() -> None:
    with pytest.raises(ValueError, match="groupOffset.*num_groups"):
        analyze(
            _build_graph(
                8,
                768,
                256,
                128,
                reduction_mode=cudnn.reduction_mode.AMAX,
                reduction_dims=(1, 1, 1),
                reduction_group_offset=True,
            )
        )


def test_analyzer_rejects_gather() -> None:
    E, S, N, K = 8, 768, 256, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(
        name="token",
        dim=[1, S, K],
        stride=[S * K, K, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    w = g.tensor(
        name="weight",
        dim=[E, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.BFLOAT16,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[E, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    idx = g.tensor(name="idx", dim=[S, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(
        tok,
        w,
        fto,
        token_index=idx,
        mode=cudnn.moe_grouped_matmul_mode.GATHER,
        name="moe",
    )
    out.set_output(True)
    with pytest.raises(NotImplementedError, match="mode=NONE"):
        analyze(g)


# --------------------------------------------------------------------------- #
# End-to-end correctness (GPU)
# --------------------------------------------------------------------------- #


def _offsets(group_sizes, S, dtype=torch.int32):
    starts, cur = [], 0
    for gs in group_sizes:
        starts.append(cur)
        cur += gs
    assert cur == S
    return torch.tensor(starts, dtype=dtype, device="cuda")


def _ref(token, weight, offsets, S, N, E):
    out = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    starts = offsets.tolist()
    for g in range(len(starts)):
        b = starts[g]
        e = starts[g + 1] if g + 1 < len(starts) else S
        if b == e:
            continue
        out[b:e] = token[0, b:e].float() @ weight[g % E].float().T
    return out.to(torch.bfloat16)


def _ref_f32(token, weight, offsets, S, N, E):
    out = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    starts = offsets.tolist()
    for g in range(len(starts)):
        b = starts[g]
        e = starts[g + 1] if g + 1 < len(starts) else S
        if b == e:
            continue
        out[b:e] = token[0, b:e].float() @ weight[g % E].float().T
    return out


def _block_quant_ref(x, block_size, out_dtype, scale_dtype):
    blocks = x.view(1, x.shape[0], x.shape[1] // block_size, block_size)
    output_max = 448.0 if out_dtype is torch.float8_e4m3fn else 57344.0
    scale_f = blocks.abs().amax(dim=-1) / output_max
    if scale_dtype is torch.float8_e8m0fnu:
        safe = torch.where(scale_f > 0, scale_f, 1.0)
        scale_f = torch.where(scale_f > 0, torch.pow(2.0, torch.ceil(torch.log2(safe))), 0.0)
    scale = scale_f.to(scale_dtype)
    inv = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    q = (blocks * inv.unsqueeze(-1)).clamp(-output_max, output_max)
    q = q.to(out_dtype).view(1, x.shape[0], x.shape[1])
    return q, scale


def _block_quant_q_atol(scale_dtype) -> float:
    # Non-pow2 E4M3 scales use the kernel's approximate reciprocal → up to one
    # smallest E4M3 output step off the torch reference.
    return 1.0 / 512.0 if scale_dtype is torch.float8_e4m3fn else 0.0


def _reduction_ref(x: torch.Tensor, mode, dims: tuple[int, ...]) -> torch.Tensor:
    if mode == cudnn.reduction_mode.AMAX:
        return x.abs().amax(dim=dims, keepdim=True)
    if mode == cudnn.reduction_mode.MAX:
        return x.amax(dim=dims, keepdim=True)
    if mode == cudnn.reduction_mode.MIN:
        return x.amin(dim=dims, keepdim=True)
    return x.sum(dim=dims, keepdim=True)


def _group_reduction_ref(
    x: torch.Tensor,
    offsets: torch.Tensor,
    mode,
    out_dims: tuple[int, int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    group_count, _, n = out_dims
    starts = offsets.tolist()
    out = torch.empty(out_dims, dtype=out_dtype, device=x.device)
    if mode in (cudnn.reduction_mode.ADD, cudnn.reduction_mode.AMAX):
        out.fill_(0)
    elif mode == cudnn.reduction_mode.MAX:
        out.fill_(-(2**31) if out_dtype == torch.int32 else -float("inf"))
    elif mode == cudnn.reduction_mode.MIN:
        out.fill_(2**31 - 1 if out_dtype == torch.int32 else float("inf"))
    else:
        raise AssertionError(f"unsupported reduction mode {mode!r}")
    for g in range(group_count):
        begin = starts[g]
        end = starts[g + 1] if g + 1 < group_count else x.shape[0]
        if begin == end:
            continue
        src = x[begin:end].to(out_dtype) if out_dtype == torch.int32 else x[begin:end]
        if out_dims[1:] == (1, 1):
            reduce_dims = (0, 1)
            out[g, 0, 0] = _reduction_ref(src, mode, reduce_dims)
        elif out_dims[1:] == (1, x.shape[1]):
            out[g, 0, :n] = _reduction_ref(src, mode, (0,)).view(-1)
        else:
            raise AssertionError(f"unsupported group reduction dims {out_dims}")
    return out


def _group_sizes_from_offsets(offsets: list[int], total: int) -> list[int]:
    return [(offsets[i + 1] if i + 1 < len(offsets) else total) - offsets[i] for i in range(len(offsets))]


def _reduction_dims(out_dims: tuple[int, int, int], full: tuple[int, int, int]):
    return tuple(i for i, (out_extent, full_extent) in enumerate(zip(out_dims, full)) if out_extent == 1 and full_extent != 1)


def _mk_nonpacked_data(S, N, K, E, mode):
    torch.manual_seed(0)
    if mode == "zero_stride":
        token_base = torch.randn(K, dtype=torch.bfloat16, device="cuda")
        weight_base = torch.randn(K, dtype=torch.bfloat16, device="cuda")
        token = torch.as_strided(token_base, (1, S, K), (0, 0, 1))
        weight = torch.as_strided(weight_base, (E, N, K), (0, 0, 1))
    else:
        pad = 16
        token_storage = torch.randn(1, S, K + pad, dtype=torch.bfloat16, device="cuda")
        weight_storage = torch.randn(E, N, K + pad, dtype=torch.bfloat16, device="cuda")
        token = token_storage[:, :, :K]
        weight = weight_storage[:, :, :K]
    output_storage = torch.zeros(1, S, N + 16, dtype=torch.bfloat16, device="cuda")
    return token, weight, output_storage[:, :, :N]


# first_token_offset accepts INT32 or INT64; the kernel bakes the dtype at JIT
# and casts reads to Int32 internally.
_OFFSET_DTYPES = [
    (cudnn.data_type.INT32, torch.int32),
    (cudnn.data_type.INT64, torch.int64),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize("offset_cudnn_dt,offset_torch_dt", _OFFSET_DTYPES)
@pytest.mark.parametrize(
    "group_sizes",
    [
        [64, 0, 200, 128, 100, 12, 196, 68],  # uneven + one empty group
        [96, 96, 96, 96, 96, 96, 96, 96],  # balanced
        [768, 0, 0, 0, 0, 0, 0, 0],  # all tokens in group 0
    ],
)
def test_moe_e2e(group_sizes, offset_cudnn_dt, offset_torch_dt, cfg_name, cta_group) -> None:
    E, N, K = 8, 256, 128
    S = sum(group_sizes)
    cfg = next(c for c in CATALOG if c.name == cfg_name)
    compiled = _plan(
        _build_graph(E, S, N, K, offset_dt=offset_cudnn_dt),
        config=cfg,
        cta_group=cta_group,
    )

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = _offsets(group_sizes, S, dtype=offset_torch_dt)

    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(output[0], _ref(token, weight, offsets, S, N, E), atol=1e-1, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize(
    "case_name,out_dt,out_torch_dt,scale_dt,scale_torch_dt,scale_reorder,group_sizes,N",
    _QUANT_CASES,
    ids=[case[0] for case in _QUANT_CASES],
)
def test_moe_block_quant_epilogue(
    cfg_name,
    cta_group,
    case_name,
    out_dt,
    out_torch_dt,
    scale_dt,
    scale_torch_dt,
    scale_reorder,
    group_sizes,
    N,
) -> None:
    E, K = 4, 128
    N = int(N)
    S = sum(group_sizes)
    scale_shape = _quant_scale_shape(S, N, scale_reorder)
    cfg = next(c for c in CATALOG if c.name == cfg_name)
    compiled = _plan(
        _build_graph(
            E,
            S,
            N,
            K,
            quant=True,
            quant_out_dt=out_dt,
            quant_scale_dt=scale_dt,
            quant_scale_reorder=scale_reorder,
            quant_scale_dim=scale_shape if scale_reorder else None,
        ),
        config=cfg,
        cta_group=cta_group,
    )

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    q = torch.empty(1, S, N, dtype=out_torch_dt, device="cuda")
    if scale_reorder:
        q_scale = torch.zeros(*scale_shape, dtype=scale_torch_dt, device="cuda")
    else:
        q_scale = torch.empty(*scale_shape, dtype=scale_torch_dt, device="cuda")
    offsets = _offsets(group_sizes, S)

    compiled(_vp_moe(compiled, token, weight, offsets, [q, q_scale]))
    torch.cuda.synchronize()

    ref = _ref_f32(token, weight, offsets, S, N, E)
    q_ref, scale_ref = _block_quant_ref(ref, 32, out_torch_dt, scale_torch_dt)
    if scale_reorder:
        scale_ref = _to_blocked(scale_ref[0]).view_as(q_scale)
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(
        q.float(),
        q_ref.float(),
        atol=_block_quant_q_atol(scale_torch_dt),
        rtol=0,
    )


def _run_moe_reduction(
    cfg_name,
    cta_group,
    mode,
    red_dims,
    *,
    E=4,
    N=128,
    K=128,
    red_stride=None,
    red_dt=cudnn.data_type.FLOAT,
    red_torch_dt=torch.float32,
    red_compute_dt=None,
    integer_inputs=False,
    group_sizes=None,
    group_reduction=False,
) -> None:
    if group_sizes is None:
        group_sizes = [64, 0, 120, 72]
    S = sum(group_sizes)
    cfg = next(c for c in CATALOG if c.name == cfg_name)
    compiled = _plan(
        _build_graph(
            E,
            S,
            N,
            K,
            reduction_mode=mode,
            reduction_dims=tuple(red_dims),
            reduction_stride=red_stride,
            reduction_dt=red_dt,
            reduction_compute_dt=red_compute_dt,
            reduction_group_offset=group_reduction,
            num_groups=len(group_sizes),
        ),
        config=cfg,
        cta_group=cta_group,
    )

    torch.manual_seed(0)
    if integer_inputs:
        token = torch.randint(-2, 3, (1, S, K), device="cuda").to(torch.bfloat16)
        weight = torch.randint(-2, 3, (E, N, K), device="cuda").to(torch.bfloat16)
    else:
        token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(1, S, N, dtype=torch.bfloat16, device="cuda")
    if red_stride is None:
        red = torch.empty(*red_dims, dtype=red_torch_dt, device="cuda")
    else:
        red = torch.empty_strided(red_dims, red_stride, dtype=red_torch_dt, device="cuda")
    offsets = _offsets(group_sizes, S)

    compiled(_vp_moe(compiled, token, weight, offsets, [output, red]))
    torch.cuda.synchronize()

    ref = _ref_f32(token, weight, offsets, S, N, E)
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    red_src = ref.to(red_torch_dt) if red_torch_dt == torch.int32 else ref
    if group_reduction:
        red_ref = _group_reduction_ref(red_src, offsets, mode, tuple(red_dims), red_torch_dt)
    else:
        ref_dims = _reduction_dims(tuple(red_dims), (1, S, N))
        red_ref = _reduction_ref(red_src.view(1, S, N), mode, ref_dims).to(red_torch_dt)
    torch.testing.assert_close(
        red,
        red_ref,
        atol=1e-1 if red_torch_dt == torch.float32 else 0,
        rtol=1e-2 if red_torch_dt == torch.float32 else 0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.AMAX,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
)
def test_moe_reduction_scalar_fp32(mode, cfg_name, cta_group) -> None:
    _run_moe_reduction(cfg_name, cta_group, mode, [1, 1, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize(
    "mode,red_dims,red_stride",
    [
        (cudnn.reduction_mode.ADD, [1, 256, 1], [0, 2, 1]),
        (cudnn.reduction_mode.AMAX, [1, 1, 128], [0, 0, 2]),
    ],
)
def test_moe_reduction_partial_strided_fp32(mode, red_dims, red_stride) -> None:
    _run_moe_reduction(
        _CFG,
        2,
        mode,
        red_dims,
        red_stride=red_stride,
        integer_inputs=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.AMAX,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
)
def test_moe_reduction_scalar_int32(mode) -> None:
    _run_moe_reduction(
        _GEOMETRIES[1][0],
        _GEOMETRIES[1][1],
        mode,
        [1, 1, 1],
        red_dt=cudnn.data_type.INT32,
        red_torch_dt=torch.int32,
        red_compute_dt=cudnn.data_type.INT32,
        integer_inputs=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
def test_moe_group_reduction_amax_scalar_fp32(cfg_name, cta_group) -> None:
    _run_moe_reduction(
        cfg_name,
        cta_group,
        cudnn.reduction_mode.AMAX,
        [4, 1, 1],
        group_sizes=[64, 0, 120, 72],
        group_reduction=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_moe_group_reduction_full_expert_amax_fp32() -> None:
    group_sizes = _group_sizes_from_offsets(_FULL_EXPERT_REDUCE_OFFSETS, 2000)
    _run_moe_reduction(
        _CFG,
        2,
        cudnn.reduction_mode.AMAX,
        [36, 1, 1],
        E=9,
        N=248,
        K=520,
        group_sizes=group_sizes,
        group_reduction=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
)
def test_moe_group_reduction_per_col_fp32(mode) -> None:
    _run_moe_reduction(
        _CFG,
        2,
        mode,
        [4, 1, 128],
        group_sizes=[32, 96, 0, 128],
        group_reduction=True,
        integer_inputs=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize(
    "cfg_name,cta_group,mode",
    [
        ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2, "padded"),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", 1, "padded"),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", 1, "zero_stride"),
    ],
)
def test_moe_nonpacked_tensors(cfg_name, cta_group, mode) -> None:
    group_sizes = [64, 0, 200, 128, 100, 12, 196, 68]
    E, N, K = 8, 256, 128
    S = sum(group_sizes)
    cfg = next(c for c in CATALOG if c.name == cfg_name)
    compiled = _plan(_build_graph(E, S, N, K), config=cfg, cta_group=cta_group)

    token, weight, output = _mk_nonpacked_data(S, N, K, E, mode)
    offsets = _offsets(group_sizes, S)
    assert not token.is_contiguous() or not weight.is_contiguous()
    assert not output.is_contiguous()

    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[0],
        _ref(token, weight, offsets, S, N, E),
        atol=1e-1,
        rtol=1e-2,
    )


def _ref_bxe(token, weight, offsets, S, N, num_experts, num_groups):
    """Reference for BxE > E: group g uses expert g % num_experts."""
    out = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    starts = offsets.tolist()
    for g in range(num_groups):
        b = starts[g]
        e = starts[g + 1] if g + 1 < num_groups else S
        if b == e:
            continue
        out[b:e] = token[0, b:e].float() @ weight[g % num_experts].float().T
    return out.to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
def test_moe_bxe_gt_e(cfg_name, cta_group) -> None:
    """num_groups (BxE) > num_experts (E): expert = group % E."""
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
    num_groups = len(offset_values)  # 36 > E=9
    cfg = next(c for c in CATALOG if c.name == cfg_name)
    compiled = _plan(_build_graph(E, S, N, K), config=cfg, cta_group=cta_group)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(offset_values, dtype=torch.int32, device="cuda")

    # num_experts/num_groups are derived from weight.shape[0] /
    # first_token_offset.shape[0] inside the call.
    compiled(_vp_moe(compiled, token, weight, offsets, output))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[0],
        _ref_bxe(token, weight, offsets, S, N, E, num_groups),
        atol=2e-1,
        rtol=5e-2,
    )
