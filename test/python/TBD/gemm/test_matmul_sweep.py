"""Correctness sweep over (config × dtype-pair × shape) — the matmul regression gate.

Each case builds a frontend graph, JITs through the GEMM hook, and asserts
bit-tight equality vs torch-fp32 (small-integer inputs keep the reduction exact).
Parametrize order is (config, dtype, shape) so each (config, dtype) block reuses
one compiled kernel. CUDNN_GEMM_TEST_FULL=1 expands the config axis to the whole
CATALOG. Also runnable as a script (forwards argv to pytest).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Iterable

import pytest
import torch

# Module-wide GPU gate — every test here is end-to-end and needs a B200.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


import cudnn
import cudnn.TBD.gemm  # noqa: F401  — installs the cudnn.pygraph recorder hook
from cudnn.TBD.gemm.compiler import _current_sm, jit_from_cudnn_graph
from cudnn.TBD.gemm.tile_config import CATALOG


class _Plan:
    """JIT-compiles a recorded graph with a forced tile config (bypassing the
    TBD engine's auto-select). Exposes chain / binding / block_scale / aux_names;
    callable with a variant pack."""

    def __init__(self, graph, config=None, cta_group=2, scheduler="clc", force_stg_epi=False):
        self.g = graph
        kw = dict(cta_group=cta_group, scheduler=scheduler, force_stg_epi=force_stg_epi)
        if config is not None:
            kw["config"] = config
        self._compiled = jit_from_cudnn_graph(graph, **kw)
        self.chain = self._compiled.chain
        self.binding = self._compiled.binding
        self.block_scale = self.chain.has_block_scale
        self.aux_names = [t.name for t in self.chain.aux_tensors]

    def __call__(self, variant_pack):
        return self._compiled(variant_pack)


def _plan(graph, config=None, cta_group=2, scheduler="clc", force_stg_epi=False):
    return _Plan(
        graph,
        config=config,
        cta_group=cta_group,
        scheduler=scheduler,
        force_stg_epi=force_stg_epi,
    )


def _vp(compiled, a, b, outs, *aux):
    """Variant-pack dict {cuDNN tensor: buffer}: A/B operands, outputs, then aux."""
    bd = compiled.binding
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {bd.a_operands[0]: a, bd.b_operands[0]: b}
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


# INT8 matmul runs only on SM 100 or SM 110 (disjoint range).
_INT8_SM_RANGES = ((100, 101), (110, 111))


_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
_CUDNN_DTYPE = {
    "bf16": cudnn.data_type.BFLOAT16,
    "fp16": cudnn.data_type.HALF,
    "fp8_e4m3": cudnn.data_type.FP8_E4M3,
    "fp8_e5m2": cudnn.data_type.FP8_E5M2,
}
_ELEM_BYTES = {"bf16": 2, "fp16": 2, "fp8_e4m3": 1, "fp8_e5m2": 1}


# Shape menu: tile-aligned baseline + M-OOB + K-OOB + combined M+K-OOB. N stays
# aligned across the whole menu — see _compatible() for why.
_WEIRD_SHAPES: tuple[tuple[int, int, int], ...] = (
    # Tile-aligned baseline.
    (384, 768, 384),
    (640, 384, 512),
    (256, 1280, 256),
    (512, 1024, 640),  # K = 5×128
    # M-OOB (N aligned, K aligned).
    (255, 256, 256),  # one row short of a tile
    (200, 256, 256),  # deep inside a partial tile
    # K-OOB (M aligned, N aligned).
    (
        256,
        256,
        200,
    ),  # partial K-tile (valid for BF16/FP16, SKIP for FP8: 16B TMA stride)
    (256, 256, 96),  # smaller than one K_BYTES=128 BF16 tile
    # M + K OOB.
    (255, 256, 240),
)

# (input_dtype, output_dtype) pairs: same-dtype BF16/FP16, FP8 E4M3/E5M2 → FP16,
# and one mixed FP8 → BF16.
_CORE_DTYPE_PAIRS: tuple[tuple[str, str], ...] = (
    ("bf16", "bf16"),
    ("fp16", "fp16"),
    ("fp8_e4m3", "fp16"),
    ("fp8_e5m2", "fp16"),
    ("fp8_e4m3", "bf16"),
)

# Curated config subset — each entry covers a distinct template-architectural
# corner. Full CATALOG sweep is opt-in via CUDNN_GEMM_TEST_FULL=1.
_QUICK_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",  # baseline cta1 single-CTA
    "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",  # large N
    "CONFIG_sm100_128x128x64_128x128x32_cluster1x1_1ctamma",  # K_BYTES=64
    "CONFIG_sm100_128x32x128_128x32x32_cluster1x1_1ctamma",  # smallest N=32
    "CONFIG_sm100_64x128x128_64x128x32_cluster2x1_1ctamma",  # cta1 + B-multicast
    "CONFIG_sm100_128x64x128_128x64x32_cluster1x2_1ctamma",  # cta1 + A-multicast
    "CONFIG_sm100_64x64x128_64x64x32_cluster2x2_1ctamma",  # cta1 both multicasts
    "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",  # cta2 baseline
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",  # cta2 large N
    "CONFIG_sm100_128x256x64_128x256x32_cluster2x1_2ctamma",  # cta2 K_BYTES=64
    "CONFIG_sm100_128x128x128_128x128x32_cluster4x2_2ctamma",  # cta2 big cluster
    "CONFIG_sm100_64x64x128_64x64x32_cluster2x4_2ctamma",  # cta2 cluster-m=128 (cta_tile_m=64)
    # K_BYTES=64 large-cluster coverage — mirrors the K_BYTES=128 entries above.
    "CONFIG_sm100_64x64x64_64x64x32_cluster2x2_1ctamma",  # cta1 both multicasts, K_BYTES=64
    "CONFIG_sm100_128x128x64_128x128x32_cluster4x2_2ctamma",  # cta2 big cluster, K_BYTES=64
    "CONFIG_sm100_64x64x64_64x64x32_cluster2x4_2ctamma",  # cta2 cluster-m=128, K_BYTES=64
    "CONFIG_sm100_128x64x128_128x64x32_cluster1x4_1ctamma_static",  # static scheduler (cta1)
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",  # static scheduler (cta2)
)

_BATCHED_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",  # rank-3 baseline
    "CONFIG_sm100_64x64x128_64x64x32_cluster2x2_1ctamma",  # rank-3 + TMA multicast
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",  # rank-3 + CTA_2 MMA
)

_BATCHED_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    # _WEIRD_SHAPES classes prefixed with batch. Batches in {1,2,3}; the runtime
    # binds to the graph batch, so each distinct batch needs its own compiled anchor.
    (1, 384, 768, 384),
    (2, 640, 384, 512),
    (3, 256, 1280, 256),
    (2, 512, 1024, 640),
    (3, 255, 256, 256),  # M-OOB
    (1, 200, 256, 256),  # M-OOB
    (2, 256, 256, 200),  # K-OOB: SKIP for FP8
    (3, 256, 256, 96),  # K-OOB
    (2, 255, 256, 240),  # M + K OOB
)

_BATCH_BROADCAST_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    # _WEIRD_SHAPES M/N/K classes, output batch > 1 so one input is a real broadcast.
    (2, 384, 768, 384),
    (3, 640, 384, 512),
    (2, 256, 1280, 256),
    (3, 512, 1024, 640),
    (2, 255, 256, 256),  # M-OOB
    (3, 200, 256, 256),  # M-OOB
    (2, 256, 256, 200),  # K-OOB: SKIP for FP8
    (3, 256, 256, 96),  # K-OOB
    (2, 255, 256, 240),  # M + K OOB
)

_BATCH_BROADCAST_CASES = tuple((side, shape) for side in ("A", "B") for shape in _BATCH_BROADCAST_SHAPES)

_INPUT_LAYOUTS: tuple[tuple[str, str], ...] = (
    ("k", "k"),
    ("m", "k"),
    ("k", "n"),
    ("m", "n"),
)
_LAYOUT_DTYPE_PAIRS: tuple[tuple[str, str], ...] = (
    ("bf16", "bf16"),
    ("fp16", "fp16"),
    ("fp8_e4m3", "fp16"),
)
_LAYOUT_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
)
_NONPACKED_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    "CONFIG_sm100_128x64x128_128x64x32_cluster1x4_1ctamma_static",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",
)
_NONPACKED_LAYOUTS: tuple[tuple[str, str], ...] = (
    ("k", "k"),
    ("m", "n"),
)
_NONPACKED_DTYPE_PAIRS: tuple[tuple[str, str], ...] = (
    ("bf16", "bf16"),
    ("fp8_e4m3", "fp16"),
)


_NAME_TO_CFG = {c.name: c for c in CATALOG}

_LEGACY_RE = re.compile(r"^(CONFIG_sm100_\d+x\d+x\d+_\d+x\d+x\d+_cluster\d+x\d+)_([12])ctamma(_static)?$")


def _resolve(legacy_name):
    """Legacy config-name (with _Nctamma/_static, kept as readable test IDs) ->
    (pure-geometry config, cta_group, scheduler)."""
    m = _LEGACY_RE.match(legacy_name)
    assert m, legacy_name
    return (
        _NAME_TO_CFG[m.group(1)],
        int(m.group(2)),
        "static" if m.group(3) else "clc",
    )


def _sweep_config_names() -> list[str]:
    """Quick subset by default; full catalog under CUDNN_GEMM_TEST_FULL=1."""
    if os.environ.get("CUDNN_GEMM_TEST_FULL", "0") == "1":
        return [c.name for c in CATALOG]
    return list(_QUICK_CONFIGS)


# Pretty IDs (drive the `-k` filter and the failure report line).


def _shape_id(s: tuple[int, int, int]) -> str:
    return f"{s[0]}x{s[1]}x{s[2]}"


def _batched_shape_id(s: tuple[int, int, int, int]) -> str:
    return f"B{s[0]}_{s[1]}x{s[2]}x{s[3]}"


def _batch_broadcast_id(p: tuple[str, tuple[int, int, int, int]]) -> str:
    side, s = p
    return f"broadcast{side}_B{s[0]}_{s[1]}x{s[2]}x{s[3]}"


def _dtype_id(p: tuple[str, str]) -> str:
    return f"{p[0]}->{p[1]}"


def _config_id(name: str) -> str:
    """Strip the redundant CONFIG_ prefix and _sm100 suffix from pytest IDs."""
    out = name
    if out.startswith("CONFIG_"):
        out = out[len("CONFIG_") :]
    if out.endswith("_sm100"):
        out = out[: -len("_sm100")]
    return out


def _layout_id(p: tuple[str, str]) -> str:
    return f"A{p[0]}_B{p[1]}"


# Compatibility gate.


def _compatible(
    cfg,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
) -> tuple[bool, str]:
    """Reject only shapes the kernel can't service. Returns (ok, reason)."""
    in_eb = _ELEM_BYTES[in_dtype]
    out_eb = _ELEM_BYTES[out_dtype]
    if cfg.cta_tile_k_bytes % in_eb != 0:
        return False, (f"K_BYTES={cfg.cta_tile_k_bytes} not divisible by in_elem_bytes={in_eb} " f"(catalog × dtype mismatch)")
    a_contig_extent = K if a_major == "k" else M
    b_contig_extent = K if b_major == "k" else N
    if (a_contig_extent * in_eb) % 16 != 0:
        return False, (
            f"A {a_major}-major contiguous extent * in_eb="
            f"{a_contig_extent * in_eb} not 16B-aligned. "
            f"{in_dtype!r} needs that extent % {16 // in_eb} == 0."
        )
    if (b_contig_extent * in_eb) % 16 != 0:
        return False, (
            f"B {b_major}-major contiguous extent * in_eb="
            f"{b_contig_extent * in_eb} not 16B-aligned. "
            f"{in_dtype!r} needs that extent % {16 // in_eb} == 0."
        )
    cta_smem_m, cta_smem_n, _ = cfg.cta_smem_tile_mnk(in_eb, cta_group)
    mn_group_elems = cfg.cta_tile_k_bytes // in_eb
    if a_major == "m" and (cta_smem_m < mn_group_elems or cta_smem_m % mn_group_elems != 0):
        return False, (f"A M-major per-CTA SMEM M={cta_smem_m} is not compatible with " f"the {mn_group_elems}-element swizzle group")
    if b_major == "n" and (cta_smem_n < mn_group_elems or cta_smem_n % mn_group_elems != 0):
        return False, (f"B N-major per-CTA SMEM N={cta_smem_n} is not compatible with " f"the {mn_group_elems}-element swizzle group")
    if (N * out_eb) % 32 != 0:
        return False, (
            f"N*out_eb={N * out_eb} not 32B-aligned — STG full-vec store bakes " f"alignment=VEC_BYTES=32 at JIT. {out_dtype!r} needs N % {32 // out_eb} == 0."
        )
    return True, ""


# Graph + data + reference.


def _a_stride_batched(M: int, K: int, a_major: str) -> list[int]:
    return [M * K, K, 1] if a_major == "k" else [M * K, 1, M]


def _b_stride_batched(N: int, K: int, b_major: str) -> list[int]:
    return [N * K, 1, K] if b_major == "k" else [N * K, N, 1]


def _build_graph(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, b_major))
    C = g.matmul(A=A, B=B, name="mm")
    if out_major == "m":
        C.set_stride([M * N, 1, M])
    C.set_output(True)
    if out_dtype != in_dtype:
        C.set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _build_block_quant_graph(
    M: int,
    N: int,
    K: int,
    block_size: int = 32,
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    Q, QS = g.block_scale_quantize(input=C, block_size=block_size, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    return g


def _build_batched_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, b_major))
    C = g.matmul(A=A, B=B, name="mm")
    if out_major == "m":
        C.set_stride([M * N, 1, M])
    C.set_output(True)
    if out_dtype != in_dtype:
        C.set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _build_batch_broadcast_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    broadcast_side: str,
    a_major: str = "k",
    b_major: str = "k",
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a_batch = 1 if broadcast_side == "A" else batch
    b_batch = 1 if broadcast_side == "B" else batch
    A = g.tensor(name="A", dim=[a_batch, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[b_batch, K, N], stride=_b_stride_batched(N, K, b_major))
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    if out_dtype != in_dtype:
        C.set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _mkdata(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
):
    """Small-integer inputs ⇒ exact FP32 reduction ⇒ kernel and reference differ
    only by the final deterministic downcast. All shapes rank-3, batch=1."""
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    a_shape = (1, M, K) if a_major == "k" else (1, K, M)
    b_shape = (1, N, K) if b_major == "k" else (1, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    if out_major == "m":
        c = torch.empty(1, N, M, dtype=_TORCH_DTYPE[out_dtype], device="cuda").transpose(1, 2)
    else:
        c = torch.empty(1, M, N, dtype=_TORCH_DTYPE[out_dtype], device="cuda")
    return a, b, c


def _mkbatched_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    a_shape = (batch, M, K) if a_major == "k" else (batch, K, M)
    b_shape = (batch, N, K) if b_major == "k" else (batch, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    if out_major == "m":
        c = torch.empty(batch, N, M, dtype=_TORCH_DTYPE[out_dtype], device="cuda").transpose(1, 2)
    else:
        c = torch.empty(batch, M, N, dtype=_TORCH_DTYPE[out_dtype], device="cuda")
    return a, b, c


def _mkbatch_broadcast_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    broadcast_side: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    a_batch = 1 if broadcast_side == "A" else batch
    b_batch = 1 if broadcast_side == "B" else batch
    a_shape = (a_batch, M, K) if a_major == "k" else (a_batch, K, M)
    b_shape = (b_batch, N, K) if b_major == "k" else (b_batch, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    c = torch.empty(batch, M, N, dtype=_TORCH_DTYPE[out_dtype], device="cuda")
    return a, b, c


def _mkbatched_nonpacked_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    pad = 16
    device = "cuda"
    if a_major == "k":
        a_storage = torch.empty(batch, M, K + pad, dtype=torch.int32, device=device).random_(*rng)
        a_storage = a_storage.to(dtype=_TORCH_DTYPE[in_dtype])
        a = a_storage[:, :, :K]
    else:
        a_storage = torch.empty(batch, K, M + pad, dtype=torch.int32, device=device).random_(*rng)
        a_storage = a_storage.to(dtype=_TORCH_DTYPE[in_dtype])
        a = a_storage[:, :, :M].transpose(1, 2)
    if b_major == "k":
        b_storage = torch.empty(batch, N, K + pad, dtype=torch.int32, device=device).random_(*rng)
        b_storage = b_storage.to(dtype=_TORCH_DTYPE[in_dtype])
        b = b_storage[:, :, :K]
    else:
        b_storage = torch.empty(batch, K, N + pad, dtype=torch.int32, device=device).random_(*rng)
        b_storage = b_storage.to(dtype=_TORCH_DTYPE[in_dtype])
        b = b_storage[:, :, :N].transpose(1, 2)
    c_storage = torch.empty(batch, M, N + pad, dtype=_TORCH_DTYPE[out_dtype], device=device)
    c = c_storage[:, :, :N]
    return a, b, c


def _mkbatched_zero_stride_input_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    seed: int = 0,
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    device = "cuda"
    a_base = torch.empty(K, dtype=torch.int32, device=device).random_(*rng)
    b_base = torch.empty(K, dtype=torch.int32, device=device).random_(*rng)
    a_base = a_base.to(dtype=_TORCH_DTYPE[in_dtype])
    b_base = b_base.to(dtype=_TORCH_DTYPE[in_dtype])
    a = torch.as_strided(a_base, (batch, M, K), (0, 0, 1))
    b = torch.as_strided(b_base, (batch, N, K), (0, 0, 1))
    c_storage = torch.empty(batch, M, N + 16, dtype=_TORCH_DTYPE[out_dtype], device=device)
    c = c_storage[:, :, :N]
    return a, b, c


def _reference(a: torch.Tensor, b: torch.Tensor, out_dtype: str) -> torch.Tensor:
    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    return ref.to(_TORCH_DTYPE[out_dtype])


def _block_quant_reference(
    x: torch.Tensor,
    block_size: int,
    out_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, M, N = x.shape
    blocks = x.view(B, M, N // block_size, block_size)
    output_max = 448.0 if out_dtype is torch.float8_e4m3fn else 57344.0
    scale_f = blocks.abs().amax(dim=-1) / output_max
    if scale_dtype is torch.float8_e8m0fnu:
        safe = torch.where(scale_f > 0, scale_f, 1.0)
        scale_f = torch.where(
            scale_f > 0,
            torch.pow(2.0, torch.ceil(torch.log2(safe))),
            0.0,
        )
    scale = scale_f.to(scale_dtype)
    inv = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    q = (blocks * inv.unsqueeze(-1)).clamp(-output_max, output_max)
    return q.to(out_dtype).view(B, M, N), scale


def _batched_reference(a: torch.Tensor, b: torch.Tensor, out_dtype: str) -> torch.Tensor:
    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    return ref.to(_TORCH_DTYPE[out_dtype])


def _batch_broadcast_reference(a: torch.Tensor, b: torch.Tensor, batch: int, out_dtype: str) -> torch.Tensor:
    a_full = a.to(torch.float32).expand(batch, -1, -1)
    b_full = b.to(torch.float32).expand(batch, -1, -1)
    ref = torch.einsum("bmk,bnk->bmn", a_full, b_full)
    return ref.to(_TORCH_DTYPE[out_dtype])


def _tolerance(in_dtype: str, out_dtype: str) -> float:
    """One out-dtype ULP: both accumulator and reference are exact FP32, so the
    only rounding is the deterministic downcast."""
    return 1.0 if in_dtype.startswith("fp8") else 0.5


# Compile cache (session-scoped) — one compile per (config, in_dt, out_dt).


@pytest.fixture(scope="session")
def _compile_cache() -> dict:
    """Maps (config_name, in_dt, out_dt) → CompiledFusedGemm | str(error).
    Cases visit in (config, dtype, shape) order, so each (config, dtype) block
    of 9 shapes shares one compile."""
    return {}


def _pick_anchor(
    cfg,
    in_dt: str,
    out_dt: str,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
) -> tuple[int, int, int] | None:
    """First menu shape compatible with (cfg, in_dt, out_dt), as the JIT anchor.
    The kernel is M/N/K-symbolic so any compatible shape works.

    Foot-gun: the C row-stride alignment is baked at JIT from the anchor's N, so
    the menu must be uniform in N-alignment class (enforced by `_compatible`)
    for runtime shapes to be drop-in.
    """
    for shape in _WEIRD_SHAPES:
        ok, _ = _compatible(cfg, *shape, in_dt, out_dt, a_major, b_major, cta_group)
        if ok:
            return shape
    return None


def _get_compiled(
    cache: dict,
    cfg,
    in_dt: str,
    out_dt: str,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
    scheduler: str = "clc",
    out_major: str = "n",
):
    """Return the cached compiled kernel, building it on first miss."""
    key = (cfg.name, in_dt, out_dt, a_major, b_major, cta_group, scheduler, out_major)
    if key in cache:
        entry = cache[key]
        if isinstance(entry, str):
            pytest.fail(entry, pytrace=False)
        return entry

    anchor = _pick_anchor(cfg, in_dt, out_dt, a_major, b_major, cta_group)
    if anchor is None:
        # No compatible anchor to build against (rare; e.g. K_BYTES vs elem_bytes mismatch).
        msg = f"no menu shape is compatible with ({cfg.name}, {in_dt}->{out_dt})"
        cache[key] = msg
        pytest.skip(msg)

    try:
        g = _build_graph(*anchor, in_dt, out_dt, a_major, b_major, out_major)
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        msg = f"JIT compile failed: {type(e).__name__}: {first[:200]}"
        cache[key] = msg
        pytest.fail(msg, pytrace=False)

    cache[key] = compiled
    return compiled


def _pick_batched_anchor(
    cfg,
    in_dt: str,
    out_dt: str,
    batch: int,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
) -> tuple[int, int, int] | None:
    """Pick a compatible M/N/K anchor for a fixed batch size."""
    for b, M, N, K in _BATCHED_SHAPES:
        if b != batch:
            continue
        ok, _ = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
        if ok:
            return M, N, K
    return None


def _get_batched_compiled(
    cache: dict,
    cfg,
    in_dt: str,
    out_dt: str,
    batch: int,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
    scheduler: str = "clc",
    out_major: str = "n",
):
    """Return the cached rank-3 compiled kernel for this graph batch."""
    key = (
        "batched",
        cfg.name,
        in_dt,
        out_dt,
        batch,
        a_major,
        b_major,
        cta_group,
        scheduler,
        out_major,
    )
    if key in cache:
        entry = cache[key]
        if isinstance(entry, str):
            pytest.fail(entry, pytrace=False)
        return entry

    anchor = _pick_batched_anchor(cfg, in_dt, out_dt, batch, a_major, b_major, cta_group)
    if anchor is None:
        msg = f"no batched menu shape is compatible with " f"({cfg.name}, {in_dt}->{out_dt}, batch={batch})"
        cache[key] = msg
        pytest.skip(msg)

    try:
        g = _build_batched_graph(batch, *anchor, in_dt, out_dt, a_major, b_major, out_major)
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        msg = f"JIT compile failed: {type(e).__name__}: {first[:200]}"
        cache[key] = msg
        pytest.fail(msg, pytrace=False)

    cache[key] = compiled
    return compiled


def _pick_batch_broadcast_anchor(
    cfg,
    in_dt: str,
    out_dt: str,
    batch: int,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
) -> tuple[int, int, int] | None:
    """Pick a compatible M/N/K anchor for a fixed broadcast output batch."""
    for b, M, N, K in _BATCH_BROADCAST_SHAPES:
        if b != batch:
            continue
        ok, _ = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
        if ok:
            return M, N, K
    return None


def _get_batch_broadcast_compiled(
    cache: dict,
    cfg,
    in_dt: str,
    out_dt: str,
    batch: int,
    broadcast_side: str,
    a_major: str = "k",
    b_major: str = "k",
    cta_group: int = 2,
    scheduler: str = "clc",
):
    """Return the cached rank-3 compiled kernel for a batch-broadcast graph."""
    key = (
        "batch_broadcast",
        broadcast_side,
        cfg.name,
        in_dt,
        out_dt,
        batch,
        a_major,
        b_major,
        cta_group,
        scheduler,
    )
    if key in cache:
        entry = cache[key]
        if isinstance(entry, str):
            pytest.fail(entry, pytrace=False)
        return entry

    anchor = _pick_batch_broadcast_anchor(cfg, in_dt, out_dt, batch, a_major, b_major, cta_group)
    if anchor is None:
        msg = f"no batch-broadcast menu shape is compatible with " f"({cfg.name}, {in_dt}->{out_dt}, batch={batch})"
        cache[key] = msg
        pytest.skip(msg)

    try:
        g = _build_batch_broadcast_graph(batch, *anchor, in_dt, out_dt, broadcast_side, a_major, b_major)
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        msg = f"JIT compile failed: {type(e).__name__}: {first[:200]}"
        cache[key] = msg
        pytest.fail(msg, pytrace=False)

    cache[key] = compiled
    return compiled


@pytest.mark.parametrize("shape", _WEIRD_SHAPES, ids=[_shape_id(s) for s in _WEIRD_SHAPES])
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CORE_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _CORE_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _sweep_config_names(),
    ids=[_config_id(n) for n in _sweep_config_names()],
)
def test_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    shape: tuple[int, int, int],
) -> None:
    """One (config, dtype-pair, shape); incompatible combos SKIP, else bit-tight."""
    cfg, cta_group, scheduler = _resolve(config_name)
    ok, reason = _compatible(cfg, *shape, in_dt, out_dt, cta_group=cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_compiled(_compile_cache, cfg, in_dt, out_dt, cta_group=cta_group, scheduler=scheduler)

    M, N, K = shape
    a, b, c = _mkdata(M, N, K, in_dt, out_dt)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())
    max_diff = float(diff.max().item())
    max_ref = float(ref.abs().max().item())

    # Rich diagnostic so a CI failure is self-explanatory without re-running.
    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  shape:     {M}x{N}x{K}"
        f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
        f"\n  max|diff|: {max_diff:.4g}  (tol={tol})"
        f"\n  max|ref|:  {max_ref:.4g}"
        f"\n  hint:      sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
        f"\n             sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
    )


def test_dense_block_scale_quant_epilogue() -> None:
    """Plain dense GEMM can use terminal block_scale_quantize epilogue."""
    config_name = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
    cfg, cta_group, scheduler = _resolve(config_name)
    M = N = K = 128
    block_size = 32
    g = _build_block_quant_graph(M, N, K, block_size)
    compiled = _plan(
        g,
        config=cfg,
        cta_group=cta_group,
        scheduler=scheduler,
    )
    assert not compiled.block_scale
    assert compiled.chain.block_quant is not None

    a, b, _ = _mkdata(M, N, K, "bf16", "bf16")
    q = torch.empty(1, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    q_scale = torch.empty(1, M, N // block_size, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp(compiled, a, b, [q, q_scale]))
    torch.cuda.synchronize()

    ref_mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    q_ref, scale_ref = _block_quant_reference(
        ref_mm,
        block_size,
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
    )
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "a_major,b_major",
    _INPUT_LAYOUTS,
    ids=[_layout_id(p) for p in _INPUT_LAYOUTS],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _LAYOUT_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _LAYOUT_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _LAYOUT_CONFIGS,
    ids=[_config_id(n) for n in _LAYOUT_CONFIGS],
)
def test_input_layout_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    a_major: str,
    b_major: str,
) -> None:
    """Pure matmul coverage for A K/M-major and B K/N-major inputs."""
    M, N, K = 256, 256, 256
    cfg, cta_group, scheduler = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        a_major,
        b_major,
        cta_group=cta_group,
        scheduler=scheduler,
    )
    a, b, c = _mkdata(M, N, K, in_dt, out_dt, a_major=a_major, b_major=b_major)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())
    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  layout:    A{a_major}/B{b_major}"
        f"\n  max|diff|: {float(diff.max().item()):.4g}  (tol={tol})"
    )


@pytest.mark.parametrize(
    "a_major,b_major",
    _NONPACKED_LAYOUTS,
    ids=[_layout_id(p) for p in _NONPACKED_LAYOUTS],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _NONPACKED_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _NONPACKED_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _NONPACKED_CONFIGS,
    ids=[_config_id(n) for n in _NONPACKED_CONFIGS],
)
def test_nonpacked_batched_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    a_major: str,
    b_major: str,
) -> None:
    """Padded A/B/C views exercise dynamic strides in TMA descriptors and stores."""
    batch, M, N, K = 2, 256, 256, 256
    cfg, cta_group, scheduler = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        a_major,
        b_major,
        cta_group=cta_group,
        scheduler=scheduler,
    )
    a, b, c = _mkbatched_nonpacked_data(batch, M, N, K, in_dt, out_dt, a_major=a_major, b_major=b_major)
    assert not a.is_contiguous() and not b.is_contiguous() and not c.is_contiguous()

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _batched_reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())
    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  layout:    A{a_major}/B{b_major}"
        f"\n  strides:   A{tuple(a.stride())} B{tuple(b.stride())} C{tuple(c.stride())}"
        f"\n  max|diff|: {float(diff.max().item()):.4g}  (tol={tol})"
    )


@pytest.mark.parametrize(
    "config_name",
    _NONPACKED_CONFIGS[:2],
    ids=[_config_id(n) for n in _NONPACKED_CONFIGS[:2]],
)
def test_zero_stride_broadcast_input_matmul(
    _compile_cache,
    config_name: str,
) -> None:
    batch, M, N, K = 2, 256, 256, 256
    in_dt = out_dt = "bf16"
    cfg, cta_group, scheduler = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, "k", "k", cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        "k",
        "k",
        cta_group=cta_group,
        scheduler=scheduler,
    )
    a, b, c = _mkbatched_zero_stride_input_data(batch, M, N, K, in_dt, out_dt)
    assert a.stride() == (0, 0, 1)
    assert b.stride() == (0, 0, 1)
    assert not c.is_contiguous()

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _batched_reference(a, b, out_dt)
    torch.testing.assert_close(c, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    "shape",
    _BATCHED_SHAPES,
    ids=[_batched_shape_id(s) for s in _BATCHED_SHAPES],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CORE_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _CORE_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _BATCHED_CONFIGS,
    ids=[_config_id(n) for n in _BATCHED_CONFIGS],
)
def test_batched_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    shape: tuple[int, int, int, int],
) -> None:
    """Rank-3 matmul keeps batch as the native L mode and maps it to grid.z."""
    batch, M, N, K = shape
    cfg, cta_group, scheduler = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, cta_group=cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        cta_group=cta_group,
        scheduler=scheduler,
    )

    a, b, c = _mkbatched_data(batch, M, N, K, in_dt, out_dt)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _batched_reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())

    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  shape:     B{batch} {M}x{N}x{K}"
        f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
        f"\n  max|diff|: {float(diff.max().item()):.4g}  (tol={tol})"
        f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
        f"\n  hint:      sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
        f"\n             sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
    )


@pytest.mark.parametrize(
    "a_major,b_major",
    _INPUT_LAYOUTS,
    ids=[_layout_id(p) for p in _INPUT_LAYOUTS],
)
def test_input_layout_batched_matmul(
    _compile_cache,
    a_major: str,
    b_major: str,
) -> None:
    """Rank-3 layout coverage keeps batch native while varying A/B major mode."""
    config_name = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
    cfg, cta_group, scheduler = _resolve(config_name)
    batch, M, N, K = 2, 256, 256, 256
    in_dt = out_dt = "bf16"
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        a_major,
        b_major,
        cta_group=cta_group,
        scheduler=scheduler,
    )
    a, b, c = _mkbatched_data(batch, M, N, K, in_dt, out_dt, a_major=a_major, b_major=b_major)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _batched_reference(a, b, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    assert int((diff > _tolerance(in_dt, out_dt)).sum().item()) == 0


@pytest.mark.parametrize(
    "broadcast_side,shape",
    _BATCH_BROADCAST_CASES,
    ids=[_batch_broadcast_id(p) for p in _BATCH_BROADCAST_CASES],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CORE_DTYPE_PAIRS,
    ids=[_dtype_id(p) for p in _CORE_DTYPE_PAIRS],
)
@pytest.mark.parametrize(
    "config_name",
    _BATCHED_CONFIGS,
    ids=[_config_id(n) for n in _BATCHED_CONFIGS],
)
def test_batch_broadcast_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    broadcast_side: str,
    shape: tuple[int, int, int, int],
) -> None:
    """Rank-3 matmul with one input broadcast across the output batch."""
    batch, M, N, K = shape
    cfg, cta_group, scheduler = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, cta_group=cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batch_broadcast_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        broadcast_side,
        cta_group=cta_group,
        scheduler=scheduler,
    )

    a, b, c = _mkbatch_broadcast_data(batch, M, N, K, in_dt, out_dt, broadcast_side)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _batch_broadcast_reference(a, b, batch, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = _tolerance(in_dt, out_dt)
    bad = int((diff > tol).sum().item())

    assert bad == 0, (
        f"\n  config:    {config_name}"
        f"\n  dtype:     {in_dt} -> {out_dt}"
        f"\n  broadcast: {broadcast_side}"
        f"\n  shape:     B{batch} {M}x{N}x{K}"
        f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
        f"\n  max|diff|: {float(diff.max().item()):.4g}  (tol={tol})"
        f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
        f"\n  hint:      sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
        f"\n             sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
    )


@pytest.mark.parametrize(
    "a_major,b_major",
    _INPUT_LAYOUTS,
    ids=[_layout_id(p) for p in _INPUT_LAYOUTS],
)
@pytest.mark.parametrize("broadcast_side", ("A", "B"), ids=("bcastA", "bcastB"))
def test_input_layout_batch_broadcast_matmul(
    _compile_cache,
    broadcast_side: str,
    a_major: str,
    b_major: str,
) -> None:
    """Input-layout coverage when one operand is broadcast across batch."""
    config_name = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
    cfg, cta_group, scheduler = _resolve(config_name)
    batch, M, N, K = 3, 256, 256, 256
    in_dt = out_dt = "bf16"
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, a_major, b_major, cta_group)
    if not ok:
        pytest.skip(reason)

    compiled = _get_batch_broadcast_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        broadcast_side,
        a_major,
        b_major,
        cta_group=cta_group,
        scheduler=scheduler,
    )
    a, b, c = _mkbatch_broadcast_data(
        batch,
        M,
        N,
        K,
        in_dt,
        out_dt,
        broadcast_side,
        a_major=a_major,
        b_major=b_major,
    )
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = _batch_broadcast_reference(a, b, batch, out_dt)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    assert int((diff > _tolerance(in_dt, out_dt)).sum().item()) == 0


# Mixed FP8: tcgen05 F8F6F4 takes A/B FP8 variants independently, so every
# {E4M3,E5M2}² pair is valid. The main sweep only drives A==B; this covers mixed.

_FP8_AB_PAIRS = [
    ("fp8_e4m3", "fp8_e4m3"),
    ("fp8_e5m2", "fp8_e5m2"),
    ("fp8_e4m3", "fp8_e5m2"),  # mixed
    ("fp8_e5m2", "fp8_e4m3"),  # mixed
]
_FP8_MIXED_CONFIGS = [
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
]


@pytest.mark.parametrize("a_dt,b_dt", _FP8_AB_PAIRS, ids=[f"{a}_x_{b}" for a, b in _FP8_AB_PAIRS])
@pytest.mark.parametrize("config_name", _FP8_MIXED_CONFIGS, ids=[_config_id(n) for n in _FP8_MIXED_CONFIGS])
def test_mixed_fp8_matmul(config_name: str, a_dt: str, b_dt: str) -> None:
    """A and B each an arbitrary FP8 variant -> FP16 out, bit-exact vs fp32."""
    cfg, cta_group, scheduler = _resolve(config_name)
    M = N = K = 256

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=_a_stride_batched(M, K, "k"),
        data_type=_CUDNN_DTYPE[a_dt],
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=_b_stride_batched(N, K, "k"),
        data_type=_CUDNN_DTYPE[b_dt],
    )
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    C.set_data_type(cudnn.data_type.HALF)

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    assert compiled.chain.matmul.a_dtype == a_dt
    assert compiled.chain.matmul.b_dtype == b_dt

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=_TORCH_DTYPE[a_dt], device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=_TORCH_DTYPE[b_dt], device="cuda")
    c = torch.empty(1, M, N, dtype=torch.float16, device="cuda")

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)).to(torch.float16)
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    assert int((diff > 1e-1).sum().item()) == 0, f"{config_name} {a_dt} x {b_dt}: max|diff|={diff.max().item():.4g}"


# INT8 × INT8 → INT32 (integer tensor-core MMA). Epilogue widens int32 → fp32;
# bit-exact vs an fp32 reference (small-magnitude products are exact).

_INT8_CONFIGS = [
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_static",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma_static",
]


# Output dtype → (cudnn enum, torch dtype, input value range). fp8 needs tiny
# inputs so the int32 accumulator stays in fp8's range; others use a wider range.
_INT8_OUT_DTYPES = {
    "fp32": (cudnn.data_type.FLOAT, torch.float32, 8),
    "bf16": (cudnn.data_type.BFLOAT16, torch.bfloat16, 8),
    "fp16": (cudnn.data_type.HALF, torch.float16, 8),
    "int32": (cudnn.data_type.INT32, torch.int32, 8),
    "fp8_e4m3": (cudnn.data_type.FP8_E4M3, torch.float8_e4m3fn, 1),
}


@pytest.mark.parametrize("config_name", _INT8_CONFIGS, ids=[_config_id(n) for n in _INT8_CONFIGS])
@pytest.mark.parametrize("out_dt", list(_INT8_OUT_DTYPES))
def test_int8_matmul(config_name: str, out_dt: str) -> None:
    """INT8×INT8→INT32, output ∈ {fp32,bf16,fp16,int32,fp8}; bit-exact vs a
    rounded integer reference (values small enough that the rounding is exact)."""
    sm = _current_sm()
    if sm is not None and not any(lo <= sm < hi for lo, hi in _INT8_SM_RANGES):
        pytest.skip(f"int8 matmul unsupported on sm_{sm} (SM 100/110 only)")
    cfg, cta_group, scheduler = _resolve(config_name)
    M = N = K = 256
    cudnn_dt, torch_dt, vmax = _INT8_OUT_DTYPES[out_dt]

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.INT8,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.INT32,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    C.set_data_type(cudnn_dt)

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    assert compiled.chain.matmul.accum_dtype == "int32"
    assert compiled.chain.output_dtype == out_dt

    torch.manual_seed(0)
    a = torch.randint(-vmax, vmax, (1, M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-vmax, vmax, (1, N, K), dtype=torch.int8, device="cuda")
    c = torch.empty(1, M, N, dtype=torch_dt, device="cuda")

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.cpu().to(torch.int64), b.cpu().to(torch.int64))
    if out_dt == "int32":
        diff = (c.cpu().to(torch.int64) - ref).abs().max().item()
    else:
        diff = (c.float().cpu() - ref.to(torch_dt).float()).abs().max().item()
    assert diff == 0.0, f"{config_name} -> {out_dt}: max|diff|={diff} (expected bit-exact)"


# M-major batched output.
_M_MAJOR_BATCHED_CASES: tuple[tuple[str, str, str], ...] = (
    ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma", "bf16", "bf16"),
    ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma", "bf16", "fp16"),
    ("CONFIG_sm100_64x128x128_64x128x32_cluster1x1_1ctamma", "bf16", "bf16"),
)


@pytest.mark.parametrize(
    "config_name,in_dt,out_dt",
    _M_MAJOR_BATCHED_CASES,
    ids=[f"{_config_id(c)}-{i}-{o}" for c, i, o in _M_MAJOR_BATCHED_CASES],
)
def test_m_major_output_batched(_compile_cache, config_name: str, in_dt: str, out_dt: str) -> None:
    """M-major + batch>1: covers TMA-store and STG m-major store paths."""
    batch, M, N, K = 3, 256, 256, 256
    cfg, cta_group, scheduler = _resolve(config_name)
    ok, reason = _compatible(cfg, M, N, K, in_dt, out_dt, cta_group=cta_group)
    if not ok:
        pytest.skip(reason)
    compiled = _get_batched_compiled(
        _compile_cache,
        cfg,
        in_dt,
        out_dt,
        batch,
        cta_group=cta_group,
        scheduler=scheduler,
        out_major="m",
    )
    a, b, c = _mkbatched_data(batch, M, N, K, in_dt, out_dt, out_major="m")
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()
    torch.testing.assert_close(c, _reference(a, b, out_dt), atol=_tolerance(in_dt, out_dt), rtol=0)


# Standalone CLI shim — forwards remaining argv to pytest on this file.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
