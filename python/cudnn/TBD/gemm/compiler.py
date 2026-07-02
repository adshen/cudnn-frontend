"""Compiler driver: cudnn graph -> rendered GEMM kernel.py -> compiled callable.

Phase 5: stitches together graph_analyzer, epilogue_codegen, and the kernel
template. Workflow:

  1. analyze() the FusedGraph -> FusionChain
  2. generate() the codegen snippets
  3. render the template (`sm100_matmul_1ctamma.py` for
     cta_group=1 or `sm100_matmul_2ctamma.py` for cta_group=2,
     picked via TileConfig.template_file) by replacing the `# @@INJECT_*@@`
     markers
  4. write the rendered module to a content-addressed dir under
     CUDNN_GEMM_KERNEL_CACHE (defaults to $XDG_CACHE_HOME/cudnn_gemm/kernel_cache,
     i.e. ~/.cache/cudnn_gemm/kernel_cache — never inside the project tree)
  5. dynamic-import the rendered module and return its `compile(K)` callable
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cudnn

from .epilogue_codegen import EpilogueSnippets, generate
from .fusion_ir import ZERO_PRESERVING_OPS, FusionChain, TensorRef
from .graph_analyzer import (
    GemmBinding,
    analyze_with_binding,
    resolve_variant_pack,
)
from .tile_config import DEFAULT_CONFIG, TileConfig

_TEMPLATE_DIR = Path(__file__).parent / "kernel_templates"


# ---------------------------------------------------------------------------
# Symbolic-shape helpers for aux fake tensors
# ---------------------------------------------------------------------------


def _aux_fake_shape_code(aux: TensorRef) -> str:
    """Python expression for the shape tuple of an aux fake tensor.

    Uses sym_l / sym_m / sym_n where the aux dim should match the matmul.
    Concrete 1s elsewhere.
    """
    if len(aux.dim) == 3:
        batch = "1" if aux.dim[0] == 1 else "sym_l"
        if aux.bcast_mode == "scalar":
            return f"({batch}, 1, 1)"
        if aux.bcast_mode == "per_row":
            return f"({batch}, sym_m, 1)"
        if aux.bcast_mode == "per_col":
            return f"({batch}, 1, sym_n)"
        if aux.bcast_mode == "per_elem":
            return f"({batch}, sym_m, sym_n)"
    else:
        if aux.bcast_mode == "scalar":
            return "(1, 1)"
        if aux.bcast_mode == "per_row":
            return "(sym_m, 1)"
        if aux.bcast_mode == "per_col":
            return "(1, sym_n)"
        if aux.bcast_mode == "per_elem":
            return "(sym_m, sym_n)"
    raise AssertionError(f"unknown bcast_mode {aux.bcast_mode!r}")


def _aux_fake_stride_order(aux: TensorRef) -> str:
    if len(aux.dim) == 3:
        return "(2, 1, 0)"
    return "(1, 0)"


def _aux_can_use_explicit_fake_stride(aux: TensorRef) -> bool:
    # Rank-1 aux tensors are represented as rank-2 broadcastable fake tensors
    # during compile, so their raw rank-1 stride is not a valid fake stride.
    if len(aux.dim) not in (2, 3):
        return False
    stride1_dims = [i for i, stride in enumerate(aux.stride) if stride == 1]
    if len(stride1_dims) <= 1:
        return True
    nontrivial = [i for i in stride1_dims if aux.dim[i] != 1]
    return len(nontrivial) == 1


_DTYPE_TO_DSL = {
    "bf16": "cutlass.BFloat16",
    "fp16": "cutlass.Float16",
    "fp32": "cutlass.Float32",
    "int8": "cutlass.Int8",
    # FN = "finite, no NaN" variant — what sm100 tcgen05 actually accepts.
    "fp8_e4m3": "cutlass.Float8E4M3FN",
    "fp8_e5m2": "cutlass.Float8E5M2",
    "fp8_e8m0": "cutlass.Float8E8M0FNU",
    # FP4 (e2m1) is always handled packed 2-per-byte as Float4E2M1FNx2.
    "fp4_e2m1": "cutlass.Float4E2M1FNx2",
    "uint8": "cutlass.Uint8",
    "int32": "cutlass.Int32",
    "int64": "cutlass.Int64",
}

# cutlass dtype names (used for SMEM allocation / fake-tensor element type).
_DTYPE_TO_CUTLASS = {
    "bf16": "cutlass.BFloat16",
    "fp16": "cutlass.Float16",
    "fp32": "cutlass.Float32",
    "int8": "cutlass.Int8",
    "fp8_e4m3": "cutlass.Float8E4M3FN",
    "fp8_e5m2": "cutlass.Float8E5M2",
    "fp8_e8m0": "cutlass.Float8E8M0FNU",
    "fp4_e2m1": "cutlass.Float4E2M1FNx2",
    "uint8": "cutlass.Uint8",
    "int32": "cutlass.Int32",
    "int64": "cutlass.Int64",
}

# Tcgen05 MMA family enum (keyed by the MMA *input* dtype).
_DTYPE_TO_MMA_KIND = {
    "bf16": "nvvm.Tcgen05MMAKind.F16",
    "fp16": "nvvm.Tcgen05MMAKind.F16",
    "fp8_e4m3": "nvvm.Tcgen05MMAKind.F8F6F4",
    "fp8_e5m2": "nvvm.Tcgen05MMAKind.F8F6F4",
    "int8": "nvvm.Tcgen05MMAKind.INT8",
}


def _aux_fake_block(aux_tensors: list[TensorRef], *, dynamic_strides: bool = False) -> str:
    """Build the lines that declare `fake_<name>` for each aux tensor. Joined
    with bare ``\\n`` — the marker-replacement layer re-applies the marker
    line's indent to every replacement line, so baking indent in here would
    double it."""
    lines = []
    for aux in aux_tensors:
        shape = _aux_fake_shape_code(aux)
        dtype = _DTYPE_TO_DSL[aux.dtype]
        if dynamic_strides and _aux_can_use_explicit_fake_stride(aux):
            stride = "(" + ", ".join(str(s) for s in aux.stride) + ")"
            lines.append(f"fake_{aux.name} = make_fake_tensor({dtype}, {shape}, " f"stride={stride}, assumed_align=16)")
        else:
            stride_order = _aux_fake_stride_order(aux)
            lines.append(f"fake_{aux.name} = make_fake_compact_tensor({dtype}, {shape}, " f"stride_order={stride_order}, assumed_align=16)")
    return "\n".join(lines) if lines else "pass"


def _aux_signature_block(aux_tensors: list[TensorRef]) -> str:
    """Comma-separated params (one per line) for a signature list. No baked
    indent — see ``_aux_fake_block`` for the rationale."""
    if not aux_tensors:
        return ""
    return ",\n".join(f"{aux.name}: cute.Tensor" for aux in aux_tensors) + ","


def _aux_call_block(aux_tensors: list[TensorRef], prefix: str = "") -> str:
    """Comma-separated args (one per line) for a call list. No baked indent."""
    if not aux_tensors:
        return ""
    return ",\n".join(f"{prefix}{aux.name}" for aux in aux_tensors) + ","


def _reduction_stride_kernel_params(chain: FusionChain) -> str:
    params: list[str] = []
    for i in range(len(chain.reductions)):
        params.extend(
            [
                f"red_stride_m_{i}: cutlass.Int64",
                f"red_stride_n_{i}: cutlass.Int64",
                f"red_stride_l_{i}: cutlass.Int64",
            ]
        )
    if chain.block_quant is not None:
        params.extend(
            [
                "quant_scale_stride_m: cutlass.Int64",
                "quant_scale_stride_n: cutlass.Int64",
                "quant_scale_stride_l: cutlass.Int64",
            ]
        )
    return ",\n".join(params) + "," if params else ""


def _reduction_stride_host_unpack(chain: FusionChain) -> str:
    if not chain.reductions and chain.block_quant is None:
        return ""
    lines = ["_stride_idx += 3"]
    for i in range(len(chain.reductions)):
        lines.extend(
            [
                f"red_stride_m_{i} = problem_size[_stride_idx]",
                f"red_stride_n_{i} = problem_size[_stride_idx + 1]",
                f"red_stride_l_{i} = problem_size[_stride_idx + 2]",
                "_stride_idx += 3",
            ]
        )
    if chain.block_quant is not None:
        lines.extend(
            [
                "quant_scale_stride_m = problem_size[_stride_idx]",
                "quant_scale_stride_n = problem_size[_stride_idx + 1]",
                "quant_scale_stride_l = problem_size[_stride_idx + 2]",
                "_stride_idx += 3",
            ]
        )
    return "\n".join(lines)


def _reduction_stride_host_unpack_from(chain: FusionChain, start_index: int) -> str:
    if not chain.reductions and chain.block_quant is None:
        return ""
    lines: list[str] = []
    for i in range(len(chain.reductions)):
        base = start_index + 3 * i
        lines.extend(
            [
                f"red_stride_m_{i} = problem_size[{base}]",
                f"red_stride_n_{i} = problem_size[{base + 1}]",
                f"red_stride_l_{i} = problem_size[{base + 2}]",
            ]
        )
    if chain.block_quant is not None:
        base = start_index + 3 * len(chain.reductions)
        lines.extend(
            [
                f"quant_scale_stride_m = problem_size[{base}]",
                f"quant_scale_stride_n = problem_size[{base + 1}]",
                f"quant_scale_stride_l = problem_size[{base + 2}]",
            ]
        )
    return "\n".join(lines)


def _reduction_stride_host_pass(chain: FusionChain) -> str:
    args: list[str] = []
    for i in range(len(chain.reductions)):
        args.extend(
            [
                f"red_stride_m_{i}",
                f"red_stride_n_{i}",
                f"red_stride_l_{i}",
            ]
        )
    if chain.block_quant is not None:
        args.extend(
            [
                "quant_scale_stride_m",
                "quant_scale_stride_n",
                "quant_scale_stride_l",
            ]
        )
    return ",\n".join(args) + "," if args else ""


def _reduction_stride_compile_decls(chain: FusionChain) -> str:
    lines: list[str] = []
    for i in range(len(chain.reductions)):
        lines.extend(
            [
                f"sym_red_stride_m_{i} = cute.sym_int64()",
                f"sym_red_stride_n_{i} = cute.sym_int64()",
                f"sym_red_stride_l_{i} = cute.sym_int64()",
            ]
        )
    if chain.block_quant is not None:
        lines.extend(
            [
                "sym_quant_scale_stride_m = cute.sym_int64()",
                "sym_quant_scale_stride_n = cute.sym_int64()",
                "sym_quant_scale_stride_l = cute.sym_int64()",
            ]
        )
    return "\n".join(lines)


def _reduction_stride_compile_symbols(chain: FusionChain) -> str:
    args: list[str] = []
    for i in range(len(chain.reductions)):
        args.extend(
            [
                f"sym_red_stride_m_{i}",
                f"sym_red_stride_n_{i}",
                f"sym_red_stride_l_{i}",
            ]
        )
    if chain.block_quant is not None:
        args.extend(
            [
                "sym_quant_scale_stride_m",
                "sym_quant_scale_stride_n",
                "sym_quant_scale_stride_l",
            ]
        )
    return ",\n".join(args) + "," if args else ""


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _mainloop_chain_zero_preserving(ops) -> bool:
    """True iff applying this mainloop op chain to 0 provably yields 0.

    Each op is zero-preserving iff it is in ``ZERO_PRESERVING_OPS`` (the single
    source of truth in fusion_ir; an unlisted op is conservatively treated as
    NON-zero-preserving). Once the running value is no longer provably zero the
    chain is reported not-zero-preserving — which only makes the K-OOB mask fire
    more often (still correct, just slightly more work)."""
    zero = True
    for op in ops:
        if not zero:
            return False
        zero = op.op in ZERO_PRESERVING_OPS
    return zero


def _render_tile_constants(cfg: TileConfig, chain: FusionChain, cta_group: int) -> str:
    """Override module-level tile + dtype constants for the requested config/chain.

    The template has default values immediately above the marker; this snippet
    is appended (assignments to the same names) and the last assignment wins.
    Tile constants are dtype-agnostic in :class:`TileConfig` (K stored in
    bytes); here we resolve them into *element* counts using the chain's A
    dtype. Dtype constants (`ab_dtype`, `mma_a_dtype`, `ab_tma_dtype`,
    `mma_kind`, ...) are derived from `chain.matmul.{a,b}_dtype` and
    `chain.output_dtype`.
    """
    # a_dt / b_dt: global-memory dtypes (what TMA loads from GMEM).
    # mma_a_dt / mma_b_dt: the dtype the MMA instruction uses — equal to the
    # global dtype (no implicit cast).
    a_dt = chain.matmul.a_dtype
    b_dt = chain.matmul.b_dtype
    mma_a_dt = _mma_a_dtype(chain)
    mma_b_dt = _mma_b_dtype(chain)
    accum_dt = chain.matmul.accum_dtype
    out_dt = chain.output_dtype
    # elem_bytes drives K_TILE, swizzle, and MMA instruction sizing — always
    # the MMA dtype bytes (e.g. 2 for BF16 even when global A is fp32).
    elem_bytes = _DTYPE_BYTES[mma_a_dt]
    # SMEM-row swizzle is determined by K-tile width in bytes (the inner-most
    # SMEM row). Templates allocate (M, K_bytes) and (N, K_bytes) tiles, so:
    #   K_bytes = 128 -> SWIZZLE_128B, TMA s128b, stride_byte_offset = 8 * 128
    #   K_bytes =  64 -> SWIZZLE_64B,  TMA s64b,  stride_byte_offset = 8 *  64
    #   K_bytes =  32 -> SWIZZLE_32B,  TMA s32b,  stride_byte_offset = 8 *  32
    # 8 rows-per-stride-chunk comes from the 8×16B tcgen05 core matrix.
    _SWIZZLE_TABLE = {
        128: ("SWIZZLE_128B", "s128b"),
        64: ("SWIZZLE_64B", "s64b"),
        32: ("SWIZZLE_32B", "s32b"),
    }
    if cfg.cta_tile_k_bytes not in _SWIZZLE_TABLE:
        raise ValueError(f"TileConfig {cfg.name!r}: unsupported cta_tile_k_bytes=" f"{cfg.cta_tile_k_bytes} (supported: {sorted(_SWIZZLE_TABLE)})")
    smem_swizzle_name, tma_swizzle_name = _SWIZZLE_TABLE[cfg.cta_tile_k_bytes]
    smem_swizzle_bytes = cfg.cta_tile_k_bytes
    smem_desc_stride_byte_offset = 8 * cfg.cta_tile_k_bytes
    mma_inst_k_bytes = cfg.mma_inst_mnk(elem_bytes, cta_group)[2] * elem_bytes
    cta_smem_m, cta_smem_n, _cta_smem_k = cfg.cta_smem_tile_mnk(elem_bytes, cta_group)
    mn_group_elems = smem_swizzle_bytes // elem_bytes

    def _smem_desc_params(
        is_mn_major: bool,
        mn_extent: int,
        operand_name: str,
    ) -> tuple[int, int, int, int]:
        if not is_mn_major:
            return 16, smem_desc_stride_byte_offset, mma_inst_k_bytes, 1
        if mn_extent < mn_group_elems or mn_extent % mn_group_elems != 0:
            raise ValueError(
                f"TileConfig {cfg.name!r} cannot use {operand_name}-major input: "
                f"SMEM extent {mn_extent} is not a multiple of the "
                f"{mn_group_elems}-element swizzle group"
            )
        group_elems = mn_group_elems
        return (
            cfg.cta_tile_k_bytes * group_elems,
            8 * group_elems * elem_bytes,
            mma_inst_k_bytes * group_elems,
            group_elems,
        )

    a_lbo, a_sbo, a_k_step, a_tma_group_elems = _smem_desc_params(chain.matmul.a_major == "m", cta_smem_m, "M")
    b_lbo, b_sbo, b_k_step, b_tma_group_elems = _smem_desc_params(chain.matmul.b_major == "n", cta_smem_n, "N")

    lines = [
        f"# Tile config: {cfg.name}",
        f"mma_inst_shape_mnk = {cfg.mma_inst_mnk(elem_bytes, cta_group)}",
        f"cgrp_tile_mnk = {cfg.cgrp_tile_mnk(elem_bytes)}",
        # Template uses `cta_tile_mnk` for per-CTA SMEM/TMA box dims, which
        # is the SMEM tile (B's N halved under 2-CTA MMA), not the logical
        # per-CTA tile exposed by TileConfig.
        f"cta_tile_mnk = {cfg.cta_smem_tile_mnk(elem_bytes, cta_group)}",
        f"epi_tile_mn = {cfg.epi_tile_mn}",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"matmul_batch = {chain.matmul.batch}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"a_is_m_major = {chain.matmul.a_major == 'm'}",
        f"b_is_n_major = {chain.matmul.b_major == 'n'}",
        f"mma_a_major = {1 if chain.matmul.a_major == 'm' else 0}",
        f"mma_b_major = {1 if chain.matmul.b_major == 'n' else 0}",
        f"ab_stages = {cfg.max_ab_stages(cta_group)}",
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b(cta_group)}",
        f"ab_smem_swizzle = cutlass.primitives.Tcgen05SmemSwizzle.{smem_swizzle_name}",
        f"ab_smem_swizzle_bytes = {smem_swizzle_bytes}",
        f"ab_smem_desc_stride_byte_offset = {smem_desc_stride_byte_offset}",
        f"a_smem_desc_leading_byte_offset = {a_lbo}",
        f"a_smem_desc_stride_byte_offset = {a_sbo}",
        f"a_smem_k_step_bytes = {a_k_step}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_smem_desc_leading_byte_offset = {b_lbo}",
        f"b_smem_desc_stride_byte_offset = {b_sbo}",
        f"b_smem_k_step_bytes = {b_k_step}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        f"ab_tma_swizzle = _tma.TensorMapSwizzle.{tma_swizzle_name}",
        "",
        f"# Dtype family: A={a_dt}->MMA{mma_a_dt}, B={b_dt}->MMA{mma_b_dt}, out={out_dt} (K_BYTES={cfg.cta_tile_k_bytes})",
        # ab_dtype: MMA operand dtype (what SMEM holds / MMA reads).
        f"ab_dtype = {_DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"cd_dtype = {_DTYPE_TO_CUTLASS[out_dt]}",
        f"mma_a_dtype = {_DTYPE_TO_DSL[mma_a_dt]}",
        f"mma_b_dtype = {_DTYPE_TO_DSL[mma_b_dt]}",
        # Accumulator dtype: int32 for INT8 MMA, fp32 otherwise. The idesc is
        # built directly from mma_a/b/c_dtype (Int8/Int32 included — GEMM's
        # InstrDesc.build now maps integer input formats). `acc_widen_to_fp32`
        # tells the epilogue to convert the int32 TMEM load to fp32 before the
        # (fp32-only) op chain + cast. We DON'T widen when the output is int32
        # (raw accumulator): widening through fp32 would lose precision past
        # 2**24, so int32 output passes the accumulator straight through.
        f"mma_c_dtype = {_DTYPE_TO_DSL[accum_dt]}",
        f"acc_widen_to_fp32 = {accum_dt == 'int32' and out_dt != 'int32'}",
        # ab_tma_dtype: the A/B TMA-descriptor element dtype. Used by all
        # templates (matmul + mainloop) for both A and B — TMA only cares about
        # the element byte width, which is identical across an a/b dtype pair.
        f"ab_tma_dtype = {_DTYPE_TO_DSL[mma_a_dt]}",
        f"cd_tma_dtype = {_DTYPE_TO_DSL[out_dt]}",
        f"mma_kind = {_DTYPE_TO_MMA_KIND[mma_a_dt]}",
        f"cd_out_is_m_major = {chain.matmul.out_major == 'm'}",
        # M-major TMA-store C-descriptor inner-M box = swizzle span(128 B) /
        # elem_bytes (inner box bytes == swizzle span).
        f"cd_mmajor_atom_m = {128 // _DTYPE_BYTES[out_dt]}",
    ]
    # Persistent kernel always: double-TMEM + L2 N-super-block swizzle.
    lines.append(f"acc_stages = {cfg.acc_stages}")
    lines.append(f"tile_swizzle_n = {cfg.tile_swizzle_n}")
    # Multi-GEMM (parallel matmuls sharing the epilogue). Always emitted so the
    # template can reference these unconditionally; single-GEMM = (1, 1, 1).
    # gemm_a_idx[g] / gemm_b_idx[g] pick GEMM g's operand from the distinct A / B
    # pools. Each GEMM owns its own TMEM accumulator region (cta_tile_n cols), so
    # one acc_stage spans `num_gemms * cta_tile_n` cols; we keep double-buffering
    # (acc_stages=2) only when both stages fit the arch TMEM budget, else drop to
    # a single acc stage. (cta_group is always 1 here — the only multi-GEMM
    # template — so cta_tile_n == the per-CTA N.)
    lines.append(f"num_gemms = {chain.num_gemms}")
    lines.append(f"num_a_operands = {chain.num_a_operands}")
    lines.append(f"num_b_operands = {chain.num_b_operands}")
    lines.append(f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}")
    lines.append(f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}")
    if chain.is_multi_gemm:
        total_tmem = _tmem_cols_for_arch(cfg.arch)
        region = chain.num_gemms * cfg.cta_tile_n
        if region > total_tmem:
            raise NotImplementedError(
                f"multi-GEMM: {chain.num_gemms} GEMMs × cta_tile_n={cfg.cta_tile_n} "
                f"= {region} acc cols exceed {total_tmem} TMEM (single stage). "
                f"Pick a smaller cta_tile_n or fewer GEMMs."
            )
        acc_stages_mg = 2 if 2 * region <= total_tmem else 1
        lines.append(f"acc_stages = {acc_stages_mg}  # multi-GEMM: {chain.num_gemms}×{cfg.cta_tile_n} cols/stage")
        # ab_stages must account for one SMEM buffer per DISTINCT operand
        # (num_a A-tiles + num_b B-tiles per stage), not the single A+B that
        # smem_max_ab_stages assumes.
        from .tile_config import _SM100_SMEM_BUDGET_BYTES, _AB_STAGES_CAP

        # Per-CTA SMEM B-tile N is halved under 2-CTA MMA (the pair splits B's N).
        smem_n = cfg.cta_tile_n // cta_group
        per_stage = (chain.num_a_operands * cfg.cta_tile_m + chain.num_b_operands * smem_n) * cfg.cta_tile_k_bytes + 2 * 8
        fixed = 2 * acc_stages_mg * 8 + 8
        avail = _SM100_SMEM_BUDGET_BYTES - fixed
        ab_stages_mg = min(avail // per_stage, _AB_STAGES_CAP)
        if ab_stages_mg < 1:
            raise NotImplementedError(
                f"multi-GEMM: {chain.num_a_operands} A + {chain.num_b_operands} B " f"operand tiles per stage exceed SMEM budget at this geometry"
            )
        lines.append(f"ab_stages = {ab_stages_mg}  # multi-GEMM: {chain.num_a_operands}A+{chain.num_b_operands}B per stage")
    # MoE grouped matmul: the grouped persistent scheduler launches a FIXED
    # number of clusters (≈ NUM_SMS / cluster_size) and strides through tiles by
    # this count. The host grid and the kernel's persistent stride share it.
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        # first_token_offset element dtype (int32 / int64) — drives the compile()
        # fake tensor; the scheduler casts reads to Int32 internally.
        lines.append(f"offset_cutlass_dtype = {_DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
    # Mainloop fusion (Phase 6): the 12-warp template adds 4 mainloop-fusion
    # warps (+128 threads). Override threads_per_cta and expose the warp count.
    # Emitted *after* the earlier `threads_per_cta` line so the override wins.
    if chain.has_mainloop_fusion:
        lines.append("num_mainloop_warps = 4")
        lines.append("threads_per_cta = 384")
        # Per-operand fusion flags (const_expr-gated in the template).
        lines.append(f"mainloop_fuse_a = {chain.has_mainloop_fusion_a}")
        lines.append(f"mainloop_fuse_b = {chain.has_mainloop_fusion_b}")
        # K-OOB mask: when BOTH operands are fused and NEITHER chain maps
        # 0 -> 0, the TMA K-tail zero-fill is corrupted by the transform
        # (e.g. cos(0)=1) and the OOB K-elements would add f_a(0)*f_b(0) to
        # every accumulator. The mainloop then zeros A's OOB K-elements (using
        # the swizzle-aware load/store below) so the product with B's OOB is 0.
        koob_fix = (
            chain.has_mainloop_fusion_a
            and chain.has_mainloop_fusion_b
            and not _mainloop_chain_zero_preserving(chain.mainloop_a_ops)
            and not _mainloop_chain_zero_preserving(chain.mainloop_b_ops)
        )
        lines.append(f"mainloop_koob_fix = {koob_fix}")
        # CuTe XOR swizzle matching the MMA-dtype SMEM layout
        # (s128b=Swizzle(3,4,3), s64b=(2,4,3), s32b=(1,4,3)).
        # bbits = log2(K_MMA_bytes / 16) = log2(K_BYTES / 16).
        _bbits = (cfg.cta_tile_k_bytes // 16).bit_length() - 1
        lines.append(f"ab_swizzle = cutlass.Swizzle({_bbits}, 4, 3)")
        # Mixed-input mainloop (dtype cast): a fused operand may be LOADED at a
        # narrower dtype than the MMA reads (e.g. int8 A -> bf16 MMA). The TMA
        # loads the narrow tile into a separate LOAD SMEM buffer; the mainloop
        # warps widen it into the wide MMA SMEM buffer (store_swizzled, so the
        # MMA's swizzle is respected). load==MMA ⇒ no cast (in-place path).
        lines.append(f"mainloop_a_cast = {chain.mainloop_a_cast}")
        lines.append(f"mainloop_b_cast = {chain.mainloop_b_cast}")
        load_a_dt = chain.mainloop_a_load_dtype or chain.matmul.a_dtype
        load_b_dt = chain.mainloop_b_load_dtype or chain.matmul.b_dtype
        lines.append(f"ab_load_a_dtype = {_DTYPE_TO_DSL[load_a_dt]}")
        lines.append(f"ab_load_b_dtype = {_DTYPE_TO_DSL[load_b_dt]}")

    # Epilogue store vector width — largest power-of-2 in {32,16,8,4} that
    # divides the C row stride in bytes. Picked so that st.global.v{1,2,4,8}.b32
    # natural-alignment requirements always hold across every row.
    vec_bytes_epi = _compute_output_vec_bytes(chain)
    lines.append(f"vec_bytes_epi = {vec_bytes_epi}")
    # Matmul-output tap (Phase 1 multi-output): when set, an extra GMEM
    # output buffer holds the raw fp32 accumulator cast to `tap_dtype`.
    # Phase 2 multi-output: per-tap `vec_bytes_tap_<i>` constants are emitted
    # via `EpilogueSnippets.tap_constants` and appended in `_render_template`.
    # Epilogue store mode: TMA-store-via-SMEM-buffer (preferred — wide bulk
    # transfer with no per-thread STG overhead) vs per-thread STG (fallback).
    # See _use_tma_store_epi() for gating logic.
    use_tma = (not _FORCE_STG_EPI) and _use_tma_store_epi(chain, cfg, vec_bytes_epi, cta_group)
    lines.append(f"use_tma_store_epi = {use_tma}")
    # Final ab_stages override: account for the TMA-D SMEM buffer (fixed bytes,
    # when the TMA-store epilogue is active) AND a mixed-input mainloop's narrow
    # LOAD buffer (per-stage bytes, beside the wide MMA tile). Otherwise leave
    # the plain max.
    smem_d_bytes = _smem_d_bytes(cfg, chain) if use_tma else 0
    cast_extra_per_stage = 0
    if chain.has_mainloop_fusion and (chain.mainloop_a_cast or chain.mainloop_b_cast):
        smem_n = cfg.cta_tile_n // cta_group
        k_elems = cfg.cta_tile_k_bytes // _DTYPE_BYTES[chain.matmul.a_dtype]
        if chain.mainloop_a_cast:
            cast_extra_per_stage += cfg.cta_tile_m * k_elems * _DTYPE_BYTES[chain.mainloop_a_load_dtype]
        if chain.mainloop_b_cast:
            cast_extra_per_stage += smem_n * k_elems * _DTYPE_BYTES[chain.mainloop_b_load_dtype]
    if smem_d_bytes > 0 or cast_extra_per_stage > 0:
        new_ab = cfg.max_ab_stages(
            cta_group,
            extra_smem_bytes=smem_d_bytes,
            extra_per_stage_bytes=cast_extra_per_stage,
        )
        lines.append(f"ab_stages = {new_ab}  # SMEM-D {smem_d_bytes}B fixed" f" + cast LOAD {cast_extra_per_stage}B/stage")
    return "\n".join(lines)


def _tmem_cols_for_arch(arch: str) -> int:
    """TMEM column count for a config/template arch family. The block-scale
    renderer keys off this (not the active GPU) so the kernel budgets the right
    column count even when rendered on a different / no GPU (render-only, CI)."""
    if arch == "sm100":
        return 512
    raise NotImplementedError(f"TMEM column count not known for arch {arch!r}; add it to " f"_tmem_cols_for_arch")


def _current_sm() -> int | None:
    """Active GPU's SM number (major*10 + minor, e.g. 100 for cc 10.0), or
    ``None`` when no CUDA device is visible (render-only / CI without a GPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return major * 10 + minor
    except Exception:  # noqa: BLE001 — render path must work without a GPU
        pass
    return None


def _grid_num_clusters(cfg: TileConfig) -> int:
    """Fixed persistent cluster count for the MoE grouped-matmul scheduler:
    roughly ``NUM_SMS / cluster_size`` so one cluster lands per SM-group. The
    exact value only affects occupancy, not correctness (the host grid and the
    kernel's persistent stride share it). Falls back to 148 (sm100 B200) when no
    GPU is visible (render-only / CI)."""
    cluster_size = cfg.cgrp_size_m * cfg.cgrp_size_n
    sm_count = 148
    try:
        import torch

        if torch.cuda.is_available():
            sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:  # noqa: BLE001 — render path must work without a GPU
        pass
    return max(1, sm_count // cluster_size)


def _template_arch_family(template_file: str) -> str:
    """Leading ``sm<NNN>`` arch-family token of a template filename."""
    m = re.match(r"(sm\d+)_", template_file)
    if m is None:
        raise ValueError(f"cannot determine arch family from template {template_file!r} " f"(expected a leading 'sm<NNN>_' token)")
    return m.group(1)


def _render_block_scale_tile_constants(cfg: TileConfig, chain: FusionChain, cta_group: int, *, use_tma_store_epi: bool = False) -> str:
    """Emit module-level constants for the block-scaled-matmul template.

    Unlike the generic matmul path this does NOT go through the dtype-byte
    machinery (FP4 is 0.5 bytes / element). It resolves everything from the
    chain's BlockScaleSpec + the geometry config, including the SF SMEM/TMEM
    sizing and the SMEM→TMEM (utccp) copy schedules.
    """
    from .tile_config import validate_block_scale_config

    bs = chain.block_scale
    assert bs is not None
    is_fp4 = bs.is_fp4
    # Data K per tile, in elements. K_BYTES is fixed at 128 (validated).
    data_elem_bits = 4 if is_fp4 else 8
    cta_k_elems = cfg.cta_tile_k_bytes * 8 // data_elem_bits  # fp4: 256, fp8: 128
    validate_block_scale_config(cfg, bs.block_size, cta_k_elems)

    cta_m = cfg.cta_tile_m
    cta_n = cfg.cta_tile_n
    # MMA K-instruction width (sm100 → 32 bytes): fp4 → 64 elems, fp8 → 32.
    mma_inst_k_bytes = cfg.mma_inst_k_bytes
    mma_inst_k_elems = mma_inst_k_bytes * 8 // data_elem_bits
    num_kblocks = cta_k_elems // mma_inst_k_elems

    # --- Operand major (K- / M- / N-major) -----------------------------------
    # FP4 (nvfp4 / mxfp4) is K-major only — the sub-byte (Float4E2M1FNx2)
    # packing makes an M/N-contiguous descriptor mis-stride. mxfp8 (1 B/elem)
    # may be M-major (A) / N-major (B). The scale-factor layout is unchanged
    # regardless of the data major.
    a_major = chain.matmul.a_major
    b_major = chain.matmul.b_major
    if is_fp4 and (a_major != "k" or b_major != "k"):
        raise ValueError(f"FP4 block-scaled inputs must be K-major (got A={a_major}-major, " f"B={b_major}-major); only mxfp8 supports M/N-major operands.")
    # Major-dependent SMEM descriptor params, mirroring the generic matmul
    # path's _smem_desc_params. K-major: leading=16 B, stride=8*K_bytes,
    # k_step=32 B (the s128b MMA K-inst), group=1. M/N-major (mxfp8, 1 B/elem):
    # the contiguous MN dim is loaded in `mn_group_elems`-wide swizzle chunks.
    ab_elem_bytes = 1  # FP8; FP4 is rejected above for non-K
    mn_group_elems = cfg.cta_tile_k_bytes // ab_elem_bytes
    cta_smem_m = cta_m
    cta_smem_n = cta_n // cta_group

    def _bs_smem_desc_params(is_mn_major, mn_extent, name):
        if not is_mn_major:
            return 16, 8 * cfg.cta_tile_k_bytes, mma_inst_k_bytes, 1
        if mn_extent < mn_group_elems or mn_extent % mn_group_elems != 0:
            raise ValueError(
                f"block-scale config {cfg.name!r} cannot use {name}-major input: "
                f"SMEM extent {mn_extent} is not a multiple of the "
                f"{mn_group_elems}-element swizzle group"
            )
        g = mn_group_elems
        return (
            cfg.cta_tile_k_bytes * g,
            8 * g * ab_elem_bytes,
            mma_inst_k_bytes * g,
            g,
        )

    a_lbo, a_sbo, a_k_step, a_tma_group_elems = _bs_smem_desc_params(a_major == "m", cta_smem_m, "M")
    b_lbo, b_sbo, b_k_step, b_tma_group_elems = _bs_smem_desc_params(b_major == "n", cta_smem_n, "N")

    # Packed (Float4E2M1FNx2 / Float8) element count per row for SMEM.
    pack = 2 if is_fp4 else 1
    ab_packed_per_row = cta_k_elems // pack  # ab_dtype elems per K-row
    sA_packed_elems = cta_m * ab_packed_per_row
    sB_packed_elems = (cta_n // cta_group) * ab_packed_per_row

    # --- Scale factors --------------------------------------------------------
    sf_k = cta_k_elems // bs.block_size  # SF values along K per tile
    sf_k4 = sf_k // 4  # 4 SF-K per utccp atom
    nb_m = cta_m // 128
    nb_n = cta_n // 128
    # --- SF scheme: small FIXED TMEM "word" per block + per-word utccp refresh.
    # SF lives in a tiny fixed TMEM region (NUM_BLOCKS × REGISTERS_PER_BLOCK
    # cols), refreshed by utccp before the MMA(s) that read it — the MMA reads
    # the same fixed word and selects the active scale via the idesc
    # a_sf_id/b_sf_id field. A 128×4 utccp atom = 4 TMEM cols holding 4 K-scales.
    # One MMA K-inst consumes `scales_per_inst = mma_inst_k/block_size` scales
    # (<= 4 here): the word is ONE atom (4 cols) and `insts_per_word =
    # 4/scales_per_inst` consecutive K-insts share it (a_sf_id = j*scales_per_inst).
    # TMEM footprint stays block-size-independent and tiny, freeing TMEM for acc.
    _REGISTERS_PER_ATOM = 4  # cols per 128×4 utccp atom
    scales_per_inst = mma_inst_k_elems // bs.block_size
    word_scales = max(_REGISTERS_PER_ATOM, scales_per_inst)  # cols per block-word
    word_atoms = word_scales // _REGISTERS_PER_ATOM  # atoms copied per word
    insts_per_word = max(_REGISTERS_PER_ATOM // scales_per_inst, 1)
    num_sf_words = max(num_kblocks // insts_per_word, 1)  # utccp refreshes / k-tile
    _REGISTERS_PER_BLOCK = word_scales  # SF word width per block
    sfa_tmem_cols = nb_m * _REGISTERS_PER_BLOCK  # fixed SF word width (SFA)
    sfb_tmem_cols = nb_n * _REGISTERS_PER_BLOCK  # fixed SF word width (SFB)
    # utccp SMEM-source offsets (16-byte units). One 128×4 atom = 512 B = 32;
    # consecutive K-atoms are one atom apart; each M/N-block of 128 rows is
    # `sf_k4` atoms further along the SF SMEM tile.
    sf_atom_desc_stride = 32  # K-atom stride
    sf_block_desc_stride = 32 * sf_k4  # M/N-block stride

    # SF SMEM bytes per stage (sf_dtype 1 byte; whole k-tile loaded by TMA).
    sfa_smem_bytes = cta_m * sf_k
    sfb_smem_bytes = cta_n * sf_k
    # SF TMA box (fp16-recast trick): inner dim 256 fp16 (=512 B = one
    # 128×4 SF atom block, hardcoded in the template), box-K = sf_k4 atoms.
    sf_tma_box_k = sf_k4

    # --- TMEM budget → acc_stages (+ optional overlap trick) -----------------
    # acc accumulator needs `cta_n` cols/stage; SF needs `sf_total_cols`. With
    # two NON-overlapping acc buffers we'd use 2*cta_n + sf. When that doesn't
    # fit (e.g. cta_n=256), use the OVERLAP trick: the two acc buffers overlap by
    # `acc_overlap_cols` (a multiple of the 32-col epilogue load), so the acc
    # region spans only `2*cta_n - overlap`. The epilogue loads the overlap
    # subtile FIRST and arrives acc_empty early, so the MMA of the next tile can
    # reuse the overlap region while the epilogue drains the rest — preserving
    # the double-TMEM pipelining at acc_stages=1's single mbar.
    # Key off the config's ARCH (not the active GPU) so the kernel budgets the
    # right TMEM column count even when rendered on a different / no GPU.
    total_tmem = _tmem_cols_for_arch(cfg.arch)

    def _align16(x: int) -> int:
        return (x + 15) & ~15

    # SF is a single fixed word PER DISTINCT OPERAND (a shared A → one SFA word,
    # read by every GEMM that uses it), refreshed by the utccp once per 128x4
    # atom right before the MMA that reads it.
    # --- TMEM acc-stage + overlap (per-GEMM budget; arch- & count-agnostic) ---
    # Each GEMM's accumulator gets its OWN, NON-overlapping TMEM region; the two
    # tile-stage buffers OVERLAP *within* that region. Budget SF first, split the
    # rest evenly across GEMMs, then run the ordinary single-GEMM stage/overlap
    # decision on that per-GEMM budget:
    #   per_gemm = (total_tmem - sf) // num_gemms          (>= cta_n, else reject)
    #   2*cta_n <= per_gemm        -> acc_stages = 2 (full per-GEMM double-buffer)
    #   else overlap = ceil32(2*cta_n - per_gemm):
    #         overlap < cta_n      -> acc_stages = 1 + overlap (parity toggle)
    #         else                 -> acc_stages = 1, no overlap
    # GEMM g occupies [g*acc_gemm_stride, (g+1)*acc_gemm_stride); within it stage s
    # is at +s*acc_stage_stride. SF follows all GEMM regions. Single-GEMM collapses
    # to the legacy behaviour; the formula is independent of num_gemms and total
    # TMEM (sm100=512).
    num_gemms = chain.num_gemms
    na, nb = chain.num_a_operands, chain.num_b_operands
    sf_total_cols = na * sfa_tmem_cols + nb * sfb_tmem_cols
    per_gemm = (total_tmem - sf_total_cols) // num_gemms
    if per_gemm < cta_n:
        raise NotImplementedError(
            f"block-scale {cfg.name!r}: per-GEMM TMEM budget {per_gemm} < one acc "
            f"({cta_n}) for {num_gemms} GEMMs + SF({sf_total_cols}). "
            f"Smaller cta_n / fewer GEMMs."
        )
    acc_overlap_cols = 0
    if 2 * cta_n <= per_gemm:
        acc_stages = 2  # full per-GEMM double-buffer
    else:
        acc_stages = 1
        gran = 32  # epilogue TMEM-load drain unit (cols)
        ov = ((2 * cta_n - per_gemm + gran - 1) // gran) * gran
        if ov < cta_n:  # else no room → plain 1-stage
            acc_overlap_cols = ov
    use_acc_overlap = acc_overlap_cols > 0
    # within-GEMM per-stage stride + per-GEMM region size:
    acc_stage_stride = (cta_n - acc_overlap_cols) if use_acc_overlap else cta_n
    if acc_stages == 2:
        acc_gemm_stride = 2 * cta_n
    elif use_acc_overlap:
        acc_gemm_stride = 2 * cta_n - acc_overlap_cols
    else:
        acc_gemm_stride = cta_n
    acc_overlap_subtiles = acc_overlap_cols // 32
    acc_region_cols = cta_n  # per-stage stride WITHIN a GEMM

    sf_region_base = _align16(num_gemms * acc_gemm_stride)
    # Per-distinct-operand SF word col bases. Single-GEMM → length-1 lists, the
    # first matching the legacy sfa_col_base / sfb_col_base.
    sfa_col_bases = [sf_region_base + i * sfa_tmem_cols for i in range(na)]
    sfb_col_bases = [sf_region_base + na * sfa_tmem_cols + j * sfb_tmem_cols for j in range(nb)]
    sfa_col_base = sfa_col_bases[0]
    sfb_col_base = sfb_col_bases[0]
    # tcgen05.alloc requires a power-of-2 column count; allocate the full TMEM.
    used_cols = sf_region_base + sf_total_cols
    num_tmem_alloc_cols = total_tmem
    assert used_cols <= num_tmem_alloc_cols, f"block-scale {cfg.name!r}: used {used_cols} > TMEM {num_tmem_alloc_cols}"

    # --- AB SMEM pipeline depth ----------------------------------------------
    # Per-stage SMEM = (packed data + SF) per DISTINCT operand + 2 mbar.
    per_stage = na * (sA_packed_elems + sfa_smem_bytes) + nb * (sB_packed_elems + sfb_smem_bytes) + 2 * 8
    from .tile_config import _SM100_SMEM_BUDGET_BYTES, _AB_STAGES_CAP

    fixed = 2 * acc_stages * 8 + 8
    # TMA-store epilogue stages output subtiles through a fixed SMEM-D buffer;
    # reserve it before sizing the AB pipeline (else SMEM overflows the cap).
    if use_tma_store_epi:
        fixed += _smem_d_bytes(cfg, chain)
    ab_stages = max(1, min((_SM100_SMEM_BUDGET_BYTES - fixed) // per_stage, _AB_STAGES_CAP))

    out_dt = chain.output_dtype
    vec_bytes_epi = _compute_output_vec_bytes(chain)

    # Instruction-descriptor operand dtype. fp4 (mxf4nvf4) MMA: the template
    # uses the Tcgen05MxInstrDesc with the E5M2 piggy-back. fp8 (mxf8f6f4) MMA:
    # the real fp8 dtype.
    if is_fp4:
        idesc_a = idesc_b = "cutlass.Float8E5M2"
    else:
        idesc_a = _DTYPE_TO_DSL[bs.a_dtype]
        idesc_b = _DTYPE_TO_DSL[bs.b_dtype]

    # FP4 needs the explicit B4X16 (4-bit packed) TMA format; FP8 uses the
    # auto-derived byte format (tma_format=None → default).
    ab_tma_format = "_tma.TensorMapDataFormat.B4X16" if is_fp4 else "None"

    lines = [
        f"# Block-scale config: {cfg.name} combo={bs.combo}",
        f"cta_tile_m = {cta_m}",
        f"cta_tile_n = {cta_n}",
        f"cta_tile_k_elems = {cta_k_elems}",
        f"cta_tile_mnk = ({cta_m}, {cta_n // cta_group}, {cta_k_elems})",
        # MMA instruction M = cta_m × cta_group (256 for the 2-CTA pair).
        f"mma_inst_shape_mnk = ({cta_m * cta_group}, {cta_n}, {mma_inst_k_elems})",
        f"cgrp_tile_mnk = ({cta_m * cfg.cgrp_size_m}, {cta_n * cfg.cgrp_size_n}, {cta_k_elems})",
        f"cgrp_tile_m = {cta_m * cfg.cgrp_size_m}",
        f"cgrp_tile_n = {cta_n * cfg.cgrp_size_n}",
        f"epi_tile_mn = {cfg.epi_tile_mn}",
        f"threads_per_cta = 256",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"ab_stages = {ab_stages}",
        f"acc_stages = {acc_stages}",
        f"use_acc_overlap = {use_acc_overlap}",
        f"acc_stage_stride = {acc_stage_stride}",
        f"acc_overlap_subtiles = {acc_overlap_subtiles}",
        # Multi-GEMM (parallel block-scale matmuls sharing the epilogue). Always
        # emitted; single-GEMM = (1,1,1). Each GEMM owns cta_n acc cols within an
        # acc_stage (acc_region_cols total); each distinct operand owns one SF word.
        f"num_gemms = {num_gemms}",
        f"num_a_operands = {na}",
        f"num_b_operands = {nb}",
        f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}",
        f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}",
        f"acc_region_cols = {acc_region_cols}",
        # Per-GEMM TMEM region size: GEMM g's acc lives at base + g*acc_gemm_stride
        # (regions are disjoint; the 2 tile-stage buffers overlap *within* a region).
        f"acc_gemm_stride = {acc_gemm_stride}",
        f"sfa_col_bases = {tuple(sfa_col_bases)}",
        f"sfb_col_bases = {tuple(sfb_col_bases)}",
        f"tile_swizzle_n = {cfg.tile_swizzle_n}",
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b(cta_group)}",
        "",
        f"# packed data SMEM",
        f"ab_dtype = {_DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"ab_packed_per_row = {ab_packed_per_row}",
        f"sA_packed_elems = {sA_packed_elems}",
        f"sB_packed_elems = {sB_packed_elems}",
        f"ab_tma_dtype = {_DTYPE_TO_DSL[bs.a_dtype]}",
        # TMA-descriptor element dtype. FP4 uses the NATIVE 4-bit Float4E2M1FN
        # (not the packed-pair Float4E2M1FNx2) so cute scales the descriptor's
        # strides/box by width=4 itself — no manual stride halving, matching the
        # block-scale reference. FP8 is 1 B/elem, same as ab_tma_dtype.
        f"ab_tma_desc_dtype = {'cutlass.Float4E2M1FN' if is_fp4 else _DTYPE_TO_DSL[bs.a_dtype]}",
        f"ab_tma_format = {ab_tma_format}",
        "ab_tma_swizzle = _tma.TensorMapSwizzle.s128b",
        "ab_smem_swizzle = cutlass.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B",
        f"a_smem_desc_leading_byte_offset = {a_lbo}",
        f"a_smem_desc_stride_byte_offset = {a_sbo}",
        f"a_smem_k_step_bytes = {a_k_step}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_smem_desc_leading_byte_offset = {b_lbo}",
        f"b_smem_desc_stride_byte_offset = {b_sbo}",
        f"b_smem_k_step_bytes = {b_k_step}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        # A/B operand major. FP4 is K-major only (rejected above otherwise);
        # mxfp8 may be M-major (A) / N-major (B), handled by the templates'
        # a_is_m_major / b_is_n_major TMA + descriptor branches. SF layout is
        # unchanged regardless.
        f"a_is_m_major = {a_major == 'm'}",
        f"b_is_n_major = {b_major == 'n'}",
        # MMA-instruction-descriptor major flags (1 = MN-major operand). The
        # block-scale idesc needs these so tcgen05 interprets the SMEM matrix
        # descriptor with the right operand orientation.
        f"mma_a_major = {1 if a_major == 'm' else 0}",
        f"mma_b_major = {1 if b_major == 'n' else 0}",
        "",
        f"# output",
        f"cd_dtype = {_DTYPE_TO_CUTLASS[out_dt]}",
        f"cd_tma_dtype = {_DTYPE_TO_DSL[out_dt]}",
        f"vec_bytes_epi = {vec_bytes_epi}",
        f"use_tma_store_epi = {use_tma_store_epi}",
        f"cd_out_is_m_major = {chain.matmul.out_major == 'm'}",
        # M-major TMA-store C-descriptor inner-M box = swizzle span(128 B) /
        # elem_bytes (inner box bytes == swizzle span).
        f"cd_mmajor_atom_m = {128 // _DTYPE_BYTES[out_dt]}",
        "",
        f"# block-scale MMA",
        f"mma_block_scale_kind = nvvm.MMABlockScaleKind.{bs.mma_block_scale_kind}",
        f"scale_vec_size = nvvm.Tcgen05MMABlockScale.{bs.scale_vec_size}",
        f"idesc_a_dtype = {idesc_a}",
        f"idesc_b_dtype = {idesc_b}",
        f"sf_scale_format = {bs.sf_scale_format}",
        f"mma_m_dim = {cta_m * cta_group}",
        f"mma_n_dim = {cta_n}",
        "",
        f"# scale factors",
        f"block_size = {bs.block_size}",
        f"sf_cutlass_dtype = {_DTYPE_TO_CUTLASS[bs.sf_dtype]}",
        f"sf_k = {sf_k}",
        f"sf_scales_per_inst = {scales_per_inst}",
        f"sf_insts_per_atom = {insts_per_word}",
        f"num_sf_atoms = {num_sf_words}",
        f"word_atoms = {word_atoms}",
        f"num_blocks_m = {nb_m}",
        f"num_blocks_n = {nb_n}",
        f"registers_per_block = {_REGISTERS_PER_BLOCK}",
        f"registers_per_atom = {_REGISTERS_PER_ATOM}",
        f"sf_atom_desc_stride = {sf_atom_desc_stride}",
        f"sf_block_desc_stride = {sf_block_desc_stride}",
        f"sfa_col_base = {sfa_col_base}",
        f"sfb_col_base = {sfb_col_base}",
        f"num_tmem_alloc_cols = {num_tmem_alloc_cols}",
        f"sfa_smem_bytes = {sfa_smem_bytes}",
        f"sfb_smem_bytes = {sfb_smem_bytes}",
        f"sf_tma_box_k = {sf_tma_box_k}",
        f"sfa_tma_box_mn = {nb_m}",
        f"sfb_tma_box_mn = {nb_n}",
    ]
    # MoE grouped block-scale matmul: the grouped persistent scheduler launches a
    # FIXED number of clusters (≈ NUM_SMS / cluster_size) and strides through
    # tiles by this count; the host grid and the kernel's stride share it. The
    # first_token_offset element dtype (int32/int64) drives the compile() fake.
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {_DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
        # A's per-row byte stride = k * ab_data_elem_bits / 8 (FP4: 4 bits/elem →
        # k/2 bytes). The descriptor-replacement base offset for a routed group is
        # group_begin rows × this. (NOT ab_dtype.width — that is the *packed*
        # Float4E2M1FNx2 = 8-bit type.)
        lines.append(f"ab_data_elem_bits = {data_elem_bits}")
    return "\n".join(lines)


def _resolve_path_blocks(src: str, use_tma_store_epi: bool) -> str:
    """Strip mode-specific blocks from the kernel template *before* the
    `@@INJECT_*@@` markers are filled in.

    Why this exists: `cutlass.const_expr` in cute selects which branch's IR
    to emit, but **both** branches are still type-checked at parse time.
    The TMA and STG epilogue paths intentionally bind ``vec_f32`` to vectors
    of different shapes (full t2r_inst_repx vs a vsize-element slice), so
    cute would flag the assignment in the dynamic ``if col_j + vsize <= N:``
    inside the STG branch as a type-change when the TMA branch also
    declares ``vec_f32``. Stripping the dead branch at render time avoids
    the type-consistency requirement entirely.

    Marker syntax (one block per pair, no nesting):

        # @@TMA_STORE_ONLY:BEGIN@@
        ...TMA-only code, including any @@INJECT_*@@ markers...
        # @@TMA_STORE_ONLY:END@@

        # @@STG_ONLY:BEGIN@@
        ...STG-only code, including any @@INJECT_*@@ markers...
        # @@STG_ONLY:END@@
    """
    keep_marker = "TMA_STORE_ONLY" if use_tma_store_epi else "STG_ONLY"
    drop_marker = "STG_ONLY" if use_tma_store_epi else "TMA_STORE_ONLY"
    keep_pat = re.compile(
        rf"^[ \t]*# *@@{keep_marker}:BEGIN@@[ \t]*\n(.*?)" rf"^[ \t]*# *@@{keep_marker}:END@@[ \t]*\n",
        flags=re.MULTILINE | re.DOTALL,
    )
    drop_pat = re.compile(
        rf"^[ \t]*# *@@{drop_marker}:BEGIN@@[ \t]*\n.*?" rf"^[ \t]*# *@@{drop_marker}:END@@[ \t]*\n",
        flags=re.MULTILINE | re.DOTALL,
    )
    src = keep_pat.sub(r"\1", src)
    src = drop_pat.sub("", src)
    return src


def _mainloop_template_file(base_template_file: str) -> str:
    """Map the ordinary template filename to its mainloop-fusion variant.

    ``sm100_matmul_1ctamma.py`` ->
    ``sm100_matmul_mainloop_1ctamma.py``.
    """
    return base_template_file.replace("_matmul_", "_matmul_mainloop_")


def _render_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
    cta_group: int,
    scheduler: str,
) -> str:
    # Template file is selected by the kernel registry (single source of truth)
    # from the pure-geometry config + the requested execution strategy
    # (cta_group / scheduler); mainloop / graph_type come from the chain.
    from .kernel_registry import select_template

    tmpl = select_template(chain, config, cta_group, scheduler)
    template_path = _TEMPLATE_DIR / tmpl.file
    src = template_path.read_text()
    # Strip the unused epilogue-store path *first* so its @@INJECT_EPILOGUE@@
    # marker doesn't survive into the marker-replacement step.
    vec_bytes_epi = _compute_output_vec_bytes(chain)
    use_tma = (not _FORCE_STG_EPI) and _use_tma_store_epi(chain, config, vec_bytes_epi, cta_group)
    src = _resolve_path_blocks(src, use_tma)

    aux_tensors = chain.aux_tensors

    # ---- Multi-GEMM A/B operand plumbing (1ctamma only) ------------------
    # One TMA descriptor + SMEM buffer + runtime tensor per DISTINCT operand
    # (deduped by the analyzer). Single-GEMM = 1 A + 1 B (suffix _0), so the
    # rendered loops have length 1 and behave exactly like the legacy code.
    na, nb = chain.num_a_operands, chain.num_b_operands
    kernel_ab_desc_params = (
        ",\n".join(
            [f"tma_a_desc_{i}: cutlass.GridConstant[_tma.TensorMap]" for i in range(na)]
            + [f"tma_b_desc_{j}: cutlass.GridConstant[_tma.TensorMap]" for j in range(nb)]
        )
        + ","
    )
    ab_desc_lists = (
        "tma_a_descs = [" + ", ".join(f"tma_a_desc_{i}" for i in range(na)) + "]\n" "tma_b_descs = [" + ", ".join(f"tma_b_desc_{j}" for j in range(nb)) + "]"
    )
    host_ab_params = ",\n".join([f"a_{i}: cute.Tensor" for i in range(na)] + [f"b_{j}: cute.Tensor" for j in range(nb)]) + ","
    host_ab_lists = "_a_operands = [" + ", ".join(f"a_{i}" for i in range(na)) + "]\n" "_b_operands = [" + ", ".join(f"b_{j}" for j in range(nb)) + "]"
    host_kernel_desc_pass = ",\n".join([f"tma_a_desc_list[{i}]" for i in range(na)] + [f"tma_b_desc_list[{j}]" for j in range(nb)]) + ","
    compile_ab_fakes = "\n".join([f"fake_a_{i} = _make_fake_a()" for i in range(na)] + [f"fake_b_{j} = _make_fake_b()" for j in range(nb)])
    compile_ab_pass = ",\n".join([f"fake_a_{i}" for i in range(na)] + [f"fake_b_{j}" for j in range(nb)]) + ","
    # Per-GEMM register-vector bindings in the STG inner loop: GEMM 0 is bound by
    # the template (vec_f32); the rest (vec_f32_1, ...) are injected here.
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}][j * vsize : (j + 1) * vsize]" for g in range(1, chain.num_gemms)) or "pass"
    # MoE multi-GEMM: the kernel also takes the raw A (token) tensor per distinct
    # A operand (to compute the per-group patched base address). These are the
    # same runtime tensors the host uses to build the A descriptors (a_0, ...).
    moe_kernel_ma_params = ",\n".join(f"mA_{i}: cute.Tensor" for i in range(na))
    if moe_kernel_ma_params:
        moe_kernel_ma_params += ","
    moe_ma_list = "mA_list = [" + ", ".join(f"mA_{i}" for i in range(na)) + "]"
    moe_host_ma_pass = ",\n".join(f"a_{i}" for i in range(na))
    if moe_host_ma_pass:
        moe_host_ma_pass += ","

    # Indentation matches the marker's column in the template (8 spaces inside
    # _kernel/_host signatures, 4 inside compile() body).
    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(
        aux_tensors,
        dynamic_strides=not chain.has_mainloop_fusion and not chain.is_multi_gemm,
    )
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_tile_constants(config, chain, cta_group)
    if snippets.tap_constants:
        tile_constants += "\n" + "\n".join(snippets.tap_constants)
    # Multi-output tap plumbing (Phase 2). Empty lists → markers expand to
    # nothing; the kernel signature shrinks back to the single-output form.
    kernel_tap_params = ",\n".join(snippets.tap_kernel_params)
    if kernel_tap_params:
        kernel_tap_params += ","
    host_tap_params = ",\n".join(snippets.tap_host_params)
    if host_tap_params:
        host_tap_params += ","
    host_tap_pass = ",\n".join(snippets.tap_host_pass)
    if host_tap_pass:
        host_tap_pass += ","
    compile_tap_fakes = "\n".join(snippets.tap_compile_fakes)
    compile_tap_pass = ",\n".join(snippets.tap_compile_pass)
    if compile_tap_pass:
        compile_tap_pass += ","
    tap_ptr_binds = "\n".join(snippets.tap_ptr_binds) if snippets.tap_ptr_binds else "pass"
    red_kernel_stride_params = _reduction_stride_kernel_params(chain)
    red_host_stride_unpack = _reduction_stride_host_unpack_from(chain, 14) if chain.has_moe else _reduction_stride_host_unpack(chain)
    red_host_stride_pass = _reduction_stride_host_pass(chain)
    red_compile_stride_decls = _reduction_stride_compile_decls(chain)
    red_compile_stride_symbols = _reduction_stride_compile_symbols(chain)

    # The @@INJECT_*@@ markers appear inside a comment-only line so we just
    # locate the *line* containing the marker and replace it. A marker may
    # appear multiple times at different indent levels (e.g. @@INJECT_EPILOGUE@@
    # inside both the TMA and STG epilogue branches); use a per-match
    # substitution callback so each occurrence gets re-indented at its own
    # column rather than the first one's.
    def _replace_marker_line(marker: str, replacement: str) -> None:
        nonlocal src
        pattern = re.compile(rf"^([ \t]*)# *@@{marker}@@[ \t]*\n", flags=re.MULTILINE)
        if not pattern.search(src):
            raise RuntimeError(f"template missing marker @@{marker}@@")
        if replacement == "":
            src = pattern.sub("", src)
        else:

            def repl(m: re.Match) -> str:
                indent = m.group(1)
                return indent + replacement.replace("\n", "\n" + indent) + "\n"

            src = pattern.sub(repl, src)

    _replace_marker_line("INJECT_TILE_CONSTANTS", tile_constants)
    _replace_marker_line("INJECT_KERNEL_AUX_PARAMS", kernel_aux_params)
    _replace_marker_line("INJECT_HOST_AUX_PARAMS", host_aux_params)
    _replace_marker_line("INJECT_HOST_AUX_PASS", host_aux_pass)
    _replace_marker_line("INJECT_COMPILE_AUX_FAKES", compile_aux_fakes)
    _replace_marker_line("INJECT_COMPILE_AUX_PASS", compile_aux_pass)
    _replace_marker_line("INJECT_KERNEL_TAP_PARAMS", kernel_tap_params)
    _replace_marker_line("INJECT_HOST_TAP_PARAMS", host_tap_params)
    _replace_marker_line("INJECT_HOST_TAP_PASS", host_tap_pass)
    _replace_marker_line("INJECT_COMPILE_TAP_FAKES", compile_tap_fakes)
    _replace_marker_line("INJECT_COMPILE_TAP_PASS", compile_tap_pass)
    _replace_marker_line("INJECT_TAP_PTRS", tap_ptr_binds)
    _replace_marker_line("INJECT_AUX_VIEWS", snippets.aux_views)
    _replace_marker_line("INJECT_EPILOGUE", snippets.epilogue)
    for marker, replacement in (
        ("INJECT_KERNEL_REDUCTION_STRIDE_PARAMS", red_kernel_stride_params),
        ("INJECT_HOST_REDUCTION_STRIDES", red_host_stride_unpack),
        ("INJECT_HOST_REDUCTION_STRIDE_PASS", red_host_stride_pass),
        ("INJECT_COMPILE_REDUCTION_STRIDE_DECLS", red_compile_stride_decls),
        ("INJECT_COMPILE_REDUCTION_STRIDE_SYMBOLS", red_compile_stride_symbols),
    ):
        if f"@@{marker}@@" in src:
            _replace_marker_line(marker, replacement)
        elif chain.reductions or chain.block_quant is not None:
            raise RuntimeError(f"template missing marker @@{marker}@@")
    # Multi-GEMM A/B operand plumbing — only the 1ctamma template carries these
    # markers (the only multi-GEMM-capable template this pass). Other templates
    # keep their fixed single-A/single-B signature.
    if "@@INJECT_KERNEL_AB_DESC_PARAMS@@" in src:
        _replace_marker_line("INJECT_KERNEL_AB_DESC_PARAMS", kernel_ab_desc_params)
        _replace_marker_line("INJECT_AB_DESC_LISTS", ab_desc_lists)
        _replace_marker_line("INJECT_HOST_AB_PARAMS", host_ab_params)
        _replace_marker_line("INJECT_HOST_AB_LISTS", host_ab_lists)
        _replace_marker_line("INJECT_HOST_KERNEL_DESC_PASS", host_kernel_desc_pass)
        _replace_marker_line("INJECT_COMPILE_AB_FAKES", compile_ab_fakes)
        _replace_marker_line("INJECT_COMPILE_AB_PASS", compile_ab_pass)
    # Per-GEMM STG vector bindings — present on every STG-epilogue template
    # (mainloop included; single-GEMM → `pass`). Filled ungated so the mainloop
    # epilogue can share the matmul's per-GEMM read/store pipeline verbatim.
    if "@@INJECT_STG_VEC_BINDINGS@@" in src:
        _replace_marker_line("INJECT_STG_VEC_BINDINGS", stg_vec_bindings)
    # MoE-specific raw-A-tensor plumbing — only the MoE grouped-matmul templates
    # carry these (the kernel patches each A descriptor per routed group).
    if "@@INJECT_MOE_KERNEL_MA_PARAMS@@" in src:
        _replace_marker_line("INJECT_MOE_KERNEL_MA_PARAMS", moe_kernel_ma_params)
        _replace_marker_line("INJECT_MOE_MA_LIST", moe_ma_list)
        _replace_marker_line("INJECT_MOE_HOST_MA_PASS", moe_host_ma_pass)
    # Mainloop-fusion transforms — only the 12-warp templates carry these.
    if chain.has_mainloop_fusion:
        _replace_marker_line("INJECT_MAINLOOP_A", snippets.mainloop_transform_a)
        _replace_marker_line("INJECT_MAINLOOP_B", snippets.mainloop_transform_b)

    # Tag the kernel function name with the template + geometry so nsys gives
    # each (template, config) a distinct GPU kernel symbol, e.g.
    # `_kernel_sm100_matmul_2ctamma_128x256x128_128x256x32_cluster2x1_...`.
    # cta_group / static come from the template stem, the geometry from config.
    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{tmpl.file.removesuffix('.py')}_{config.geometry_name}")
    src = re.sub(r"\b_kernel\(", f"_kernel_{tag}(", src)

    return src


def _render_block_scale_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
    cta_group: int,
    scheduler: str,
) -> str:
    """Render the block-scaled-matmul kernel template. Picks the TMA-store
    epilogue when `_use_tma_store_epi` allows (single tensor output, bf16/fp16,
    cta_tile_m=128, ...) and STG otherwise; SF TMA descriptors are hardcoded
    in the template (not injected). Epilogue aux/tap markers still work."""
    from .kernel_registry import select_template

    tmpl = select_template(chain, config, cta_group, scheduler)
    template_path = _TEMPLATE_DIR / tmpl.file
    src = template_path.read_text()
    vec_bytes_epi = _compute_output_vec_bytes(chain)
    use_tma = (not _FORCE_STG_EPI) and _use_tma_store_epi(chain, config, vec_bytes_epi, cta_group)
    src = _resolve_path_blocks(src, use_tma_store_epi=use_tma)

    aux_tensors = chain.aux_tensors
    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(aux_tensors)
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_block_scale_tile_constants(config, chain, cta_group, use_tma_store_epi=use_tma)
    if snippets.tap_constants:
        tile_constants += "\n" + "\n".join(snippets.tap_constants)

    kernel_tap_params = ",\n".join(snippets.tap_kernel_params)
    if kernel_tap_params:
        kernel_tap_params += ","
    host_tap_params = ",\n".join(snippets.tap_host_params)
    if host_tap_params:
        host_tap_params += ","
    host_tap_pass = ",\n".join(snippets.tap_host_pass)
    if host_tap_pass:
        host_tap_pass += ","
    compile_tap_fakes = "\n".join(snippets.tap_compile_fakes)
    compile_tap_pass = ",\n".join(snippets.tap_compile_pass)
    if compile_tap_pass:
        compile_tap_pass += ","
    tap_ptr_binds = "\n".join(snippets.tap_ptr_binds) if snippets.tap_ptr_binds else "pass"
    red_kernel_stride_params = _reduction_stride_kernel_params(chain)
    red_host_stride_unpack = _reduction_stride_host_unpack_from(chain, 14 if chain.has_moe else 13)
    red_host_stride_pass = _reduction_stride_host_pass(chain)
    red_compile_stride_decls = _reduction_stride_compile_decls(chain)
    red_compile_stride_symbols = _reduction_stride_compile_symbols(chain)

    # ---- Multi-GEMM A/B + SF operand plumbing (block-scale) ----------------
    # One (packed data + SF) descriptor pair per DISTINCT operand. Single-GEMM =
    # 1 A + 1 B (suffix _0). The SF travels with its data operand.
    # Operand order is GROUPED BY KIND — all A data, all B data, all SFA, all
    # SFB — so single-GEMM (na=nb=1) is exactly (a, b, sfa, sfb): the legacy
    # block-scale runtime-call order (CompiledFusedGemm passes a, b, sfa, sfb).
    na, nb = chain.num_a_operands, chain.num_b_operands
    _G = "cutlass.GridConstant[_tma.TensorMap]"
    kernel_ab_desc_params = (
        ",\n".join(
            [f"tma_a_desc_{i}: {_G}" for i in range(na)]
            + [f"tma_b_desc_{j}: {_G}" for j in range(nb)]
            + [f"tma_sfa_desc_{i}: {_G}" for i in range(na)]
            + [f"tma_sfb_desc_{j}: {_G}" for j in range(nb)]
        )
        + ","
    )
    ab_desc_lists = (
        "tma_a_descs = [" + ", ".join(f"tma_a_desc_{i}" for i in range(na)) + "]\n"
        "tma_b_descs = [" + ", ".join(f"tma_b_desc_{j}" for j in range(nb)) + "]\n"
        "tma_sfa_descs = [" + ", ".join(f"tma_sfa_desc_{i}" for i in range(na)) + "]\n"
        "tma_sfb_descs = [" + ", ".join(f"tma_sfb_desc_{j}" for j in range(nb)) + "]"
    )
    host_ab_params = (
        ",\n".join(
            [f"a_{i}: cute.Tensor" for i in range(na)]
            + [f"b_{j}: cute.Tensor" for j in range(nb)]
            + [f"sfa_{i}: cute.Tensor" for i in range(na)]
            + [f"sfb_{j}: cute.Tensor" for j in range(nb)]
        )
        + ","
    )
    host_ab_lists = (
        "_a_operands = [" + ", ".join(f"a_{i}" for i in range(na)) + "]\n"
        "_b_operands = [" + ", ".join(f"b_{j}" for j in range(nb)) + "]\n"
        "_sfa_operands = [" + ", ".join(f"sfa_{i}" for i in range(na)) + "]\n"
        "_sfb_operands = [" + ", ".join(f"sfb_{j}" for j in range(nb)) + "]"
    )
    host_kernel_desc_pass = (
        ",\n".join(
            [f"tma_a_desc_list[{i}]" for i in range(na)]
            + [f"tma_b_desc_list[{j}]" for j in range(nb)]
            + [f"tma_sfa_desc_list[{i}]" for i in range(na)]
            + [f"tma_sfb_desc_list[{j}]" for j in range(nb)]
        )
        + ","
    )
    compile_ab_fakes = "\n".join(
        [f"fake_a_{i} = _make_fake_a()" for i in range(na)]
        + [f"fake_b_{j} = _make_fake_b()" for j in range(nb)]
        + [f"fake_sfa_{i} = _make_fake_sfa()" for i in range(na)]
        + [f"fake_sfb_{j} = _make_fake_sfb()" for j in range(nb)]
    )
    compile_ab_pass = (
        ",\n".join(
            [f"fake_a_{i}" for i in range(na)]
            + [f"fake_b_{j}" for j in range(nb)]
            + [f"fake_sfa_{i}" for i in range(na)]
            + [f"fake_sfb_{j}" for j in range(nb)]
        )
        + ","
    )
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}][j * vsize : (j + 1) * vsize]" for g in range(1, chain.num_gemms)) or "pass"
    # MoE block-scale multi-GEMM: the kernel also takes the raw A (token) tensor
    # per distinct A operand for the per-routed-group descriptor patch.
    moe_kernel_ma_params = ",\n".join(f"mA_{i}: cute.Tensor" for i in range(na))
    if moe_kernel_ma_params:
        moe_kernel_ma_params += ","
    moe_ma_list = "mA_list = [" + ", ".join(f"mA_{i}" for i in range(na)) + "]"
    moe_host_ma_pass = ",\n".join(f"a_{i}" for i in range(na))
    if moe_host_ma_pass:
        moe_host_ma_pass += ","

    def _replace_marker_line(marker: str, replacement: str) -> None:
        nonlocal src
        pattern = re.compile(rf"^([ \t]*)# *@@{marker}@@[ \t]*\n", flags=re.MULTILINE)
        if not pattern.search(src):
            raise RuntimeError(f"block-scale template missing marker @@{marker}@@")
        if replacement == "":
            src = pattern.sub("", src)
        else:

            def repl(m: re.Match) -> str:
                indent = m.group(1)
                return indent + replacement.replace("\n", "\n" + indent) + "\n"

            src = pattern.sub(repl, src)

    _replace_marker_line("INJECT_TILE_CONSTANTS", tile_constants)
    _replace_marker_line("INJECT_KERNEL_AUX_PARAMS", kernel_aux_params)
    _replace_marker_line("INJECT_HOST_AUX_PARAMS", host_aux_params)
    _replace_marker_line("INJECT_HOST_AUX_PASS", host_aux_pass)
    _replace_marker_line("INJECT_COMPILE_AUX_FAKES", compile_aux_fakes)
    _replace_marker_line("INJECT_COMPILE_AUX_PASS", compile_aux_pass)
    _replace_marker_line("INJECT_KERNEL_TAP_PARAMS", kernel_tap_params)
    _replace_marker_line("INJECT_HOST_TAP_PARAMS", host_tap_params)
    _replace_marker_line("INJECT_HOST_TAP_PASS", host_tap_pass)
    _replace_marker_line("INJECT_COMPILE_TAP_FAKES", compile_tap_fakes)
    _replace_marker_line("INJECT_COMPILE_TAP_PASS", compile_tap_pass)
    _replace_marker_line("INJECT_TAP_PTRS", tap_ptr_binds)
    _replace_marker_line("INJECT_AUX_VIEWS", snippets.aux_views)
    _replace_marker_line("INJECT_EPILOGUE", snippets.epilogue)
    for marker, replacement in (
        ("INJECT_KERNEL_REDUCTION_STRIDE_PARAMS", red_kernel_stride_params),
        ("INJECT_HOST_REDUCTION_STRIDES", red_host_stride_unpack),
        ("INJECT_HOST_REDUCTION_STRIDE_PASS", red_host_stride_pass),
        ("INJECT_COMPILE_REDUCTION_STRIDE_DECLS", red_compile_stride_decls),
        ("INJECT_COMPILE_REDUCTION_STRIDE_SYMBOLS", red_compile_stride_symbols),
    ):
        if f"@@{marker}@@" in src:
            _replace_marker_line(marker, replacement)
        elif chain.reductions or chain.block_quant is not None:
            raise RuntimeError(f"block-scale template missing marker @@{marker}@@")
    # Multi-GEMM A/B + SF operand plumbing — present only on the block-scale
    # templates that carry these markers (gated; MoE block-scale lacks them).
    if "@@INJECT_KERNEL_AB_DESC_PARAMS@@" in src:
        _replace_marker_line("INJECT_KERNEL_AB_DESC_PARAMS", kernel_ab_desc_params)
        _replace_marker_line("INJECT_AB_DESC_LISTS", ab_desc_lists)
        _replace_marker_line("INJECT_HOST_AB_PARAMS", host_ab_params)
        _replace_marker_line("INJECT_HOST_AB_LISTS", host_ab_lists)
        _replace_marker_line("INJECT_HOST_KERNEL_DESC_PASS", host_kernel_desc_pass)
        _replace_marker_line("INJECT_COMPILE_AB_FAKES", compile_ab_fakes)
        _replace_marker_line("INJECT_COMPILE_AB_PASS", compile_ab_pass)
        if "@@INJECT_STG_VEC_BINDINGS@@" in src:
            _replace_marker_line("INJECT_STG_VEC_BINDINGS", stg_vec_bindings)
    # MoE block-scale raw-A-tensor plumbing (per-routed-group descriptor patch).
    if "@@INJECT_MOE_KERNEL_MA_PARAMS@@" in src:
        _replace_marker_line("INJECT_MOE_KERNEL_MA_PARAMS", moe_kernel_ma_params)
        _replace_marker_line("INJECT_MOE_MA_LIST", moe_ma_list)
        _replace_marker_line("INJECT_MOE_HOST_MA_PASS", moe_host_ma_pass)

    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{tmpl.file.removesuffix('.py')}_{config.geometry_name}")
    src = re.sub(r"\b_kernel\(", f"_kernel_{tag}(", src)
    return src


# ---------------------------------------------------------------------------
# Cache / import
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    base = os.environ.get("CUDNN_GEMM_KERNEL_CACHE")
    if not base:
        # Default to a per-user cache OUTSIDE the source tree / installed package
        # so we never write a cache dir into the project (or site-packages).
        # Honor XDG_CACHE_HOME, else ~/.cache.
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "cudnn_gemm", "kernel_cache")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_dsl_array_offset() -> None:
    """Re-enable ``cutlass.Array`` pointer-offset arithmetic (``arr + n``).

    The cute DSL removed ``Array.__add__`` (to avoid confusion with element-wise
    vector math), but the kernel templates use it pervasively for mbar / SMEM
    pointers, always meaning "sub-Array at element offset n". Bind it to the
    DSL's sanctioned replacement, ``subview(n)`` (still supports ``.data_ptr()``
    / indexing / nvvm ops). Lazy + idempotent so importing this package never
    requires the DSL to be installed — only the actual JIT-compile path does."""
    import cutlass

    if not getattr(cutlass.Array, "_gemm_offset_add", False):
        cutlass.Array.__add__ = lambda self, n: self.subview(n)
        cutlass.Array._gemm_offset_add = True


def _import_kernel(src: str) -> object:
    """Write rendered source to a content-addressed dir and dynamic-import it."""
    _ensure_dsl_array_offset()
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    out_dir = _cache_dir() / f"gen_{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generated_kernel.py"
    if not out_file.exists() or out_file.read_text() != src:
        out_file.write_text(src)

    mod_name = f"_cudnn_gemm_generated_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, out_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Layout-ambiguity helper — see CompiledFusedGemm.__call__ for the policy
# ---------------------------------------------------------------------------


def _is_layout_ambiguous(t: object) -> bool:
    """True iff cute's auto-deduce of ``leading_dim`` would fault on this tensor.

    cute's algorithm (from ``cute.runtime._Tensor.mark_layout_dynamic`` docs):

    * exactly one dim has stride 1                                 -> OK
    * multiple dims have stride 1, exactly one of them has size>1  -> OK
    * otherwise                                                    -> raises

    The third case is what we want to detect — most commonly a (1, 1) scalar
    aux tensor with stride (1, 1), where both dims have stride 1 and neither
    has size > 1.
    """
    strides = getattr(t, "stride", None)
    shape = getattr(t, "shape", None)
    if strides is None or shape is None:
        return False
    try:
        strides = list(strides())  # torch.Tensor.stride is a method
    except TypeError:
        strides = list(strides)  # numpy etc.
    sizes = list(shape)
    stride1_dims = [i for i, s in enumerate(strides) if s == 1]
    if len(stride1_dims) <= 1:
        return False
    # Multiple dims have stride 1 -> only safe if exactly one has size > 1.
    big = [i for i in stride1_dims if sizes[i] > 1]
    return len(big) != 1


# Project convention for explicit leading_dim, applied when cute's auto-deduce
# would fault. Indexed by position in CompiledFusedGemm.__call__'s signature.
# All matmul tensors are rank-3 at the runtime API; per-element leading-dim
# refers to the inner (M, K)/(N, K)/(M, N) plane (batch axis is permuted to
# the outermost dim before the kernel sees it).
# A runtime tensors are always shaped (batch, M, K); K-major has contiguous K,
# M-major has contiguous M through a transposed view.
# B runtime tensors are always shaped (batch, N, K); K-major has contiguous K,
# N-major has contiguous N through a transposed view. Per project convention
# we treat B's leading dim as -2 when cute needs an explicit leading_dim.
# C: (batch, M, N) row-major, contiguous dim is N (-1).
# aux: broadcasts onto the (..., M, N) output, C-like -> -1.
_LEADING_DIM_A = -1
_LEADING_DIM_B = -2
_LEADING_DIM_C = -1
_LEADING_DIM_AUX = -1


def _maybe_wrap_layout(t: object, leading_dim: int) -> object:
    """If ``t``'s layout would defeat cute's auto-deduce, pre-wrap it via
    DLPack + explicit ``mark_layout_dynamic(leading_dim)``. The returned cute
    ``_Tensor`` exposes ``__c_pointers__``, so the JIT executor's argument
    loop bypasses ``TensorAdapter`` (which always calls the parameterless
    overload and would fault here). For non-ambiguous tensors we pass
    through unchanged to keep the fast path."""
    if not _is_layout_ambiguous(t):
        return t
    # Lazy import — cute is only needed on the runtime fault path, and the
    # smoke tier of `cudnn.TBD.gemm.verify` runs without GPU.
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t).mark_layout_dynamic(leading_dim)


def _wrap_raw_tensor(t: object) -> object:
    """Wrap a tensor when the kernel only consumes its raw pointer."""
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t)


_REDUCTION_INIT_VALUE = {
    "fp32": {
        "add": 0.0,
        "amax": 0.0,
        "max": -float("inf"),
        "min": float("inf"),
    },
    "int32": {
        "add": 0,
        "amax": 0,
        "max": -(2**31),
        "min": 2**31 - 1,
    },
}


def _expected_output_shape(spec, chain: FusionChain, mnk) -> tuple[int, int, int]:
    full = (chain.matmul.batch, int(mnk[0]), int(mnk[1]))
    if spec.is_quant_scale:
        assert spec.dim is not None
        return spec.dim
    if not spec.is_reduction:
        return full
    assert spec.dim is not None
    red_idx = int(spec.source.rsplit("_", 1)[1])
    if chain.reductions[red_idx].grouped_by_moe:
        return spec.dim
    return tuple(1 if spec.dim[i] == 1 else full[i] for i in range(3))


def _initialize_reduction_outputs(chain: FusionChain, outputs) -> None:
    for spec, tensor in zip(chain.outputs, outputs):
        if not spec.is_reduction:
            continue
        red_idx = int(spec.source.rsplit("_", 1)[1])
        red = chain.reductions[red_idx]
        tensor.fill_(_REDUCTION_INIT_VALUE[red.compute_dtype][red.mode])


@dataclass
class CompiledFusedGemm:
    """One compiled fused-GEMM, directly callable with runtime tensors.

    M, N, K are all symbolic in the rendered kernel — this single object
    handles any valid problem size, including shapes that are NOT multiples
    of the cluster tile. The TMA descriptor's default OOB-fill behavior
    zero-fills any element loaded past `global_dims`, so partial K-tail /
    M-tail / N-tail tiles contribute 0 to the FP32 accumulator. The STG
    epilogue path additionally gates per-element on `row < M` and
    `col + vsize <= N` to avoid spurious writes, and the TMA-store path
    relies on TMA hardware dropping OOB coordinates.

    Construct via :func:`jit_from_cudnn_graph`, then invoke like a regular
    function::

        compiled = jit_from_cudnn_graph(g)
        # tensors are rank-3 (batch_a/M/K, batch_b/K/N, batch/M/N)
        compiled(a, b, c, (M, N, K))            # run
        compiled(a2, b2, c2, (M2, N2, K2))      # same compiled binary
        print(compiled.chain.summary())          # inspection

    Layout-ambiguity policy (see ``_maybe_wrap_layout``). For any tensor
    where cute's auto-deduce of ``leading_dim`` would fault (multiple
    stride-1 dims and no dim with size > 1 — i.e., a scalar (1,1) aux),
    we pre-wrap with explicit ``leading_dim`` per project convention:
    A=-1, B=-2, C=-1, aux=-1. Tensors that auto-deduce cleanly pass
    through unchanged to keep the fast path.

    Note on the TMA OOB-fill enum naming. The CUDA driver API exposes
    `CUtensorMapFloatOOBfill` with values NONE (=0) and NAN_REQUEST_ZERO_FMA
    (=1). The "NONE" name is misleading — empirically (and per the driver's
    `cudaTmaDescOobFillMode::TENSOR_ZFILL = 0` naming), bit 0
    means zero-fill, not "undefined". Setting `nan_request_zero_fma` on
    sm100 is actively harmful: tcgen05.mma propagates the NaN straight
    through the accumulator.
    """

    chain: FusionChain
    config: TileConfig
    aux_names: list[str]  # in aux-tensor order
    generated_path: Path
    _launchable: Callable  # cute-compiled, accepts (a, b, c, mnk, *aux)
    block_scale: bool = False  # block-scaled matmul (FP4/FP8 + SF)
    binding: "GemmBinding | None" = None  # role -> cuDNN tensor (variant-pack call)

    def __call__(self, variant_pack):
        # The runtime call is a variant-pack dict keyed by cuDNN tensor object
        # (or uid / name) -> buffer; (M, N, K) is inferred from the buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        if self.binding is None:
            raise NotImplementedError("variant-pack call is not wired up for this graph type")
        b = self.binding
        resolved = resolve_variant_pack(variant_pack, b)

        def pull(t, role):
            if t is None or id(t) not in resolved:
                raise KeyError(f"variant pack is missing a buffer for {role}")
            return resolved[id(t)]

        a_bufs = [pull(t, "A operand") for t in b.a_operands]
        b_bufs = [pull(t, "B operand") for t in b.b_operands]
        out_bufs = [pull(t, "output") for t in b.outputs]
        aux_bufs = [pull(t, f"aux {self.aux_names[i]!r}") for i, t in enumerate(b.aux)]
        # (M, N, K) from buffer shapes: A is (batch, M, K) — for FP4-packed A the
        # K dim is stored at K/2 elements/byte, so scale it back up. M, N come
        # from the terminal output (batch, M, N).
        k_factor = 2 if self.chain.matmul.a_dtype == "fp4_e2m1" else 1
        M = a_bufs[0].shape[1]
        K = a_bufs[0].shape[2] * k_factor
        N = out_bufs[0].shape[2]
        mnk = (M, N, K)
        c_arg = out_bufs if len(out_bufs) > 1 else out_bufs[0]

        if self.chain.is_multi_gemm:
            if self.block_scale:
                sfa = [pull(t, "SFA") for t in b.sfa_operands]
                sfb = [pull(t, "SFB") for t in b.sfb_operands]
                pairs = [((a_bufs[ai], sfa[ai]), (b_bufs[bi], sfb[bi])) for ai, bi in self.chain.gemm_operands]
            else:
                pairs = [(a_bufs[ai], b_bufs[bi]) for ai, bi in self.chain.gemm_operands]
            return self._call_positional(pairs, c_arg, mnk, *aux_bufs)

        if self.block_scale:
            sfa = pull(b.sfa_operands[0], "SFA")
            sfb = pull(b.sfb_operands[0], "SFB")
            return self._call_positional(a_bufs[0], b_bufs[0], c_arg, mnk, sfa, sfb, *aux_bufs)
        return self._call_positional(a_bufs[0], b_bufs[0], c_arg, mnk, *aux_bufs)

    def _call_positional(self, *args):
        # Internal launcher (driven by __call__ after it resolves the variant
        # pack). Single-GEMM args: (a, b, c, (M,N,K), *aux). Multi-GEMM: the
        # first arg is a list of per-GEMM (a, b) tensor pairs, deduped by tensor
        # identity into the distinct-A / distinct-B slots fixed at JIT.
        if self.chain.is_multi_gemm:
            if self.block_scale:
                return self._call_block_scale_multi_gemm(*args)
            return self._call_multi_gemm(*args)
        a, b, c, mnk, *aux = args
        # `c` may be a single Tensor (single-output, current default) or a
        # list/tuple of Tensors in the order declared by `self.chain.outputs`
        # — slot 0 = terminal, slot 1 = matmul tap (if any).
        outputs_spec = self.chain.outputs
        if isinstance(c, (list, tuple)):
            cs = list(c)
        else:
            cs = [c]
        if len(cs) != len(outputs_spec):
            raise ValueError(
                f"this graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(cs)} runtime "
                f"output tensor(s). Pass a list of tensors in slot order."
            )

        expected_a_batch = self.chain.matmul.a_batch
        expected_b_batch = self.chain.matmul.b_batch
        bad_shapes = len(a.shape) != 3 or len(b.shape) != 3 or a.shape[0] != expected_a_batch or b.shape[0] != expected_b_batch
        for spec, ci in zip(outputs_spec, cs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, self.chain, mnk):
                bad_shapes = True
        if bad_shapes:
            raise ValueError(
                f"runtime tensors must be rank-3 with shapes matching the graph "
                f"A batch={expected_a_batch}, B batch={expected_b_batch}, "
                f"outputs={[ _expected_output_shape(o, self.chain, mnk) for o in outputs_spec ]}; "
                f"got A={tuple(a.shape)}, B={tuple(b.shape)}, "
                f"C={[tuple(ci.shape) for ci in cs]}"
            )
        _initialize_reduction_outputs(self.chain, cs)

        base_problem = (mnk[0], mnk[1], mnk[2], cs[0].shape[0])
        # Block-scaled matmul: the first two aux are the scale factors SFA/SFB
        # (in the 128x4-blocked layout). They go right after a/b in the kernel
        # signature; remaining aux are epilogue-fusion tensors.
        sf_args: tuple = ()
        if self.block_scale:
            if len(aux) < 2:
                raise ValueError("block-scaled matmul call needs sfa, sfb after c: " "compiled(a, b, c, (M,N,K), sfa, sfb, *epilogue_aux)")
            sfa, sfb = aux[0], aux[1]
            aux = aux[2:]
            sfa = _maybe_wrap_layout(sfa.permute(1, 2, 0), _LEADING_DIM_AUX)
            sfb = _maybe_wrap_layout(sfb.permute(1, 2, 0), _LEADING_DIM_AUX)
            sf_args = (sfa, sfb)
        a = a.permute(1, 2, 0)
        b = b.permute(1, 2, 0)
        cs = [ci.permute(1, 2, 0) for ci in cs]
        side_output_strides = tuple(stride for spec, ci in zip(outputs_spec, cs) if spec.is_reduction or spec.is_quant_scale for stride in ci.stride())
        if not self.chain.has_mainloop_fusion and not self.chain.has_moe:
            problem_size = (
                *base_problem,
                *tuple(a.stride()),
                *tuple(b.stride()),
                *tuple(cs[0].stride()),
                *side_output_strides,
            )
        elif self.chain.has_mainloop_fusion:
            # Mainloop: pass the C strides so the epilogue / C TMA descriptor
            # honor arbitrary (padded) output row strides, exactly like matmul
            # (A/B still loaded packed). cs[0] is permuted to (M, N, L), so its
            # stride is (c_stride_m, c_stride_n, c_stride_l).
            problem_size = (*base_problem, *tuple(cs[0].stride()))
        elif side_output_strides:
            problem_size = (*base_problem, *side_output_strides)
        else:
            problem_size = base_problem
        a = _maybe_wrap_layout(a, _LEADING_DIM_A)
        b = _maybe_wrap_layout(b, _LEADING_DIM_B)
        cs = [
            _wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C) for spec, ci in zip(outputs_spec, cs)
        ]
        aux = tuple(_maybe_wrap_layout(t, _LEADING_DIM_AUX) for t in aux)
        # cute.compile fixes the kernel signature at JIT time with one
        # parameter per output, so the runtime call passes them flat:
        #   plain:       (a, b, c_terminal, c_tap_0, ..., mnk, *aux)
        #   block-scale: (a, b, sfa, sfb, c_terminal, ..., mnk, *aux)
        return self._launchable(a, b, *sf_args, *cs, problem_size, *aux)

    def _call_multi_gemm(self, gemm_pairs, c, mnk, *aux):
        """Multi-GEMM call: ``compiled([(A,B0),(A,B1),...], c, (M,N,K), *aux)``.

        Dedup the per-GEMM (a, b) pairs by tensor identity into the distinct
        A / B slots fixed at JIT time, verify the sharing pattern matches the
        compiled chain, then pass ``(a_0..a_na-1, b_0..b_nb-1, c, mnk4, *aux)``
        in the order the rendered kernel signature expects."""
        chain = self.chain
        if not isinstance(gemm_pairs, (list, tuple)) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in gemm_pairs):
            raise ValueError("multi-GEMM call expects a list of (a, b) tensor pairs as the " f"first argument; got {type(gemm_pairs).__name__}")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} GEMM(s); got " f"{len(gemm_pairs)} (a, b) pair(s)")
        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na
        b_slots: list = [None] * nb
        for (A_g, B_g), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, t, role in ((a_slots, ai, A_g, "A"), (b_slots, bi, B_g, "B")):
                if slots[idx] is None:
                    slots[idx] = t
                elif slots[idx].data_ptr() != t.data_ptr():
                    raise ValueError(
                        f"multi-GEMM operand sharing mismatch: distinct {role} slot "
                        f"{idx} was given two different tensors. The runtime sharing "
                        "pattern must match the graph the kernel was compiled from."
                    )
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("multi-GEMM: not every distinct A/B operand slot was filled")

        outputs_spec = chain.outputs
        cs = list(c) if isinstance(c, (list, tuple)) else [c]
        if len(cs) != len(outputs_spec):
            raise ValueError(
                f"this graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(cs)}. "
                "Pass a list of output tensors in slot order."
            )
        for spec, ci in zip(outputs_spec, cs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, mnk):
                raise ValueError(f"multi-GEMM output {spec.source!r} must have shape " f"{_expected_output_shape(spec, chain, mnk)}; " f"got {tuple(ci.shape)}")
        for role, slots in (("A", a_slots), ("B", b_slots)):
            for t in slots:
                if len(t.shape) != 3:
                    raise ValueError(f"multi-GEMM {role} operand must be rank-3; got {tuple(t.shape)}")
        _initialize_reduction_outputs(chain, cs)

        base_problem = (mnk[0], mnk[1], mnk[2], cs[0].shape[0])
        a_permuted = [t.permute(1, 2, 0) for t in a_slots]
        b_permuted = [t.permute(1, 2, 0) for t in b_slots]
        c_permuted = [ci.permute(1, 2, 0) for ci in cs]
        side_output_strides = tuple(stride for spec, ci in zip(outputs_spec, c_permuted) if spec.is_reduction or spec.is_quant_scale for stride in ci.stride())
        if not self.block_scale and not self.chain.has_mainloop_fusion and not self.chain.has_moe:
            problem_size = (
                *base_problem,
                *(x for t in a_permuted for x in t.stride()),
                *(x for t in b_permuted for x in t.stride()),
                *tuple(c_permuted[0].stride()),
                *side_output_strides,
            )
        else:
            problem_size = base_problem
        a_wrapped = [_maybe_wrap_layout(t, _LEADING_DIM_A) for t in a_permuted]
        b_wrapped = [_maybe_wrap_layout(t, _LEADING_DIM_B) for t in b_permuted]
        cs_wrapped = [
            _wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C)
            for spec, ci in zip(outputs_spec, c_permuted)
        ]
        aux = tuple(_maybe_wrap_layout(t, _LEADING_DIM_AUX) for t in aux)
        return self._launchable(*a_wrapped, *b_wrapped, *cs_wrapped, problem_size, *aux)

    def _call_block_scale_multi_gemm(self, gemm_pairs, c, mnk, *aux):
        """Block-scale multi-GEMM call:
        ``compiled([((A,SFA),(B0,SFB0)), ((A,SFA),(B1,SFB1))], c, (M,N,K), *epi_aux)``.

        Each operand is a (packed_data, scale_factor) pair; dedup by the
        packed-data tensor identity (the SF travels with its data → a shared
        dequant collapses to one distinct operand). Pass
        ``(a_0, sfa_0, .., b_0, sfb_0, .., c, mnk4, *epi_aux)`` in the order the
        rendered block-scale kernel signature expects."""
        chain = self.chain
        ok = (
            isinstance(gemm_pairs, (list, tuple))
            and gemm_pairs
            and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(o, (list, tuple)) and len(o) == 2 for o in p) for p in gemm_pairs)
        )
        if not ok:
            raise ValueError("block-scale multi-GEMM call expects a list of " "((a,sfa),(b,sfb)) pairs as the first argument")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} GEMM(s); got {len(gemm_pairs)} pair(s)")
        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na  # (packed_a, sfa)
        b_slots: list = [None] * nb
        for ((A_g, SFA_g), (B_g, SFB_g)), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, data, sf, role in ((a_slots, ai, A_g, SFA_g, "A"), (b_slots, bi, B_g, SFB_g, "B")):
                if slots[idx] is None:
                    slots[idx] = (data, sf)
                elif slots[idx][0].data_ptr() != data.data_ptr():
                    raise ValueError(f"block-scale multi-GEMM operand sharing mismatch: distinct " f"{role} slot {idx} got two different packed tensors.")
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("block-scale multi-GEMM: not every distinct operand slot was filled")

        # Outputs in chain.outputs slot order: slot 0 = terminal, slots 1.. =
        # taps (the no-epilogue case has one output per GEMM: GEMM 0 terminal,
        # GEMMs >0 taps). A fused epilogue has a single (terminal) output.
        outputs_spec = chain.outputs
        cs = list(c) if isinstance(c, (list, tuple)) else [c]
        if len(cs) != len(outputs_spec):
            raise ValueError(
                f"this graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(cs)}. "
                f"Pass a list of output tensors in slot order."
            )
        for spec, ci in zip(outputs_spec, cs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, mnk):
                raise ValueError(
                    f"block-scale multi-GEMM output {spec.source!r} must have " f"shape {_expected_output_shape(spec, chain, mnk)}; " f"got {tuple(ci.shape)}"
                )
        _initialize_reduction_outputs(chain, cs)
        base_problem = (mnk[0], mnk[1], mnk[2], cs[0].shape[0])
        # Grouped by kind: all A data, all B data, all SFA, all SFB (matches the
        # block-scale kernel signature; single-GEMM collapses to a, b, sfa, sfb).
        a_permuted = [d.permute(1, 2, 0) for d, _ in a_slots]
        b_permuted = [d.permute(1, 2, 0) for d, _ in b_slots]
        c_permuted = [ci.permute(1, 2, 0) for ci in cs]
        a_stride = tuple(a_permuted[0].stride())
        b_stride = tuple(b_permuted[0].stride())
        if any(tuple(t.stride()) != a_stride for t in a_permuted[1:]):
            raise ValueError("block-scale multi-GEMM requires all distinct A operands to share layout")
        if any(tuple(t.stride()) != b_stride for t in b_permuted[1:]):
            raise ValueError("block-scale multi-GEMM requires all distinct B operands to share layout")
        side_output_strides = tuple(stride for spec, ci in zip(outputs_spec, c_permuted) if spec.is_reduction or spec.is_quant_scale for stride in ci.stride())
        problem_size = (
            *base_problem,
            *a_stride,
            *b_stride,
            *tuple(c_permuted[0].stride()),
            *side_output_strides,
        )
        a_w = [_maybe_wrap_layout(t, _LEADING_DIM_A) for t in a_permuted]
        b_w = [_maybe_wrap_layout(t, _LEADING_DIM_B) for t in b_permuted]
        sfa_w = [_maybe_wrap_layout(s.permute(1, 2, 0), _LEADING_DIM_AUX) for _, s in a_slots]
        sfb_w = [_maybe_wrap_layout(s.permute(1, 2, 0), _LEADING_DIM_AUX) for _, s in b_slots]
        cs_w = [
            _wrap_raw_tensor(t) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(t, _LEADING_DIM_C)
            for spec, t in zip(outputs_spec, c_permuted)
        ]
        aux = tuple(_maybe_wrap_layout(t, _LEADING_DIM_AUX) for t in aux)
        return self._launchable(*a_w, *b_w, *sfa_w, *sfb_w, *cs_w, problem_size, *aux)


_DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "int8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp8_e8m0": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
}


def _mma_a_dtype(chain: FusionChain) -> str:
    """MMA-instruction A dtype = the graph-declared operand dtype (mainloop
    transforms are dtype-preserving; there is no implicit cast)."""
    return chain.matmul.a_dtype


def _mma_b_dtype(chain: FusionChain) -> str:
    """MMA-instruction B dtype = the graph-declared operand dtype."""
    return chain.matmul.b_dtype


def _check_supported(chain: FusionChain, config: TileConfig) -> None:
    """Reject a plain-matmul (input/acc dtype combo × active arch) the project
    can't run. Delegates to the unified MMA-type support table in
    ``kernel_registry`` (single source of truth — formerly the local
    ``_PIPELINE_DTYPE_ARCH`` table) and raises on rejection. ``config`` is taken
    for call-site symmetry but unused (mma-type support is config-independent)."""
    from .kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.MATMUL)
    if reason is not None:
        raise NotImplementedError(reason)


def _check_dtype_config_compat(chain: FusionChain, config: TileConfig, cta_group: int) -> None:
    """Reject (chain, config) pairs where the config's K_BYTES is not a
    multiple of the MMA dtype's element width. ``cta_group`` (the template's)
    sets the per-CTA SMEM N for the N-major-B swizzle-group check."""
    mma_dt = _mma_a_dtype(chain)
    elem_bytes = _DTYPE_BYTES.get(mma_dt)
    if elem_bytes is None:
        raise ValueError(f"unsupported MMA a_dtype {mma_dt!r}")
    if config.cta_tile_k_bytes % elem_bytes != 0:
        raise ValueError(
            f"TileConfig {config.name!r} has cta_tile_k_bytes="
            f"{config.cta_tile_k_bytes} which is not divisible by "
            f"elem_bytes={elem_bytes} for dtype {chain.matmul.a_dtype!r}."
        )
    mn_group_elems = config.cta_tile_k_bytes // elem_bytes
    if chain.matmul.a_major == "m":
        if config.cta_tile_m < mn_group_elems:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use M-major A for "
                f"dtype={chain.matmul.a_dtype!r}: cta_tile_m={config.cta_tile_m} "
                f"is smaller than the {mn_group_elems}-element swizzle group"
            )
        if config.cta_tile_m % mn_group_elems != 0:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use M-major A: " f"cta_tile_m={config.cta_tile_m} is not divisible by " f"swizzle group {mn_group_elems}"
            )
    if chain.matmul.b_major == "n":
        smem_n = config.cta_smem_tile_mnk(elem_bytes, cta_group)[1]
        if smem_n < mn_group_elems:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use N-major B for "
                f"dtype={chain.matmul.b_dtype!r}: per-CTA SMEM N={smem_n} "
                f"is smaller than the {mn_group_elems}-element swizzle group"
            )
        if smem_n % mn_group_elems != 0:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use N-major B: " f"per-CTA SMEM N={smem_n} is not divisible by " f"swizzle group {mn_group_elems}"
            )


def _check_input_alignment(chain: FusionChain) -> None:
    """The contiguous input dimension must be 16-byte aligned for TMA.
    Uses each operand's global-memory dtype."""
    M = chain.matmul.M
    N = chain.matmul.N
    K = chain.matmul.K
    a_elem_bytes = _DTYPE_BYTES[chain.matmul.a_dtype]
    b_elem_bytes = _DTYPE_BYTES[chain.matmul.b_dtype]
    a_major_extent = K if chain.matmul.a_major == "k" else M
    b_major_extent = K if chain.matmul.b_major == "k" else N
    bad: list[str] = []
    elem_bytes = a_elem_bytes
    if (a_major_extent * a_elem_bytes) % 16 != 0:
        bad.append(f"A {chain.matmul.a_major}-major extent {a_major_extent} * " f"{a_elem_bytes}B")
    if (b_major_extent * b_elem_bytes) % 16 != 0:
        bad.append(f"B {chain.matmul.b_major}-major extent {b_major_extent} * " f"{b_elem_bytes}B")
    _ = elem_bytes  # used below in error message
    if bad:
        raise ValueError(
            "input contiguous dimensions must be 16-byte aligned for TMA; "
            f"violations: {', '.join(bad)}. "
            f"Pad the contiguous dimension to a multiple of "
            f"{16 // a_elem_bytes} elements (A) / {16 // b_elem_bytes} elements (B)."
        )


# How many SMEM-D slots to double-buffer (lets the TMA store of one subtile
# overlap the sts of the next). 2 is the minimum useful value. More slots
# would help only if the TMA-store latency exceeds the sts cost of one
# subtile, which on B200 isn't generally the case.
_EPI_SMEM_STAGES = 2


def _smem_d_bytes(cfg, chain) -> int:
    """SMEM-D buffer size in bytes for the TMA-store epilogue.

    Holds `_EPI_SMEM_STAGES` slots of `cta_tile_m × epi_tile_mn[1]` elements
    (one subtile per slot). The 16-byte alignment pad mirrors the existing
    smem_a/b allocations.
    """
    elem_bytes = _DTYPE_BYTES[chain.output_dtype]
    return _EPI_SMEM_STAGES * cfg.cta_tile_m * cfg.epi_tile_mn[1] * elem_bytes + 16


def _use_tma_store_epi(chain, cfg, vec_bytes_epi: int, cta_group: int) -> bool:
    """Phase-1 gate for the TMA-store epilogue path.

    Requirements:
      - **single tensor output**: no aux tensors. Aux fusion ops do
        per-column / per-element loads with a vector width tied to the
        STG vsize, which doesn't line up with the full t2r_inst_repx
        vector that the TMA path stages into SMEM. Unary-only fusions
        (relu, gelu, ...) are allowed because they're shape-polymorphic.
      - **output row stride ≥ 16 bytes for N-major output**: PTX `cp.async.bulk.tensor`
        requires the SMEM source layout to be aligned with the TMA
        descriptor's swizzle (s64b for the typical 32-col BF16 subtile).
        With < 16-byte stride the descriptor cannot be constructed.
      - **cta_tile_m == 128**: Phase 1 only wires the 128-rows-per-CTA
        thread→row layout. cta_tile_m=64 (cluster-m=128 2×2 DP) uses a
        different epilogue layout (lane<16 active) and is left for later.
      - **out dtype ∈ {bf16, fp16}**: matches the s64b swizzle we hard-code
        for the 32-col subtile (32 × 2 bytes = 64 bytes/row).
      - **M-major (col-major) output, 16B-aligned M**: via 16x256b TMEM-load +
        stmatrix.trans + tma_store.
    """
    if chain.has_moe:
        # MoE scatters output rows by routed group (epilogue gates on
        # group_begin / row < group_end). The TMA-store path writes contiguous
        # tiles with no group offset, so MoE is STG-only — independent of the
        # jit path's _FORCE_STG_EPI.
        return False
    if chain.is_multi_gemm:
        # Multi-GEMM stages multiple accumulators into one fused output; the
        # TMA-store path has no multi-accumulator hook → STG only.
        return False
    if chain.aux_tensors:
        return False
    if chain.matmul.output_tap:
        # Multi-output: the matmul-tap STG path runs alongside the terminal
        # store in the per-vector inner loop. The TMA-store path stages full
        # t2r_inst_repx subtiles and doesn't have a hook for the tap.
        return False
    if chain.reductions:
        # Reduction taps use per-element atomic updates from the STG epilogue.
        return False
    if chain.block_quant is not None:
        # Quant scale is a per-vector side output from the STG epilogue.
        return False
    if chain.matmul.out_major == "n" and vec_bytes_epi < 16:
        return False
    if cfg.cta_tile_m != 128:
        return False
    if chain.output_dtype not in ("bf16", "fp16"):
        return False
    # The TMA descriptor uses a fixed 32-col subtile (epi_tile_mn[1]==32 in
    # every catalog entry). Under cta_group=2 each CTA holds half of B's N,
    # so its per-CTA col count is cta_tile_n//2 — needs ≥ 32 to make a full
    # subtile fit. cta_tile_n=32 with cta_group=2 (per-CTA n=16) would split
    # a 32-col tile across two CTAs, which the current TMA path doesn't
    # support; fall back to STG there.
    if cta_group == 2 and cfg.cta_tile_n < 64:
        return False
    # m major output which meets the alignment requirement.
    if chain.matmul.out_major == "m":
        m_align = 16 // _DTYPE_BYTES[chain.output_dtype]
        return chain.matmul.M % m_align == 0
    return True


def _compute_output_vec_bytes(chain: FusionChain) -> int:
    """Largest power-of-2 in {32, 16, 8, 4} that divides the C row stride in
    bytes. This becomes the epilogue store width: 32 → st.global.v8.b32,
    16 → v4.b32, 8 → v2.b32, 4 → b32. Every row of C must satisfy the PTX
    natural-alignment requirement for the chosen store width, and the row
    stride is what governs row-to-row alignment for all rows beyond 0.

    For m-major output, each row is strided by M, so vsize=1 → scalar-store
    fallback or stmatrix.trans + tma_store"""
    elem_bytes = _DTYPE_BYTES[chain.output_dtype]
    if chain.matmul.out_major == "m":
        return elem_bytes
    stride_bytes = chain.matmul.N * elem_bytes
    for vb in (32, 16, 8, 4):
        if stride_bytes % vb == 0:
            return vb
    raise ValueError(
        f"output row stride must be at least 4-byte aligned but got "
        f"N * elem_bytes = {chain.matmul.N} * {elem_bytes} = {stride_bytes} "
        f"bytes (dtype={chain.output_dtype!r}). PTX scalar store requires "
        f"4-byte natural alignment; sub-32-bit element stores are not "
        f"supported by this kernel."
    )


def _block_quant_cols_per_acc_stage(config: TileConfig, cta_group: int) -> int:
    cols_per_acc_stage = config.cta_tile_n
    if cta_group == 2 and config.cta_tile_m == 64:
        cols_per_acc_stage //= 2
    return cols_per_acc_stage


def _check_block_quant_supported(
    chain: FusionChain,
    vec_bytes_epi: int,
    config: TileConfig,
    cta_group: int,
) -> None:
    q = chain.block_quant
    if q is None:
        return
    if chain.has_mainloop_fusion:
        raise NotImplementedError("block_scale_quantize epilogue is not supported with mainloop fusion")
    if chain.matmul.out_major != "n":
        raise NotImplementedError("block_scale_quantize epilogue currently supports only N-major output")
    elem_bytes = _DTYPE_BYTES[chain.output_dtype]
    vsize = vec_bytes_epi // elem_bytes
    if q.block_size != vsize:
        raise NotImplementedError(
            "block_scale_quantize epilogue requires block_size to match the " f"STG vector width; got block_size={q.block_size}, vsize={vsize}"
        )
    cols_per_acc_stage = _block_quant_cols_per_acc_stage(config, cta_group)
    if cols_per_acc_stage < q.block_size or cols_per_acc_stage % q.block_size != 0:
        raise NotImplementedError(
            "block_scale_quantize epilogue requires each CTA epilogue drain to "
            "cover whole quantization blocks; got "
            f"cols_per_acc_stage={cols_per_acc_stage}, block_size={q.block_size}, "
            f"config={config.name}, cta_group={cta_group}"
        )
    if q.scale_reorder == "F8_128x4":
        expected_scale_dim = (
            chain.matmul.batch,
            ((chain.matmul.M + 127) // 128) * 128,
            (((chain.matmul.N // q.block_size) + 3) // 4) * 4,
        )
        if q.scale_dim != expected_scale_dim:
            raise NotImplementedError("F8_128x4 block_scale_quantize scale output currently requires " f"scale_dim={expected_scale_dim}; got {q.scale_dim}")


_FORCE_STG_EPI = False


def probe_supported(
    graph: cudnn.pygraph,
    config: TileConfig = DEFAULT_CONFIG,
    *,
    cta_group: int = 2,
    scheduler: str = "clc",
) -> None:
    """Cheap eligibility check — the gates :func:`jit_from_cudnn_graph` runs, but
    WITHOUT ``cute.compile``. Raises ``NotImplementedError`` / ``ValueError`` if
    the GEMM engine cannot run the graph. Used by the TBD engine probe to decide
    whether to list ``TBD_eng0`` in the plan list (see cudnn.TBD.heuristics).

    Block-scale / MoE have their per-side gates inside the specialized ``_jit_*``
    compile paths; analysis succeeding is treated as eligible for those and the
    full validation surfaces at compile time (when the engine is selected)."""
    chain, _binding = analyze_with_binding(graph)
    if chain.has_moe or chain.has_block_scale:
        return  # specialized paths validate at compile
    if chain.is_multi_gemm:
        from .kernel_registry import select_template

        tmpl = select_template(chain, config, cta_group, scheduler)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1ctamma CLC template this pass; got cta_group={cta_group}, "
                f"scheduler={scheduler!r} → {tmpl.file}."
            )
    _check_supported(chain, config)
    _check_dtype_config_compat(chain, config, cta_group)
    _check_input_alignment(chain)


def jit_from_cudnn_graph(
    graph: cudnn.pygraph,
    config: TileConfig = DEFAULT_CONFIG,
    *,
    cta_group: int = 2,
    scheduler: str = "clc",
    force_stg_epi: bool = False,
) -> CompiledFusedGemm:
    """End-to-end: cuDNN frontend graph -> rendered + cute-compiled GEMM kernel.

    Eagerly performs analyze → codegen → render → import → cute.compile.
    The returned :class:`CompiledFusedGemm` is directly callable.

    `graph` is a ``cudnn.pygraph`` built with the standard frontend API
    after ``import cudnn.TBD.gemm`` (the import installs the op-recording hook).
    `config` is a PURE-GEOMETRY tile shape from `tile_config.CATALOG`. The
    execution strategy is chosen here: ``cta_group`` ∈ {1, 2} (1-CTA vs 2-CTA
    MMA) and ``scheduler`` ∈ {"clc", "static"} pick the kernel template
    (mainloop fusion is auto-detected from the graph). Defaults match the
    previous DEFAULT_CONFIG (2-CTA MMA + CLC). Set ``force_stg_epi=True`` to
    skip the TMA-store path even when its gate would accept.
    """
    chain, binding = analyze_with_binding(graph)
    # MoE grouped block-scale matmul = both pattern-matches at once (dequant +
    # moe_grouped_matmul). Must be checked BEFORE the single-feature gates.
    if chain.has_moe and chain.has_block_scale:
        return _jit_moe_block_scale(chain, config, cta_group, scheduler, binding=binding)
    # Block-scale matmul is gated independently (its own per-side case table);
    # route to it before the plain-matmul gate.
    if chain.has_block_scale:
        return _jit_block_scale(chain, config, cta_group, scheduler, binding=binding)
    # MoE grouped matmul: own graph type / template (grouped persistent scheduler
    # + per-group A TMA descriptor replacement). Gated via the MOE mma table.
    if chain.has_moe:
        return _jit_moe(chain, config, cta_group, scheduler, binding=binding)
    # Multi-GEMM is only implemented in the 1ctamma CLC template this pass.
    # select_template deliberately skips capability gates (single-point
    # probing), so reject an unsupported strategy here with a clear message
    # rather than fault deep in cute on a missing vec_f32_<g> binding.
    if chain.is_multi_gemm:
        from .kernel_registry import select_template

        tmpl = select_template(chain, config, cta_group, scheduler)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1ctamma CLC template this pass; got cta_group={cta_group}, "
                f"scheduler={scheduler!r} → {tmpl.file}. Use cta_group=1, scheduler='clc'."
            )
    # Plain-matmul (pipeline × input/acc dtype combo × active arch) gate.
    _check_supported(chain, config)
    _check_dtype_config_compat(chain, config, cta_group)
    _check_input_alignment(chain)
    # Side effect: this also raises if output alignment < 4 bytes (the floor
    # supported by PTX scalar st.b32). Called eagerly so callers see the
    # rejection at JIT time, before _render_template re-computes it.
    _compute_output_vec_bytes(chain)
    global _FORCE_STG_EPI
    prev_force = _FORCE_STG_EPI
    _FORCE_STG_EPI = force_stg_epi
    try:
        vec_bytes_epi = _compute_output_vec_bytes(chain)
        _check_block_quant_supported(chain, vec_bytes_epi, config, cta_group)
        snippets = generate(
            chain,
            vec_bytes_epi=vec_bytes_epi,
            output_elem_bytes=_DTYPE_BYTES[chain.output_dtype],
        )
        src = _render_template(chain, snippets, config, cta_group, scheduler)
    finally:
        _FORCE_STG_EPI = prev_force
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return CompiledFusedGemm(
        chain=chain,
        config=config,
        aux_names=[aux.name for aux in chain.aux_tensors],
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),  # one-shot cute.compile (lru_cached inside mod)
        binding=binding,
    )


# --- sm100_block_scale_matmul: arch/dtype support -------------------------
# The supported per-side cases (data/SF dtype, block size, reorder, dequant
# compute/out) now live in the unified `kernel_registry.MMA_TYPE_SUPPORT` table
# — the single source of truth, merged with the plain-matmul dtype table. This
# gate delegates to it.


def _check_block_scale_supported(chain: FusionChain) -> None:
    """Reject any block-scale matmul the `sm100_block_scale_matmul` template
    can't run — a FULL per-side match (data dtype, SF dtype, block size, reorder
    layout, dequant compute/out) + active arch, judged by
    `kernel_registry.MMA_TYPE_SUPPORT`. Raises on rejection."""
    from .kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.BLOCK_SCALE_MATMUL)
    if reason is not None:
        raise NotImplementedError(reason)


def _resolve_moe_variant_pack(compiled, variant_pack: dict):
    """Resolve a MoE variant-pack dict into the buffers the positional MoE call
    needs, inferring (S, N, K) from buffer shapes. Returns
    ``(a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, (S, N, K))``."""
    b = compiled.binding
    if b is None:
        raise NotImplementedError("variant-pack call is not yet wired up for this graph type")
    resolved = resolve_variant_pack(variant_pack, b)

    def pull(t, role):
        if t is None or id(t) not in resolved:
            raise KeyError(f"variant pack is missing a buffer for {role}")
        return resolved[id(t)]

    a_bufs = [pull(t, "token") for t in b.a_operands]
    b_bufs = [pull(t, "weight") for t in b.b_operands]
    out_bufs = [pull(t, "output") for t in b.outputs]
    aux_bufs = [pull(t, "aux") for t in b.aux]
    fto = pull(b.first_token_offset, "first_token_offset")
    sfa = [pull(t, "SFA") for t in b.sfa_operands]
    sfb = [pull(t, "SFB") for t in b.sfb_operands]
    k_factor = 2 if compiled.chain.matmul.a_dtype == "fp4_e2m1" else 1
    S = a_bufs[0].shape[1]
    K = a_bufs[0].shape[2] * k_factor
    N = out_bufs[0].shape[2]
    return a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, (S, N, K)


@dataclass
class CompiledMoeGemm:
    """A compiled MoE grouped matmul forward pass, directly callable.

    Runtime call::

        compiled = jit_from_cudnn_graph(g)           # g built with moe_grouped_matmul
        compiled(token, weight, first_token_offset, output, (S, N, K, E))

    Layouts (rank-3, matching the cuDNN frontend / project conventions):
      * ``token``  — (1, S, K) row-major  (A; single token plane).
      * ``weight`` — (E, N, K) row-major  (B; per-expert, same N-K convention as
        the plain-matmul B operand). The cuDNN ``[E, H, N]`` column-major-in-H×N
        weight is bit-identical to this ``(E, N, K)`` row-major layout in memory.
      * ``first_token_offset`` — (E,) int32: group g spans token rows
        ``[first_token_offset[g], first_token_offset[g+1])`` (last group → S).
      * ``output`` — (1, S, N) row-major.

    The per-CTA A-descriptor workspace is allocated and owned here (the user API
    stays clean); it is reused across calls and grown if the grid changes."""

    chain: FusionChain
    config: TileConfig
    generated_path: Path
    _launchable: Callable
    _grid_ctas: int = 0
    _workspace: object = None  # torch.Tensor (lazy), int64, 128B-aligned
    aux_names: list = field(default_factory=list)
    binding: "GemmBinding | None" = None  # role -> cuDNN tensor (variant-pack call)

    def _make_workspace(self, n_slots, device):
        """Lazily allocate the per-CTA A-descriptor GMEM workspace (16 int64 per
        slot, 128-byte aligned). ``n_slots`` = grid_ctas * num_a_operands."""
        import torch

        if self._workspace is None:
            self._workspace = torch.empty(n_slots * 16, dtype=torch.int64, device=device)
            if self._workspace.data_ptr() % 128 != 0:
                raise RuntimeError("A TMA descriptor workspace must be 128-byte aligned; got " f"0x{self._workspace.data_ptr():x}")
        return self._workspace

    def __call__(self, variant_pack):
        # Variant-pack dict {cuDNN tensor | uid | name: buffer}; (S, N, K) is
        # inferred from buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        return self._call_variant_pack(variant_pack)

    def _launch_single(self, token, weight, first_token_offset, output, snke):
        import torch

        if len(snke) < 3:
            raise ValueError("MoE call needs problem_size (S, N, K[, ...]); " f"got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])
        outputs_spec = self.chain.outputs
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(outputs_spec):
            raise ValueError(
                f"this MoE graph has {len(outputs_spec)} output(s) " f"({[o.source for o in outputs_spec]}); got {len(outputs)}. " "Pass outputs in slot order."
            )
        for name, t, rank in (("token", token, 3), ("weight", weight, 3), ("first_token_offset", first_token_offset, 1)):
            if len(t.shape) != rank:
                raise ValueError(f"MoE {name} must be rank-{rank}; got shape {tuple(t.shape)}")
        for spec, t in zip(outputs_spec, outputs):
            if len(t.shape) != 3 or tuple(t.shape) != _expected_output_shape(spec, self.chain, (S, N, K)):
                raise ValueError(
                    f"MoE output {spec.source!r} must have shape " f"{_expected_output_shape(spec, self.chain, (S, N, K))}; " f"got {tuple(t.shape)}"
                )
        _initialize_reduction_outputs(self.chain, outputs)
        # num_experts = weight's batch (E); num_groups = first_token_offset length
        # (BxE — may exceed E; group g uses expert g % E). Derived from the
        # runtime tensors, so the call is robust to BxE > E.
        num_experts = int(weight.shape[0])
        num_groups = int(first_token_offset.shape[0])
        # Permute to the kernel's (S,K,1)/(N,K,E)/(S,N,1) layout.
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        if token.stride(-1) != 1 or weight.stride(-1) != 1 or outputs[0].stride(-1) != 1:
            raise ValueError(
                "MoE non-packed tensors require contiguous innermost dimensions: "
                f"got token stride {tuple(token.stride())}, "
                f"weight stride {tuple(weight.stride())}, "
                f"output stride {tuple(outputs[0].stride())}"
            )
        side_output_strides = tuple(stride for spec, ci in zip(outputs_spec, c_perms) if spec.is_reduction or spec.is_quant_scale for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *tuple(a_perm.stride()),
            *tuple(b_perm.stride()),
            *tuple(c_perms[0].stride()),
            *side_output_strides,
        )
        a = _maybe_wrap_layout(a_perm, _LEADING_DIM_A)
        b = _maybe_wrap_layout(b_perm, _LEADING_DIM_B)
        cs = [
            _wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C)
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        # A-descriptor workspace: grid_num_clusters * cluster_m * cluster_n slots
        # of 16 int64 (128-byte tensormap), 128-byte aligned.
        workspace = self._make_workspace(self._grid_ctas, token.device)
        return self._launchable(
            a,
            b,
            cs[0],
            first_token_offset,
            workspace,
            *cs[1:],
            problem_size,
        )

    def _call_variant_pack(self, variant_pack: dict):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, _sfa, _sfb, snk = _resolve_moe_variant_pack(self, variant_pack)
        out = out_bufs if len(out_bufs) > 1 else out_bufs[0]
        if self.chain.is_multi_gemm:
            pairs = [(a_bufs[ai], b_bufs[bi]) for ai, bi in self.chain.gemm_operands]
            return self._call_multi_gemm(pairs, fto, out, snk, *aux_bufs)
        return self._launch_single(a_bufs[0], b_bufs[0], fto, out, snk)

    def _call_multi_gemm(self, gemm_pairs, first_token_offset, output, snke, *aux):
        """Multi-GEMM MoE call:
        ``compiled([(tok, w0), (tok, w1), ...], fto, out, (S, N, K[, ...]), *aux)``.

        Each pair is a (token, weight) tuple; tokens/weights are deduped by
        tensor identity into the distinct A/B slots fixed at JIT time (a shared
        token → one distinct A operand). All grouped matmuls share ``fto``.
        ``out`` is the single fused (terminal) output."""
        import torch

        chain = self.chain
        if not isinstance(gemm_pairs, (list, tuple)) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in gemm_pairs):
            raise ValueError("multi-GEMM MoE call expects a list of (token, weight) pairs as " f"the first argument; got {type(gemm_pairs).__name__}")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} grouped matmul(s); got " f"{len(gemm_pairs)} (token, weight) pair(s)")
        if len(snke) < 3:
            raise ValueError(f"MoE call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])

        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na
        b_slots: list = [None] * nb
        for (tok, w), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, t, role in ((a_slots, ai, tok, "token"), (b_slots, bi, w, "weight")):
                if slots[idx] is None:
                    slots[idx] = t
                elif slots[idx].data_ptr() != t.data_ptr():
                    raise ValueError(
                        f"multi-GEMM MoE operand sharing mismatch: distinct {role} "
                        f"slot {idx} was given two different tensors. The runtime "
                        "sharing pattern must match the compiled graph."
                    )
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("multi-GEMM MoE: not every distinct token/weight slot was filled")

        outputs_spec = chain.outputs
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outs) != len(outputs_spec):
            raise ValueError(f"fused multi-GEMM MoE has {len(outputs_spec)} output(s); got {len(outs)}")
        out = outs[0]
        for name, t, rank in (("output", out, 3), ("first_token_offset", first_token_offset, 1)):
            if len(t.shape) != rank:
                raise ValueError(f"MoE {name} must be rank-{rank}; got shape {tuple(t.shape)}")
        for role, slots in (("token", a_slots), ("weight", b_slots)):
            for t in slots:
                if len(t.shape) != 3 or t.stride(-1) != 1:
                    raise ValueError(
                        f"multi-GEMM MoE {role} must be rank-3 with contiguous " f"innermost dim; got shape {tuple(t.shape)} stride {tuple(t.stride())}"
                    )
        if out.stride(-1) != 1:
            raise ValueError("multi-GEMM MoE output requires contiguous innermost dim")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, (S, N, K)):
                raise ValueError(
                    f"multi-GEMM MoE output {spec.source!r} must have shape " f"{_expected_output_shape(spec, chain, (S, N, K))}; got {tuple(ci.shape)}"
                )
        _initialize_reduction_outputs(chain, outs)

        num_experts = int(b_slots[0].shape[0])
        num_groups = int(first_token_offset.shape[0])
        # All A (resp. B) operands share the same layout → use slot 0's strides.
        a_perm0 = a_slots[0].permute(1, 2, 0)
        b_perm0 = b_slots[0].permute(1, 2, 0)
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        side_output_strides = tuple(stride for spec, ci in zip(outputs_spec, c_perms) if spec.is_reduction or spec.is_quant_scale for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *tuple(a_perm0.stride()),
            *tuple(b_perm0.stride()),
            *tuple(c_perms[0].stride()),
            *side_output_strides,
        )
        a_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_A) for t in a_slots]
        b_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_B) for t in b_slots]
        cs = [
            _wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C)
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        aux = tuple(_maybe_wrap_layout(t, _LEADING_DIM_AUX) for t in aux)
        # Workspace: one 128-B A descriptor per distinct A operand per CTA.
        workspace = self._make_workspace(self._grid_ctas * na, out.device)
        return self._launchable(
            *a_wrapped,
            *b_wrapped,
            cs[0],
            first_token_offset,
            workspace,
            *cs[1:],
            problem_size,
            *aux,
        )


def _jit_moe(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
    scheduler: str = "clc",
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeGemm:
    """JIT path for a MoE grouped matmul forward pass (mode=NONE)."""
    from .kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.MOE)
    if reason is not None:
        raise NotImplementedError(reason)
    if chain.matmul.a_major != "k" or chain.matmul.b_major != "k":
        raise NotImplementedError(
            "MoE grouped matmul supports only K-major token / weight in the POC "
            f"(got token {chain.matmul.a_major}-major, weight "
            f"{chain.matmul.b_major}-major)"
        )
    _check_dtype_config_compat(chain, config, cta_group)
    _check_input_alignment(chain)
    _compute_output_vec_bytes(chain)
    global _FORCE_STG_EPI
    prev_force = _FORCE_STG_EPI
    _FORCE_STG_EPI = True  # MoE epilogue is STG-only
    try:
        vec_bytes_epi = _compute_output_vec_bytes(chain)
        _check_block_quant_supported(chain, vec_bytes_epi, config, cta_group)
        snippets = generate(
            chain,
            vec_bytes_epi=vec_bytes_epi,
            output_elem_bytes=_DTYPE_BYTES[chain.output_dtype],
        )
        src = _render_template(chain, snippets, config, cta_group, scheduler)
    finally:
        _FORCE_STG_EPI = prev_force
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cgrp_size_m, config.cgrp_size_n
    grid_ctas = _grid_num_clusters(config) * cluster_m * cluster_n
    return CompiledMoeGemm(
        chain=chain,
        config=config,
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _grid_ctas=grid_ctas,
        aux_names=[aux.name for aux in chain.aux_tensors],
        binding=binding,
    )


def _jit_block_scale(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
    scheduler: str = "clc",
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledFusedGemm:
    """JIT path for block-scaled (FP4 / FP8 + per-block SF) matmul.

    Bypasses the generic dtype-byte alignment checks (FP4 is 0.5 B/elem) and
    routes to the block-scale template via :func:`_render_block_scale_template`.
    Validation of the (config, block_size) pair happens inside the tile-constant
    renderer (``validate_block_scale_config``)."""
    # Reject anything the block-scale template can't run — exact per-side match
    # against the supported cases (+ family arch). This subsumes the both-sided
    # requirement (single-sided matches no case).
    _check_block_scale_supported(chain)
    if chain.reductions:
        if config.arch != "sm100":
            raise NotImplementedError("block-scale reduction is supported only on sm100 templates")
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("block-scale reduction supports only fp32 compute/output")
    # Per-template active-GPU SM gate: reject early on a GPU outside the
    # template's range rather than fault at launch. No-op when no GPU is visible.
    from .kernel_registry import select_template

    _tmpl = select_template(chain, config, cta_group, scheduler)
    if chain.is_multi_gemm and not _tmpl.supports_multi_gemm:
        raise NotImplementedError(
            f"block-scale multi-GEMM ({chain.num_gemms} GEMMs) is not supported by " f"{_tmpl.file} (cta_group={cta_group}, scheduler={scheduler!r})."
        )
    _arch_reason = _tmpl.arch_active_reject()
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)  # eager: rejects bad output alignment
    vec_bytes_epi = _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, vec_bytes_epi, config, cta_group)
    snippets = generate(
        chain,
        vec_bytes_epi=vec_bytes_epi,
        output_elem_bytes=_DTYPE_BYTES[chain.output_dtype],
    )
    src = _render_block_scale_template(chain, snippets, config, cta_group, scheduler)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return CompiledFusedGemm(
        chain=chain,
        config=config,
        aux_names=[aux.name for aux in chain.aux_tensors],
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        block_scale=True,
        binding=binding,
    )


@dataclass
class CompiledMoeBlockScaleGemm:
    """A compiled MoE grouped *block-scale* (FP4/FP8 + per-block SF) matmul fwd.

    Runtime call::

        compiled = jit_from_cudnn_graph(g)   # g built with block_scale_dequantize
                                             #   on token+weight → moe_grouped_matmul
        compiled(token, weight, sfa, sfb, first_token_offset, output, (S, N, K, E))

    Layouts (rank-3, matching the cuDNN frontend / project conventions):
      * ``token``  — (1, S, Kp) packed FP4/FP8  (A; single token plane).
      * ``weight`` — (E, N, Kp) packed FP4/FP8  (B; per-expert).
      * ``sfa``    — token scale factors, F8_128x4-reordered + padded to 128 rows
        PER GROUP, then concatenated (Σ ceil(group_m/128) blocks).
      * ``sfb``    — weight scale factors, F8_128x4-reordered, per-expert.
      * ``first_token_offset`` — (num_groups,) int32/int64; group g spans token
        rows ``[fto[g], fto[g+1])`` (last → S). Group sizes are arbitrary (need
        NOT be 128-aligned); the scheduler tracks each group's start SF-block.
      * ``output`` — (1, S, N) row-major.

    The per-CTA A-descriptor workspace is allocated/owned here (clean user API)."""

    chain: FusionChain
    config: TileConfig
    generated_path: Path
    _launchable: Callable
    _grid_ctas: int = 0
    _workspace: object = None  # torch.Tensor (lazy), int64, 128B-aligned
    aux_names: list = field(default_factory=list)
    binding: "GemmBinding | None" = None  # role -> cuDNN tensor (variant-pack call)

    def _make_workspace(self, n_slots, device):
        import torch

        if self._workspace is None:
            self._workspace = torch.empty(n_slots * 16, dtype=torch.int64, device=device)
            if self._workspace.data_ptr() % 128 != 0:
                raise RuntimeError("A TMA descriptor workspace must be 128-byte aligned; got " f"0x{self._workspace.data_ptr():x}")
        return self._workspace

    def __call__(self, variant_pack):
        # Variant-pack dict {cuDNN tensor | uid | name: buffer}; (S, N, K) is
        # inferred from buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        return self._call_variant_pack(variant_pack)

    def _launch_single(self, token, weight, sfa, sfb, first_token_offset, output, snke):
        import torch

        if len(snke) < 3:
            raise ValueError("MoE block-scale call needs problem_size (S, N, K[, ...]); " f"got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])
        outputs_spec = self.chain.outputs
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(outputs_spec):
            raise ValueError(
                f"this MoE block-scale graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(outputs)}. "
                "Pass outputs in slot order."
            )
        for name, t, rank in (("token", token, 3), ("weight", weight, 3), ("sfa", sfa, 3), ("sfb", sfb, 3), ("first_token_offset", first_token_offset, 1)):
            if len(t.shape) != rank:
                raise ValueError(f"MoE block-scale {name} must be rank-{rank}; " f"got shape {tuple(t.shape)}")
        for spec, t in zip(outputs_spec, outputs):
            if len(t.shape) != 3 or tuple(t.shape) != _expected_output_shape(spec, self.chain, (S, N, K)):
                raise ValueError(
                    f"MoE block-scale output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, self.chain, (S, N, K))}; "
                    f"got {tuple(t.shape)}"
                )
        _initialize_reduction_outputs(self.chain, outputs)
        # num_experts = weight's batch (E); num_groups = first_token_offset length
        # (BxE — may exceed E; group g uses expert g % E). Derived from runtime
        # tensors, so the call is robust to BxE > E.
        num_experts = int(weight.shape[0])
        num_groups = int(first_token_offset.shape[0])
        # Permute to the kernel's inner-plane layouts (batch axis last). The host
        # rebuilds the SF descriptors from .iterator only, so the SF permute just
        # needs to preserve the base pointer.
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        if token.stride(-1) != 1 or weight.stride(-1) != 1 or outputs[0].stride(-1) != 1:
            raise ValueError(
                "MoE block-scale non-packed tensors require contiguous innermost "
                f"dimensions: got token stride {tuple(token.stride())}, "
                f"weight stride {tuple(weight.stride())}, "
                f"output stride {tuple(outputs[0].stride())}"
            )
        side_output_strides = tuple(stride for spec, ci in zip(outputs_spec, c_perms) if spec.is_reduction or spec.is_quant_scale for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *tuple(a_perm.stride()),
            *tuple(b_perm.stride()),
            *tuple(c_perms[0].stride()),
            *side_output_strides,
        )
        a = _maybe_wrap_layout(a_perm, _LEADING_DIM_A)
        b = _maybe_wrap_layout(b_perm, _LEADING_DIM_B)
        cs = [
            _wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C)
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        msfa = _maybe_wrap_layout(sfa.permute(1, 2, 0), _LEADING_DIM_AUX)
        msfb = _maybe_wrap_layout(sfb.permute(1, 2, 0), _LEADING_DIM_AUX)
        workspace = self._make_workspace(self._grid_ctas, token.device)
        return self._launchable(
            a,
            b,
            msfa,
            msfb,
            cs[0],
            first_token_offset,
            workspace,
            *cs[1:],
            problem_size,
        )

    def _call_variant_pack(self, variant_pack: dict):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, snk = _resolve_moe_variant_pack(self, variant_pack)
        out = out_bufs if len(out_bufs) > 1 else out_bufs[0]
        if self.chain.is_multi_gemm:
            pairs = [((a_bufs[ai], sfa[ai]), (b_bufs[bi], sfb[bi])) for ai, bi in self.chain.gemm_operands]
            return self._call_multi_gemm(pairs, fto, out, snk, *aux_bufs)
        return self._launch_single(a_bufs[0], b_bufs[0], sfa[0], sfb[0], fto, out, snk)

    def _call_multi_gemm(self, gemm_pairs, first_token_offset, output, snke, *aux):
        """Multi-GEMM MoE block-scale call:
        ``compiled([((tok,sfa),(w0,sfb0)), ((tok,sfa),(w1,sfb1)), ...], fto,
        out, (S, N, K[, ...]), *aux)``.

        Each GEMM is a ((token, sfa), (weight, sfb)) pair; dedup by the PACKED
        data tensor identity (SF travels with its data → a shared token+sfa
        collapses to one distinct A operand). All grouped matmuls share ``fto``;
        ``out`` is the single fused (terminal) output."""
        import torch

        chain = self.chain
        ok = (
            isinstance(gemm_pairs, (list, tuple))
            and gemm_pairs
            and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(o, (list, tuple)) and len(o) == 2 for o in p) for p in gemm_pairs)
        )
        if not ok:
            raise ValueError("multi-GEMM MoE block-scale call expects a list of " "((token,sfa),(weight,sfb)) pairs as the first argument")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} grouped matmul(s); got " f"{len(gemm_pairs)} pair(s)")
        if len(snke) < 3:
            raise ValueError(f"MoE block-scale call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])

        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na  # (packed token, sfa)
        b_slots: list = [None] * nb  # (packed weight, sfb)
        for ((tok, sfa), (w, sfb)), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, data, sf, role in ((a_slots, ai, tok, sfa, "token"), (b_slots, bi, w, sfb, "weight")):
                if slots[idx] is None:
                    slots[idx] = (data, sf)
                elif slots[idx][0].data_ptr() != data.data_ptr():
                    raise ValueError(f"multi-GEMM MoE block-scale {role} sharing mismatch: distinct " f"slot {idx} was given two different packed tensors.")
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("multi-GEMM MoE block-scale: not every distinct operand slot was filled")

        outputs_spec = chain.outputs
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outs) != len(outputs_spec):
            raise ValueError(f"fused multi-GEMM MoE block-scale has {len(outputs_spec)} output(s); got {len(outs)}")
        out = outs[0]
        for name, t, rank in (("output", out, 3), ("first_token_offset", first_token_offset, 1)):
            if len(t.shape) != rank:
                raise ValueError(f"MoE block-scale {name} must be rank-{rank}; got {tuple(t.shape)}")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, (S, N, K)):
                raise ValueError(
                    f"multi-GEMM MoE block-scale output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, chain, (S, N, K))}; got {tuple(ci.shape)}"
                )
        _initialize_reduction_outputs(chain, outs)
        num_experts = int(b_slots[0][0].shape[0])
        num_groups = int(first_token_offset.shape[0])
        a0, b0 = a_slots[0][0], b_slots[0][0]
        a_perm0 = a0.permute(1, 2, 0)
        b_perm0 = b0.permute(1, 2, 0)
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        if a0.stride(-1) != 1 or b0.stride(-1) != 1 or out.stride(-1) != 1:
            raise ValueError("multi-GEMM MoE block-scale tensors require contiguous innermost dim")
        side_output_strides = tuple(stride for spec, ci in zip(outputs_spec, c_perms) if spec.is_reduction or spec.is_quant_scale for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *tuple(a_perm0.stride()),
            *tuple(b_perm0.stride()),
            *tuple(c_perms[0].stride()),
            *side_output_strides,
        )
        # Grouped by kind — all A data, all B data, all SFA, all SFB — matching
        # the rendered _host signature (single-GEMM → a,b,sfa,sfb).
        a_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_A) for (t, _sf) in a_slots]
        b_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_B) for (t, _sf) in b_slots]
        sfa_wrapped = [_maybe_wrap_layout(sf.permute(1, 2, 0), _LEADING_DIM_AUX) for (_t, sf) in a_slots]
        sfb_wrapped = [_maybe_wrap_layout(sf.permute(1, 2, 0), _LEADING_DIM_AUX) for (_t, sf) in b_slots]
        cs = [
            _wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C)
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        aux = tuple(_maybe_wrap_layout(t, _LEADING_DIM_AUX) for t in aux)
        workspace = self._make_workspace(self._grid_ctas * na, out.device)
        return self._launchable(
            *a_wrapped,
            *b_wrapped,
            *sfa_wrapped,
            *sfb_wrapped,
            cs[0],
            first_token_offset,
            workspace,
            *cs[1:],
            problem_size,
            *aux,
        )


def _jit_moe_block_scale(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
    scheduler: str = "clc",
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeBlockScaleGemm:
    """JIT path for a MoE grouped block-scale matmul (dequant + moe_grouped).

    Combines the block-scale SF machinery (own per-side case table) with the MoE
    grouped persistent scheduler + per-group A TMA descriptor replacement. STG
    epilogue only (forced inside :func:`_render_block_scale_template`)."""
    from .kernel_registry import (
        GraphType,
        mma_arch_reject,
        select_template,
    )

    reason = mma_arch_reject(chain, GraphType.MOE_BLOCK_SCALE)
    if reason is not None:
        raise NotImplementedError(reason)
    if chain.matmul.a_major != "k" or chain.matmul.b_major != "k":
        raise NotImplementedError(
            "MoE block-scale matmul supports only K-major token / weight in the "
            f"POC (got token {chain.matmul.a_major}-major, weight "
            f"{chain.matmul.b_major}-major)"
        )
    if chain.reductions:
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("MoE block-scale reduction supports only fp32 compute/output")
    # Per-template active-GPU SM gate.
    _tmpl = select_template(chain, config, cta_group, scheduler)
    _arch_reason = _tmpl.arch_active_reject()
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)
    vec_bytes_epi = _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, vec_bytes_epi, config, cta_group)
    snippets = generate(
        chain,
        vec_bytes_epi=vec_bytes_epi,
        output_elem_bytes=_DTYPE_BYTES[chain.output_dtype],
    )
    src = _render_block_scale_template(chain, snippets, config, cta_group, scheduler)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cgrp_size_m, config.cgrp_size_n
    grid_ctas = _grid_num_clusters(config) * cluster_m * cluster_n
    return CompiledMoeBlockScaleGemm(
        chain=chain,
        config=config,
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _grid_ctas=grid_ctas,
        binding=binding,
    )
