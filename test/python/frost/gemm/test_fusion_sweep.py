"""Correctness sweep over fused epilogue chains (op × broadcast_mode × config × dtype).

Companion to test_matmul_sweep (pure matmul); this exercises epilogue codegen +
template integration. Small-integer inputs keep the FP32 reduction exact, so
atol=1e-1/rtol=1e-2 only has to absorb dtype downcast + transcendental ULPs.
Runs as a script too (forwards argv to pytest on this file).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

# Module-wide GPU gate — every test here is end-to-end and needs a B200.
pytestmark = [pytest.mark.L0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")]


import cudnn
import cudnn.frost.gemm  # noqa: F401  — installs the cudnn.pygraph recorder hook
import re

from cudnn.frost.gemm.compiler import jit_from_cudnn_graph
from cudnn.frost.gemm.tile_config import CATALOG


class _Plan:
    """JIT-compiles a recorded graph with a forced tile config (sweeps pin a
    config directly). Exposes chain/binding/block_scale/aux_names; callable
    with a variant pack."""

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
    """Variant-pack dict from the compiled binding (A/B + outputs + aux)."""
    bd = compiled.binding
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {bd.a_operands[0]: a, bd.b_operands[0]: b}
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


# Dtype tables (dup of test_matmul_sweep to keep the files independent)

_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e8m0": torch.float8_e8m0fnu,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
}
_CUDNN_DTYPE = {
    "bf16": cudnn.data_type.BFLOAT16,
    "fp16": cudnn.data_type.HALF,
    "fp32": cudnn.data_type.FLOAT,
    "fp8_e4m3": cudnn.data_type.FP8_E4M3,
    "fp8_e5m2": cudnn.data_type.FP8_E5M2,
    "fp8_e8m0": cudnn.data_type.FP8_E8M0,
    "int8": cudnn.data_type.INT8,
    "uint8": cudnn.data_type.UINT8,
    "int32": cudnn.data_type.INT32,
}


# Chain spec — list of (op_name, broadcast_mode | None) tuples
Op = str  # cuDNN frontend method name, e.g. "relu", "add", "bias"
Bcast = str  # "scalar" | "per_row" | "per_col" | "per_elem"
Step = tuple[Op, Bcast | None]  # (op, bcast); bcast=None means unary
Chain = tuple[Step, ...]


def _chain_id(chain: Chain) -> str:
    parts = []
    for op, bcast in chain:
        parts.append(op if bcast is None else f"{op}_{bcast}")
    return "__".join(parts)


def _layout_id(p: tuple[str, str]) -> str:
    return f"A{p[0]}_B{p[1]}"


# Default sweep axes — one config, one dtype, one shape per fusion-API case.
_DEFAULT_CONFIG = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
_DEFAULT_IN_DT = "bf16"
_DEFAULT_OUT_DT = "bf16"
_DEFAULT_SHAPE: tuple[int, int, int] = (256, 256, 128)
_DEFAULT_BATCHED_SHAPE: tuple[int, int, int, int] = (2, 256, 256, 128)
_INPUT_LAYOUTS: tuple[tuple[str, str], ...] = (
    ("k", "k"),
    ("m", "k"),
    ("k", "n"),
    ("m", "n"),
)
_NONPACKED_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
)
_NONPACKED_AUX_BCAST_MODES: tuple[Bcast, ...] = ("per_col", "per_elem")


# Group A: every unary op (skip `exp` — magnitude overflow with small-int inputs)
_UNARY_CHAINS: tuple[Chain, ...] = tuple(
    ((op, None),)
    for op in (
        "identity",
        "relu",
        "gelu",
        "gelu_approx_tanh",
        "swish",
        "sigmoid",
        "tanh",
        "abs",
        "neg",
    )
)

# Group B: every binary op × every broadcast mode (skip `div` — div-by-zero)
_BCAST_MODES: tuple[Bcast, ...] = ("scalar", "per_row", "per_col", "per_elem")
_BINARY_CHAINS: tuple[Chain, ...] = tuple(((op, bcast),) for op in ("add", "mul", "sub") for bcast in _BCAST_MODES)

# `bias` uses a different cuDNN API method than `add` (separate recorder patch)
_BIAS_CHAINS: tuple[Chain, ...] = ((("bias", "per_col"),),)

# Group C: canonical multi-op chains
_MULTI_CHAINS: tuple[Chain, ...] = (
    (("bias", "per_col"), ("gelu_approx_tanh", None)),  # FFN canonical
    (("bias", "per_row"), ("swish", None)),  # per-row bias
    (("bias", "scalar"), ("relu", None)),  # scalar bias
    (("add", "per_elem"), ("sigmoid", None)),  # per-elem aux, biggest IO
    (("mul", "per_col"), ("tanh", None)),  # scale → tanh
)

_LAYOUT_CHAINS: tuple[Chain, ...] = (
    (("relu", None),),
    (("bias", "per_col"), ("gelu_approx_tanh", None)),
    (("add", "per_elem"), ("sigmoid", None)),
)

# Default-axis cases: chain varies, everything else fixed.
_DEFAULT_AXIS_CHAINS: tuple[Chain, ...] = _UNARY_CHAINS + _BINARY_CHAINS + _BIAS_CHAINS + _MULTI_CHAINS


# Cross-config subset: canonical chains × {cta2 basic, cta2 cluster-m=128}
_CROSS_CONFIG_NAMES: tuple[str, ...] = (
    "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",  # cta_group=2 baseline
    "CONFIG_sm100_64x64x128_64x64x32_cluster2x4_2ctamma",  # cta2 cluster-m=128 (cta_tile_m=64)
)
_CROSS_CHAINS: tuple[Chain, ...] = (
    (("relu", None),),
    (("bias", "per_col"), ("gelu_approx_tanh", None)),
    (("mul", "per_col"), ("tanh", None)),
)


# Cross-dtype subset: canonical chains × {fp16, fp8_e4m3→fp16}
_CROSS_DTYPE_PAIRS: tuple[tuple[str, str], ...] = (
    ("fp16", "fp16"),
    ("fp8_e4m3", "fp16"),
)
_EPILOGUE_DTYPES: tuple[str, ...] = (
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_e8m0",
    "int8",
    "uint8",
    "bf16",
    "fp16",
    "fp32",
    "int32",
)


@dataclass(frozen=True)
class BatchedFusionCase:
    chain: Chain
    batched_aux: bool = False


_BATCHED_FUSION_CASES: tuple[BatchedFusionCase, ...] = (
    BatchedFusionCase((("relu", None),)),
    BatchedFusionCase((("bias", "per_col"), ("gelu_approx_tanh", None))),
    BatchedFusionCase((("add", "per_elem"), ("sigmoid", None))),
    BatchedFusionCase((("add", "scalar"), ("relu", None)), batched_aux=True),
    BatchedFusionCase((("bias", "per_row"), ("swish", None)), batched_aux=True),
    BatchedFusionCase((("sub", "per_col"),), batched_aux=True),
    BatchedFusionCase((("mul", "per_elem"), ("tanh", None)), batched_aux=True),
)


def _batched_case_id(case: BatchedFusionCase) -> str:
    aux = "batched_aux" if case.batched_aux else "shared_aux"
    return f"{_chain_id(case.chain)}__{aux}"


@dataclass(frozen=True)
class Rank3BroadcastCase:
    aux_shape: tuple[int, int, int]
    aux_on_rhs: bool


@dataclass(frozen=True)
class MixedDtypeBroadcastCase:
    aux_shape: tuple[int, int, int]
    aux_on_rhs: bool
    aux_dtype: str
    out_dtype: str


def _rank3_broadcast_case_id(case: Rank3BroadcastCase) -> str:
    parts = ["B" if case.aux_shape[0] != 1 else "1"]
    parts.append("M" if case.aux_shape[1] != 1 else "1")
    parts.append("N" if case.aux_shape[2] != 1 else "1")
    side = "rhs" if case.aux_on_rhs else "lhs"
    return f"{''.join(parts)}__{side}"


_RANK3_BROADCAST_CASES: tuple[Rank3BroadcastCase, ...] = tuple(
    Rank3BroadcastCase((b, m, n), aux_on_rhs)
    for aux_on_rhs in (True, False)
    for b in (1, _DEFAULT_BATCHED_SHAPE[0])
    for m in (1, _DEFAULT_BATCHED_SHAPE[1])
    for n in (1, _DEFAULT_BATCHED_SHAPE[2])
)


def _mixed_dtype_broadcast_case_id(case: MixedDtypeBroadcastCase) -> str:
    base = _rank3_broadcast_case_id(Rank3BroadcastCase(case.aux_shape, case.aux_on_rhs))
    return f"{base}__aux_{case.aux_dtype}__out_{case.out_dtype}"


_MIXED_DTYPE_BROADCAST_CASES: tuple[MixedDtypeBroadcastCase, ...] = (
    MixedDtypeBroadcastCase((1, 1, 1), True, "fp8_e4m3", "fp32"),
    MixedDtypeBroadcastCase((1, 1, 1), False, "int8", "bf16"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], 1, 1), True, "fp8_e5m2", "fp16"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], 1, 1), False, "uint8", "int32"),
    MixedDtypeBroadcastCase((1, _DEFAULT_BATCHED_SHAPE[1], 1), True, "fp8_e8m0", "fp8_e4m3"),
    MixedDtypeBroadcastCase((1, _DEFAULT_BATCHED_SHAPE[1], 1), False, "bf16", "fp8_e5m2"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], _DEFAULT_BATCHED_SHAPE[1], 1), True, "fp16", "uint8"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], _DEFAULT_BATCHED_SHAPE[1], 1), False, "fp32", "int8"),
    MixedDtypeBroadcastCase((1, 1, _DEFAULT_BATCHED_SHAPE[2]), True, "int32", "fp8_e8m0"),
    MixedDtypeBroadcastCase((1, 1, _DEFAULT_BATCHED_SHAPE[2]), False, "fp8_e4m3", "fp32"),
    MixedDtypeBroadcastCase((_DEFAULT_BATCHED_SHAPE[0], 1, _DEFAULT_BATCHED_SHAPE[2]), True, "int8", "bf16"),
    MixedDtypeBroadcastCase(
        (_DEFAULT_BATCHED_SHAPE[0], 1, _DEFAULT_BATCHED_SHAPE[2]),
        False,
        "fp8_e5m2",
        "fp16",
    ),
    MixedDtypeBroadcastCase(
        (1, _DEFAULT_BATCHED_SHAPE[1], _DEFAULT_BATCHED_SHAPE[2]),
        True,
        "uint8",
        "int32",
    ),
    MixedDtypeBroadcastCase(
        (1, _DEFAULT_BATCHED_SHAPE[1], _DEFAULT_BATCHED_SHAPE[2]),
        False,
        "fp8_e8m0",
        "fp8_e4m3",
    ),
    MixedDtypeBroadcastCase(
        (
            _DEFAULT_BATCHED_SHAPE[0],
            _DEFAULT_BATCHED_SHAPE[1],
            _DEFAULT_BATCHED_SHAPE[2],
        ),
        True,
        "bf16",
        "fp8_e5m2",
    ),
    MixedDtypeBroadcastCase(
        (
            _DEFAULT_BATCHED_SHAPE[0],
            _DEFAULT_BATCHED_SHAPE[1],
            _DEFAULT_BATCHED_SHAPE[2],
        ),
        False,
        "fp16",
        "uint8",
    ),
)


# Reference (torch fp32, then downcast)

_TORCH_UNARY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda x: x,
    "relu": torch.relu,
    "gelu": lambda x: torch.nn.functional.gelu(x),  # erf-based
    "gelu_approx_tanh": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
    "swish": torch.nn.functional.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "abs": torch.abs,
    "neg": torch.neg,
    "exp": torch.exp,
    "ceil": torch.ceil,
    "floor": torch.floor,
    "erf": torch.erf,
    "log": torch.log,
    "reciprocal": torch.reciprocal,
    "rsqrt": torch.rsqrt,
    "sqrt": torch.sqrt,
}

_TORCH_BINARY: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "add": lambda x, y: x + y,
    "bias": lambda x, y: x + y,
    "mul": lambda x, y: x * y,
    "sub": lambda x, y: x - y,
    "div": lambda x, y: x / y,
    "max": torch.maximum,
    "min": torch.minimum,
    "pow": torch.pow,
    "add_square": lambda x, y: x + y * y,
}


# Aux shape / runtime data helpers


def _aux_dim_stride(bcast: Bcast, M: int, N: int) -> tuple[list[int], list[int]]:
    """cuDNN dim+stride for an aux broadcasting in the given mode. The analyzer
    infers bcast_mode from these dims, so the shape drives the codegen branch."""
    if bcast == "scalar":
        return [1, 1], [1, 1]
    if bcast == "per_row":
        return [M, 1], [1, 1]
    if bcast == "per_col":
        return [1, N], [N, 1]
    if bcast == "per_elem":
        return [M, N], [N, 1]
    raise AssertionError(f"unknown bcast {bcast!r}")


def _aux_dim_stride_batched(bcast: Bcast, batch: int, M: int, N: int, batched_aux: bool) -> tuple[list[int], list[int]]:
    if not batched_aux:
        return _aux_dim_stride(bcast, M, N)
    if bcast == "scalar":
        return [batch, 1, 1], [1, 1, 1]
    if bcast == "per_row":
        return [batch, M, 1], [M, 1, 1]
    if bcast == "per_col":
        return [batch, 1, N], [N, N, 1]
    if bcast == "per_elem":
        return [batch, M, N], [M * N, N, 1]
    raise AssertionError(f"unknown bcast {bcast!r}")


def _rank3_aux_dim_stride(
    aux_shape: tuple[int, int, int],
) -> tuple[list[int], list[int]]:
    b, m, n = aux_shape
    return [b, m, n], [m * n, n, 1]


def _nonpacked_aux_dim_stride(bcast: Bcast, batch: int, M: int, N: int) -> tuple[list[int], list[int]]:
    pad = 16
    if bcast == "per_col":
        return [batch, 1, N], [N + pad, N + pad, 1]
    if bcast == "per_elem":
        return [batch, M, N], [M * (N + pad), N + pad, 1]
    raise ValueError(f"unsupported nonpacked aux bcast {bcast!r}")


def _mkaux(bcast: Bcast, M: int, N: int, dtype: str, seed: int) -> torch.Tensor:
    """Small-integer aux; range matches `_mkdata` so the FP32 reduction is exact."""
    torch.manual_seed(seed)
    if bcast == "scalar":
        shape = (1, 1)
    elif bcast == "per_row":
        shape = (M, 1)
    elif bcast == "per_col":
        shape = (1, N)
    elif bcast == "per_elem":
        shape = (M, N)
    else:
        raise AssertionError(f"unknown bcast {bcast!r}")
    return torch.empty(*shape, dtype=torch.int32).random_(-2, 2).to(dtype=_TORCH_DTYPE[dtype], device="cuda")


def _mkaux_batched(
    bcast: Bcast,
    batch: int,
    M: int,
    N: int,
    dtype: str,
    seed: int,
    batched_aux: bool,
) -> torch.Tensor:
    if not batched_aux:
        return _mkaux(bcast, M, N, dtype, seed)

    torch.manual_seed(seed)
    if bcast == "scalar":
        shape = (batch, 1, 1)
    elif bcast == "per_row":
        shape = (batch, M, 1)
    elif bcast == "per_col":
        shape = (batch, 1, N)
    elif bcast == "per_elem":
        shape = (batch, M, N)
    else:
        raise AssertionError(f"unknown bcast {bcast!r}")
    return torch.empty(*shape, dtype=torch.int32).random_(-2, 2).to(dtype=_TORCH_DTYPE[dtype], device="cuda")


def _mkaux_rank3_broadcast(
    aux_shape: tuple[int, int, int],
    dtype: str,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.empty(*aux_shape, dtype=torch.int32).random_(-2, 2).to(dtype=_TORCH_DTYPE[dtype], device="cuda")


def _mkaux_nonpacked(bcast: Bcast, batch: int, M: int, N: int, dtype: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    pad = 16
    if bcast == "per_col":
        storage = torch.empty(batch, 1, N + pad, dtype=torch.int32, device="cuda").random_(-2, 2)
    elif bcast == "per_elem":
        storage = torch.empty(batch, M, N + pad, dtype=torch.int32, device="cuda").random_(-2, 2)
    else:
        raise ValueError(f"unsupported nonpacked aux bcast {bcast!r}")
    storage = storage.to(dtype=_TORCH_DTYPE[dtype])
    return storage[:, :, :N]


def _mkaux_zero_stride_per_elem(batch: int, M: int, N: int, dtype: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    base = torch.empty(N, dtype=torch.int32, device="cuda").random_(-2, 2).to(dtype=_TORCH_DTYPE[dtype])
    return torch.as_strided(base, (batch, M, N), (0, 0, 1))


def _mkdata(
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
):
    """A, B, C tensors (rank-3, batch=1). Small-integer inputs ⇒ exact FP32 matmul."""
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dt.startswith("fp8") else (-2, 2)
    a_shape = (1, M, K) if a_major == "k" else (1, K, M)
    b_shape = (1, N, K) if b_major == "k" else (1, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dt], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dt], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    if out_major == "m":
        c = torch.empty(1, N, M, dtype=_TORCH_DTYPE[out_dt], device="cuda").transpose(1, 2)
    else:
        c = torch.empty(1, M, N, dtype=_TORCH_DTYPE[out_dt], device="cuda")
    return a, b, c


def _mkbatched_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    seed: int = 0,
    a_major: str = "k",
    b_major: str = "k",
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dt.startswith("fp8") else (-2, 2)
    a_shape = (batch, M, K) if a_major == "k" else (batch, K, M)
    b_shape = (batch, N, K) if b_major == "k" else (batch, K, N)
    a = torch.empty(*a_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dt], device="cuda")
    b = torch.empty(*b_shape, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dt], device="cuda")
    if a_major == "m":
        a = a.transpose(1, 2)
    if b_major == "n":
        b = b.transpose(1, 2)
    c = torch.empty(batch, M, N, dtype=_TORCH_DTYPE[out_dt], device="cuda")
    return a, b, c


def _mkbatched_nonpacked_data(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    seed: int = 0,
):
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dt.startswith("fp8") else (-2, 2)
    pad = 16
    a_storage = torch.empty(batch, M, K + pad, dtype=torch.int32, device="cuda").random_(*rng)
    a_storage = a_storage.to(dtype=_TORCH_DTYPE[in_dt])
    b_storage = torch.empty(batch, N, K + pad, dtype=torch.int32, device="cuda").random_(*rng)
    b_storage = b_storage.to(dtype=_TORCH_DTYPE[in_dt])
    c_storage = torch.empty(batch, M, N + pad, dtype=_TORCH_DTYPE[out_dt], device="cuda")
    a = a_storage[:, :, :K]
    b = b_storage[:, :, :K]
    c = c_storage[:, :, :N]
    return a, b, c


def _a_stride_batched(M: int, K: int, a_major: str) -> list[int]:
    return [M * K, K, 1] if a_major == "k" else [M * K, 1, M]


def _b_stride_batched(N: int, K: int, b_major: str) -> list[int]:
    return [N * K, 1, K] if b_major == "k" else [N * K, N, 1]


def _apply_unary(g: cudnn.pygraph, op: str, cur, name: str):
    return getattr(g, op)(input=cur, name=name)


def _apply_binary(g: cudnn.pygraph, op: str, lhs, rhs, name: str):
    if op == "bias":
        return g.bias(input=lhs, bias=rhs, name=name)
    if op in {"max", "min", "pow"}:
        return getattr(g, op)(input0=lhs, input1=rhs, name=name)
    return getattr(g, op)(a=lhs, b=rhs, name=name)


# Graph builder — drives the cuDNN frontend API + the recorder


def _build_graph(
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    chain: Chain,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
) -> tuple[cudnn.pygraph, list[str]]:
    """Build a `cudnn.pygraph` for a matmul + the given chain. Returns
    (graph, aux_names); aux_names is the in-chain order the call site must
    pass aux in (matching chain.aux_tensors)."""
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, b_major))
    cur = g.matmul(A=A, B=B, name="mm")

    aux_names: list[str] = []
    for i, (op, bcast) in enumerate(chain):
        if bcast is None:
            cur = _apply_unary(g, op, cur, f"u{i}")
        else:
            dim, stride = _aux_dim_stride(bcast, M, N)
            aux_name = f"aux{i}"
            aux = g.tensor(name=aux_name, dim=dim, stride=stride)
            aux_names.append(aux_name)
            cur = _apply_binary(g, op, cur, aux, f"o{i}")
    if out_major == "m":
        cur.set_stride([M * N, 1, M])
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, aux_names


def _build_nonpacked_epilogue_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    aux_bcast: Bcast,
) -> tuple[cudnn.pygraph, list[str]]:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, "k"))
    aux_dim, aux_stride = _nonpacked_aux_dim_stride(aux_bcast, batch, M, N)
    aux = g.tensor(name="aux0", dim=aux_dim, stride=aux_stride)
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(a=cur, b=aux, name="add")
    cur = g.relu(input=cur, name="relu")
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, ["aux0"]


def _build_zero_stride_aux_epilogue_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
) -> tuple[cudnn.pygraph, list[str]]:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, "k"))
    aux = g.tensor(name="aux0", dim=[batch, M, N], stride=[0, 0, 1])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(a=cur, b=aux, name="add")
    cur = g.relu(input=cur, name="relu")
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, ["aux0"]


def _build_batched_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    chain: Chain,
    *,
    batched_aux: bool,
    a_major: str = "k",
    b_major: str = "k",
) -> tuple[cudnn.pygraph, list[str]]:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, b_major))
    cur = g.matmul(A=A, B=B, name="mm")

    aux_names: list[str] = []
    for i, (op, bcast) in enumerate(chain):
        if bcast is None:
            cur = _apply_unary(g, op, cur, f"u{i}")
        else:
            dim, stride = _aux_dim_stride_batched(bcast, batch, M, N, batched_aux)
            aux_name = f"aux{i}"
            aux = g.tensor(name=aux_name, dim=dim, stride=stride)
            aux_names.append(aux_name)
            cur = _apply_binary(g, op, cur, aux, f"o{i}")
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, aux_names


def _build_rank3_broadcast_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    case: Rank3BroadcastCase,
) -> tuple[cudnn.pygraph, list[str]]:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, "k"))
    aux_dim, aux_stride = _rank3_aux_dim_stride(case.aux_shape)
    aux = g.tensor(name="aux0", dim=aux_dim, stride=aux_stride)
    cur = g.matmul(A=A, B=B, name="mm")
    if case.aux_on_rhs:
        cur = g.sub(a=cur, b=aux, name="sub")
    else:
        cur = g.sub(a=aux, b=cur, name="sub")
    cur.set_output(True)
    if out_dt != in_dt:
        cur.set_data_type(_CUDNN_DTYPE[out_dt])
    return g, ["aux0"]


def _build_mixed_dtype_broadcast_graph(
    batch: int,
    M: int,
    N: int,
    K: int,
    case: MixedDtypeBroadcastCase,
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[batch, K, N], stride=_b_stride_batched(N, K, "k"))
    aux_dim, aux_stride = _rank3_aux_dim_stride(case.aux_shape)
    aux = g.tensor(
        name="aux0",
        dim=aux_dim,
        stride=aux_stride,
        data_type=_CUDNN_DTYPE[case.aux_dtype],
    )
    cur = g.matmul(A=A, B=B, name="mm")
    if case.aux_on_rhs:
        cur = g.add(a=cur, b=aux, name="add")
    else:
        cur = g.add(a=aux, b=cur, name="add")
    cur.set_output(True).set_data_type(_CUDNN_DTYPE[case.out_dtype])
    return g


def _build_epilogue_dtype_graph(
    M: int,
    N: int,
    K: int,
    aux_dtype: str,
    out_dtype: str,
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    aux = g.tensor(
        name="aux0",
        dim=[1, 1, N],
        stride=[N, N, 1],
        data_type=_CUDNN_DTYPE[aux_dtype],
    )
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(a=cur, b=aux, name="add")
    cur.set_output(True).set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _reference(
    a: torch.Tensor,
    b: torch.Tensor,
    aux_runtime: dict[str, torch.Tensor],
    chain: Chain,
    out_dt: str,
) -> torch.Tensor:
    cur = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    for i, (op, bcast) in enumerate(chain):
        if bcast is None:
            cur = _TORCH_UNARY[op](cur)
        else:
            aux_t = aux_runtime[f"aux{i}"].to(torch.float32)
            cur = _TORCH_BINARY[op](cur, aux_t)
    return cur.to(_TORCH_DTYPE[out_dt])


def _batched_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    aux_runtime: dict[str, torch.Tensor],
    chain: Chain,
    out_dt: str,
) -> torch.Tensor:
    cur = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    for i, (op, bcast) in enumerate(chain):
        if bcast is None:
            cur = _TORCH_UNARY[op](cur)
        else:
            aux_t = aux_runtime[f"aux{i}"].to(torch.float32)
            cur = _TORCH_BINARY[op](cur, aux_t)
    return cur.to(_TORCH_DTYPE[out_dt])


def _rank3_broadcast_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    out_dt: str,
    aux_on_rhs: bool,
) -> torch.Tensor:
    cur = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    aux_f32 = aux.to(torch.float32)
    if aux_on_rhs:
        cur = cur - aux_f32
    else:
        cur = aux_f32 - cur
    return cur.to(_TORCH_DTYPE[out_dt])


# Config lookup


_NAME_TO_CFG = {c.name: c for c in CATALOG}

_LEGACY_RE = re.compile(r"^(CONFIG_sm100_\d+x\d+x\d+_\d+x\d+x\d+_cluster\d+x\d+)_([12])ctamma(_static)?$")


def _resolve(legacy_name):
    """Legacy config-name -> (pure-geometry config, cta_group, scheduler)."""
    m = _LEGACY_RE.match(legacy_name)
    assert m, legacy_name
    return (
        _NAME_TO_CFG[m.group(1)],
        int(m.group(2)),
        "static" if m.group(3) else "clc",
    )


# Test bodies — each parametrized axis is a separate test for clear IDs


def _run_case(
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    config_name: str,
    chain: Chain,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
) -> None:
    """Common run/check body: build graph, JIT, launch, compare to torch."""
    cfg, cta_group, scheduler = _resolve(config_name)
    g, aux_names = _build_graph(M, N, K, in_dt, out_dt, chain, a_major, b_major, out_major)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a, b, c = _mkdata(
        M,
        N,
        K,
        in_dt,
        out_dt,
        seed=0,
        a_major=a_major,
        b_major=b_major,
        out_major=out_major,
    )
    aux_runtime: dict[str, torch.Tensor] = {}
    for i, (op, bcast) in enumerate(chain):
        if bcast is None:
            continue
        aux_runtime[f"aux{i}"] = _mkaux(bcast, M, N, in_dt, seed=10 + i)

    compiled(_vp(compiled, a, b, c, *[aux_runtime[n] for n in aux_names]))
    torch.cuda.synchronize()

    ref = _reference(a, b, aux_runtime, chain, out_dt)
    try:
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    except AssertionError as e:
        diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
        bad = int((diff > 1e-1).sum().item())
        pytest.fail(
            f"\n  config:    {config_name}"
            f"\n  dtype:     {in_dt} -> {out_dt}"
            f"\n  shape:     {M}x{N}x{K}"
            f"\n  layout:    A{a_major}/B{b_major}"
            f"\n  chain:     {_chain_id(chain)}"
            f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
            f"\n  max|diff|: {float(diff.max().item()):.4g}"
            f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
            f"\n  sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  inner: {str(e).splitlines()[0]}",
            pytrace=False,
        )


def _run_batched_case(
    batch: int,
    M: int,
    N: int,
    K: int,
    in_dt: str,
    out_dt: str,
    config_name: str,
    case: BatchedFusionCase,
) -> None:
    cfg, cta_group, scheduler = _resolve(config_name)
    g, aux_names = _build_batched_graph(batch, M, N, K, in_dt, out_dt, case.chain, batched_aux=case.batched_aux)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a, b, c = _mkbatched_data(batch, M, N, K, in_dt, out_dt, seed=0)
    aux_runtime: dict[str, torch.Tensor] = {}
    for i, (op, bcast) in enumerate(case.chain):
        if bcast is None:
            continue
        aux_runtime[f"aux{i}"] = _mkaux_batched(
            bcast,
            batch,
            M,
            N,
            in_dt,
            seed=10 + i,
            batched_aux=case.batched_aux,
        )

    compiled(_vp(compiled, a, b, c, *[aux_runtime[n] for n in aux_names]))
    torch.cuda.synchronize()

    ref = _batched_reference(a, b, aux_runtime, case.chain, out_dt)
    try:
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    except AssertionError as e:
        diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
        bad = int((diff > 1e-1).sum().item())
        pytest.fail(
            f"\n  config:    {config_name}"
            f"\n  dtype:     {in_dt} -> {out_dt}"
            f"\n  shape:     B{batch} {M}x{N}x{K}"
            f"\n  chain:     {_batched_case_id(case)}"
            f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
            f"\n  max|diff|: {float(diff.max().item()):.4g}"
            f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
            f"\n  sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  inner: {str(e).splitlines()[0]}",
            pytrace=False,
        )


def _run_rank3_broadcast_case(case: Rank3BroadcastCase) -> None:
    batch, M, N, K = _DEFAULT_BATCHED_SHAPE
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g, aux_names = _build_rank3_broadcast_graph(batch, M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, case)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a, b, c = _mkbatched_data(batch, M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, seed=0)
    aux = _mkaux_rank3_broadcast(case.aux_shape, _DEFAULT_IN_DT, seed=23)
    compiled(_vp(compiled, a, b, c, *[aux for _ in aux_names]))
    torch.cuda.synchronize()

    ref = _rank3_broadcast_reference(a, b, aux, _DEFAULT_OUT_DT, case.aux_on_rhs)
    try:
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    except AssertionError as e:
        diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
        bad = int((diff > 1e-1).sum().item())
        pytest.fail(
            f"\n  config:    {_DEFAULT_CONFIG}"
            f"\n  dtype:     {_DEFAULT_IN_DT} -> {_DEFAULT_OUT_DT}"
            f"\n  shape:     B{batch} {M}x{N}x{K}"
            f"\n  aux shape: {case.aux_shape}"
            f"\n  aux side:  {'rhs' if case.aux_on_rhs else 'lhs'}"
            f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
            f"\n  max|diff|: {float(diff.max().item()):.4g}"
            f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
            f"\n  sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
            f"\n  inner: {str(e).splitlines()[0]}",
            pytrace=False,
        )


def _run_epilogue_dtype_case(aux_dtype: str, out_dtype: str) -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g = _build_epilogue_dtype_graph(M, N, K, aux_dtype, out_dtype)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    c = torch.empty(1, M, N, dtype=_TORCH_DTYPE[out_dtype], device="cuda")
    aux_i32 = torch.arange(N, dtype=torch.int32, device="cuda") % 2 + 1
    aux = aux_i32.reshape(1, 1, N).to(_TORCH_DTYPE[aux_dtype])

    compiled(_vp(compiled, a, b, c, aux))
    torch.cuda.synchronize()

    ref = aux.to(torch.float32).expand(1, M, N).to(_TORCH_DTYPE[out_dtype])
    # Exact compare: test values 1/2 round-trip exactly through the covered dtypes.
    torch.testing.assert_close(
        c.to(torch.float32),
        ref.to(torch.float32),
        atol=0,
        rtol=0,
    )


def _run_mixed_dtype_broadcast_case(case: MixedDtypeBroadcastCase) -> None:
    batch, M, N, K = _DEFAULT_BATCHED_SHAPE
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g = _build_mixed_dtype_broadcast_graph(batch, M, N, K, case)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(batch, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(batch, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    row_bit = torch.arange(M, dtype=torch.int32, device="cuda") % 2
    col_bit = torch.arange(N, dtype=torch.int32, device="cuda") % 2
    batch_bit = torch.arange(batch, dtype=torch.int32, device="cuda") % 2
    a[:, :, 0] = (row_bit.unsqueeze(0) + batch_bit.unsqueeze(1)).to(_TORCH_DTYPE[_DEFAULT_IN_DT])
    b[:, :, 0] = 1
    a[:, :, 1] = 1
    b[:, :, 1] = col_bit.unsqueeze(0).to(_TORCH_DTYPE[_DEFAULT_IN_DT])
    if case.out_dtype == "fp8_e8m0":
        # e8m0 has no mantissa; use exactly-representable 1s to isolate dtype
        # plumbing + broadcast indexing.
        a.zero_()
        b.zero_()
        a[:, :, 0] = 1
        b[:, :, 0] = 1
    c = torch.empty(batch, M, N, dtype=_TORCH_DTYPE[case.out_dtype], device="cuda")
    aux_elems = case.aux_shape[0] * case.aux_shape[1] * case.aux_shape[2]
    if case.out_dtype == "fp8_e8m0":
        aux_i32 = torch.ones(aux_elems, dtype=torch.int32, device="cuda")
    else:
        aux_i32 = torch.arange(aux_elems, dtype=torch.int32, device="cuda") % 2 + 1
    aux = aux_i32.reshape(case.aux_shape).to(_TORCH_DTYPE[case.aux_dtype])

    assert compiled.chain.aux_tensors[0].dtype == case.aux_dtype
    assert compiled.chain.output_dtype == case.out_dtype
    assert compiled.chain.ops[0].aux_on_rhs is case.aux_on_rhs

    compiled(_vp(compiled, a, b, c, aux))
    torch.cuda.synchronize()

    ref = (torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)) + aux.to(torch.float32)).to(_TORCH_DTYPE[case.out_dtype])
    # Exact compare: test data round-trips exactly for the chosen output dtype.
    torch.testing.assert_close(
        c.to(torch.float32),
        ref.to(torch.float32),
        atol=0,
        rtol=0,
    )


# Group A+B+bias+multi: chain varies, everything else fixed.
@pytest.mark.parametrize(
    "chain",
    _DEFAULT_AXIS_CHAINS,
    ids=[_chain_id(c) for c in _DEFAULT_AXIS_CHAINS],
)
def test_fusion_default_axis(chain: Chain) -> None:
    """API-surface gate: one case per (op, broadcast_mode) cell, default axes."""
    M, N, K = _DEFAULT_SHAPE
    _run_case(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, _DEFAULT_CONFIG, chain)


# Representative M-major chains: unary and canonical aux+unary.
_M_MAJOR_CHAINS: tuple[Chain, ...] = (
    (("relu", None),),
    (("bias", "per_col"), ("gelu_approx_tanh", None)),
)


@pytest.mark.parametrize(
    "chain",
    _M_MAJOR_CHAINS,
    ids=[_chain_id(c) for c in _M_MAJOR_CHAINS],
)
def test_fusion_m_major(chain: Chain) -> None:
    """Epilogue fusion with M-major output."""
    M, N, K = _DEFAULT_SHAPE
    _run_case(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, _DEFAULT_CONFIG, chain, out_major="m")


def test_rank1_per_col_fusion() -> None:
    """Rank-1 [N] aux broadcasts across M and must index with col_j."""
    M, N, K = _DEFAULT_SHAPE
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[N], stride=[1])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.bias(input=cur, bias=bias, name="b")
    cur.set_output(True)

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    a, b, c = _mkdata(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, seed=0)
    bias_runtime = (torch.arange(N, dtype=torch.int32) % 5 - 2).to(dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")

    compiled(_vp(compiled, a, b, c, bias_runtime))
    torch.cuda.synchronize()

    ref = (torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)) + bias_runtime.to(torch.float32)).to(_TORCH_DTYPE[_DEFAULT_OUT_DT])
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


def test_rank1_scalar_fusion() -> None:
    """Rank-1 [1] aux broadcasts across all output elements."""
    M, N, K = _DEFAULT_SHAPE
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    scale = g.tensor(name="scale", dim=[1], stride=[1])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(a=cur, b=scale, name="add")
    cur.set_output(True)

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    a, b, c = _mkdata(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, seed=0)
    scale_runtime = torch.tensor([2], dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")

    compiled(_vp(compiled, a, b, c, scale_runtime))
    torch.cuda.synchronize()

    ref = (torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)) + scale_runtime.to(torch.float32)).to(_TORCH_DTYPE[_DEFAULT_OUT_DT])
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


def test_batched_rank1_per_col_fusion() -> None:
    """Rank-1 [N] aux broadcasts across batch and M for rank-3 matmul output."""
    batch, M, N, K = _DEFAULT_BATCHED_SHAPE
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[N], stride=[1])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.bias(input=cur, bias=bias, name="b")
    cur.set_output(True)

    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    a, b, c = _mkbatched_data(batch, M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, seed=0)
    bias_runtime = (torch.arange(N, dtype=torch.int32) % 5 - 2).to(dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")

    compiled(_vp(compiled, a, b, c, bias_runtime))
    torch.cuda.synchronize()

    ref = (torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)) + bias_runtime.to(torch.float32)).to(_TORCH_DTYPE[_DEFAULT_OUT_DT])
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "case",
    _RANK3_BROADCAST_CASES,
    ids=[_rank3_broadcast_case_id(c) for c in _RANK3_BROADCAST_CASES],
)
def test_rank3_batched_broadcast_patterns(case: Rank3BroadcastCase) -> None:
    """Rank-3 aux, every [B/1, M/1, N/1] broadcast pattern. `sub` (non-commutative)
    covers both aux-on-RHS and aux-on-LHS to check port direction is preserved."""
    _run_rank3_broadcast_case(case)


@pytest.mark.parametrize(
    "aux_bcast",
    _NONPACKED_AUX_BCAST_MODES,
    ids=list(_NONPACKED_AUX_BCAST_MODES),
)
@pytest.mark.parametrize(
    "config_name",
    _NONPACKED_CONFIGS,
    ids=[n.removeprefix("CONFIG_") for n in _NONPACKED_CONFIGS],
)
def test_nonpacked_epilogue_aux_and_output(
    config_name: str,
    aux_bcast: Bcast,
) -> None:
    batch, M, N, K = 2, 256, 256, 256
    in_dt = out_dt = "bf16"
    cfg, cta_group, scheduler = _resolve(config_name)
    g, aux_names = _build_nonpacked_epilogue_graph(batch, M, N, K, in_dt, out_dt, aux_bcast)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(
            f"JIT compile failed: {type(e).__name__}: {first[:200]}",
            pytrace=False,
        )

    a, b, c = _mkbatched_nonpacked_data(batch, M, N, K, in_dt, out_dt, seed=0)
    aux = _mkaux_nonpacked(aux_bcast, batch, M, N, in_dt, seed=13)
    assert not a.is_contiguous() and not b.is_contiguous()
    assert not c.is_contiguous() and not aux.is_contiguous()

    compiled(_vp(compiled, a, b, c, *[aux for _ in aux_names]))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref = torch.relu(ref + aux.to(torch.float32)).to(_TORCH_DTYPE[out_dt])
    try:
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    except AssertionError as e:
        diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
        bad = int((diff > 1e-1).sum().item())
        pytest.fail(
            f"\n  config:    {config_name}"
            f"\n  aux bcast: {aux_bcast}"
            f"\n  strides:   A{tuple(a.stride())} B{tuple(b.stride())} "
            f"C{tuple(c.stride())} aux{tuple(aux.stride())}"
            f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
            f"\n  max|diff|: {float(diff.max().item()):.4g}"
            f"\n  inner:     {str(e).splitlines()[0]}",
            pytrace=False,
        )


@pytest.mark.parametrize(
    "config_name",
    _NONPACKED_CONFIGS,
    ids=[n.removeprefix("CONFIG_") for n in _NONPACKED_CONFIGS],
)
def test_zero_stride_epilogue_aux_broadcast(
    config_name: str,
) -> None:
    batch, M, N, K = 2, 256, 256, 256
    in_dt = out_dt = "bf16"
    cfg, cta_group, scheduler = _resolve(config_name)
    g, aux_names = _build_zero_stride_aux_epilogue_graph(batch, M, N, K, in_dt, out_dt)
    compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler)

    a, b, c = _mkbatched_nonpacked_data(batch, M, N, K, in_dt, out_dt, seed=0)
    aux = _mkaux_zero_stride_per_elem(batch, M, N, in_dt, seed=13)
    assert aux.stride() == (0, 0, 1)
    assert not c.is_contiguous()

    compiled(_vp(compiled, a, b, c, *[aux for _ in aux_names]))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref = torch.relu(ref + aux.to(torch.float32)).to(_TORCH_DTYPE[out_dt])
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)


# Cross-config subset.
@pytest.mark.parametrize(
    "chain",
    _CROSS_CHAINS,
    ids=[_chain_id(c) for c in _CROSS_CHAINS],
)
@pytest.mark.parametrize(
    "config_name",
    _CROSS_CONFIG_NAMES,
    ids=[n.replace("CONFIG_", "").replace("_sm100", "") for n in _CROSS_CONFIG_NAMES],
)
def test_fusion_cross_config(config_name: str, chain: Chain) -> None:
    """Canonical chains on non-default configs — catches template-side bugs
    (e.g. the 2×2 DP TMEM layout for cluster-m=128)."""
    M, N, K = _DEFAULT_SHAPE
    _run_case(M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, config_name, chain)


# Cross-dtype subset.
@pytest.mark.parametrize(
    "chain",
    _CROSS_CHAINS,
    ids=[_chain_id(c) for c in _CROSS_CHAINS],
)
@pytest.mark.parametrize(
    "in_dt,out_dt",
    _CROSS_DTYPE_PAIRS,
    ids=[f"{p[0]}->{p[1]}" for p in _CROSS_DTYPE_PAIRS],
)
def test_fusion_cross_dtype(in_dt: str, out_dt: str, chain: Chain) -> None:
    """Canonical chains with FP16 and FP8→FP16 — catches dtype-specific codegen bugs."""
    M, N, K = _DEFAULT_SHAPE
    _run_case(M, N, K, in_dt, out_dt, _DEFAULT_CONFIG, chain)


@pytest.mark.parametrize("aux_dtype", _EPILOGUE_DTYPES)
def test_epilogue_aux_dtypes(aux_dtype: str) -> None:
    _run_epilogue_dtype_case(aux_dtype, "fp32")


@pytest.mark.parametrize("out_dtype", _EPILOGUE_DTYPES)
def test_epilogue_output_dtypes(out_dtype: str) -> None:
    _run_epilogue_dtype_case("bf16", out_dtype)


def test_epilogue_uint8_output_preserves_values_above_int8_positive_range() -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g = _build_epilogue_dtype_graph(M, N, K, "fp32", "uint8")
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    c = torch.empty(1, M, N, dtype=torch.uint8, device="cuda")
    aux = torch.full((1, 1, N), 200.0, dtype=torch.float32, device="cuda")

    compiled(_vp(compiled, a, b, c, aux))
    torch.cuda.synchronize()

    ref = torch.full((1, M, N), 200, dtype=torch.uint8, device="cuda")
    torch.testing.assert_close(c, ref, atol=0, rtol=0)


def test_epilogue_int32_compute_type() -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    aux = g.tensor(
        name="aux0",
        dim=[1, 1, N],
        stride=[N, N, 1],
        data_type=cudnn.data_type.INT32,
    )
    cur = g.matmul(A=A, B=B, name="mm")
    cur = g.add(
        a=cur,
        b=aux,
        name="add_i32",
        compute_data_type=cudnn.data_type.INT32,
    )
    cur.set_output(True).set_data_type(cudnn.data_type.INT32)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    c = torch.empty(1, M, N, dtype=torch.int32, device="cuda")
    aux_runtime = (torch.arange(N, dtype=torch.int32, device="cuda") % 5 - 2).reshape(1, 1, N)

    compiled(_vp(compiled, a, b, c, aux_runtime))
    torch.cuda.synchronize()

    ref = aux_runtime.expand(1, M, N)
    torch.testing.assert_close(c, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    "op",
    (
        "ceil",
        "floor",
        "erf",
        "log",
        "reciprocal",
        "rsqrt",
        "sqrt",
    ),
)
def test_additional_unary_epilogue_ops(op: str) -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    cur = g.matmul(A=A, B=B, name="mm")
    cur = _apply_unary(g, op, cur, "new_unary")
    cur.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    a[:, :, 0] = 1
    b[:, :, 0] = 4
    c = torch.empty(1, M, N, dtype=torch.float32, device="cuda")

    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref = _TORCH_UNARY[op](mm)
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("bcast", _BCAST_MODES)
@pytest.mark.parametrize("aux_on_rhs", (True, False), ids=("rhs", "lhs"))
@pytest.mark.parametrize("op", ("max", "min", "pow", "add_square"))
def test_additional_binary_epilogue_ops(
    op: str,
    aux_on_rhs: bool,
    bcast: Bcast,
) -> None:
    M, N, K = 128, 128, 128
    cfg, cta_group, scheduler = _resolve(_DEFAULT_CONFIG)
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[_DEFAULT_IN_DT],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    dim, stride = _aux_dim_stride(bcast, M, N)
    aux = g.tensor(name="aux0", dim=dim, stride=stride)
    cur = g.matmul(A=A, B=B, name="mm")
    if aux_on_rhs:
        cur = _apply_binary(g, op, cur, aux, "new_binary")
    else:
        cur = _apply_binary(g, op, aux, cur, "new_binary")
    cur.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    try:
        compiled = _plan(g, config=cfg, cta_group=cta_group, scheduler=scheduler, force_stg_epi=True)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        pytest.fail(f"JIT compile failed: {type(e).__name__}: {first[:200]}", pytrace=False)

    a = torch.zeros(1, M, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    b = torch.zeros(1, N, K, dtype=_TORCH_DTYPE[_DEFAULT_IN_DT], device="cuda")
    a[:, :, 0] = 1
    b[:, :, 0] = 3
    c = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    aux_runtime = torch.full(
        tuple(dim),
        2,
        dtype=_TORCH_DTYPE[_DEFAULT_IN_DT],
        device="cuda",
    )

    compiled(_vp(compiled, a, b, c, aux_runtime))
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    aux_f32 = aux_runtime.to(torch.float32)
    if aux_on_rhs:
        ref = _TORCH_BINARY[op](mm, aux_f32)
    else:
        ref = _TORCH_BINARY[op](aux_f32, mm)
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "case",
    _MIXED_DTYPE_BROADCAST_CASES,
    ids=[_mixed_dtype_broadcast_case_id(c) for c in _MIXED_DTYPE_BROADCAST_CASES],
)
def test_mixed_dtype_rank3_broadcast_patterns(
    case: MixedDtypeBroadcastCase,
) -> None:
    _run_mixed_dtype_broadcast_case(case)


@pytest.mark.parametrize(
    "chain",
    _LAYOUT_CHAINS,
    ids=[_chain_id(c) for c in _LAYOUT_CHAINS],
)
@pytest.mark.parametrize(
    "a_major,b_major",
    _INPUT_LAYOUTS,
    ids=[_layout_id(p) for p in _INPUT_LAYOUTS],
)
def test_fusion_input_layouts(a_major: str, b_major: str, chain: Chain) -> None:
    """Representative fused epilogues with A K/M-major and B K/N-major inputs."""
    M, N, K = 256, 256, 256
    _run_case(
        M,
        N,
        K,
        _DEFAULT_IN_DT,
        _DEFAULT_OUT_DT,
        _DEFAULT_CONFIG,
        chain,
        a_major,
        b_major,
    )


@pytest.mark.parametrize(
    "case",
    _BATCHED_FUSION_CASES,
    ids=[_batched_case_id(c) for c in _BATCHED_FUSION_CASES],
)
def test_batched_fusion_default_axis(case: BatchedFusionCase) -> None:
    """Rank-3 matmul + fused epilogues: unary-only, batch-shared aux, and
    per-batch rank-3 aux."""
    batch, M, N, K = _DEFAULT_BATCHED_SHAPE
    _run_batched_case(batch, M, N, K, _DEFAULT_IN_DT, _DEFAULT_OUT_DT, _DEFAULT_CONFIG, case)


# Coverage manifest (read by `python -m cudnn.frost.gemm.verify --list`)


@dataclass(frozen=True)
class _CoverageManifest:
    default_config: str
    default_in_dt: str
    default_out_dt: str
    default_shape: tuple[int, int, int]
    default_batched_shape: tuple[int, int, int, int]
    chains_default_axis: tuple[str, ...]
    chains_batched_default_axis: tuple[str, ...]
    input_layouts: tuple[str, ...]
    chains_input_layouts: tuple[str, ...]
    chains_cross_config: tuple[str, ...]
    cross_configs: tuple[str, ...]
    cross_dtype_pairs: tuple[tuple[str, str], ...]


COVERAGE = _CoverageManifest(
    default_config=_DEFAULT_CONFIG,
    default_in_dt=_DEFAULT_IN_DT,
    default_out_dt=_DEFAULT_OUT_DT,
    default_shape=_DEFAULT_SHAPE,
    default_batched_shape=_DEFAULT_BATCHED_SHAPE,
    chains_default_axis=tuple(_chain_id(c) for c in _DEFAULT_AXIS_CHAINS),
    chains_batched_default_axis=tuple(_batched_case_id(c) for c in _BATCHED_FUSION_CASES),
    input_layouts=tuple(_layout_id(p) for p in _INPUT_LAYOUTS),
    chains_input_layouts=tuple(_chain_id(c) for c in _LAYOUT_CHAINS),
    chains_cross_config=tuple(_chain_id(c) for c in _CROSS_CHAINS),
    cross_configs=_CROSS_CONFIG_NAMES,
    cross_dtype_pairs=_CROSS_DTYPE_PAIRS,
)


# Standalone CLI shim — forwards remaining argv to pytest on this file


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
