"""Analyze a user-built ``cudnn.pygraph`` and produce a ``FusionChain``.

The user-facing API is **pure cuDNN frontend** — they write::

    import cudnn
    import cudnn.TBD.gemm                          # triggers install_recorder() at import

    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, ...)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True)

    compiled = cudnn.TBD.gemm.compiler.jit_from_cudnn_graph(g)

There is no custom ``FusedGraph`` wrapper. Importing ``cudnn.TBD.gemm`` monkey-patches
``cudnn.pygraph``'s tensor / matmul / pointwise op methods so they record the
op chain on the graph instance (``g._cudnn_gemm_state``) while still delegating
to the real cuDNN backend. ``analyze(g)`` reads that state.

This means: any ``cudnn.pygraph`` constructed AFTER ``import cudnn.TBD.gemm`` is
automatically analyzable. Graphs built BEFORE the import are not — call
``cudnn.TBD.gemm.install_recorder()`` early if you need to be safe.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, field
from typing import Any

import cudnn

_LOG = logging.getLogger(__name__)

from .fusion_ir import (
    AMajor,
    BMajor,
    BlockQuantizeSpec,
    OutMajor,
    Dtype,
    FusionChain,
    FusionOp,
    MatmulSpec,
    ReductionSpec,
    TensorRef,
    gemm_source,
)

# ---------------------------------------------------------------------------
# Dtype + op tables
# ---------------------------------------------------------------------------


_DTYPE_FROM_CUDNN: dict[Any, Dtype] = {
    cudnn.data_type.BFLOAT16: "bf16",
    cudnn.data_type.HALF: "fp16",
    cudnn.data_type.FLOAT: "fp32",
    cudnn.data_type.INT8: "int8",
    cudnn.data_type.FP8_E4M3: "fp8_e4m3",
    cudnn.data_type.FP8_E5M2: "fp8_e5m2",
    cudnn.data_type.FP8_E8M0: "fp8_e8m0",
    cudnn.data_type.FP4_E2M1: "fp4_e2m1",
    cudnn.data_type.INT8: "int8",
    cudnn.data_type.UINT8: "uint8",
    cudnn.data_type.INT32: "int32",
    cudnn.data_type.INT64: "int64",
}

_CUDNN_FROM_DTYPE: dict[Dtype, Any] = {v: k for k, v in _DTYPE_FROM_CUDNN.items()}


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


_UNARY_OP_MAP: dict[str, str] = {
    "relu": "relu",
    "gelu": "gelu",
    "gelu_approx_tanh": "gelu_tanh",
    "swish": "swish",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "exp": "exp",
    "abs": "abs",
    "neg": "neg",
    "cos": "cos",
    "sin": "sin",
    "ceil": "ceil",
    "floor": "floor",
    "erf": "erf",
    "log": "log",
    "reciprocal": "reciprocal",
    "rsqrt": "rsqrt",
    "sqrt": "sqrt",
    "identity": "identity",
}
_BINARY_OP_MAP: dict[str, str] = {
    "add": "add",
    "mul": "mul",
    "sub": "sub",
    "div": "div",
    "max": "max",
    "min": "min",
    "pow": "pow",
    "add_square": "add_square",
    "bias": "add",  # cuDNN's `bias(input, bias)` is just `input + bias`
}


# ---------------------------------------------------------------------------
# Internal recording state, attached to each cudnn.pygraph instance
# ---------------------------------------------------------------------------


@dataclass
class _RecordedOp:
    cudnn_name: str
    op_name: str
    inputs: list[int]
    output: int
    output_tensor: Any  # strong ref so id() stays valid
    compute_dtype: Dtype | None = None  # per-op compute_data_type override (None → graph default)
    # block_scale_dequantize: the [non-K, K] block size (e.g. [1, 16] for A,
    # [16, 1] for B). None for every other op.
    block_size: tuple[int, ...] | None = None
    is_negative_scale: bool = False
    # block_scale_quantize: output is the quantized tensor; scale_output is
    # the scale-factor side-output returned by cuDNN.
    scale_output: int | None = None
    scale_output_tensor: Any = None
    quant_axis: int | None = None
    quant_transpose: bool = False
    # moe_grouped_matmul: the operation mode ("none" / "gather" / "scatter").
    # None for every other op.
    moe_mode: str | None = None
    # reduction: canonical reduction mode ("add" / "amax" / "max" / "min").
    # None for every other op.
    reduction_mode: str | None = None
    # Optional cuDNN FE groupOffset input for grouped reductions. For MoE this
    # is expected to be the same tensor as first_token_offset.
    group_offset: int | None = None


@dataclass
class _TensorMeta:
    name: str
    dim: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Dtype
    is_input: bool = False
    # cuDNN tensor reordering layout name (e.g. "F8_128x4") or None for the
    # default (NONE). Captured so the block-scale SF reorder layout is visible
    # to the analyzer / compile-stage support check.
    reordering: str | None = None
    # Strong ref to the cuDNN tensor object itself. For graph inputs this is the
    # tensor returned by g.tensor(...); op outputs carry it on _RecordedOp too.
    # Used to bind each role to its cuDNN tensor (uid / name / object) so the
    # compiled kernel can be called with a variant-pack dict instead of
    # order-sensitive positional args.
    tensor: Any = None


# pybind11's cudnn.pygraph doesn't allow setting arbitrary attributes on
# instances, so we keep per-graph recording state in a WeakKeyDictionary keyed
# by the graph object itself. Entries auto-evict when graphs are GC'd, and
# we never collide via id() reuse across short-lived graphs.
_GRAPH_STATES: "weakref.WeakKeyDictionary[cudnn.pygraph, dict]" = weakref.WeakKeyDictionary()

# cuDNN ``set_output(True)`` / ``set_data_type(...)`` are tensor-instance methods
# with no corresponding getter, so we class-patch them and stash the resulting
# flags here (keyed by id(tensor)). Cleared at each pygraph __init__ to keep
# state from leaking across short-lived scripts.
_TENSOR_OUTPUT_FLAG: dict[int, bool] = {}
_TENSOR_EXPLICIT_DTYPE: dict[int, Any] = {}
_TENSOR_DIM_OVERRIDE: dict[int, tuple[int, ...]] = {}
_TENSOR_REORDERING_OVERRIDE: dict[int, str | None] = {}


def _ensure_state(graph: cudnn.pygraph) -> dict:
    state = _GRAPH_STATES.get(graph)
    if state is None:
        state = {
            "ops": [],
            "tensor_meta": {},
            "io_dtype": "bf16",
            # cuDNN graph-level defaults. ``intermediate_dtype`` is the
            # data_type a virtual (unmaterialized) tensor takes when the user
            # doesn't call set_data_type on it; ``compute_dtype`` is the
            # default math precision for every op that doesn't override it.
            "intermediate_dtype": "fp32",
            "compute_dtype": "fp32",
        }
        _GRAPH_STATES[graph] = state
    return state


def _get_state(graph: cudnn.pygraph) -> dict | None:
    return _GRAPH_STATES.get(graph)


# ---------------------------------------------------------------------------
# Variant-pack binding — maps each graph role to its cuDNN tensor
# ---------------------------------------------------------------------------


@dataclass
class GemmBinding:
    """Maps each graph role to the cuDNN tensor that fills it, so a compiled
    kernel can be called with a variant-pack dict (keyed by cuDNN tensor object,
    uid, or name) instead of order-sensitive positional args.

    Operand lists are in the distinct-slot order the rendered kernel expects;
    ``outputs`` is in :pyattr:`FusionChain.outputs` slot order (terminal, then
    taps); ``aux`` is in :pyattr:`FusionChain.aux_tensors` order. Block-scale
    fills ``sfa_operands`` / ``sfb_operands`` parallel to ``a_operands`` /
    ``b_operands``; MoE fills ``first_token_offset``."""

    a_operands: list[Any] = field(default_factory=list)
    b_operands: list[Any] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)
    aux: list[Any] = field(default_factory=list)
    sfa_operands: list[Any] = field(default_factory=list)
    sfb_operands: list[Any] = field(default_factory=list)
    first_token_offset: Any = None

    def bound_tensors(self) -> list[Any]:
        ts = [
            *self.a_operands,
            *self.b_operands,
            *self.outputs,
            *self.aux,
            *self.sfa_operands,
            *self.sfb_operands,
        ]
        if self.first_token_offset is not None:
            ts.append(self.first_token_offset)
        return [t for t in ts if t is not None]


def _make_multi_binding(
    meta: dict,
    a_ids,
    b_ids,
    a_caps,
    b_caps,
    output_objs,
    aux_objs,
    block_scale: bool,
    first_token_offset=None,
) -> GemmBinding:
    """Build a GemmBinding for the multi-operand builders (multi-GEMM / MoE),
    pulling the cuDNN tensor for each distinct A/B slot (+ its SF for the
    block-scale case) out of ``meta``."""

    def _sf(caps: dict, ids) -> list:
        objs = []
        for i in ids:
            sid = caps[i].get("sf_id")
            objs.append(meta[sid].tensor if sid is not None else None)
        return objs

    return GemmBinding(
        a_operands=[meta[i].tensor for i in a_ids],
        b_operands=[meta[i].tensor for i in b_ids],
        outputs=list(output_objs),
        aux=list(aux_objs),
        sfa_operands=_sf(a_caps, a_ids) if block_scale else [],
        sfb_operands=_sf(b_caps, b_ids) if block_scale else [],
        first_token_offset=first_token_offset,
    )


def _safe_name(t: Any) -> str | None:
    try:
        nm = t.get_name()
    except Exception:  # noqa: BLE001 — unbuilt tensors can throw
        return None
    return nm or None


def _safe_uid(t: Any) -> int | None:
    try:
        uid = t.get_uid()
    except Exception:  # noqa: BLE001
        return None
    # cuDNN leaves uid = -1 (or 0) until build_operation_graph(); only trust a
    # positive uid as a lookup key.
    return uid if isinstance(uid, int) and uid > 0 else None


def resolve_variant_pack(variant_pack: dict, binding: GemmBinding) -> dict[int, Any]:
    """Resolve a ``{key: buffer}`` variant pack to ``{id(bound_tensor): buffer}``.

    Keys may be the cuDNN tensor object (by identity), its uid (int, only once
    ``build_operation_graph()`` has assigned positive uids), or its name (str).
    Raises on an unknown key or a key that doesn't match any bound role."""
    if not isinstance(variant_pack, dict):
        raise TypeError("compiled kernels are called with a variant-pack dict " "{cudnn_tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}")
    bound = binding.bound_tensors()
    by_obj = {id(t): t for t in bound}

    name_counts: dict[str, int] = {}
    uid_counts: dict[int, int] = {}
    for t in bound:
        nm = _safe_name(t)
        if nm is not None:
            name_counts[nm] = name_counts.get(nm, 0) + 1
        uid = _safe_uid(t)
        if uid is not None:
            uid_counts[uid] = uid_counts.get(uid, 0) + 1
    by_name = {_safe_name(t): t for t in bound if name_counts.get(_safe_name(t)) == 1}
    by_uid = {_safe_uid(t): t for t in bound if uid_counts.get(_safe_uid(t)) == 1}
    by_name.pop(None, None)
    by_uid.pop(None, None)

    resolved: dict[int, Any] = {}
    for key, buf in variant_pack.items():
        if id(key) in by_obj:
            t = by_obj[id(key)]
        elif isinstance(key, int) and key in by_uid:
            t = by_uid[key]
        elif isinstance(key, str) and key in by_name:
            t = by_name[key]
        else:
            kdesc = key if isinstance(key, (int, str)) else _safe_name(key)
            raise KeyError(
                f"variant-pack key {kdesc!r} does not match any input / output "
                "tensor of this graph (key by the cuDNN tensor object, its uid, "
                "or its name)"
            )
        resolved[id(t)] = buf
    return resolved


# ---------------------------------------------------------------------------
# Monkey-patch installer
# ---------------------------------------------------------------------------


_INSTALLED = False
_ORIGINALS: dict[str, Any] = {}


def _bind(args: tuple, kwargs: dict, names: tuple[str, ...]) -> dict:
    """Merge positional + keyword args into a ``name -> value`` dict.

    The recorder patches must accept exactly what the real cuDNN API accepts —
    both positional (``g.matmul(A, B)``) and keyword (``g.matmul(A=A, B=B)``).
    Delegation always passes ``*args, **kwargs`` verbatim; this helper only
    reconstructs the named values for the best-effort op recording."""
    bound = dict(kwargs)
    for i, val in enumerate(args):
        if i < len(names):
            bound.setdefault(names[i], val)
    return bound


def _patched_init(self, *args, **kwargs):
    _ORIGINALS["__init__"](self, *args, **kwargs)
    io_dt = kwargs.get("io_data_type", cudnn.data_type.BFLOAT16)
    inter_dt = kwargs.get("intermediate_data_type", cudnn.data_type.FLOAT)
    comp_dt = kwargs.get("compute_data_type", cudnn.data_type.FLOAT)
    state = _ensure_state(self)
    state["io_dtype"] = _DTYPE_FROM_CUDNN.get(io_dt, "bf16")
    state["intermediate_dtype"] = _DTYPE_FROM_CUDNN.get(inter_dt, "fp32")
    state["compute_dtype"] = _DTYPE_FROM_CUDNN.get(comp_dt, "fp32")
    # Fresh graph → clear any stale tensor-level flags from earlier graphs.
    _TENSOR_OUTPUT_FLAG.clear()
    _TENSOR_EXPLICIT_DTYPE.clear()
    _TENSOR_DIM_OVERRIDE.clear()
    _TENSOR_REORDERING_OVERRIDE.clear()


def _patched_tensor(self, *args, **kwargs):
    state = _ensure_state(self)
    # GEMM keyword style omits data_type → default to the graph io dtype. Only
    # inject for the pure-keyword call (host callers always pass data_type, often
    # as a torch dtype — leave those untouched).
    if not args and kwargs.get("data_type") is None:
        kwargs = {**kwargs, "data_type": _CUDNN_FROM_DTYPE[state["io_dtype"]]}
    t = _ORIGINALS["tensor"](self, *args, **kwargs)
    try:
        b = _bind(args, kwargs, ("name", "dim", "stride", "data_type"))
        # Capture the SF reorder layout (e.g. F8_128x4) when set. The enum's
        # .name is "NONE" for the default; store None in that case.
        rt = b.get("reordering_type")
        reordering = rt.name if (rt is not None and getattr(rt, "name", "NONE") != "NONE") else None
        state["tensor_meta"][id(t)] = _TensorMeta(
            name=b.get("name"),
            dim=tuple(b["dim"]),
            stride=tuple(b["stride"]),
            # .get → None for torch dtypes (host graphs, never analyzed); cuDNN
            # enums (GEMM graphs) map to our literal.
            dtype=_DTYPE_FROM_CUDNN.get(b.get("data_type")),
            is_input=True,
            reordering=reordering,
            tensor=t,
        )
    except Exception:  # noqa: BLE001 — recording is best-effort; never break a real graph build
        pass
    return t


def _opt_compute_dtype(kwargs: dict) -> Dtype | None:
    """Pull a per-op ``compute_data_type`` override out of the kwargs the user
    passed to a cuDNN op, mapping it to our Dtype literal. None → "use the
    graph default" (resolved in _build_chain)."""
    dt = kwargs.get("compute_data_type")
    if dt is None:
        return None
    return _DTYPE_FROM_CUDNN.get(dt)


def _patched_matmul(self, *args, **kwargs):
    out = _ORIGINALS["matmul"](self, *args, **kwargs)
    try:
        b = _bind(args, kwargs, ("A", "B"))
        name = b.get("name", "")
        state = _ensure_state(self)
        state["ops"].append(_RecordedOp("matmul", name, [id(b["A"]), id(b["B"])], id(out), out, compute_dtype=_opt_compute_dtype(kwargs)))
        state["tensor_meta"][id(out)] = _TensorMeta(name=f"{name}::OUT_0", dim=(), stride=(), dtype="fp32")
    except Exception:  # noqa: BLE001
        pass
    return out


def _patched_block_scale_dequantize(self, *args, **kwargs):
    out = _ORIGINALS["block_scale_dequantize"](self, *args, **kwargs)
    try:
        b = _bind(args, kwargs, ("input", "descale", "block_size", "is_negative_scale"))
        name = b.get("name", "")
        state = _ensure_state(self)
        # The dequant output is virtual fp32 (per the graph json). dim/stride are
        # the same logical (batch, M, K) / (batch, K, N) as the packed input — the
        # matmul reads it as if dequantized. Record fp32 so downstream rank checks
        # see a 3D operand carrying the input's dims.
        in_meta = state["tensor_meta"].get(id(b["input"]))
        state["ops"].append(
            _RecordedOp(
                "block_scale_dequantize",
                name,
                [id(b["input"]), id(b["descale"])],
                id(out),
                out,
                compute_dtype=_opt_compute_dtype(kwargs),
                block_size=tuple(b["block_size"]),
                is_negative_scale=bool(b.get("is_negative_scale", False)),
            )
        )
        state["tensor_meta"][id(out)] = _TensorMeta(
            name=f"{name}::OUT_0",
            dim=in_meta.dim if in_meta else (),
            stride=in_meta.stride if in_meta else (),
            dtype="fp32",
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def _patched_block_scale_quantize(self, *args, **kwargs):
    out = _ORIGINALS["block_scale_quantize"](self, *args, **kwargs)
    try:
        b = _bind(args, kwargs, ("input", "block_size", "axis", "transpose"))
        name = b.get("name", "")
        quantized, scale = out
        block_size = b["block_size"]
        if isinstance(block_size, (list, tuple)):
            if len(block_size) != 1:
                raise NotImplementedError(f"block_scale_quantize expects a scalar block_size in cudnn.TBD.gemm; got {block_size!r}")
            block_size_i = int(block_size[0])
        else:
            block_size_i = int(block_size)
        axis = b.get("axis")
        transpose = bool(b.get("transpose", False))
        state = _ensure_state(self)
        in_meta = state["tensor_meta"].get(id(b["input"]))
        state["ops"].append(
            _RecordedOp(
                "block_scale_quantize",
                name,
                [id(b["input"])],
                id(quantized),
                quantized,
                compute_dtype=_opt_compute_dtype(kwargs),
                block_size=(block_size_i,),
                scale_output=id(scale),
                scale_output_tensor=scale,
                quant_axis=-1 if axis is None else int(axis),
                quant_transpose=transpose,
            )
        )
        state["tensor_meta"][id(quantized)] = _TensorMeta(
            name=f"{name}::OUT_0",
            dim=in_meta.dim if in_meta else (),
            stride=in_meta.stride if in_meta else (),
            dtype="fp32",
        )
        scale_dim: tuple[int, ...] = ()
        scale_stride: tuple[int, ...] = ()
        if in_meta is not None and len(in_meta.dim) == 3:
            bb, m, n = in_meta.dim
            scale_n = (int(n) + block_size_i - 1) // block_size_i
            scale_dim = (int(bb), int(m), scale_n)
            scale_stride = (int(m) * scale_n, scale_n, 1)
        state["tensor_meta"][id(scale)] = _TensorMeta(
            name=f"{name}::OUT_1",
            dim=scale_dim,
            stride=scale_stride,
            dtype="fp8_e8m0",
        )
    except Exception:  # noqa: BLE001
        pass
    return out


_MOE_MODE_FROM_CUDNN: dict[Any, str] = {
    cudnn.moe_grouped_matmul_mode.NONE: "none",
    cudnn.moe_grouped_matmul_mode.GATHER: "gather",
    cudnn.moe_grouped_matmul_mode.SCATTER: "scatter",
}

_REDUCTION_MODE_FROM_CUDNN: dict[Any, str] = {
    cudnn.reduction_mode.ADD: "add",
    cudnn.reduction_mode.AMAX: "amax",
    cudnn.reduction_mode.MAX: "max",
    cudnn.reduction_mode.MIN: "min",
}


def _patched_moe_grouped_matmul(self, *args, **kwargs):
    out = _ORIGINALS["moe_grouped_matmul"](self, *args, **kwargs)
    try:
        b = _bind(args, kwargs, ("token", "weight", "first_token_offset", "token_index", "token_ks", "mode"))
        name = b.get("name", "")
        mode = b.get("mode", cudnn.moe_grouped_matmul_mode.NONE)
        state = _ensure_state(self)
        state["ops"].append(
            _RecordedOp(
                "moe_grouped_matmul",
                name,
                [id(b["token"]), id(b["weight"]), id(b["first_token_offset"])],
                id(out),
                out,
                compute_dtype=_opt_compute_dtype(kwargs),
                moe_mode=_MOE_MODE_FROM_CUDNN.get(mode, "none"),
            )
        )
        state["tensor_meta"][id(out)] = _TensorMeta(name=f"{name}::OUT_0", dim=(), stride=(), dtype="fp32")
    except Exception:  # noqa: BLE001
        pass
    return out


def _patched_reduction(self, *args, **kwargs):
    out = _ORIGINALS["reduction"](self, *args, **kwargs)
    try:
        b = _bind(args, kwargs, ("input", "mode"))
        mode = b.get("mode")
        name = b.get("name", "")
        group_offset = b.get("group_offset")
        state = _ensure_state(self)
        # Record every reduction (mode mapped to our literal, or None when GEMM
        # doesn't support it). Unsupported modes are rejected later, in analyze()
        # — never here, so a non-GEMM graph using an unsupported mode still builds.
        state["ops"].append(
            _RecordedOp(
                "reduction",
                name,
                [id(b["input"])],
                id(out),
                out,
                compute_dtype=_opt_compute_dtype(kwargs),
                reduction_mode=_REDUCTION_MODE_FROM_CUDNN.get(mode),
                group_offset=(id(group_offset) if group_offset is not None else None),
            )
        )
        state["tensor_meta"][id(out)] = _TensorMeta(name=f"{name}::OUT_0", dim=(), stride=(), dtype="fp32")
    except Exception:  # noqa: BLE001
        pass
    return out


def _make_unary_patch(cudnn_name: str):
    def patched(self, *args, **kwargs):
        out = _ORIGINALS[cudnn_name](self, *args, **kwargs)
        try:
            b = _bind(args, kwargs, ("input",))
            name = b.get("name", "")
            state = _ensure_state(self)
            state["ops"].append(_RecordedOp(cudnn_name, name, [id(b["input"])], id(out), out, compute_dtype=_opt_compute_dtype(kwargs)))
            state["tensor_meta"][id(out)] = _TensorMeta(name=f"{name}::OUT_0", dim=(), stride=(), dtype="fp32")
        except Exception:  # noqa: BLE001
            pass
        return out

    return patched


def _make_binary_patch(cudnn_name: str, *, a_kw: str = "a", b_kw: str = "b"):
    def patched(self, *args, **kwargs):
        out = _ORIGINALS[cudnn_name](self, *args, **kwargs)
        try:
            bnd = _bind(args, kwargs, (a_kw, b_kw))
            name = bnd.get("name", "")
            state = _ensure_state(self)
            state["ops"].append(_RecordedOp(cudnn_name, name, [id(bnd[a_kw]), id(bnd[b_kw])], id(out), out, compute_dtype=_opt_compute_dtype(kwargs)))
            state["tensor_meta"][id(out)] = _TensorMeta(name=f"{name}::OUT_0", dim=(), stride=(), dtype="fp32")
        except Exception:  # noqa: BLE001
            pass
        return out

    return patched


def _patched_tensor_set_output(self, val):
    _TENSOR_OUTPUT_FLAG[id(self)] = bool(val)
    return _ORIGINALS["tensor.set_output"](self, val)


def _patched_tensor_set_data_type(self, dt):
    _TENSOR_EXPLICIT_DTYPE[id(self)] = dt
    return _ORIGINALS["tensor.set_data_type"](self, dt)


def _patched_tensor_set_dim(self, dim):
    _TENSOR_DIM_OVERRIDE[id(self)] = tuple(dim)
    return _ORIGINALS["tensor.set_dim"](self, dim)


def _patched_tensor_set_reordering_type(self, rt):
    _TENSOR_REORDERING_OVERRIDE[id(self)] = rt.name if getattr(rt, "name", "NONE") != "NONE" else None
    return _ORIGINALS["tensor.set_reordering_type"](self, rt)


# ---------------------------------------------------------------------------
# GEMM engine (named "TBD_eng0") — registered with the shared cudnn.TBD dispatch
# (see cudnn/TBD/heuristics.py). ``probe_gemm_plan`` decides eligibility (no
# compile); ``build_gemm_plan`` JIT-compiles when the engine is selected. The
# engine selects its own config; forced-config callers use jit_from_cudnn_graph.
# ---------------------------------------------------------------------------


def probe_gemm_plan(graph: cudnn.pygraph) -> bool:
    """Cheap eligibility check for the ``TBD_eng0`` GEMM engine — analyze + the
    support gates, NO ``cute.compile``. Returns True if the GEMM engine should be
    listed in the plan list for this graph. Never raises (a probe must not break
    the native path)."""
    state = _get_state(graph)
    if state is None or not state.get("ops"):
        return False  # graph wasn't recorded by the hook → not a GEMM candidate
    from .compiler import probe_supported

    try:
        probe_supported(graph)
    except (NotImplementedError, ValueError):
        return False
    except Exception:  # noqa: BLE001
        _LOG.debug("cudnn.TBD.gemm: probe_supported raised unexpectedly; ineligible", exc_info=True)
        return False
    return True


def build_gemm_plan(graph: cudnn.pygraph):
    """Analyze + JIT the recorded graph into a compiled GEMM plan.

    Returns a callable :class:`CompiledFusedGemm` on success; raises
    ``NotImplementedError`` / ``ValueError`` (type + message preserved) when the
    engine rejects the graph — same as calling ``jit_from_cudnn_graph`` directly.
    Raises ``ValueError`` if the graph was never recorded (import-order error)."""
    state = _get_state(graph)
    if state is None or not state.get("ops"):
        raise ValueError(
            "cudnn.TBD.gemm: this graph has no recorded ops — import "
            "cudnn.TBD.gemm BEFORE constructing the graph so the recorder hook "
            "is installed before g = cudnn.pygraph(...). The op chain cannot be "
            "reconstructed after the fact."
        )
    from .compiler import jit_from_cudnn_graph

    return jit_from_cudnn_graph(graph)


def install_recorder() -> None:
    """Monkey-patch ``cudnn.pygraph`` to record op chains. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    _ORIGINALS["__init__"] = cudnn.pygraph.__init__
    _ORIGINALS["tensor"] = cudnn.pygraph.tensor
    _ORIGINALS["matmul"] = cudnn.pygraph.matmul
    _ORIGINALS["block_scale_dequantize"] = cudnn.pygraph.block_scale_dequantize
    _ORIGINALS["block_scale_quantize"] = cudnn.pygraph.block_scale_quantize
    _ORIGINALS["moe_grouped_matmul"] = cudnn.pygraph.moe_grouped_matmul
    _ORIGINALS["reduction"] = cudnn.pygraph.reduction
    cudnn.pygraph.__init__ = _patched_init
    cudnn.pygraph.tensor = _patched_tensor
    cudnn.pygraph.matmul = _patched_matmul
    cudnn.pygraph.block_scale_dequantize = _patched_block_scale_dequantize
    cudnn.pygraph.block_scale_quantize = _patched_block_scale_quantize
    cudnn.pygraph.moe_grouped_matmul = _patched_moe_grouped_matmul
    cudnn.pygraph.reduction = _patched_reduction

    for cudnn_name in _UNARY_OP_MAP:
        _ORIGINALS[cudnn_name] = getattr(cudnn.pygraph, cudnn_name)
        setattr(cudnn.pygraph, cudnn_name, _make_unary_patch(cudnn_name))

    for cudnn_name in _BINARY_OP_MAP:
        _ORIGINALS[cudnn_name] = getattr(cudnn.pygraph, cudnn_name)
        if cudnn_name == "bias":
            setattr(
                cudnn.pygraph,
                cudnn_name,
                _make_binary_patch(cudnn_name, a_kw="input", b_kw="bias"),
            )
        elif cudnn_name in {"max", "min", "pow"}:
            setattr(
                cudnn.pygraph,
                cudnn_name,
                _make_binary_patch(cudnn_name, a_kw="input0", b_kw="input1"),
            )
        else:
            setattr(cudnn.pygraph, cudnn_name, _make_binary_patch(cudnn_name))

    # Tensor-class patches: set_output / set_data_type have no getters in
    # cudnn-frontend, so we wrap the setters and stash the flags in side-tables.
    from cudnn import _compiled_module as _cudnn_module

    tensor_cls = _cudnn_module.tensor
    _ORIGINALS["tensor.set_output"] = tensor_cls.set_output
    _ORIGINALS["tensor.set_data_type"] = tensor_cls.set_data_type
    _ORIGINALS["tensor.set_dim"] = tensor_cls.set_dim
    _ORIGINALS["tensor.set_reordering_type"] = tensor_cls.set_reordering_type
    tensor_cls.set_output = _patched_tensor_set_output
    tensor_cls.set_data_type = _patched_tensor_set_data_type
    tensor_cls.set_dim = _patched_tensor_set_dim
    tensor_cls.set_reordering_type = _patched_tensor_set_reordering_type

    _INSTALLED = True


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _infer_bcast_mode(matmul_out_dim: tuple[int, ...], aux_dim: tuple[int, ...]) -> str:
    """Infer how an aux tensor broadcasts onto the matmul output."""
    if len(aux_dim) > len(matmul_out_dim):
        raise ValueError(f"aux dim rank {len(aux_dim)} exceeds matmul-out rank " f"{len(matmul_out_dim)}: aux={aux_dim} out={matmul_out_dim}")
    aux_norm = (1,) * (len(matmul_out_dim) - len(aux_dim)) + tuple(aux_dim)
    for aux_extent, out_extent in zip(aux_norm, matmul_out_dim):
        if aux_extent not in (1, out_extent):
            raise ValueError(f"aux dim {aux_dim} is not broadcast-compatible with " f"matmul output dim {matmul_out_dim}")
    bcast_m = aux_norm[-2] == 1 and matmul_out_dim[-2] != 1
    bcast_n = aux_norm[-1] == 1 and matmul_out_dim[-1] != 1
    if bcast_m and bcast_n:
        return "scalar"
    if bcast_m and not bcast_n:
        return "per_col"
    if bcast_n and not bcast_m:
        return "per_row"
    return "per_elem"


def _infer_a_major(dim: tuple[int, ...], stride: tuple[int, ...]) -> AMajor:
    if stride[-1] == 1:
        return "k"
    if stride[-2] == 1:
        return "m"
    raise ValueError(f"A must be K-major or M-major in the inner (M,K) plane; " f"got dim={dim} stride={stride}")


def _infer_b_major(dim: tuple[int, ...], stride: tuple[int, ...]) -> BMajor:
    if stride[-2] == 1:
        return "k"
    if stride[-1] == 1:
        return "n"
    raise ValueError(f"B must be K-major or N-major in the inner (K,N) plane; " f"got dim={dim} stride={stride}")


def _infer_out_major(dim: tuple[int, ...], stride: tuple[int, ...]) -> OutMajor:
    if not stride:
        return "n"
    if stride[-1] == 1:
        return "n"
    if stride[-2] == 1:
        return "m"
    raise ValueError(f"output must be N-major or M-major in the inner (M,N) plane; " f"got dim={dim} stride={stride}")


def _resolve_out_dtype(
    out_id: int,
    output_tensor: Any,
    io_dtype: Dtype,
    intermediate_dtype: Dtype,
) -> Dtype:
    """Declared data_type of a chain tensor, by cuDNN's rules:
      1. explicit ``set_data_type(...)``  → that dtype
      2. else, if it's a materialized output (``set_output(True)``) → io_dtype
      3. else (pure virtual / intermediate) → intermediate_dtype

    The running value is rounded to this dtype before downstream ops read it,
    so a narrow declared dtype loses precision on purpose (matches cuDNN even
    for virtual tensors)."""
    explicit = _TENSOR_EXPLICIT_DTYPE.get(out_id)
    if explicit is not None and explicit in _DTYPE_FROM_CUDNN:
        return _DTYPE_FROM_CUDNN[explicit]
    if output_tensor is not None:
        try:
            dt = output_tensor.get_data_type()
        except Exception:  # noqa: BLE001 — defensive: unbuilt tensors vary
            dt = None
        if dt is not None and dt != cudnn.data_type.NOT_SET and dt in _DTYPE_FROM_CUDNN:
            return _DTYPE_FROM_CUDNN[dt]
    if _TENSOR_OUTPUT_FLAG.get(out_id, False):
        return io_dtype
    return intermediate_dtype


def _build_moe_chain(
    moe_ops: list[_RecordedOp],
    ops: list[_RecordedOp],
    meta: dict[int, _TensorMeta],
    io_dtype: Dtype,
    intermediate_dtype: Dtype,
    compute_dtype: Dtype,
) -> FusionChain:
    """Build a FusionChain for a MoE grouped matmul forward pass.

    ``out[first_token_offset[g] : first_token_offset[g+1]] =
        token[range] @ weight[g % E].T`` per routed group g.

    POC scope: one ``moe_grouped_matmul`` per graph, ``mode == "none"`` only,
    with an optional terminal ``block_scale_quantize`` epilogue. token = A
    ``(1, M=T, K=H)``, weight = B ``(E, K=H, N)``; the per-expert selection is
    carried in :class:`MoeSpec` (num_experts), NOT via the MatmulSpec batch
    machinery (output is a single ``(1, T, N)`` plane)."""
    from .fusion_ir import MoeSpec

    if len(moe_ops) != 1 or len([o for o in ops if o.cudnn_name == "matmul"]):
        raise ValueError("POC scope is exactly one moe_grouped_matmul per graph and no plain " f"matmul; found {len(moe_ops)} moe op(s)")
    moe = moe_ops[0]
    if moe.moe_mode != "none":
        raise NotImplementedError(f"MoE grouped matmul mode {moe.moe_mode!r} is out of POC scope; " "only mode=NONE is supported (gather / scatter rejected)")
    token_id, weight_id, fto_id = moe.inputs
    fto_meta = meta.get(fto_id)
    # first_token_offset dtype (int32 / int64) — both are valid cuDNN inputs;
    # baked into the kernel. Default int32 if the tensor wasn't a graph input.
    offset_dtype = fto_meta.dtype if fto_meta is not None else "int32"
    num_groups = int(fto_meta.dim[0]) if fto_meta is not None and fto_meta.dim else 1

    # ----- Block-scaled MoE detection (structural pattern-match) -------------
    # If the moe op's token / weight come from block_scale_dequantize nodes, fold
    # the dequant(s) + moe into one block-scale MoE matmul: redirect token/weight
    # to the packed (FP4 / FP8) data tensors and capture their SFA/SFB. Mirrors
    # the plain-matmul block-scale detection in _build_chain; NO validation here
    # (the compiler decides which combos run). The dims/major come from the
    # dequant-output meta (which inherits the packed input's dim/stride); the MMA
    # input dtype is the packed data dtype (captured below).
    from .fusion_ir import BlockScaleSpec

    dequant_by_output = {op.output: op for op in ops if op.cudnn_name == "block_scale_dequantize"}
    block_scale_spec: "BlockScaleSpec | None" = None
    a_dtype = meta[token_id].dtype
    b_dtype = meta[weight_id].dtype
    if token_id in dequant_by_output or weight_id in dequant_by_output:

        def _capture_side(operand_id: int):
            deq = dequant_by_output.get(operand_id)
            if deq is None:
                return dict(data_dtype=meta[operand_id].dtype, block_size_2d=None, sf_dtype=None, sf_reorder=None, deq_compute=None, deq_out=None)
            data_id, sf_id = deq.inputs
            sf_meta = meta[sf_id]
            deq_compute = deq.compute_dtype if deq.compute_dtype is not None else compute_dtype
            deq_out = _resolve_out_dtype(deq.output, deq.output_tensor, io_dtype, intermediate_dtype)
            return dict(
                data_dtype=meta[data_id].dtype,
                block_size_2d=(tuple(deq.block_size) if deq.block_size else None),
                sf_dtype=sf_meta.dtype,
                sf_reorder=sf_meta.reordering,
                deq_compute=deq_compute,
                deq_out=deq_out,
            )

        a = _capture_side(token_id)
        b = _capture_side(weight_id)
        block_scale_spec = BlockScaleSpec(
            a_dtype=a["data_dtype"],
            b_dtype=b["data_dtype"],
            block_size_a=a["block_size_2d"],
            block_size_b=b["block_size_2d"],
            sf_dtype_a=a["sf_dtype"],
            sf_dtype_b=b["sf_dtype"],
            sfa_reorder=a["sf_reorder"],
            sfb_reorder=b["sf_reorder"],
            dequant_compute_a=a["deq_compute"],
            dequant_compute_b=b["deq_compute"],
            dequant_out_a=a["deq_out"],
            dequant_out_b=b["deq_out"],
        )
        a_dtype, b_dtype = a["data_dtype"], b["data_dtype"]

    token_meta = meta[token_id]
    weight_meta = meta[weight_id]
    if len(token_meta.dim) != 3 or len(weight_meta.dim) != 3:
        raise ValueError(f"moe operands must be 3D; got token={token_meta.dim} " f"weight={weight_meta.dim}")
    # token [1, T, H] → A=(1, M=T, K=H). weight [E, H, N] → B=(E, K=H, N).
    _bt, M, Ka = token_meta.dim
    E, Kb, N = weight_meta.dim
    if Ka != Kb:
        raise ValueError(f"moe K mismatch: token K={Ka} (hidden) vs weight K={Kb}")
    set_output_ids_in_order = [tid for tid in _TENSOR_OUTPUT_FLAG if _TENSOR_OUTPUT_FLAG[tid]]
    quant_ops = [op for op in ops if op.cudnn_name == "block_scale_quantize" and len(op.inputs) == 1 and op.inputs[0] == moe.output]
    terminal_quant: _RecordedOp | None = None
    for tid in reversed(set_output_ids_in_order):
        qop = next((q for q in quant_ops if q.output == tid), None)
        if qop is not None:
            terminal_quant = qop
            break

    terminal_tensor = terminal_quant.output_tensor if terminal_quant is not None else moe.output_tensor
    terminal_id = terminal_quant.output if terminal_quant is not None else moe.output
    output_dtype = _resolve_out_dtype(terminal_id, terminal_tensor, io_dtype, intermediate_dtype)
    term_dim = tuple(terminal_tensor.get_dim()) if terminal_tensor is not None else ()
    term_stride = tuple(terminal_tensor.get_stride()) if terminal_tensor is not None else ()
    if terminal_quant is not None and (not term_dim or not term_stride):
        term_meta = meta.get(terminal_quant.output)
        if term_meta is not None:
            term_dim = term_meta.dim
            term_stride = term_meta.stride

    mm_compute = moe.compute_dtype if moe.compute_dtype is not None else compute_dtype
    matmul_out_dtype = _resolve_out_dtype(moe.output, moe.output_tensor, io_dtype, intermediate_dtype)
    # b_major over the inner (K, N) plane of the weight tensor.
    matmul_spec = MatmulSpec(
        M=int(M),
        N=int(N),
        K=int(Ka),
        batch=1,
        a_batch=1,
        b_batch=1,
        a_major=_infer_a_major(token_meta.dim, token_meta.stride),
        b_major=_infer_b_major(weight_meta.dim, weight_meta.stride),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        accum_dtype=mm_compute,
        out_dtype=matmul_out_dtype,
        out_major=_infer_out_major(term_dim, term_stride),
    )
    full_dim = (1, int(M), int(N))

    def _reduction_output(red: _RecordedOp) -> tuple[tuple[int, int, int], bool]:
        dim = _TENSOR_DIM_OVERRIDE.get(red.output)
        if dim is None:
            try:
                dim = tuple(red.output_tensor.get_dim())
            except Exception:  # noqa: BLE001
                dim = ()
        if len(dim) != 3:
            raise ValueError(f"reduction {red.op_name!r} must set a rank-3 output dim; got {dim}")
        grouped_by_moe = False
        if red.group_offset is not None:
            if red.group_offset != fto_id:
                raise ValueError(f"reduction {red.op_name!r} groupOffset must be the MoE " "first_token_offset tensor")
            if int(dim[0]) != num_groups:
                raise ValueError(f"reduction {red.op_name!r} with groupOffset must use " f"output dim[0] == num_groups ({num_groups}); got {dim}")
            grouped_by_moe = True
        axis0_extent = num_groups if grouped_by_moe else full_dim[0]
        compat_full = (axis0_extent, full_dim[1], full_dim[2])
        for axis, (out_extent, full_extent) in enumerate(zip(dim, compat_full)):
            if out_extent not in (1, full_extent):
                raise ValueError(
                    f"reduction {red.op_name!r} output dim {dim} is not compatible " f"with moe output {full_dim}: axis {axis} must be 1 or {full_extent}"
                )
        if all(out_extent == full_extent for out_extent, full_extent in zip(dim, compat_full)):
            raise ValueError(f"reduction {red.op_name!r} output dim {dim} does not reduce any axis")
        return (int(dim[0]), int(dim[1]), int(dim[2])), grouped_by_moe

    reductions: list[ReductionSpec] = []
    reduction_objs: list[Any] = []
    for red in ops:
        if red.cudnn_name != "reduction":
            continue
        if not _TENSOR_OUTPUT_FLAG.get(red.output, False):
            continue
        (input_id,) = red.inputs
        if input_id != moe.output:
            raise ValueError(f"reduction {red.op_name!r} input is not produced by this " "MoE epilogue chain")
        compute = red.compute_dtype if red.compute_dtype is not None else compute_dtype
        dtype = _resolve_out_dtype(red.output, red.output_tensor, io_dtype, intermediate_dtype)
        if red.reduction_mode is None:
            raise NotImplementedError(f"reduction {red.op_name!r} mode is not supported by cudnn.TBD.gemm; " "supported modes are ADD, AMAX, MAX, and MIN")
        red_dim, grouped_by_moe = _reduction_output(red)
        reductions.append(
            ReductionSpec(
                mode=red.reduction_mode,  # type: ignore[arg-type]
                source_ref=gemm_source(0),
                dim=red_dim,
                dtype=dtype,
                compute_dtype=compute,
                grouped_by_moe=grouped_by_moe,
            )
        )
        reduction_objs.append(red.output_tensor)

    block_quant: BlockQuantizeSpec | None = None
    quant_scale_obj: Any = None
    if terminal_quant is not None:
        if terminal_quant.scale_output is None or terminal_quant.scale_output_tensor is None:
            raise AssertionError("block_scale_quantize recorded without scale output")
        if not _TENSOR_OUTPUT_FLAG.get(terminal_quant.scale_output, False):
            raise ValueError("block_scale_quantize scale output must be materialized with set_output(True)")
        scale_dtype = _resolve_out_dtype(
            terminal_quant.scale_output,
            terminal_quant.scale_output_tensor,
            io_dtype,
            intermediate_dtype,
        )
        scale_reorder = _TENSOR_REORDERING_OVERRIDE.get(terminal_quant.scale_output)
        if scale_reorder is None:
            scale_meta = meta.get(terminal_quant.scale_output)
            scale_reorder = scale_meta.reordering if scale_meta is not None else None
        bs = int(terminal_quant.block_size[0]) if terminal_quant.block_size else 0
        if bs <= 0:
            raise ValueError(f"block_scale_quantize block_size must be positive; got {bs}")
        if int(N) % bs != 0:
            raise ValueError(f"block_scale_quantize requires N divisible by block_size; got N={N}, block_size={bs}")
        logical_scale_dim = (1, int(M), int(N) // bs)
        expected_scale_dim = logical_scale_dim
        if scale_reorder == "F8_128x4":
            expected_scale_dim = (
                logical_scale_dim[0],
                _round_up(logical_scale_dim[1], 128),
                _round_up(logical_scale_dim[2], 4),
            )
        scale_dim = _TENSOR_DIM_OVERRIDE.get(terminal_quant.scale_output)
        if scale_dim is None:
            scale_meta = meta.get(terminal_quant.scale_output)
            scale_dim = scale_meta.dim if scale_meta is not None else ()
        if not scale_dim:
            scale_dim = expected_scale_dim
        if len(scale_dim) != 3:
            raise ValueError(f"block_scale_quantize scale output must be rank-3; got {scale_dim}")
        if tuple(scale_dim) != expected_scale_dim:
            raise ValueError(f"block_scale_quantize scale dim must be {expected_scale_dim}; got {scale_dim}")
        compute = terminal_quant.compute_dtype if terminal_quant.compute_dtype is not None else compute_dtype
        block_quant = BlockQuantizeSpec(
            source_ref=gemm_source(0),
            block_size=bs,
            axis=-1 if terminal_quant.quant_axis is None else terminal_quant.quant_axis,
            transpose=terminal_quant.quant_transpose,
            scale_dtype=scale_dtype,
            scale_dim=tuple(scale_dim),
            scale_reorder=scale_reorder,
            compute_dtype=compute,
        )
        quant_scale_obj = terminal_quant.scale_output_tensor

    chain = FusionChain(
        matmul=matmul_spec,
        output_dtype=output_dtype,
        moe=MoeSpec(num_experts=int(E), mode=moe.moe_mode, offset_dtype=offset_dtype),
        block_scale=block_scale_spec,
        reductions=reductions,
        block_quant=block_quant,
    )
    # Binding: token/weight resolve through any block_scale_dequantize to the
    # packed data tensors (the user passes the packed token/weight); the SF is
    # the dequant's 2nd input. The terminal output is either the raw MoE output
    # or the quantized tensor, with the quant scale as a side output.
    deq_tok = dequant_by_output.get(token_id)
    deq_w = dequant_by_output.get(weight_id)
    a_data_id = deq_tok.inputs[0] if deq_tok else token_id
    b_data_id = deq_w.inputs[0] if deq_w else weight_id
    binding = GemmBinding(
        a_operands=[meta[a_data_id].tensor],
        b_operands=[meta[b_data_id].tensor],
        outputs=([terminal_tensor] + reduction_objs + ([quant_scale_obj] if quant_scale_obj is not None else [])),
        first_token_offset=meta[fto_id].tensor,
        sfa_operands=([meta[deq_tok.inputs[1]].tensor] if deq_tok else [None]) if block_scale_spec is not None else [],
        sfb_operands=([meta[deq_w.inputs[1]].tensor] if deq_w else [None]) if block_scale_spec is not None else [],
    )
    return chain, binding


def _build_multi_moe_chain(
    moe_ops: list[_RecordedOp],
    ops: list[_RecordedOp],
    meta: dict[int, _TensorMeta],
    io_dtype: Dtype,
    intermediate_dtype: Dtype,
    compute_dtype: Dtype,
) -> FusionChain:
    """Build a FusionChain for K parallel MoE grouped matmuls sharing one
    ``first_token_offset`` and one pointwise epilogue DAG (e.g. grouped SwiGLU
    ``silu(tok @ w0) * (tok @ w1) * scale``).

    Each ``moe_grouped_matmul`` op is a GEMM; they must share the routed-group
    layout (same ``first_token_offset``), the same shape / major / dtype, and
    the same expert count. Token / weight operands are deduped by tensor id, so
    a shared token collapses to one distinct A operand. The MoE specifics (the
    per-expert weight selection + group ranges) live in :class:`MoeSpec`; the
    matmul output is a single ``(1, S, N)`` plane. POC scope: ``mode == "none"``,
    no mainloop fusion; the terminal must be a fusion op. Block-scale is supported
    (a ``block_scale_dequantize`` feeding the moe token / weight folds into a
    shared :class:`BlockScaleSpec`; a shared dequant collapses to one distinct
    operand, exactly like the plain-matmul multi-GEMM case)."""
    from .fusion_ir import BlockScaleSpec, MoeSpec

    for moe in moe_ops:
        if moe.moe_mode != "none":
            raise NotImplementedError(
                f"MoE grouped matmul mode {moe.moe_mode!r} is out of POC scope; " "only mode=NONE is supported (gather / scatter rejected)"
            )

    # All GEMMs must share the SAME first_token_offset (so they have identical
    # routed-group layout → identical output shape per group).
    fto_id = moe_ops[0].inputs[2]
    for moe in moe_ops[1:]:
        if moe.inputs[2] != fto_id:
            raise ValueError("parallel MoE grouped matmuls must share the same " "first_token_offset tensor")
    fto_meta = meta.get(fto_id)
    offset_dtype = fto_meta.dtype if fto_meta is not None else "int32"
    num_groups = int(fto_meta.dim[0]) if fto_meta is not None and fto_meta.dim else 1

    # ----- Resolve each moe operand through any block_scale_dequantize, then
    # dedup by the PACKED data tensor id (shared dequant → one distinct operand,
    # the SF travels with its data). Mirrors _build_multi_gemm_chain.
    dequant_by_output = {op.output: op for op in ops if op.cudnn_name == "block_scale_dequantize"}

    def _capture_side(operand_id: int) -> dict:
        deq = dequant_by_output.get(operand_id)
        if deq is None:
            return dict(
                data_id=operand_id,
                data_dtype=meta[operand_id].dtype,
                block_size_2d=None,
                sf_dtype=None,
                sf_reorder=None,
                deq_compute=None,
                deq_out=None,
                sf_id=None,
            )
        data_id, sf_id = deq.inputs
        sf_meta = meta[sf_id]
        deq_compute = deq.compute_dtype if deq.compute_dtype is not None else compute_dtype
        deq_out = _resolve_out_dtype(deq.output, deq.output_tensor, io_dtype, intermediate_dtype)
        return dict(
            data_id=data_id,
            data_dtype=meta[data_id].dtype,
            block_size_2d=(tuple(deq.block_size) if deq.block_size else None),
            sf_dtype=sf_meta.dtype,
            sf_reorder=sf_meta.reordering,
            deq_compute=deq_compute,
            deq_out=deq_out,
            sf_id=sf_id,
        )

    a_ids: list[int] = []  # distinct PACKED token (A) data ids
    b_ids: list[int] = []  # distinct PACKED weight (B) data ids
    a_caps: dict[int, dict] = {}
    b_caps: dict[int, dict] = {}
    gemm_operands: list[tuple[int, int]] = []
    for moe in moe_ops:
        a_cap = _capture_side(moe.inputs[0])
        b_cap = _capture_side(moe.inputs[1])
        a_pid, b_pid = a_cap["data_id"], b_cap["data_id"]
        if a_pid not in a_ids:
            a_ids.append(a_pid)
            a_caps[a_pid] = a_cap
        if b_pid not in b_ids:
            b_ids.append(b_pid)
            b_caps[b_pid] = b_cap
        gemm_operands.append((a_ids.index(a_pid), b_ids.index(b_pid)))

    is_block_scale = any(c["sf_dtype"] is not None for c in (*a_caps.values(), *b_caps.values()))

    def _moe_geometry(token_id: int, weight_id: int):
        token_meta = meta[token_id]
        weight_meta = meta[weight_id]
        if len(token_meta.dim) != 3 or len(weight_meta.dim) != 3:
            raise ValueError(f"moe operands must be 3D; got token={token_meta.dim} " f"weight={weight_meta.dim}")
        _bt, M, Ka = token_meta.dim  # token [1, T, H]
        E, Kb, N = weight_meta.dim  # weight [E, H, N]
        if Ka != Kb:
            raise ValueError(f"moe K mismatch: token K={Ka} vs weight K={Kb}")
        return (
            int(M),
            int(N),
            int(Ka),
            int(E),
            _infer_a_major(token_meta.dim, token_meta.stride),
            _infer_b_major(weight_meta.dim, weight_meta.stride),
            token_meta.dtype,
            weight_meta.dtype,
        )

    geom0 = _moe_geometry(a_ids[gemm_operands[0][0]], b_ids[gemm_operands[0][1]])
    for ai, bi in gemm_operands[1:]:
        if _moe_geometry(a_ids[ai], b_ids[bi]) != geom0:
            raise ValueError("parallel MoE grouped matmuls must share shape / layout / dtype " "/ expert count; heterogeneous GEMMs are out of POC scope")
    M, N, K, E, a_major, b_major, a_dtype, b_dtype = geom0
    matmul_out_dim = (1, M, N)

    # ----- Shared BlockScaleSpec (every distinct operand same combo) --------
    block_scale_spec = None
    if is_block_scale:
        a0 = a_caps[a_ids[gemm_operands[0][0]]]
        b0 = b_caps[b_ids[gemm_operands[0][1]]]

        def _combo_key(cap):
            return (cap["data_dtype"], cap["block_size_2d"], cap["sf_dtype"], cap["sf_reorder"], cap["deq_compute"], cap["deq_out"])

        for cap in a_caps.values():
            if _combo_key(cap) != _combo_key(a0):
                raise ValueError("all token operands of a block-scale multi-MoE must share the same SF combo")
        for cap in b_caps.values():
            if _combo_key(cap) != _combo_key(b0):
                raise ValueError("all weight operands of a block-scale multi-MoE must share the same SF combo")
        block_scale_spec = BlockScaleSpec(
            a_dtype=a0["data_dtype"],
            b_dtype=b0["data_dtype"],
            block_size_a=a0["block_size_2d"],
            block_size_b=b0["block_size_2d"],
            sf_dtype_a=a0["sf_dtype"],
            sf_dtype_b=b0["sf_dtype"],
            sfa_reorder=a0["sf_reorder"],
            sfb_reorder=b0["sf_reorder"],
            dequant_compute_a=a0["deq_compute"],
            dequant_compute_b=b0["deq_compute"],
            dequant_out_a=a0["deq_out"],
            dequant_out_b=b0["deq_out"],
        )
    mm_compute = moe_ops[0].compute_dtype if moe_ops[0].compute_dtype is not None else compute_dtype

    # ----- Epilogue DAG over MULTIPLE roots (each MoE GEMM output) ----------
    gemm_idx_by_output: dict[int, int] = {mm.output: g for g, mm in enumerate(moe_ops)}
    consumers_by_input: dict[int, list[_RecordedOp]] = {}
    for op in ops:
        for inp in op.inputs:
            consumers_by_input.setdefault(inp, []).append(op)

    reachable_op_ids: set[int] = set()
    bfs_queue: list[int] = [mm.output for mm in moe_ops]
    visited_tensors: set[int] = set()
    while bfs_queue:
        tid = bfs_queue.pop(0)
        if tid in visited_tensors:
            continue
        visited_tensors.add(tid)
        for op in consumers_by_input.get(tid, []):
            if op.cudnn_name == "moe_grouped_matmul":
                continue
            if op.output not in reachable_op_ids:
                reachable_op_ids.add(op.output)
                bfs_queue.append(op.output)
    reachable_ops = [op for op in ops if op.output in reachable_op_ids and op.cudnn_name not in {"moe_grouped_matmul", "matmul", "reduction"}]

    def _is_in_chain(tid: int) -> bool:
        return tid in gemm_idx_by_output or tid in reachable_op_ids

    in_chain_deps: dict[int, list[int]] = {op.output: [inp for inp in op.inputs if _is_in_chain(inp)] for op in reachable_ops}
    placed: set[int] = set(gemm_idx_by_output)
    remaining = list(reachable_ops)
    ordered_ops: list[_RecordedOp] = []
    while remaining:
        ready_idx = next(
            (i for i, op in enumerate(remaining) if all(d in placed for d in in_chain_deps[op.output])),
            None,
        )
        if ready_idx is None:
            raise AssertionError(f"cycle / unsatisfiable deps: {[op.op_name for op in remaining]}")
        op = remaining.pop(ready_idx)
        ordered_ops.append(op)
        placed.add(op.output)

    aux_tensors: list[TensorRef] = []
    aux_objs: list[Any] = []
    aux_seen: set[int] = set()
    op_position_by_id: dict[int, int] = {}

    def _operand_ref(tid: int) -> int:
        if tid in gemm_idx_by_output:
            return gemm_source(gemm_idx_by_output[tid])
        return op_position_by_id[tid]

    pending_ops: list[tuple[FusionOp, int]] = []
    for next_op in ordered_ops:
        if next_op.cudnn_name in _UNARY_OP_MAP:
            (parent_id,) = next_op.inputs
            fop = FusionOp(op=_UNARY_OP_MAP[next_op.cudnn_name], parent_idx=_operand_ref(parent_id))
        elif next_op.cudnn_name in _BINARY_OP_MAP:
            inp0, inp1 = next_op.inputs
            in0, in1 = _is_in_chain(inp0), _is_in_chain(inp1)
            if in0 and in1:
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=None,
                    aux_on_rhs=True,
                    parent_idx=_operand_ref(inp0),
                    parent_idx_b=_operand_ref(inp1),
                )
            elif in0 or in1:
                if in0:
                    chain_id, aux_id, aux_on_rhs = inp0, inp1, True
                else:
                    chain_id, aux_id, aux_on_rhs = inp1, inp0, False
                aux_meta = meta[aux_id]
                if not aux_meta.is_input:
                    raise ValueError(f"aux input {aux_meta.name!r} of op {next_op.op_name!r} is " "not a graph input — POC supports only graph-input aux")
                if aux_id not in aux_seen:
                    aux_seen.add(aux_id)
                    bcast = _infer_bcast_mode(matmul_out_dim, aux_meta.dim)
                    aux_tensors.append(
                        TensorRef(
                            name=aux_meta.name,
                            dim=aux_meta.dim,
                            stride=aux_meta.stride,
                            dtype=aux_meta.dtype,
                            bcast_mode=bcast,
                        )
                    )
                    aux_objs.append(aux_meta.tensor)
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=aux_meta.name,
                    aux_on_rhs=aux_on_rhs,
                    parent_idx=_operand_ref(chain_id),
                )
            else:
                raise ValueError(f"binary op {next_op.op_name!r} has no in-chain operand")
        else:
            raise ValueError(f"op {next_op.cudnn_name!r} (name={next_op.op_name!r}) is not in " "the POC pointwise subset; out-of-scope")
        pending_ops.append((fop, next_op.output))
        op_position_by_id[next_op.output] = len(pending_ops) - 1

    if not pending_ops:
        raise ValueError("multi-MoE graph has no fusion op; parallel grouped matmuls must " "share a pointwise epilogue (the no-epilogue case is out of scope)")

    set_output_ids_in_order = [tid for tid in _TENSOR_OUTPUT_FLAG if _TENSOR_OUTPUT_FLAG[tid]]
    terminal_id: int | None = None
    for tid in reversed(set_output_ids_in_order):
        if tid in op_position_by_id:
            terminal_id = tid
            break
    if terminal_id is None:
        raise ValueError("multi-MoE graph has no fusion-op output marked set_output(True); " "the fused (terminal) output must be a pointwise op")

    from dataclasses import replace as _replace

    recorded_by_out = {op.output: op for op in ordered_ops}
    fusion_ops: list[FusionOp] = []
    for fop, out_id in pending_ops:
        recorded = recorded_by_out[out_id]
        op_compute = recorded.compute_dtype if recorded.compute_dtype is not None else compute_dtype
        if out_id == terminal_id:
            fusion_ops.append(_replace(fop, compute_dtype=op_compute))
        else:
            op_out_dtype = _resolve_out_dtype(out_id, recorded.output_tensor, io_dtype, intermediate_dtype)
            fusion_ops.append(
                _replace(
                    fop,
                    compute_dtype=op_compute,
                    out_dtype=op_out_dtype,
                    output_tap=_TENSOR_OUTPUT_FLAG.get(out_id, False),
                )
            )

    output_dtype: Dtype = io_dtype
    for op in ops:
        if op.output == terminal_id and op.output_tensor is not None:
            explicit = op.output_tensor.get_data_type()
            if explicit != cudnn.data_type.NOT_SET and explicit in _DTYPE_FROM_CUDNN:
                output_dtype = _DTYPE_FROM_CUDNN[explicit]
            break

    terminal_op_idx_explicit = -2
    for pos, (_, out_id) in enumerate(pending_ops):
        if out_id == terminal_id:
            terminal_op_idx_explicit = pos
            break

    def _reduction_output(red: _RecordedOp) -> tuple[tuple[int, int, int], bool]:
        dim = _TENSOR_DIM_OVERRIDE.get(red.output)
        if dim is None:
            try:
                dim = tuple(red.output_tensor.get_dim())
            except Exception:  # noqa: BLE001
                dim = ()
        if len(dim) != 3:
            raise ValueError(f"reduction {red.op_name!r} must set a rank-3 output dim; got {dim}")
        full = (1, int(M), int(N))
        grouped_by_moe = False
        if red.group_offset is not None:
            if red.group_offset != fto_id:
                raise ValueError(f"reduction {red.op_name!r} groupOffset must be the MoE " "first_token_offset tensor")
            if int(dim[0]) != num_groups:
                raise ValueError(f"reduction {red.op_name!r} with groupOffset must use " f"output dim[0] == num_groups ({num_groups}); got {dim}")
            grouped_by_moe = True
        axis0_extent = num_groups if grouped_by_moe else full[0]
        compat_full = (axis0_extent, full[1], full[2])
        for axis, (out_extent, full_extent) in enumerate(zip(dim, compat_full)):
            if out_extent not in (1, full_extent):
                raise ValueError(
                    f"reduction {red.op_name!r} output dim {dim} is not compatible " f"with moe output {full}: axis {axis} must be 1 or {full_extent}"
                )
        if all(out_extent == full_extent for out_extent, full_extent in zip(dim, compat_full)):
            raise ValueError(f"reduction {red.op_name!r} output dim {dim} does not reduce any axis")
        return (int(dim[0]), int(dim[1]), int(dim[2])), grouped_by_moe

    reductions: list[ReductionSpec] = []
    reduction_objs: list[Any] = []
    for red in ops:
        if red.cudnn_name != "reduction":
            continue
        if not _TENSOR_OUTPUT_FLAG.get(red.output, False):
            continue
        (input_id,) = red.inputs
        if input_id in gemm_idx_by_output or input_id in op_position_by_id:
            source_ref = _operand_ref(input_id)
        else:
            raise ValueError(f"reduction {red.op_name!r} input is not produced by this " "multi-MoE epilogue chain")
        compute = red.compute_dtype if red.compute_dtype is not None else compute_dtype
        dtype = _resolve_out_dtype(red.output, red.output_tensor, io_dtype, intermediate_dtype)
        if red.reduction_mode is None:
            raise NotImplementedError(f"reduction {red.op_name!r} mode is not supported by cudnn.TBD.gemm; " "supported modes are ADD, AMAX, MAX, and MIN")
        red_dim, grouped_by_moe = _reduction_output(red)
        reductions.append(
            ReductionSpec(
                mode=red.reduction_mode,  # type: ignore[arg-type]
                source_ref=source_ref,
                dim=red_dim,
                dtype=dtype,
                compute_dtype=compute,
                grouped_by_moe=grouped_by_moe,
            )
        )
        reduction_objs.append(red.output_tensor)

    matmul_out_dtype = _resolve_out_dtype(moe_ops[0].output, moe_ops[0].output_tensor, io_dtype, intermediate_dtype)
    matmul_spec = MatmulSpec(
        M=M,
        N=N,
        K=K,
        batch=1,
        a_batch=1,
        b_batch=1,
        a_major=a_major,
        b_major=b_major,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        accum_dtype=mm_compute,
        out_dtype=matmul_out_dtype,
    )
    chain = FusionChain(
        matmul=matmul_spec,
        aux_tensors=aux_tensors,
        ops=fusion_ops,
        output_dtype=output_dtype,
        terminal_op_idx=terminal_op_idx_explicit,
        num_a_operands=len(a_ids),
        num_b_operands=len(b_ids),
        gemm_operands=gemm_operands,
        moe=MoeSpec(num_experts=int(E), mode=moe_ops[0].moe_mode, offset_dtype=offset_dtype),
        block_scale=block_scale_spec,
        reductions=reductions,
    )
    output_objs: list[Any] = [recorded_by_out[terminal_id].output_tensor]
    for i, fop in enumerate(fusion_ops):
        if fop.output_tap:
            output_objs.append(recorded_by_out[pending_ops[i][1]].output_tensor)
    output_objs.extend(reduction_objs)
    binding = _make_multi_binding(
        meta,
        a_ids,
        b_ids,
        a_caps,
        b_caps,
        output_objs,
        aux_objs,
        block_scale_spec is not None,
        first_token_offset=meta[fto_id].tensor,
    )
    return chain, binding


def _build_multi_gemm_chain(
    matmuls: list[_RecordedOp],
    ops: list[_RecordedOp],
    meta: dict[int, _TensorMeta],
    io_dtype: Dtype,
    intermediate_dtype: Dtype,
    compute_dtype: Dtype,
) -> FusionChain:
    """Build a FusionChain for K parallel GEMMs sharing one pointwise epilogue.

    Each ``matmul`` op is a GEMM; they must share shape / layout / dtype but may
    use shared or distinct A / B operands (deduped by tensor id). Their outputs
    are the DAG roots; a pointwise op references the GEMM output it reads via a
    negative ``parent_idx`` (``gemm_source(g)``). Block-scale is supported: a
    matmul operand produced by ``block_scale_dequantize`` folds into the packed
    data tensor (the SF travels with it), and a dequant shared by several
    matmuls collapses to ONE distinct operand. POC scope: no mainloop fusion, no
    per-GEMM matmul taps; the terminal must be a fusion op (the fused output)."""
    from .fusion_ir import BlockScaleSpec

    # ----- Resolve each matmul operand through any block_scale_dequantize, then
    # dedup by the PACKED data tensor id (a shared dequant → same packed id →
    # one distinct operand). This matches the runtime dedup (by packed-data
    # identity) and the shared-dequant wrinkle (one dequant feeds many matmuls).
    dequant_by_output = {op.output: op for op in ops if op.cudnn_name == "block_scale_dequantize"}

    def _capture_side(operand_id: int) -> dict:
        """For a (possibly dequantized) matmul operand, return its packed data
        id + block-scale fields (all None for a non-dequantized side)."""
        deq = dequant_by_output.get(operand_id)
        if deq is None:
            return dict(
                data_id=operand_id,
                data_dtype=meta[operand_id].dtype,
                block_size_2d=None,
                sf_dtype=None,
                sf_reorder=None,
                deq_compute=None,
                deq_out=None,
                sf_id=None,
            )
        data_id, sf_id = deq.inputs
        sf_meta = meta[sf_id]
        deq_compute = deq.compute_dtype if deq.compute_dtype is not None else compute_dtype
        deq_out = _resolve_out_dtype(deq.output, deq.output_tensor, io_dtype, intermediate_dtype)
        return dict(
            data_id=data_id,
            data_dtype=meta[data_id].dtype,
            block_size_2d=(tuple(deq.block_size) if deq.block_size else None),
            sf_dtype=sf_meta.dtype,
            sf_reorder=sf_meta.reordering,
            deq_compute=deq_compute,
            deq_out=deq_out,
            sf_id=sf_id,
        )

    a_ids: list[int] = []  # distinct PACKED A data ids
    b_ids: list[int] = []
    a_caps: dict[int, dict] = {}
    b_caps: dict[int, dict] = {}
    gemm_operands: list[tuple[int, int]] = []
    for mm in matmuls:
        a_cap = _capture_side(mm.inputs[0])
        b_cap = _capture_side(mm.inputs[1])
        a_pid, b_pid = a_cap["data_id"], b_cap["data_id"]
        if a_pid not in a_ids:
            a_ids.append(a_pid)
            a_caps[a_pid] = a_cap
        if b_pid not in b_ids:
            b_ids.append(b_pid)
            b_caps[b_pid] = b_cap
        gemm_operands.append((a_ids.index(a_pid), b_ids.index(b_pid)))

    is_block_scale = any(c["sf_dtype"] is not None for c in (*a_caps.values(), *b_caps.values()))

    # ----- Validate every GEMM shares shape / layout / dtype ----------------
    def _gemm_geometry(a_pid: int, b_pid: int):
        A_meta = meta[a_pid]
        B_meta = meta[b_pid]
        if len(A_meta.dim) != 3 or len(B_meta.dim) != 3:
            raise ValueError(f"matmul operands must be 3D; got A={A_meta.dim} B={B_meta.dim}")
        Ba, M, Ka = A_meta.dim
        Bb, Kb, N = B_meta.dim
        if Ka != Kb:
            raise ValueError(f"K dim mismatch: A={A_meta.dim} B={B_meta.dim}")
        batch = max(Ba, Bb)
        return (
            int(M),
            int(N),
            int(Ka),
            int(batch),
            int(Ba),
            int(Bb),
            _infer_a_major(A_meta.dim, A_meta.stride),
            _infer_b_major(B_meta.dim, B_meta.stride),
            A_meta.dtype,
            B_meta.dtype,
        )

    geom0 = _gemm_geometry(a_ids[gemm_operands[0][0]], b_ids[gemm_operands[0][1]])
    for ai, bi in gemm_operands[1:]:
        if _gemm_geometry(a_ids[ai], b_ids[bi]) != geom0:
            raise ValueError("parallel GEMMs must share shape / layout / dtype; multi-GEMM " "with heterogeneous GEMMs is out of POC scope")
    M, N, K, batch, Ba, Bb, a_major, b_major, a_dtype, b_dtype = geom0

    # ----- Shared BlockScaleSpec (all GEMMs same combo) --------------------
    block_scale_spec = None
    if is_block_scale:
        a0 = a_caps[a_ids[gemm_operands[0][0]]]
        b0 = b_caps[b_ids[gemm_operands[0][1]]]

        # Every distinct A operand (resp. B) must share the same block-scale
        # combo as GEMM 0's — a single shared BlockScaleSpec describes them all.
        def _combo_key(cap):
            return (cap["data_dtype"], cap["block_size_2d"], cap["sf_dtype"], cap["sf_reorder"], cap["deq_compute"], cap["deq_out"])

        for cap in a_caps.values():
            if _combo_key(cap) != _combo_key(a0):
                raise ValueError("all A operands of a block-scale multi-GEMM must share the same SF combo")
        for cap in b_caps.values():
            if _combo_key(cap) != _combo_key(b0):
                raise ValueError("all B operands of a block-scale multi-GEMM must share the same SF combo")
        block_scale_spec = BlockScaleSpec(
            a_dtype=a0["data_dtype"],
            b_dtype=b0["data_dtype"],
            block_size_a=a0["block_size_2d"],
            block_size_b=b0["block_size_2d"],
            sf_dtype_a=a0["sf_dtype"],
            sf_dtype_b=b0["sf_dtype"],
            sfa_reorder=a0["sf_reorder"],
            sfb_reorder=b0["sf_reorder"],
            dequant_compute_a=a0["deq_compute"],
            dequant_compute_b=b0["deq_compute"],
            dequant_out_a=a0["deq_out"],
            dequant_out_b=b0["deq_out"],
        )
    mm_compute = matmuls[0].compute_dtype if matmuls[0].compute_dtype is not None else compute_dtype
    matmul_out_dim = (batch, M, N)

    # ----- Epilogue DAG over MULTIPLE roots (each GEMM output) --------------
    gemm_idx_by_output: dict[int, int] = {mm.output: g for g, mm in enumerate(matmuls)}
    consumers_by_input: dict[int, list[_RecordedOp]] = {}
    for op in ops:
        for inp in op.inputs:
            consumers_by_input.setdefault(inp, []).append(op)

    # Pass 1: reachable op set (BFS from all GEMM outputs).
    reachable_op_ids: set[int] = set()
    bfs_queue: list[int] = [mm.output for mm in matmuls]
    visited_tensors: set[int] = set()
    while bfs_queue:
        tid = bfs_queue.pop(0)
        if tid in visited_tensors:
            continue
        visited_tensors.add(tid)
        for op in consumers_by_input.get(tid, []):
            if op.cudnn_name == "matmul":
                continue
            if op.output not in reachable_op_ids:
                reachable_op_ids.add(op.output)
                bfs_queue.append(op.output)
    reachable_ops = [op for op in ops if op.output in reachable_op_ids and op.cudnn_name not in {"matmul", "reduction"}]

    def _is_in_chain(tid: int) -> bool:
        return tid in gemm_idx_by_output or tid in reachable_op_ids

    in_chain_deps: dict[int, list[int]] = {op.output: [inp for inp in op.inputs if _is_in_chain(inp)] for op in reachable_ops}

    # Pass 2: Kahn topo sort (placed seeded with all GEMM outputs).
    placed: set[int] = set(gemm_idx_by_output)
    remaining = list(reachable_ops)
    ordered_ops: list[_RecordedOp] = []
    while remaining:
        ready_idx = next(
            (i for i, op in enumerate(remaining) if all(d in placed for d in in_chain_deps[op.output])),
            None,
        )
        if ready_idx is None:
            raise AssertionError(f"cycle / unsatisfiable deps: {[op.op_name for op in remaining]}")
        op = remaining.pop(ready_idx)
        ordered_ops.append(op)
        placed.add(op.output)

    aux_tensors: list[TensorRef] = []
    aux_objs: list[Any] = []
    aux_seen: set[int] = set()
    op_position_by_id: dict[int, int] = {}

    def _operand_ref(tid: int) -> int:
        """Resolve an in-chain operand id to a producing-operation reference:
        ``gemm_source(g)`` (< 0) for a GEMM output, else the prior op's index."""
        if tid in gemm_idx_by_output:
            return gemm_source(gemm_idx_by_output[tid])
        return op_position_by_id[tid]

    pending_ops: list[tuple[FusionOp, int]] = []
    for next_op in ordered_ops:
        if next_op.cudnn_name in _UNARY_OP_MAP:
            (parent_id,) = next_op.inputs
            fop = FusionOp(op=_UNARY_OP_MAP[next_op.cudnn_name], parent_idx=_operand_ref(parent_id))
        elif next_op.cudnn_name in _BINARY_OP_MAP:
            inp0, inp1 = next_op.inputs
            in0, in1 = _is_in_chain(inp0), _is_in_chain(inp1)
            if in0 and in1:
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=None,
                    aux_on_rhs=True,
                    parent_idx=_operand_ref(inp0),
                    parent_idx_b=_operand_ref(inp1),
                )
            elif in0 or in1:
                if in0:
                    chain_id, aux_id, aux_on_rhs = inp0, inp1, True
                else:
                    chain_id, aux_id, aux_on_rhs = inp1, inp0, False
                aux_meta = meta[aux_id]
                if not aux_meta.is_input:
                    raise ValueError(f"aux input {aux_meta.name!r} of op {next_op.op_name!r} is " "not a graph input — POC supports only graph-input aux")
                if aux_id not in aux_seen:
                    aux_seen.add(aux_id)
                    bcast = _infer_bcast_mode(matmul_out_dim, aux_meta.dim)
                    aux_tensors.append(
                        TensorRef(
                            name=aux_meta.name,
                            dim=aux_meta.dim,
                            stride=aux_meta.stride,
                            dtype=aux_meta.dtype,
                            bcast_mode=bcast,
                        )
                    )
                    aux_objs.append(aux_meta.tensor)
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=aux_meta.name,
                    aux_on_rhs=aux_on_rhs,
                    parent_idx=_operand_ref(chain_id),
                )
            else:
                raise ValueError(f"binary op {next_op.op_name!r} has no in-chain operand")
        else:
            raise ValueError(f"op {next_op.cudnn_name!r} (name={next_op.op_name!r}) is not in " "the POC pointwise subset; out-of-scope")
        pending_ops.append((fop, next_op.output))
        op_position_by_id[next_op.output] = len(pending_ops) - 1

    # No fusion epilogue: every GEMM output is materialized directly to its own
    # GMEM buffer (K parallel matmuls, same shape, no shared epilogue — e.g. the
    # DualBlockScaleMatmul benchmark). Each GEMM output must set_output(True);
    # GEMM 0 = terminal (slot 0), GEMMs >0 = taps (slots 1..). Cheaper than the
    # fused case (just cast+store per GEMM, no pointwise compute).
    if not pending_ops:
        per_gemm_dtypes: list[Dtype] = []
        for mm in matmuls:
            if not _TENSOR_OUTPUT_FLAG.get(mm.output, False):
                raise ValueError("no-epilogue multi-GEMM: every GEMM output must be " "set_output(True) (no fusion op materializes it)")
            per_gemm_dtypes.append(_resolve_out_dtype(mm.output, mm.output_tensor, io_dtype, intermediate_dtype))
        matmul_spec = MatmulSpec(
            M=M,
            N=N,
            K=K,
            batch=batch,
            a_batch=Ba,
            b_batch=Bb,
            a_major=a_major,
            b_major=b_major,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=mm_compute,
            out_dtype=per_gemm_dtypes[0],
        )
        chain = FusionChain(
            matmul=matmul_spec,
            aux_tensors=aux_tensors,
            ops=[],
            output_dtype=per_gemm_dtypes[0],
            terminal_op_idx=-2,
            num_a_operands=len(a_ids),
            num_b_operands=len(b_ids),
            gemm_operands=gemm_operands,
            per_gemm_outputs=per_gemm_dtypes,
            block_scale=block_scale_spec,
        )
        # No-epilogue multi-GEMM outputs = each GEMM's own buffer (slot 0 =
        # GEMM 0, then gemm_<g>), matching chain.outputs.
        binding = _make_multi_binding(
            meta,
            a_ids,
            b_ids,
            a_caps,
            b_caps,
            [mm.output_tensor for mm in matmuls],
            aux_objs,
            block_scale_spec is not None,
        )
        return chain, binding

    # Terminal = last set_output(True) that is a fusion op.
    set_output_ids_in_order = [tid for tid in _TENSOR_OUTPUT_FLAG if _TENSOR_OUTPUT_FLAG[tid]]
    terminal_id: int | None = None
    for tid in reversed(set_output_ids_in_order):
        if tid in op_position_by_id:
            terminal_id = tid
            break
    if terminal_id is None:
        raise ValueError("multi-GEMM graph has no fusion-op output marked set_output(True); " "the fused (terminal) output must be a pointwise op")

    from dataclasses import replace as _replace

    recorded_by_out = {op.output: op for op in ordered_ops}
    fusion_ops: list[FusionOp] = []
    for fop, out_id in pending_ops:
        recorded = recorded_by_out[out_id]
        op_compute = recorded.compute_dtype if recorded.compute_dtype is not None else compute_dtype
        if out_id == terminal_id:
            fusion_ops.append(_replace(fop, compute_dtype=op_compute))
        else:
            op_out_dtype = _resolve_out_dtype(out_id, recorded.output_tensor, io_dtype, intermediate_dtype)
            fusion_ops.append(
                _replace(
                    fop,
                    compute_dtype=op_compute,
                    out_dtype=op_out_dtype,
                    output_tap=_TENSOR_OUTPUT_FLAG.get(out_id, False),
                )
            )

    output_dtype: Dtype = io_dtype
    for op in ops:
        if op.output == terminal_id and op.output_tensor is not None:
            explicit = op.output_tensor.get_data_type()
            if explicit != cudnn.data_type.NOT_SET and explicit in _DTYPE_FROM_CUDNN:
                output_dtype = _DTYPE_FROM_CUDNN[explicit]
            break

    terminal_op_idx_explicit = -2
    for pos, (_, out_id) in enumerate(pending_ops):
        if out_id == terminal_id:
            terminal_op_idx_explicit = pos
            break

    def _reduction_output_dim(red: _RecordedOp) -> tuple[int, int, int]:
        dim = _TENSOR_DIM_OVERRIDE.get(red.output)
        if dim is None:
            try:
                dim = tuple(red.output_tensor.get_dim())
            except Exception:  # noqa: BLE001
                dim = ()
        if len(dim) != 3:
            raise ValueError(f"reduction {red.op_name!r} must set a rank-3 output dim; got {dim}")
        full = (int(batch), int(M), int(N))
        for axis, (out_extent, full_extent) in enumerate(zip(dim, full)):
            if out_extent not in (1, full_extent):
                raise ValueError(
                    f"reduction {red.op_name!r} output dim {dim} is not compatible " f"with matmul output {full}: axis {axis} must be 1 or {full_extent}"
                )
        if all(out_extent == full_extent for out_extent, full_extent in zip(dim, full)):
            raise ValueError(f"reduction {red.op_name!r} output dim {dim} does not reduce any axis")
        return (int(dim[0]), int(dim[1]), int(dim[2]))

    reductions: list[ReductionSpec] = []
    reduction_objs: list[Any] = []
    for red in ops:
        if red.cudnn_name != "reduction":
            continue
        if not _TENSOR_OUTPUT_FLAG.get(red.output, False):
            continue
        (input_id,) = red.inputs
        if input_id in gemm_idx_by_output or input_id in op_position_by_id:
            source_ref = _operand_ref(input_id)
        else:
            raise ValueError(f"reduction {red.op_name!r} input is not produced by this " "multi-GEMM epilogue chain")
        compute = red.compute_dtype if red.compute_dtype is not None else compute_dtype
        dtype = _resolve_out_dtype(red.output, red.output_tensor, io_dtype, intermediate_dtype)
        if red.reduction_mode is None:
            raise NotImplementedError(f"reduction {red.op_name!r} mode is not supported by cudnn.TBD.gemm; " "supported modes are ADD, AMAX, MAX, and MIN")
        reductions.append(
            ReductionSpec(
                mode=red.reduction_mode,  # type: ignore[arg-type]
                source_ref=source_ref,
                dim=_reduction_output_dim(red),
                dtype=dtype,
                compute_dtype=compute,
            )
        )
        reduction_objs.append(red.output_tensor)

    # GEMM output declared dtype (virtual fp32 in the canonical case) — the
    # epilogue rounds each accumulator to it before the op chain, matching
    # cuDNN's per-tensor semantics. All GEMMs share it (same-layout invariant);
    # resolve from GEMM 0's output tensor.
    matmul_out_dtype = _resolve_out_dtype(matmuls[0].output, matmuls[0].output_tensor, io_dtype, intermediate_dtype)

    _mg_term = recorded_by_out[terminal_id].output_tensor
    out_major = _infer_out_major(
        tuple(_mg_term.get_dim()) if _mg_term is not None else (),
        tuple(_mg_term.get_stride()) if _mg_term is not None else (),
    )
    matmul_spec = MatmulSpec(
        M=M,
        N=N,
        K=K,
        batch=batch,
        a_batch=Ba,
        b_batch=Bb,
        a_major=a_major,
        b_major=b_major,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        accum_dtype=mm_compute,
        out_dtype=matmul_out_dtype,
        out_major=out_major,
    )
    chain = FusionChain(
        matmul=matmul_spec,
        aux_tensors=aux_tensors,
        ops=fusion_ops,
        output_dtype=output_dtype,
        terminal_op_idx=terminal_op_idx_explicit,
        num_a_operands=len(a_ids),
        num_b_operands=len(b_ids),
        gemm_operands=gemm_operands,
        block_scale=block_scale_spec,
        reductions=reductions,
    )
    # Outputs in chain.outputs slot order: terminal, op taps (chain order),
    # reductions. (Multi-GEMM has no matmul tap.)
    output_objs: list[Any] = [recorded_by_out[terminal_id].output_tensor]
    for i, fop in enumerate(fusion_ops):
        if fop.output_tap:
            output_objs.append(recorded_by_out[pending_ops[i][1]].output_tensor)
    output_objs.extend(reduction_objs)
    binding = _make_multi_binding(
        meta,
        a_ids,
        b_ids,
        a_caps,
        b_caps,
        output_objs,
        aux_objs,
        block_scale_spec is not None,
    )
    return chain, binding


def _build_chain(
    ops: list[_RecordedOp],
    meta: dict[int, _TensorMeta],
    io_dtype: Dtype,
    intermediate_dtype: Dtype = "fp32",
    compute_dtype: Dtype = "fp32",
) -> FusionChain:
    # ----- MoE grouped matmul (own graph type) ------------------------------
    moe_ops = [op for op in ops if op.cudnn_name == "moe_grouped_matmul"]
    if len(moe_ops) > 1:
        # K parallel MoE grouped matmuls sharing one first_token_offset + one
        # pointwise epilogue (e.g. grouped SwiGLU). Same multi-GEMM machinery as
        # the plain-matmul case, with MoE geometry + a shared MoeSpec.
        return _build_multi_moe_chain(moe_ops, ops, meta, io_dtype, intermediate_dtype, compute_dtype)
    if moe_ops:
        return _build_moe_chain(moe_ops, ops, meta, io_dtype, intermediate_dtype, compute_dtype)

    matmuls = [op for op in ops if op.cudnn_name == "matmul"]
    if len(matmuls) == 0:
        raise ValueError("POC scope is >=1 matmul per graph; found 0")
    if len(matmuls) > 1:
        # Parallel GEMMs sharing one epilogue (multi-GEMM). Block-scale is
        # handled inside the builder (a shared dequant folds into one distinct
        # operand). Mainloop fusion alongside multiple matmuls is out of scope —
        # the builder only walks the epilogue DAG, so a pre-MMA pointwise op
        # would surface as a non-graph-input operand and raise there.
        return _build_multi_gemm_chain(matmuls, ops, meta, io_dtype, intermediate_dtype, compute_dtype)
    mm = matmuls[0]
    A_id, B_id = mm.inputs

    # ----- Block-scaled matmul detection (structural pattern-match) ---------
    # If the matmul's A and/or B operand is produced by a
    # block_scale_dequantize node, fold the dequant(s) + matmul into one
    # block-scale matmul. Three shapes match: dequant(A) @ B, A @ dequant(B),
    # dequant(A) @ dequant(B). This is purely STRUCTURAL — we apply NO dtype /
    # block-size / arch rules here; which combinations are actually runnable
    # (today: both sides dequantized, valid family) is decided at compile time.
    # For each dequantized side, the matmul operand is redirected to the packed
    # (FP4 / FP8) data tensor and its scale factor is captured.
    from .fusion_ir import BlockScaleSpec

    dequant_by_output = {op.output: op for op in ops if op.cudnn_name == "block_scale_dequantize"}
    block_scale_spec: "BlockScaleSpec | None" = None
    sfa_obj = None  # cuDNN SF tensors (block-scale), for the variant-pack binding
    sfb_obj = None
    if A_id in dequant_by_output or B_id in dequant_by_output:

        def _capture_side(operand_id: int):
            """For a (possibly) dequantized operand, return a dict of structural
            block-scale fields (all None for a non-dequantized side, plus
            ``data_id``/``data_dtype`` of the plain operand). No validation.

            ``deq_compute`` = the dequant op's math precision; ``deq_out`` = the
            dequant output dtype (the MMA's logical input type for this side)."""
            deq = dequant_by_output.get(operand_id)
            if deq is None:
                return dict(
                    data_id=operand_id, data_dtype=meta[operand_id].dtype, block_size_2d=None, sf_dtype=None, sf_reorder=None, deq_compute=None, deq_out=None
                )
            data_id, sf_id = deq.inputs
            sf_meta = meta[sf_id]
            deq_compute = deq.compute_dtype if deq.compute_dtype is not None else compute_dtype
            deq_out = _resolve_out_dtype(deq.output, deq.output_tensor, io_dtype, intermediate_dtype)
            return dict(
                data_id=data_id,
                data_dtype=meta[data_id].dtype,
                block_size_2d=(tuple(deq.block_size) if deq.block_size else None),
                sf_dtype=sf_meta.dtype,
                sf_reorder=sf_meta.reordering,
                deq_compute=deq_compute,
                deq_out=deq_out,
            )

        a = _capture_side(A_id)
        b = _capture_side(B_id)

        block_scale_spec = BlockScaleSpec(
            a_dtype=a["data_dtype"],
            b_dtype=b["data_dtype"],
            block_size_a=a["block_size_2d"],
            block_size_b=b["block_size_2d"],
            sf_dtype_a=a["sf_dtype"],
            sf_dtype_b=b["sf_dtype"],
            sfa_reorder=a["sf_reorder"],
            sfb_reorder=b["sf_reorder"],
            dequant_compute_a=a["deq_compute"],
            dequant_compute_b=b["deq_compute"],
            dequant_out_a=a["deq_out"],
            dequant_out_b=b["deq_out"],
        )
        a_data_id, b_data_id = a["data_id"], b["data_id"]
        # Capture the SF tensor objects (2nd input of each dequant) for binding,
        # while A_id/B_id still hold the dequant-output ids.
        deq_a = dequant_by_output.get(A_id)
        deq_b = dequant_by_output.get(B_id)
        sfa_obj = meta[deq_a.inputs[1]].tensor if deq_a else None
        sfb_obj = meta[deq_b.inputs[1]].tensor if deq_b else None
        # Redirect each scaled operand to its packed data tensor.
        A_id, B_id = a_data_id, b_data_id

    # ----- Mainloop fusion detection (Phase 6) -----------------------------
    # Walk backwards from the matmul's A input through any chain of unary
    # pointwise ops whose ultimate source is a graph-input tensor. Each such
    # op runs on the dedicated mainloop-fusion warps: it reads the freshly
    # TMA'd A tile out of SMEM, transforms it in registers (fp32 compute), and
    # writes it back in place before the MMA consumes it. ``A' = op(A)`` then
    # ``C = A' @ B``. POC scope: a linear chain of unary ops on A only.
    op_by_output = {op.output: op for op in ops if op.cudnn_name != "matmul"}

    # Aux tensors (shared across mainloop + epilogue). Defined here so the
    # mainloop walk can register its scalar auxes; the epilogue DAG walk below
    # keeps appending. Runtime aux order = this list's order. aux_objs holds the
    # parallel cuDNN tensor objects for the variant-pack binding.
    aux_tensors: list[TensorRef] = []
    aux_objs: list[Any] = []
    aux_seen: set[int] = set()

    def _is_scalar_input(tid: int) -> bool:
        m = meta.get(tid)
        return m is not None and m.is_input and len(m.dim) > 0 and all(d == 1 for d in m.dim)

    def _walk_mainloop(operand_id: int, label: str) -> tuple[int, list[FusionOp]]:
        """Walk backwards from a matmul operand through a chain of pointwise ops
        (unary, or binary with a single SCALAR graph-input aux) to the root
        graph input. Returns (root_tensor_id, mainloop_ops) in
        graph-input -> ... -> operand' order. Registers scalar auxes in the
        shared aux_tensors list."""
        # Each entry: (cudnn_name, is_binary, aux_name_or_None, aux_on_rhs).
        steps: list[tuple] = []
        cur = operand_id
        while cur in op_by_output:
            producer = op_by_output[cur]
            if producer.cudnn_name in _UNARY_OP_MAP:
                steps.append((producer.cudnn_name, False, None, True))
                (cur,) = producer.inputs
            elif producer.cudnn_name in _BINARY_OP_MAP:
                i0, i1 = producer.inputs
                s0, s1 = _is_scalar_input(i0), _is_scalar_input(i1)
                if s0 and not s1:
                    aux_id, chain_id, aux_on_rhs = i0, i1, False
                elif s1 and not s0:
                    aux_id, chain_id, aux_on_rhs = i1, i0, True
                else:
                    raise ValueError(
                        f"matmul {label} operand is produced by binary op "
                        f"{producer.op_name!r} ({producer.cudnn_name!r}) — mainloop "
                        "fusion only supports a binary op with exactly one SCALAR "
                        "graph-input aux (e.g. A * alpha); per-row/col/elem aux on an "
                        "operand is out of POC scope (needs swizzle-aware indexing)"
                    )
                aux_meta = meta[aux_id]
                if aux_id not in aux_seen:
                    aux_seen.add(aux_id)
                    aux_tensors.append(
                        TensorRef(
                            name=aux_meta.name,
                            dim=aux_meta.dim,
                            stride=aux_meta.stride,
                            dtype=aux_meta.dtype,
                            bcast_mode="scalar",
                        )
                    )
                    aux_objs.append(aux_meta.tensor)
                steps.append((producer.cudnn_name, True, aux_meta.name, aux_on_rhs))
                cur = chain_id
            else:
                raise ValueError(
                    f"matmul {label} operand is produced by op {producer.op_name!r} "
                    f"({producer.cudnn_name!r}) which is not a pointwise op — mainloop "
                    "fusion supports unary ops and scalar-aux binary ops in the POC"
                )
        steps.reverse()
        fops: list[FusionOp] = []
        for idx, (cudnn_name, is_binary, aux_name, aux_on_rhs) in enumerate(steps):
            parent = idx - 1 if idx > 0 else -1
            if is_binary:
                fops.append(FusionOp(op=_BINARY_OP_MAP[cudnn_name], aux=aux_name, aux_on_rhs=aux_on_rhs, parent_idx=parent))
            else:
                fops.append(FusionOp(op=_UNARY_OP_MAP[cudnn_name], parent_idx=parent))
        return cur, fops

    if block_scale_spec is not None:
        # Block-scaled matmul: operands are the packed FP4/FP8 graph inputs;
        # the dequant happens inside the MMA, so there is no mainloop fusion
        # and no dtype-cast staging.
        root_A_id, mainloop_a_ops = A_id, []
        root_B_id, mainloop_b_ops = B_id, []
    else:
        root_A_id, mainloop_a_ops = _walk_mainloop(A_id, "A")
        root_B_id, mainloop_b_ops = _walk_mainloop(B_id, "B")
    A_meta = meta[root_A_id]
    B_meta = meta[root_B_id]

    # MMA dtype vs LOAD dtype. The MMA reads the operand at:
    #   * the EXPLICIT ``set_data_type`` on the tensor feeding the matmul, if the
    #     user declared one (e.g. `after_pw0 = identity(A_int8)` with
    #     `set_data_type(bf16)` — the mixed-input case); else
    #   * the root graph input's dtype (the ordinary dtype-preserving mainloop,
    #     where the transform rounds back to the root dtype and stores in place,
    #     and the intermediate's *implicit* fp32 compute dtype is NOT the storage
    #     dtype — so abs(bf16)@B stays bf16, not fp32).
    # The LOAD dtype is always the root graph input's dtype. They differ only for
    # a mixed-input mainloop (int8 load -> bf16 MMA); then `mainloop_*_load_dtype`
    # records the narrow load dtype and the mainloop warps stage the widen. The
    # walk preserves layout, so A_meta still drives dim/stride/major.
    def _mma_operand_dtype(operand_id: int, root_meta: _TensorMeta) -> Dtype:
        explicit = _TENSOR_EXPLICIT_DTYPE.get(operand_id)
        if explicit is not None and explicit in _DTYPE_FROM_CUDNN:
            return _DTYPE_FROM_CUDNN[explicit]
        return root_meta.dtype

    mma_a_dtype = _mma_operand_dtype(A_id, A_meta)
    mma_b_dtype = _mma_operand_dtype(B_id, B_meta)
    mainloop_a_load_dtype = A_meta.dtype if A_meta.dtype != mma_a_dtype else None
    mainloop_b_load_dtype = B_meta.dtype if B_meta.dtype != mma_b_dtype else None

    # The matmul's compute dtype IS the accumulator dtype — faithfully recorded
    # from the cuDNN graph's compute_data_type (graph default or per-op
    # override). Whether the (a, b, accum) combo is actually runnable on the
    # target arch is validated by the compiler (`_check_supported`), not here —
    # the IR/analyzer have no arch info. (Pointwise-op compute dtypes are still
    # checked at FusionOp construction.)
    mm_compute = mm.compute_dtype if mm.compute_dtype is not None else compute_dtype

    if len(A_meta.dim) != 3 or len(B_meta.dim) != 3:
        raise ValueError(
            f"matmul operands must be 3D (batch, ...); got A={A_meta.dim} B={B_meta.dim}. "
            f"For un-batched matmul use a leading dim of 1 (e.g. dim=[1, M, K])."
        )
    Ba, M, Ka = A_meta.dim
    Bb, Kb, N = B_meta.dim
    if Ka != Kb:
        raise ValueError(f"K dim mismatch: A={A_meta.dim} B={B_meta.dim}")
    K = Ka
    batch = max(Ba, Bb)
    if Ba not in (1, batch) or Bb not in (1, batch):
        raise ValueError(f"batch dims must match or broadcast from 1; got A={A_meta.dim} B={B_meta.dim}")

    matmul_spec_kwargs = dict(
        M=int(M),
        N=int(N),
        K=int(K),
        batch=int(batch),
        a_batch=int(Ba),
        b_batch=int(Bb),
        a_major=_infer_a_major(A_meta.dim, A_meta.stride),
        b_major=_infer_b_major(B_meta.dim, B_meta.stride),
        a_dtype=mma_a_dtype,
        b_dtype=mma_b_dtype,
        accum_dtype=mm_compute,
    )

    matmul_out_dim = (batch, M, N)

    # (aux_tensors / aux_seen were defined above so the mainloop walk could
    # register its scalar auxes; the epilogue DAG walk keeps appending here.)
    # DAG walk from matmul output. Two passes:
    #   1. BFS to find every op reachable from `mm.output`.
    #   2. Kahn topological sort — emit each op only after all its in-chain
    #      inputs are placed. This naturally handles both fan-out (one
    #      tensor consumed by several ops) and fan-in (a binary op whose two
    #      inputs are both prior op results).
    consumers_by_input: dict[int, list[_RecordedOp]] = {}
    for op in ops:
        for inp in op.inputs:
            consumers_by_input.setdefault(inp, []).append(op)

    # Pass 1: reachable set.
    reachable_op_ids: set[int] = set()
    bfs_queue: list[int] = [mm.output]
    visited_tensors: set[int] = set()
    while bfs_queue:
        tid = bfs_queue.pop(0)
        if tid in visited_tensors:
            continue
        visited_tensors.add(tid)
        for op in consumers_by_input.get(tid, []):
            if op.output not in reachable_op_ids:
                reachable_op_ids.add(op.output)
                bfs_queue.append(op.output)
    reachable_ops = [op for op in ops if (op.output in reachable_op_ids and op.cudnn_name not in ("reduction", "block_scale_quantize"))]
    reachable_quant_ops = [op for op in ops if op.cudnn_name == "block_scale_quantize" and op.output in reachable_op_ids]

    # For each reachable op, identify its in-chain dependencies (= inputs
    # that are either the matmul output or another reachable op's output).
    in_chain_deps: dict[int, list[int]] = {}
    for op in reachable_ops:
        deps = [inp for inp in op.inputs if inp == mm.output or inp in reachable_op_ids]
        in_chain_deps[op.output] = deps

    # Pass 2: Kahn topo sort, preserving recorder order to break ties.
    placed: set[int] = {mm.output}
    remaining = list(reachable_ops)
    ordered_ops: list[_RecordedOp] = []
    while remaining:
        ready_idx = next(
            (i for i, op in enumerate(remaining) if all(d in placed for d in in_chain_deps[op.output])),
            None,
        )
        if ready_idx is None:
            stuck_names = [op.op_name for op in remaining]
            raise AssertionError(f"cycle / unsatisfiable deps in op graph: {stuck_names}")
        op = remaining.pop(ready_idx)
        ordered_ops.append(op)
        placed.add(op.output)

    # Build FusionOps in topo order.
    pending_ops: list[tuple[FusionOp, int]] = []
    op_position_by_id: dict[int, int] = {mm.output: -1}
    for next_op in ordered_ops:
        if next_op.cudnn_name in _UNARY_OP_MAP:
            (parent_id,) = next_op.inputs
            fop = FusionOp(
                op=_UNARY_OP_MAP[next_op.cudnn_name],
                parent_idx=op_position_by_id[parent_id],
            )
        elif next_op.cudnn_name in _BINARY_OP_MAP:
            inp0, inp1 = next_op.inputs
            in_chain_0 = inp0 == mm.output or inp0 in op_position_by_id
            in_chain_1 = inp1 == mm.output or inp1 in op_position_by_id
            if in_chain_0 and in_chain_1:
                # Phase-4 fan-in: both operands are in-chain.
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=None,
                    aux_on_rhs=True,
                    parent_idx=op_position_by_id[inp0],
                    parent_idx_b=op_position_by_id[inp1],
                )
            elif in_chain_0 or in_chain_1:
                if in_chain_0:
                    chain_id, aux_id, aux_on_rhs = inp0, inp1, True
                else:
                    chain_id, aux_id, aux_on_rhs = inp1, inp0, False
                aux_meta = meta[aux_id]
                if not aux_meta.is_input:
                    raise ValueError(
                        f"aux input {aux_meta.name!r} of op {next_op.op_name!r} is not a " "graph input — POC supports only graph-input aux tensors"
                    )
                if aux_id not in aux_seen:
                    aux_seen.add(aux_id)
                    bcast = _infer_bcast_mode(matmul_out_dim, aux_meta.dim)
                    aux_tensors.append(
                        TensorRef(
                            name=aux_meta.name,
                            dim=aux_meta.dim,
                            stride=aux_meta.stride,
                            dtype=aux_meta.dtype,
                            bcast_mode=bcast,
                        )
                    )
                    aux_objs.append(aux_meta.tensor)
                fop = FusionOp(
                    op=_BINARY_OP_MAP[next_op.cudnn_name],
                    aux=aux_meta.name,
                    aux_on_rhs=aux_on_rhs,
                    parent_idx=op_position_by_id[chain_id],
                )
            else:
                raise ValueError(f"binary op {next_op.op_name!r} has neither operand from the " "matmul chain — POC requires at least one in-chain operand")
        else:
            raise ValueError(f"op {next_op.cudnn_name!r} (name={next_op.op_name!r}) is not in the " "POC pointwise subset; out-of-scope")
        pending_ops.append((fop, next_op.output))
        op_position_by_id[next_op.output] = len(pending_ops) - 1

    # Identify the terminal op (whose dtype lives in `chain.output_dtype` and
    # whose result is `vec_out` at the trailing store). Rule: the *last*
    # tensor with `set_output(True)` in recorder discovery order — that
    # matches the existing single-output and Phase-1/2 conventions where the
    # caller marks `Y.set_output(True)` last.
    set_output_ids_in_order = [tid for tid in _TENSOR_OUTPUT_FLAG.keys() if _TENSOR_OUTPUT_FLAG[tid]]
    terminal_id: int | None = None
    terminal_quant: _RecordedOp | None = None
    terminal_source_ref: int | None = None
    for tid in reversed(set_output_ids_in_order):
        qop = next((q for q in reachable_quant_ops if q.output == tid), None)
        if qop is None:
            continue
        (input_id,) = qop.inputs
        if input_id == mm.output:
            terminal_source_ref = gemm_source(0)
        elif input_id in op_position_by_id:
            terminal_source_ref = op_position_by_id[input_id]
        else:
            raise ValueError(f"block_scale_quantize {qop.op_name!r} input is not produced by " "this matmul epilogue chain")
        terminal_id = qop.output
        terminal_quant = qop
        break
    for tid in reversed(set_output_ids_in_order):
        if terminal_id is not None:
            break
        if tid in op_position_by_id and op_position_by_id[tid] >= 0:
            terminal_id = tid
            break
    if terminal_id is None:
        # No fusion-op output was marked set_output → the matmul itself is
        # the terminal (single-output matmul case).
        terminal_id = mm.output

    from dataclasses import replace as _replace

    recorded_by_out = {op.output: op for op in ordered_ops}

    fusion_ops: list[FusionOp] = []
    for fop, out_id in pending_ops:
        recorded = recorded_by_out[out_id]
        # Per-op compute precision: graph default unless overridden by the op.
        # FusionOp.__post_init__ validates the supported pointwise compute dtypes.
        op_compute = recorded.compute_dtype if recorded.compute_dtype is not None else compute_dtype
        if out_id == terminal_id or (terminal_quant is not None and terminal_source_ref == op_position_by_id[out_id]):
            # The terminal's declared dtype is the chain's output_dtype, applied
            # by the trailing `vec_out` cast — so no mid-chain rounding here.
            fusion_ops.append(_replace(fop, compute_dtype=op_compute))
        else:
            # Req 2 & 3: round this op's result to its declared output dtype
            # (even for pure-virtual tensors) before the next op reads it.
            op_out_dtype = _resolve_out_dtype(out_id, recorded.output_tensor, io_dtype, intermediate_dtype)
            fusion_ops.append(
                _replace(
                    fop,
                    compute_dtype=op_compute,
                    out_dtype=op_out_dtype,
                    output_tap=_TENSOR_OUTPUT_FLAG.get(out_id, False),
                )
            )

    # Matmul-output tap: materialize a second GMEM output (the accumulator at
    # out_dtype) when C.set_output(True) and the matmul isn't also the terminal.
    matmul_output_tap = mm.output != terminal_id and _TENSOR_OUTPUT_FLAG.get(mm.output, False)
    # out_dtype = C's declared data_type (virtual or materialized). The epilogue
    # rounds the fp32 accumulator to it before any fusion op / output — honoring
    # the cuDNN graph's per-tensor dtype, like every other op.
    matmul_out_dtype = _resolve_out_dtype(mm.output, mm.output_tensor, io_dtype, intermediate_dtype)

    if terminal_quant is not None:
        _term_tensor = terminal_quant.output_tensor
    else:
        _term_tensor = mm.output_tensor if terminal_id == mm.output else recorded_by_out[terminal_id].output_tensor
    _term_dim = tuple(_term_tensor.get_dim()) if _term_tensor is not None else ()
    _term_stride = tuple(_term_tensor.get_stride()) if _term_tensor is not None else ()
    if terminal_quant is not None and (not _term_dim or not _term_stride):
        term_meta = meta.get(terminal_quant.output)
        if term_meta is not None:
            _term_dim = term_meta.dim
            _term_stride = term_meta.stride
    out_major = _infer_out_major(_term_dim, _term_stride)

    matmul_spec = MatmulSpec(
        **matmul_spec_kwargs,
        output_tap=matmul_output_tap,
        out_dtype=matmul_out_dtype,
        out_major=out_major,
    )

    # The final tensor's explicit set_data_type (if any) overrides io_dtype —
    # this is the canonical cuDNN pattern for FP8-in / FP16-out matmul where
    # the user does `Y.set_data_type(cudnn.data_type.HALF)` to downcast.
    output_dtype: Dtype = io_dtype
    if terminal_quant is not None:
        explicit = terminal_quant.output_tensor.get_data_type()
        if explicit != cudnn.data_type.NOT_SET and explicit in _DTYPE_FROM_CUDNN:
            output_dtype = _DTYPE_FROM_CUDNN[explicit]
    else:
        for op in ops:
            if op.output == terminal_id and op.output_tensor is not None:
                explicit = op.output_tensor.get_data_type()
                if explicit != cudnn.data_type.NOT_SET and explicit in _DTYPE_FROM_CUDNN:
                    output_dtype = _DTYPE_FROM_CUDNN[explicit]
                break

    # Resolve which position in `fusion_ops` is the terminal. For linear
    # chains this is always the last op (FusionChain's default sentinel
    # already handles that case); for fan-out DAGs the analyzer must set
    # it explicitly because the terminal may not be the last in BFS order.
    terminal_op_idx_explicit = -2  # auto
    if terminal_quant is not None:
        assert terminal_source_ref is not None
        terminal_op_idx_explicit = terminal_source_ref
    elif fusion_ops and terminal_id != mm.output:
        # Walk pending_ops and find the position whose output_id matches.
        for pos, (_, out_id) in enumerate(pending_ops):
            if out_id == terminal_id:
                terminal_op_idx_explicit = pos
                break

    def _reduction_output_dim(red: _RecordedOp) -> tuple[int, int, int]:
        dim = _TENSOR_DIM_OVERRIDE.get(red.output)
        if dim is None:
            try:
                dim = tuple(red.output_tensor.get_dim())
            except Exception:  # noqa: BLE001
                dim = ()
        if len(dim) != 3:
            raise ValueError(f"reduction {red.op_name!r} must set a rank-3 output dim; got {dim}")
        full = (int(batch), int(M), int(N))
        for axis, (out_extent, full_extent) in enumerate(zip(dim, full)):
            if out_extent not in (1, full_extent):
                raise ValueError(
                    f"reduction {red.op_name!r} output dim {dim} is not compatible " f"with matmul output {full}: axis {axis} must be 1 or {full_extent}"
                )
        if all(out_extent == full_extent for out_extent, full_extent in zip(dim, full)):
            raise ValueError(f"reduction {red.op_name!r} output dim {dim} does not reduce any axis")
        return (int(dim[0]), int(dim[1]), int(dim[2]))

    reductions: list[ReductionSpec] = []
    reduction_objs: list[Any] = []  # parallel cuDNN tensors for the binding
    for red in ops:
        if red.cudnn_name != "reduction":
            continue
        if not _TENSOR_OUTPUT_FLAG.get(red.output, False):
            continue
        (input_id,) = red.inputs
        if input_id == mm.output:
            source_ref = gemm_source(0)
        elif input_id in op_position_by_id:
            source_ref = op_position_by_id[input_id]
        else:
            raise ValueError(f"reduction {red.op_name!r} input is not produced by this matmul " "epilogue chain")
        compute = red.compute_dtype if red.compute_dtype is not None else compute_dtype
        dtype = _resolve_out_dtype(red.output, red.output_tensor, io_dtype, intermediate_dtype)
        if red.reduction_mode is None:
            raise NotImplementedError(f"reduction {red.op_name!r} mode is not supported by cudnn.TBD.gemm; " "supported modes are ADD, AMAX, MAX, and MIN")
        reductions.append(
            ReductionSpec(
                mode=red.reduction_mode,  # type: ignore[arg-type]
                source_ref=source_ref,
                dim=_reduction_output_dim(red),
                dtype=dtype,
                compute_dtype=compute,
            )
        )
        reduction_objs.append(red.output_tensor)

    block_quant: BlockQuantizeSpec | None = None
    quant_scale_obj: Any = None
    if terminal_quant is not None:
        assert terminal_source_ref is not None
        if terminal_quant.scale_output is None or terminal_quant.scale_output_tensor is None:
            raise AssertionError("block_scale_quantize recorded without scale output")
        if not _TENSOR_OUTPUT_FLAG.get(terminal_quant.scale_output, False):
            raise ValueError("block_scale_quantize scale output must be materialized with set_output(True)")
        scale_dtype = _resolve_out_dtype(
            terminal_quant.scale_output,
            terminal_quant.scale_output_tensor,
            io_dtype,
            intermediate_dtype,
        )
        scale_reorder = _TENSOR_REORDERING_OVERRIDE.get(terminal_quant.scale_output)
        if scale_reorder is None:
            scale_meta = meta.get(terminal_quant.scale_output)
            scale_reorder = scale_meta.reordering if scale_meta is not None else None
        bs = int(terminal_quant.block_size[0]) if terminal_quant.block_size else 0
        if bs <= 0:
            raise ValueError(f"block_scale_quantize block_size must be positive; got {bs}")
        if int(N) % bs != 0:
            raise ValueError(f"block_scale_quantize requires N divisible by block_size; got N={N}, block_size={bs}")
        logical_scale_dim = (int(batch), int(M), int(N) // bs)
        expected_scale_dim = logical_scale_dim
        if scale_reorder == "F8_128x4":
            expected_scale_dim = (
                logical_scale_dim[0],
                _round_up(logical_scale_dim[1], 128),
                _round_up(logical_scale_dim[2], 4),
            )
        scale_dim = _TENSOR_DIM_OVERRIDE.get(terminal_quant.scale_output)
        if scale_dim is None:
            scale_meta = meta.get(terminal_quant.scale_output)
            scale_dim = scale_meta.dim if scale_meta is not None else ()
        if not scale_dim:
            scale_dim = expected_scale_dim
        if len(scale_dim) != 3:
            raise ValueError(f"block_scale_quantize scale output must be rank-3; got {scale_dim}")
        if tuple(scale_dim) != expected_scale_dim:
            raise ValueError(f"block_scale_quantize scale dim must be {expected_scale_dim}; got {scale_dim}")
        compute = terminal_quant.compute_dtype if terminal_quant.compute_dtype is not None else compute_dtype
        block_quant = BlockQuantizeSpec(
            source_ref=terminal_source_ref,
            block_size=bs,
            axis=-1 if terminal_quant.quant_axis is None else terminal_quant.quant_axis,
            transpose=terminal_quant.quant_transpose,
            scale_dtype=scale_dtype,
            scale_dim=tuple(scale_dim),
            scale_reorder=scale_reorder,
            compute_dtype=compute,
        )
        quant_scale_obj = terminal_quant.scale_output_tensor

    chain = FusionChain(
        matmul=matmul_spec,
        aux_tensors=aux_tensors,
        ops=fusion_ops,
        output_dtype=output_dtype,
        terminal_op_idx=terminal_op_idx_explicit,
        mainloop_a_ops=mainloop_a_ops,
        mainloop_b_ops=mainloop_b_ops,
        mainloop_a_load_dtype=mainloop_a_load_dtype,
        mainloop_b_load_dtype=mainloop_b_load_dtype,
        block_scale=block_scale_spec,
        reductions=reductions,
        block_quant=block_quant,
    )

    # ----- Variant-pack binding (role -> cuDNN tensor), single-GEMM ----------
    # Output objects in chain.outputs slot order: terminal, matmul tap, op taps
    # (chain order), then reduction side-outputs.
    if terminal_quant is not None:
        terminal_obj = terminal_quant.output_tensor
    else:
        terminal_obj = mm.output_tensor if terminal_id == mm.output else recorded_by_out[terminal_id].output_tensor
    output_objs: list[Any] = [terminal_obj]
    if matmul_output_tap:
        output_objs.append(mm.output_tensor)
    for i, fop in enumerate(fusion_ops):
        if fop.output_tap:
            out_id = pending_ops[i][1]
            output_objs.append(recorded_by_out[out_id].output_tensor)
    output_objs.extend(reduction_objs)
    if quant_scale_obj is not None:
        output_objs.append(quant_scale_obj)

    binding = GemmBinding(
        a_operands=[meta[root_A_id].tensor],
        b_operands=[meta[root_B_id].tensor],
        outputs=output_objs,
        aux=list(aux_objs),
        sfa_operands=[sfa_obj] if block_scale_spec is not None else [],
        sfb_operands=[sfb_obj] if block_scale_spec is not None else [],
    )
    return chain, binding


def analyze_with_binding(
    graph: cudnn.pygraph,
) -> "tuple[FusionChain, GemmBinding | None]":
    """Build the FusionChain AND a variant-pack binding (role -> cuDNN tensor).

    The binding is ``None`` for chain types whose binding is not yet wired up;
    callers that need it should check. See :class:`GemmBinding`."""
    state = _get_state(graph)
    if state is None:
        raise ValueError(
            "graph has no recorded ops — was cudnn.TBD.gemm imported BEFORE the " "graph was built? If not, call cudnn.TBD.gemm.install_recorder() first."
        )
    if not state["ops"]:
        raise ValueError("graph has no ops; nothing to compile")
    return _build_chain(
        state["ops"],
        state["tensor_meta"],
        state["io_dtype"],
        state.get("intermediate_dtype", "fp32"),
        state.get("compute_dtype", "fp32"),
    )


def analyze(graph: cudnn.pygraph) -> FusionChain:
    """Build a FusionChain from a cudnn.pygraph constructed AFTER cudnn.TBD.gemm import."""
    chain, _ = analyze_with_binding(graph)
    return chain
