"""Analyze a user-built ``cudnn.pygraph`` and produce a ``FusionChain``.

``cudnn.pygraph`` is the Python-native graph IR: it records its op DAG directly,
exposed via ``graph.nodes`` / ``graph.tensors``. ``analyze(g)`` reads that IR, so a
graph is analyzable whenever it is built — no construction-time hook required.
"""

from __future__ import annotations

import logging
import threading
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

# Dtype + op tables

from .dtypes import CUDNN_FROM_DTYPE as _CUDNN_FROM_DTYPE
from .dtypes import DTYPE_FROM_CUDNN as _DTYPE_FROM_CUDNN


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


# Internal recording state, attached to each cudnn.pygraph instance


@dataclass
class _RecordedOp:
    cudnn_name: str
    op_name: str
    inputs: list[int]
    output: int
    output_tensor: Any  # strong ref so id() stays valid
    compute_dtype: Dtype | None = None  # per-op compute_data_type override (None → graph default)
    # block_scale_dequantize: the [non-K, K] block size (e.g. [1,16] for A). None otherwise.
    block_size: tuple[int, ...] | None = None
    is_negative_scale: bool = False
    # block_scale_quantize: quantized output + the SF side-output from cuDNN.
    scale_output: int | None = None
    scale_output_tensor: Any = None
    quant_axis: int | None = None
    quant_transpose: bool = False
    moe_mode: str | None = None  # moe_grouped_matmul mode; None otherwise
    reduction_mode: str | None = None  # "add"/"amax"/"max"/"min"; None otherwise
    # Optional groupOffset input for grouped reductions (MoE: == first_token_offset).
    group_offset: int | None = None


@dataclass
class _TensorMeta:
    name: str
    dim: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Dtype
    is_input: bool = False
    # SF reorder layout name (e.g. "F8_128x4") or None for the default (NONE).
    reordering: str | None = None
    # Strong ref to the cuDNN tensor object, used to bind each role for the
    # variant-pack dict (uid / name / object) instead of positional args.
    tensor: Any = None


# The native tensor mirrors set_output / set_data_type / set_dim / set_reordering_type
# onto its own attributes; these side tables (keyed by id(tensor)) carry them into
# the analyzer, repopulated from graph.nodes each analyze (see _state_from_graph).
_TENSOR_OUTPUT_FLAG: dict[int, bool] = {}
_TENSOR_EXPLICIT_DTYPE: dict[int, Any] = {}
_TENSOR_DIM_OVERRIDE: dict[int, tuple[int, ...]] = {}
_TENSOR_REORDERING_OVERRIDE: dict[int, str | None] = {}

# The side tables above are module-global; serialize the analyze path around them.
_ANALYZE_LOCK = threading.RLock()


# Variant-pack binding — maps each graph role to its cuDNN tensor


@dataclass
class GemmBinding:
    """Maps each graph role to its cuDNN tensor so a compiled kernel takes a
    variant-pack dict (keyed by tensor object / uid / name) not positional args.

    Operand lists are in kernel distinct-slot order; ``outputs`` in
    :pyattr:`FusionChain.outputs` slot order (terminal, then taps); ``aux`` in
    :pyattr:`FusionChain.aux_tensors` order. Block-scale fills ``sfa/sfb_operands``
    parallel to ``a/b_operands``; MoE fills ``first_token_offset``."""

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
    """Build a GemmBinding for the multi-operand builders (multi-GEMM / MoE):
    the cuDNN tensor per distinct A/B slot (+ its SF for block-scale) from ``meta``."""

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
    # uid is -1/0 until build_operation_graph(); only a positive uid is a valid key.
    return uid if isinstance(uid, int) and uid > 0 else None


def resolve_variant_pack(variant_pack: dict, binding: GemmBinding) -> dict[int, Any]:
    """Resolve a ``{key: buffer}`` variant pack to ``{id(bound_tensor): buffer}``.

    Keys may be the cuDNN tensor object, its uid (once positive), or its name.
    Raises on an unknown / unmatched key."""
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


# MoE / reduction mode maps (cuDNN enum -> our literal)

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


# Reading the native cudnn.pygraph IR (graph.nodes) into analyzer state


def _map_dtype(dt: Any) -> "Dtype | None":
    """cuDNN data_type enum (or None) -> our Dtype literal (or None)."""
    if dt is None:
        return None
    return _DTYPE_FROM_CUDNN.get(dt)


def _reordering_name(t: Any) -> "str | None":
    """SF reorder layout name (e.g. ``F8_128x4``) of a tensor, or None (default)."""
    rt = getattr(t, "reordering_type", None)
    name = getattr(rt, "name", None) if rt is not None else None
    return name if (name and name != "NONE") else None


def _node_to_recorded_op(node: Any) -> "_RecordedOp | None":
    """Translate one native-IR ``Node`` into a :class:`_RecordedOp`, or None for a
    node type outside the GEMM family (ignored — the analyzer only consumes the
    matmul / pointwise / block-scale / MoE / reduction ops)."""
    node_type = node.node_type.name
    name = node.name
    compute = _map_dtype(node.compute_data_type)
    if node_type == "MATMUL":
        A, B = node.inputs["A"], node.inputs["B"]
        out = node.outputs["C"]
        return _RecordedOp("matmul", name, [id(A), id(B)], id(out), out, compute_dtype=compute)
    if node_type == "MOE_GROUPED_MATMUL":
        tok = node.inputs["token"]
        weight = node.inputs["weight"]
        fto = node.inputs["first_token_offset"]
        out = node.outputs["OUT_0"]
        mode = node.params.get("mode", cudnn.moe_grouped_matmul_mode.NONE)
        return _RecordedOp(
            "moe_grouped_matmul",
            name,
            [id(tok), id(weight), id(fto)],
            id(out),
            out,
            compute_dtype=compute,
            moe_mode=_MOE_MODE_FROM_CUDNN.get(mode, "none"),
        )
    if node_type == "REDUCTION":
        inp = node.inputs["input"]
        out = node.outputs["OUT_0"]
        group_offset = node.inputs.get("group_offset")
        return _RecordedOp(
            "reduction",
            name,
            [id(inp)],
            id(out),
            out,
            compute_dtype=compute,
            reduction_mode=_REDUCTION_MODE_FROM_CUDNN.get(node.params.get("mode")),
            group_offset=(id(group_offset) if group_offset is not None else None),
        )
    if node_type == "BLOCK_SCALE_DEQUANTIZE":
        inp = node.inputs["input"]
        descale = node.inputs["descale"]
        out = node.outputs["OUT_0"]
        block_size = node.params.get("block_size")
        return _RecordedOp(
            "block_scale_dequantize",
            name,
            [id(inp), id(descale)],
            id(out),
            out,
            compute_dtype=compute,
            block_size=tuple(block_size) if block_size is not None else None,
            is_negative_scale=bool(node.params.get("is_negative_scale", False)),
        )
    if node_type == "BLOCK_SCALE_QUANTIZE":
        inp = node.inputs["input"]
        quantized = node.outputs["Y"]
        scale = node.outputs["scale"]
        block_size = node.params.get("block_size")
        if isinstance(block_size, (list, tuple)):
            if len(block_size) != 1:
                raise NotImplementedError(f"block_scale_quantize expects a scalar block_size in cudnn.frost.gemm; got {block_size!r}")
            block_size_i = int(block_size[0])
        else:
            block_size_i = int(block_size)
        axis = node.params.get("axis")
        return _RecordedOp(
            "block_scale_quantize",
            name,
            [id(inp)],
            id(quantized),
            quantized,
            compute_dtype=compute,
            block_size=(block_size_i,),
            scale_output=id(scale),
            scale_output_tensor=scale,
            quant_axis=-1 if axis is None else int(axis),
            quant_transpose=bool(node.params.get("transpose", False)),
        )
    if node_type == "POINTWISE":
        out = node.outputs["OUT_0"]
        return _RecordedOp(
            node.params.get("mode"),
            name,
            [id(t) for t in node.inputs.values()],
            id(out),
            out,
            compute_dtype=compute,
        )
    return None


def _state_from_graph(graph: cudnn.pygraph) -> dict:
    """Read a Python-native ``cudnn.pygraph`` into the analyzer's working state:
    the op list + per-tensor metadata + graph dtype defaults, plus the tensor-flag
    side tables. The native graph exposes its op DAG directly via ``graph.nodes``,
    so nothing is recorded at construction time."""
    _TENSOR_OUTPUT_FLAG.clear()
    _TENSOR_EXPLICIT_DTYPE.clear()
    _TENSOR_DIM_OVERRIDE.clear()
    _TENSOR_REORDERING_OVERRIDE.clear()

    ctx = graph.context
    raw_io = getattr(ctx, "io_data_type", None)
    raw_intermediate = getattr(ctx, "intermediate_data_type", None)
    raw_compute = getattr(ctx, "compute_data_type", None)
    io_dtype = _map_dtype(raw_io)
    intermediate_dtype = _map_dtype(raw_intermediate)
    compute_dtype = _map_dtype(raw_compute)
    for _raw, _mapped, _field in (
        (raw_io, io_dtype, "io_data_type"),
        (raw_intermediate, intermediate_dtype, "intermediate_data_type"),
        (raw_compute, compute_dtype, "compute_data_type"),
    ):
        if _raw is not None and _mapped is None:
            raise ValueError(f"unsupported {_field}: {_raw!r}")
    io_dtype = io_dtype or "bf16"
    intermediate_dtype = intermediate_dtype or "fp32"
    compute_dtype = compute_dtype or "fp32"

    nodes = list(graph.nodes)
    produced: set[int] = set()
    for node in nodes:
        for out in node.outputs.values():
            if out is not None:
                produced.add(id(out))

    tensor_meta: dict[int, _TensorMeta] = {}

    def _register(t: Any) -> None:
        if t is None or id(t) in tensor_meta:
            return
        reordering = _reordering_name(t)
        tensor_meta[id(t)] = _TensorMeta(
            name=t.get_name(),
            dim=tuple(t.dim),
            stride=tuple(t.stride),
            dtype=_map_dtype(t.get_data_type()),
            is_input=id(t) not in produced,
            reordering=reordering,
            tensor=t,
        )
        if getattr(t, "data_type", None) is not None:
            _TENSOR_EXPLICIT_DTYPE[id(t)] = t.get_data_type()
        if getattr(t, "dim_assigned", False) and id(t) in produced:
            _TENSOR_DIM_OVERRIDE[id(t)] = tuple(t.dim)
        if reordering is not None:
            _TENSOR_REORDERING_OVERRIDE[id(t)] = reordering

    for node in nodes:
        for t in node.inputs.values():
            _register(t)
        for t in node.outputs.values():
            _register(t)

    # Materialized (non-virtual) op outputs, in node order -> terminal = last.
    for node in nodes:
        for out in node.outputs.values():
            if out is not None and not out.is_virtual:
                _TENSOR_OUTPUT_FLAG[id(out)] = True

    ops: list[_RecordedOp] = []
    for node in nodes:
        recorded = _node_to_recorded_op(node)
        if recorded is not None:
            ops.append(recorded)

    for op in ops:
        if op.cudnn_name == "block_scale_dequantize":
            in_meta = tensor_meta.get(op.inputs[0])
            out_meta = tensor_meta.get(op.output)
            if in_meta is not None and out_meta is not None:
                out_meta.dim = in_meta.dim
                out_meta.stride = in_meta.stride

    return {
        "ops": ops,
        "tensor_meta": tensor_meta,
        "io_dtype": io_dtype,
        "intermediate_dtype": intermediate_dtype,
        "compute_dtype": compute_dtype,
    }


def _graph_has_gemm(graph: cudnn.pygraph) -> bool:
    """True if the graph has any matmul / MoE grouped-matmul node (GEMM candidate)."""
    try:
        for node in graph.nodes:
            if node.node_type.name in ("MATMUL", "MOE_GROUPED_MATMUL"):
                return True
    except Exception:  # noqa: BLE001 — a probe must never break the native path
        return False
    return False


# GEMM engine ("frost_gemm_eng0"), registered with the shared cudnn.frost dispatch (see
# cudnn/frost/heuristics.py). probe_gemm_plan = eligibility (no compile);
# build_gemm_plan = JIT when selected. Forced-config callers use jit_from_cudnn_graph.


def probe_gemm_plan(graph: cudnn.pygraph) -> bool:
    """Cheap eligibility check for the GEMM engine (analyze + support gates, NO
    ``cute.compile``). Never raises (a probe must not break the native path)."""
    if not _graph_has_gemm(graph):
        return False
    from .compiler import probe_supported

    try:
        probe_supported(graph)
    except (NotImplementedError, ValueError):
        return False
    except Exception:  # noqa: BLE001
        _LOG.debug(
            "cudnn.frost.gemm: probe_supported raised unexpectedly; ineligible",
            exc_info=True,
        )
        return False
    return True


def build_gemm_plan(graph: cudnn.pygraph):
    """Analyze + JIT the graph into a compiled GEMM plan.

    Returns a callable :class:`CompiledFusedGemm`; raises ``NotImplementedError`` /
    ``ValueError`` (type + message preserved) on rejection."""
    if not _graph_has_gemm(graph):
        raise ValueError("cudnn.frost.gemm: graph has no matmul / moe_grouped_matmul node; nothing to compile")
    from .compiler import jit_from_cudnn_graph
    from .tile_config import select_config

    chain = analyze(graph)
    config, cta_group, scheduler = select_config(chain.matmul.M, chain.matmul.N, chain.num_gemms)
    return jit_from_cudnn_graph(graph, config=config, cta_group=cta_group, scheduler=scheduler)


# Analyzer


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
    """Declared data_type of a chain tensor: explicit set_data_type, else io_dtype
    if a materialized output, else intermediate_dtype.

    The running value is rounded to this dtype before downstream ops read it, so a
    narrow declared dtype loses precision on purpose (matches cuDNN, even virtual)."""
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
    """Build a FusionChain for one MoE grouped matmul forward pass.

    ``out[fto[g]:fto[g+1]] = token[range] @ weight[g % E].T`` per routed group g.
    POC scope: one moe op, mode=="none", optional terminal block_scale_quantize.
    token = A (1,M=T,K=H), weight = B (E,K=H,N); per-expert selection is in
    :class:`MoeSpec`, NOT the MatmulSpec batch (output is a single (1,T,N) plane)."""
    from .fusion_ir import MoeSpec

    if len(moe_ops) != 1 or len([o for o in ops if o.cudnn_name == "matmul"]):
        raise ValueError("POC scope is exactly one moe_grouped_matmul per graph and no plain " f"matmul; found {len(moe_ops)} moe op(s)")
    moe = moe_ops[0]
    if moe.moe_mode != "none":
        raise NotImplementedError(f"MoE grouped matmul mode {moe.moe_mode!r} is out of POC scope; " "only mode=NONE is supported (gather / scatter rejected)")
    token_id, weight_id, fto_id = moe.inputs
    fto_meta = meta.get(fto_id)
    # first_token_offset dtype (int32/int64 both valid; baked in). Default int32.
    offset_dtype = fto_meta.dtype if fto_meta is not None else "int32"
    num_groups = int(fto_meta.dim[0]) if fto_meta is not None and fto_meta.dim else 1

    # Block-scaled MoE detection (structural): if token/weight come from
    # block_scale_dequantize nodes, fold dequant(s) + moe into one block-scale
    # matmul (redirect to packed data tensors, capture SFA/SFB). NO validation
    # here (the compiler decides which combos run); mirrors _build_chain.
    from .fusion_ir import BlockScaleSpec

    dequant_by_output = {op.output: op for op in ops if op.cudnn_name == "block_scale_dequantize"}
    block_scale_spec: "BlockScaleSpec | None" = None
    a_dtype = meta[token_id].dtype
    b_dtype = meta[weight_id].dtype
    if token_id in dequant_by_output or weight_id in dequant_by_output:

        def _capture_side(operand_id: int):
            deq = dequant_by_output.get(operand_id)
            if deq is None:
                return dict(
                    data_dtype=meta[operand_id].dtype,
                    block_size_2d=None,
                    sf_dtype=None,
                    sf_reorder=None,
                    deq_compute=None,
                    deq_out=None,
                )
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
    # token [1,T,H] → A=(1,M=T,K=H). weight [E,H,N] → B=(E,K=H,N).
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
            raise NotImplementedError(f"reduction {red.op_name!r} mode is not supported by cudnn.frost.gemm; " "supported modes are ADD, AMAX, MAX, and MIN")
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
    # Binding: token/weight resolve through any dequant to the packed data tensors
    # (SF = dequant's 2nd input); terminal = raw MoE output or the quantized tensor
    # (with the quant scale as a side output).
    deq_tok = dequant_by_output.get(token_id)
    deq_w = dequant_by_output.get(weight_id)
    a_data_id = deq_tok.inputs[0] if deq_tok else token_id
    b_data_id = deq_w.inputs[0] if deq_w else weight_id
    binding = GemmBinding(
        a_operands=[meta[a_data_id].tensor],
        b_operands=[meta[b_data_id].tensor],
        outputs=([terminal_tensor] + reduction_objs + ([quant_scale_obj] if quant_scale_obj is not None else [])),
        first_token_offset=meta[fto_id].tensor,
        sfa_operands=(([meta[deq_tok.inputs[1]].tensor] if deq_tok else [None]) if block_scale_spec is not None else []),
        sfb_operands=(([meta[deq_w.inputs[1]].tensor] if deq_w else [None]) if block_scale_spec is not None else []),
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
    ``first_token_offset`` and one pointwise epilogue DAG (e.g. grouped SwiGLU).

    All GEMMs must share the routed-group layout (same fto), shape / major / dtype,
    and expert count. Operands deduped by tensor id (shared token → one A operand).
    POC scope: mode=="none", no mainloop fusion, terminal must be a fusion op or a
    block_scale_quantize fed by one. Block-scale supported (dequant folds into a
    shared :class:`BlockScaleSpec`)."""
    from .fusion_ir import BlockQuantizeSpec, BlockScaleSpec, MoeSpec

    for moe in moe_ops:
        if moe.moe_mode != "none":
            raise NotImplementedError(
                f"MoE grouped matmul mode {moe.moe_mode!r} is out of POC scope; " "only mode=NONE is supported (gather / scatter rejected)"
            )

    # All GEMMs must share the SAME first_token_offset (identical routed-group layout).
    fto_id = moe_ops[0].inputs[2]
    for moe in moe_ops[1:]:
        if moe.inputs[2] != fto_id:
            raise ValueError("parallel MoE grouped matmuls must share the same " "first_token_offset tensor")
    fto_meta = meta.get(fto_id)
    offset_dtype = fto_meta.dtype if fto_meta is not None else "int32"
    num_groups = int(fto_meta.dim[0]) if fto_meta is not None and fto_meta.dim else 1

    # Resolve each moe operand through any dequant, then dedup by PACKED data
    # tensor id (shared dequant → one distinct operand; SF travels with its data).
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

    # Shared BlockScaleSpec (every distinct operand must match GEMM 0's combo).
    block_scale_spec = None
    if is_block_scale:
        a0 = a_caps[a_ids[gemm_operands[0][0]]]
        b0 = b_caps[b_ids[gemm_operands[0][1]]]

        def _combo_key(cap):
            return (
                cap["data_dtype"],
                cap["block_size_2d"],
                cap["sf_dtype"],
                cap["sf_reorder"],
                cap["deq_compute"],
                cap["deq_out"],
            )

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

    # Epilogue DAG over multiple roots (each MoE GEMM output).
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
    reachable_ops = [
        op for op in ops if op.output in reachable_op_ids and op.cudnn_name not in {"moe_grouped_matmul", "matmul", "reduction", "block_scale_quantize"}
    ]
    reachable_quant_ops = [op for op in ops if op.cudnn_name == "block_scale_quantize" and op.output in reachable_op_ids]

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
    terminal_quant: _RecordedOp | None = None
    terminal_source_ref: int | None = None
    for tid in reversed(set_output_ids_in_order):
        qop = next((q for q in reachable_quant_ops if q.output == tid), None)
        if qop is None:
            continue
        (input_id,) = qop.inputs
        if input_id not in op_position_by_id:
            raise ValueError(f"block_scale_quantize {qop.op_name!r} input must be a fusion-op output " "of the shared multi-MoE epilogue")
        terminal_source_ref = op_position_by_id[input_id]
        terminal_id = qop.output
        terminal_quant = qop
        break
    if terminal_id is None:
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
        if out_id == terminal_id or (terminal_quant is not None and terminal_source_ref == op_position_by_id[out_id]):
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

    terminal_op_idx_explicit = -2
    if terminal_quant is not None:
        assert terminal_source_ref is not None
        terminal_op_idx_explicit = terminal_source_ref
    else:
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
            raise NotImplementedError(f"reduction {red.op_name!r} mode is not supported by cudnn.frost.gemm; " "supported modes are ADD, AMAX, MAX, and MIN")
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
        block_quant=block_quant,
    )
    if terminal_quant is not None:
        terminal_obj = terminal_quant.output_tensor
    else:
        terminal_obj = recorded_by_out[terminal_id].output_tensor
    output_objs: list[Any] = [terminal_obj]
    for i, fop in enumerate(fusion_ops):
        if fop.output_tap:
            output_objs.append(recorded_by_out[pending_ops[i][1]].output_tensor)
    output_objs.extend(reduction_objs)
    if quant_scale_obj is not None:
        output_objs.append(quant_scale_obj)
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

    All GEMMs share shape / layout / dtype but may use shared or distinct A / B
    operands (deduped by tensor id). GEMM outputs are the DAG roots; an op refs a
    GEMM output via a negative ``parent_idx`` (``gemm_source(g)``). Block-scale
    supported (dequant folds into the packed tensor; shared dequant → one operand).
    POC scope: no mainloop fusion, no per-GEMM taps, terminal must be a fusion op
    or a block_scale_quantize fed by one."""
    from .fusion_ir import BlockQuantizeSpec, BlockScaleSpec

    # Resolve each matmul operand through any dequant, then dedup by PACKED data
    # tensor id (shared dequant → one distinct operand), matching the runtime dedup.
    dequant_by_output = {op.output: op for op in ops if op.cudnn_name == "block_scale_dequantize"}

    def _capture_side(operand_id: int) -> dict:
        """Packed data id + block-scale fields for a matmul operand (all None for
        a non-dequantized side)."""
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

    # Validate every GEMM shares shape / layout / dtype.
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

    # Shared BlockScaleSpec (every distinct operand must match GEMM 0's combo).
    block_scale_spec = None
    if is_block_scale:
        a0 = a_caps[a_ids[gemm_operands[0][0]]]
        b0 = b_caps[b_ids[gemm_operands[0][1]]]

        def _combo_key(cap):
            return (
                cap["data_dtype"],
                cap["block_size_2d"],
                cap["sf_dtype"],
                cap["sf_reorder"],
                cap["deq_compute"],
                cap["deq_out"],
            )

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

    # Epilogue DAG over multiple roots (each GEMM output).
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
    reachable_ops = [op for op in ops if op.output in reachable_op_ids and op.cudnn_name not in {"matmul", "reduction", "block_scale_quantize"}]
    reachable_quant_ops = [op for op in ops if op.cudnn_name == "block_scale_quantize" and op.output in reachable_op_ids]

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
        """In-chain operand id → producing-op ref: ``gemm_source(g)`` (<0) for a
        GEMM output, else the prior op's index."""
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

    # No fusion epilogue: each GEMM output materializes directly to its own GMEM
    # buffer. Each must set_output(True); GEMM 0 = terminal (slot 0), GEMMs >0 = taps.
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
        # No-epilogue outputs = each GEMM's own buffer (slot 0 = GEMM 0).
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

    # Terminal = last set_output(True) that is a fusion op (or a terminal
    # block_scale_quantize fed by one).
    set_output_ids_in_order = [tid for tid in _TENSOR_OUTPUT_FLAG if _TENSOR_OUTPUT_FLAG[tid]]
    terminal_id: int | None = None
    terminal_quant: _RecordedOp | None = None
    terminal_source_ref: int | None = None
    for tid in reversed(set_output_ids_in_order):
        qop = next((q for q in reachable_quant_ops if q.output == tid), None)
        if qop is None:
            continue
        (input_id,) = qop.inputs
        if input_id not in op_position_by_id:
            raise ValueError(f"block_scale_quantize {qop.op_name!r} input must be a fusion-op output " "of the shared multi-GEMM epilogue")
        terminal_source_ref = op_position_by_id[input_id]
        terminal_id = qop.output
        terminal_quant = qop
        break
    if terminal_id is None:
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
        if out_id == terminal_id or (terminal_quant is not None and terminal_source_ref == op_position_by_id[out_id]):
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

    terminal_op_idx_explicit = -2
    if terminal_quant is not None:
        assert terminal_source_ref is not None
        terminal_op_idx_explicit = terminal_source_ref
    else:
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
            raise NotImplementedError(f"reduction {red.op_name!r} mode is not supported by cudnn.frost.gemm; " "supported modes are ADD, AMAX, MAX, and MIN")
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

    # GEMM output declared dtype — the epilogue rounds each accumulator to it
    # before the op chain. All GEMMs share it; resolve from GEMM 0.
    matmul_out_dtype = _resolve_out_dtype(matmuls[0].output, matmuls[0].output_tensor, io_dtype, intermediate_dtype)

    _mg_term = terminal_quant.output_tensor if terminal_quant is not None else recorded_by_out[terminal_id].output_tensor
    _mg_term_dim = tuple(_mg_term.get_dim()) if _mg_term is not None else ()
    _mg_term_stride = tuple(_mg_term.get_stride()) if _mg_term is not None else ()
    if terminal_quant is not None and (not _mg_term_dim or not _mg_term_stride):
        term_meta = meta.get(terminal_quant.output)
        if term_meta is not None:
            _mg_term_dim = term_meta.dim
            _mg_term_stride = term_meta.stride
    out_major = _infer_out_major(_mg_term_dim, _mg_term_stride)
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
        block_quant=block_quant,
    )
    # Outputs in slot order: terminal, op taps (chain order), reductions, quant
    # scale (no matmul tap).
    output_objs: list[Any] = [_mg_term]
    for i, fop in enumerate(fusion_ops):
        if fop.output_tap:
            output_objs.append(recorded_by_out[pending_ops[i][1]].output_tensor)
    output_objs.extend(reduction_objs)
    if quant_scale_obj is not None:
        output_objs.append(quant_scale_obj)
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
    # MoE grouped matmul (own graph type).
    moe_ops = [op for op in ops if op.cudnn_name == "moe_grouped_matmul"]
    if len(moe_ops) > 1:
        # K parallel MoE grouped matmuls sharing one fto + one epilogue.
        return _build_multi_moe_chain(moe_ops, ops, meta, io_dtype, intermediate_dtype, compute_dtype)
    if moe_ops:
        return _build_moe_chain(moe_ops, ops, meta, io_dtype, intermediate_dtype, compute_dtype)

    matmuls = [op for op in ops if op.cudnn_name == "matmul"]
    if len(matmuls) == 0:
        raise ValueError("POC scope is >=1 matmul per graph; found 0")
    if len(matmuls) > 1:
        # Parallel GEMMs sharing one epilogue (multi-GEMM). Block-scale handled in
        # the builder; mainloop fusion alongside multiple matmuls is out of scope
        # (the builder walks only the epilogue DAG → a pre-MMA op raises there).
        return _build_multi_gemm_chain(matmuls, ops, meta, io_dtype, intermediate_dtype, compute_dtype)
    mm = matmuls[0]
    A_id, B_id = mm.inputs

    # Block-scaled matmul detection (structural): if A and/or B is produced by a
    # block_scale_dequantize node, fold the dequant(s) + matmul into one
    # block-scale matmul (dequant(A)@B, A@dequant(B), or both). Purely STRUCTURAL
    # — NO dtype/block-size/arch rules here; runnability is decided at compile
    # time. Each dequantized operand is redirected to its packed data tensor + SF.
    from .fusion_ir import BlockScaleSpec

    dequant_by_output = {op.output: op for op in ops if op.cudnn_name == "block_scale_dequantize"}
    block_scale_spec: "BlockScaleSpec | None" = None
    sfa_obj = None  # cuDNN SF tensors (block-scale), for the variant-pack binding
    sfb_obj = None
    if A_id in dequant_by_output or B_id in dequant_by_output:

        def _capture_side(operand_id: int):
            """Structural block-scale fields for a (possibly) dequantized operand
            (all None for a non-dequantized side). No validation. ``deq_compute`` =
            dequant math precision; ``deq_out`` = dequant output dtype (the MMA's
            logical input type for this side)."""
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
        # Capture the SF tensors (2nd dequant input) before redirecting A_id/B_id.
        deq_a = dequant_by_output.get(A_id)
        deq_b = dequant_by_output.get(B_id)
        sfa_obj = meta[deq_a.inputs[1]].tensor if deq_a else None
        sfb_obj = meta[deq_b.inputs[1]].tensor if deq_b else None
        # Redirect each scaled operand to its packed data tensor.
        A_id, B_id = a_data_id, b_data_id

    # Mainloop fusion detection: walk backwards from A/B through a chain of unary
    # pointwise ops rooted at a graph input. Each op runs on the mainloop-fusion
    # warps (transforms the SMEM tile in place before the MMA). POC scope: a
    # linear chain of unary ops on A/B only.
    op_by_output = {op.output: op for op in ops if op.cudnn_name != "matmul"}

    # Aux tensors shared across mainloop + epilogue, defined here so the mainloop
    # walk can register its scalar auxes (the epilogue walk keeps appending).
    # Runtime aux order = this list's order; aux_objs = parallel cuDNN tensors.
    aux_tensors: list[TensorRef] = []
    aux_objs: list[Any] = []
    aux_seen: set[int] = set()

    def _is_scalar_input(tid: int) -> bool:
        m = meta.get(tid)
        return m is not None and m.is_input and len(m.dim) > 0 and all(d == 1 for d in m.dim)

    def _walk_mainloop(operand_id: int, label: str) -> tuple[int, list[FusionOp]]:
        """Walk backwards from a matmul operand through pointwise ops (unary, or
        binary with one SCALAR graph-input aux) to the root graph input. Returns
        (root_tensor_id, mainloop_ops) in graph-input -> operand' order; registers
        scalar auxes in aux_tensors."""
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
                fops.append(
                    FusionOp(
                        op=_BINARY_OP_MAP[cudnn_name],
                        aux=aux_name,
                        aux_on_rhs=aux_on_rhs,
                        parent_idx=parent,
                    )
                )
            else:
                fops.append(FusionOp(op=_UNARY_OP_MAP[cudnn_name], parent_idx=parent))
        return cur, fops

    if block_scale_spec is not None:
        # Block-scaled matmul: dequant happens inside the MMA, so no mainloop
        # fusion and no dtype-cast staging.
        root_A_id, mainloop_a_ops = A_id, []
        root_B_id, mainloop_b_ops = B_id, []
    else:
        root_A_id, mainloop_a_ops = _walk_mainloop(A_id, "A")
        root_B_id, mainloop_b_ops = _walk_mainloop(B_id, "B")
    A_meta = meta[root_A_id]
    B_meta = meta[root_B_id]

    def _mma_operand_dtype(operand_id: int) -> Dtype:
        om = meta.get(operand_id)
        return _resolve_out_dtype(operand_id, om.tensor if om else None, io_dtype, intermediate_dtype)

    mma_a_dtype = _mma_operand_dtype(A_id)
    mma_b_dtype = _mma_operand_dtype(B_id)
    mainloop_a_load_dtype = A_meta.dtype if A_meta.dtype != mma_a_dtype else None
    mainloop_b_load_dtype = B_meta.dtype if B_meta.dtype != mma_b_dtype else None

    # The matmul's compute dtype IS the accumulator dtype, recorded faithfully.
    # Runnability of the (a,b,accum) combo is validated by the compiler
    # (_check_supported), not here (the IR has no arch info).
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

    # DAG walk from matmul output: (1) BFS for every reachable op, (2) Kahn topo
    # sort (emit each op only after its in-chain inputs) — handles both fan-out
    # and fan-in.
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

    # In-chain deps of each reachable op = inputs that are the matmul output or
    # another reachable op's output.
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

    # Terminal op = the LAST tensor with set_output(True) in recorder order (its
    # dtype is chain.output_dtype, its result is vec_out at the trailing store).
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
        # No fusion-op output marked set_output → the matmul is the terminal.
        terminal_id = mm.output

    from dataclasses import replace as _replace

    recorded_by_out = {op.output: op for op in ordered_ops}

    fusion_ops: list[FusionOp] = []
    for fop, out_id in pending_ops:
        recorded = recorded_by_out[out_id]
        # Per-op compute precision: graph default unless overridden (validated in
        # FusionOp.__post_init__).
        op_compute = recorded.compute_dtype if recorded.compute_dtype is not None else compute_dtype
        if out_id == terminal_id or (terminal_quant is not None and terminal_source_ref == op_position_by_id[out_id]):
            # Terminal dtype = output_dtype, applied by the trailing vec_out cast
            # — no mid-chain rounding here.
            fusion_ops.append(_replace(fop, compute_dtype=op_compute))
        else:
            # Round this op's result to its declared out dtype (even virtual)
            # before the next op reads it.
            op_out_dtype = _resolve_out_dtype(out_id, recorded.output_tensor, io_dtype, intermediate_dtype)
            fusion_ops.append(
                _replace(
                    fop,
                    compute_dtype=op_compute,
                    out_dtype=op_out_dtype,
                    output_tap=_TENSOR_OUTPUT_FLAG.get(out_id, False),
                )
            )

    # Matmul-output tap: materialize the accumulator as a second GMEM output when
    # C.set_output(True) and the matmul isn't the terminal.
    matmul_output_tap = mm.output != terminal_id and _TENSOR_OUTPUT_FLAG.get(mm.output, False)
    # out_dtype = C's declared dtype; the epilogue rounds the fp32 accumulator to
    # it before any fusion op / output.
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

    # The final tensor's explicit set_data_type (if any) overrides io_dtype
    # (canonical FP8-in / FP16-out downcast pattern).
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

    # Which position in fusion_ops is the terminal. Linear chains default to the
    # last op (FusionChain sentinel); fan-out DAGs must set it explicitly since
    # the terminal may not be last in BFS order.
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
            raise NotImplementedError(f"reduction {red.op_name!r} mode is not supported by cudnn.frost.gemm; " "supported modes are ADD, AMAX, MAX, and MIN")
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

    # Variant-pack binding (role -> cuDNN tensor), single-GEMM. Output objects in
    # slot order: terminal, matmul tap, op taps (chain order), reductions.
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
    See :class:`GemmBinding`."""
    with _ANALYZE_LOCK:
        state = _state_from_graph(graph)
        if not state["ops"]:
            raise ValueError("graph has no ops; nothing to compile")
        return _build_chain(
            state["ops"],
            state["tensor_meta"],
            state["io_dtype"],
            state["intermediate_dtype"],
            state["compute_dtype"],
        )


def analyze(graph: cudnn.pygraph) -> FusionChain:
    """Build a FusionChain from a cudnn.pygraph constructed AFTER cudnn.frost.gemm import."""
    chain, _ = analyze_with_binding(graph)
    return chain
