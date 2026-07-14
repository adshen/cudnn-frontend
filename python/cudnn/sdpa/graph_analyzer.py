# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Engine-agnostic SDPA graph analysis: ``graph.nodes`` -> :class:`SdpaGraphFacts`.

``cudnn.pygraph`` is the Python-native graph IR: it records its op DAG
directly, exposed via ``graph.nodes`` — no construction-time hooks are needed.
This module extracts *facts* (what the graph asks for) without judging
supportedness; each registered engine matches the facts against its own
capability declaration (see ``cudnn.sdpa.fwd.engines``). The parse runs once
per graph and is cached.

Also hosts the graph-side runtime helpers shared by every SDPA engine:
variant-pack resolution and TensorDesc construction.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from typing import Any, Optional

import cudnn
import torch

_LOG = logging.getLogger(__name__)


_DTYPE_FROM_CUDNN = {
    cudnn.data_type.HALF: torch.float16,
    cudnn.data_type.BFLOAT16: torch.bfloat16,
}

# BHSD logical / BSHD physical, size-1 dims wildcarded.
_REQ_STRIDE_ORDER = (3, 1, 2, 0)


def _device_cc() -> Optional[tuple]:
    """Compute capability of the current CUDA device, or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(torch.cuda.current_device())


def _stride_order(dim: tuple, stride: tuple) -> tuple:
    return tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], dim[x[0]])))


def _bshd_ok(dim: tuple, stride: tuple) -> bool:
    order = _stride_order(dim, stride)
    act = tuple(ax for ax in order if dim[ax] != 1)
    exp = tuple(ax for ax in _REQ_STRIDE_ORDER if dim[ax] != 1)
    return act == exp


@dataclass(frozen=True)
class SdpaGraphFacts:
    """What a single-SDPA graph asks for. Pure description — no support judgment.

    Why extraction never judges: supportedness is per-engine knowledge (one
    kernel has paged-KV, another has sinks). The moment a shared parser
    starts rejecting, it becomes an if-ladder that must know every kernel —
    which stops scaling at the second kernel. So this record only DESCRIBES
    (``has_bias=True`` is a fact, not an error) and each engine's
    Capabilities row does the judging in ``fwd/engines.mismatch``.

    ``invalid`` is the one exception: a graph-consistency error (malformed
    regardless of which kernel would run — K/V shape mismatch, padding mask
    without seq_len_kv, ...); when set, every engine is ineligible.
    """

    invalid: Optional[str] = None

    # geometry
    b: int = 0
    h_q: int = 0
    h_kv: int = 0
    s_q: int = 0
    s_kv: int = 0
    d_qk: int = 0
    d_v: int = 0

    # dtypes / layout
    dtype: Optional[torch.dtype] = None  # Q dtype (None if outside the fp16/bf16 map)
    uniform_dtype: bool = True  # K/V/O dtypes equal Q's
    bshd_layout: bool = True  # all of Q/K/V/O in BSHD-physical order

    # masks (resolved cuDNN semantics)
    causal: bool = False  # effective causal upper bound (right band == 0)
    bottom_right: bool = False  # diagonal anchored bottom-right
    window_left: Optional[int] = None  # SWA offset W (cuDNN length - 1)
    right_band_widening: bool = False  # diagonal_band_right_bound > 0

    # requested features
    has_bias: bool = False
    has_dropout: bool = False
    has_score_mod: bool = False
    has_paged_kv: bool = False
    has_alibi: bool = False
    has_unfuse_fma: bool = False
    has_block_mask: bool = False
    has_rng_dump: bool = False
    dynamic_scale: bool = False  # attn_scale passed as a tensor
    seq_q_trim: bool = False  # seq_len_q given without a padding mask

    padded: bool = False  # per-batch KV lengths present (padding mask or THD)
    thd: bool = False  # ragged (THD) Q/K/V
    has_sink: bool = False
    wants_stats: bool = False

    scale: Optional[float] = None
    device_cc: Optional[tuple] = None  # current CUDA device capability

    # IR tensor refs for binding
    q_t: Any = None
    k_t: Any = None
    v_t: Any = None
    o_t: Any = None
    stats_t: Any = None
    sink_t: Any = None
    seq_kv_t: Any = None
    seq_q_t: Any = None


def _single_sdpa_node(graph: "cudnn.pygraph") -> Optional[Any]:
    """The graph's sole SDPA-forward node, or None if the graph is anything else."""
    try:
        nodes = graph.nodes
    except Exception:  # noqa: BLE001 — non-IR graph objects
        return None
    if len(nodes) != 1:
        return None
    node = nodes[0]
    if node.node_type != cudnn.NodeType.SDPA:
        return None
    return node


def _record_from_node(node: Any) -> dict:
    """Flatten an SDPA node into one kwargs-style dict.

    ``node.params`` holds the scalar sdpa() kwargs verbatim; tensor kwargs are
    named ports in ``node.inputs`` (port name == kwarg name); O / Stats (and
    the output-style kwargs like rng_dump) live in ``node.outputs``.
    """
    rec: dict = dict(node.params)
    for port, t in node.inputs.items():
        rec.setdefault(port, t)
    rec["o"] = node.outputs.get("O")
    rec["stats"] = node.outputs.get("Stats")
    if rec.get("rng_dump") is None:
        rec["rng_dump"] = node.outputs.get("rng_dump")
    if node.params.get("_dropout_n"):
        rec["dropout"] = True  # any dropout spec (tensors or probability) is requested
    return rec


def _invalid(reason: str) -> SdpaGraphFacts:
    return SdpaGraphFacts(invalid=f"cudnn.sdpa: {reason}")


def _extract_facts(rec: dict) -> SdpaGraphFacts:
    q, k, v, o = rec.get("q"), rec.get("k"), rec.get("v"), rec.get("o")
    if q is None or k is None or v is None or o is None:
        return _invalid("missing q/k/v/o on the sdpa() node")

    q_dim, q_stride = tuple(q.get_dim()), tuple(q.get_stride())
    k_dim, k_stride = tuple(k.get_dim()), tuple(k.get_stride())
    v_dim, v_stride = tuple(v.get_dim()), tuple(v.get_stride())
    o_dim, o_stride = tuple(o.get_dim()), tuple(o.get_stride())
    if len({len(q_dim), len(k_dim), len(v_dim), len(o_dim), 4}) != 1:
        return _invalid("Q/K/V/O must all be rank-4 (B, H, S, D)")

    b, h_q, s_q, d_qk = q_dim
    _, h_kv, s_kv, _ = k_dim
    d_v = v_dim[-1]
    if k_dim != (b, h_kv, s_kv, d_qk):
        return _invalid("K shape mismatch")
    if v_dim != (b, h_kv, s_kv, d_v):
        return _invalid("V shape mismatch")
    if o_dim != (b, h_q, s_q, d_v):
        return _invalid("O shape mismatch")
    if any(x <= 0 for x in (b, h_q, h_kv, s_q, s_kv, d_qk, d_v)):
        return _invalid("B/H/S/D must all be > 0")
    if h_q % h_kv != 0:
        return _invalid("H_q must be divisible by H_kv (GQA / MQA)")

    q_dtype = _DTYPE_FROM_CUDNN.get(q.get_data_type())
    uniform = all(_DTYPE_FROM_CUDNN.get(t.get_data_type()) == q_dtype for t in (k, v, o))
    bshd = all(_bshd_ok(d, s) for d, s in ((q_dim, q_stride), (k_dim, k_stride), (v_dim, v_stride), (o_dim, o_stride)))

    # Masks: resolve cuDNN's several spellings to (causal, bottom_right, window_left).
    use_causal = bool(rec.get("use_causal_mask", False))
    use_causal_br = bool(rec.get("use_causal_mask_bottom_right", False))
    swa_len = rec.get("sliding_window_length")
    left_bound = swa_len if swa_len is not None else rec.get("diagonal_band_left_bound")
    if use_causal or use_causal_br:
        resolved_right = 0
        align_is_br = use_causal_br
    else:
        resolved_right = rec.get("diagonal_band_right_bound")
        align_is_br = rec.get("diagonal_alignment") == cudnn.diagonal_alignment.BOTTOM_RIGHT
    right_widening = resolved_right not in (0, None)
    causal = (resolved_right is not None) and not right_widening
    if left_bound is not None and left_bound < 1:
        return _invalid(f"sliding-window length must be >= 1; got {left_bound}")
    window_left = (left_bound - 1) if left_bound is not None else None

    # Padding / THD.
    thd = getattr(q, "ragged_offset", None) is not None
    use_padding_mask = bool(rec.get("use_padding_mask", False))
    seq_len_kv = rec.get("seq_len_kv")
    seq_len_q = rec.get("seq_len_q")
    seq_q_trim = False
    if thd:
        if getattr(k, "ragged_offset", None) is None or getattr(v, "ragged_offset", None) is None:
            return _invalid("THD (ragged) requires ragged Q, K, and V")
        if seq_len_q is None or seq_len_kv is None:
            return _invalid("THD (ragged) requires seq_len_q and seq_len_kv")
        padded = True
    else:
        if use_padding_mask and seq_len_kv is None:
            return _invalid("use_padding_mask requires seq_len_kv")
        seq_q_trim = seq_len_q is not None and not use_padding_mask
        padded = use_padding_mask and seq_len_kv is not None

    sink_token = rec.get("sink_token")
    if sink_token is not None:
        sink_dim = tuple(sink_token.get_dim())
        if sink_dim != (1, h_q, 1, 1):
            return _invalid(f"sink_token must be (1, H_q, 1, 1); got {sink_dim}")

    generate_stats = rec.get("generate_stats")
    is_inference = rec.get("is_inference")
    if generate_stats is not None:
        wants_stats = bool(generate_stats)
    elif is_inference is not None:
        wants_stats = not bool(is_inference)
    else:
        wants_stats = False
    stats = rec.get("stats")
    if wants_stats and stats is None:
        return _invalid("generate_stats=True but no Stats tensor was returned")

    attn_scale = rec.get("attn_scale")
    dynamic_scale = attn_scale is not None and not isinstance(attn_scale, (int, float))
    scale = float(attn_scale) if (attn_scale is not None and not dynamic_scale) else None

    return SdpaGraphFacts(
        b=b,
        h_q=h_q,
        h_kv=h_kv,
        s_q=s_q,
        s_kv=s_kv,
        d_qk=d_qk,
        d_v=d_v,
        dtype=q_dtype,
        uniform_dtype=uniform,
        bshd_layout=bshd,
        causal=causal,
        bottom_right=bool(align_is_br),
        window_left=window_left,
        right_band_widening=right_widening,
        has_bias=rec.get("bias") is not None,
        has_dropout=rec.get("dropout") is not None,
        has_score_mod=rec.get("fn") is not None,
        has_paged_kv=(rec.get("paged_attention_k_table") is not None or rec.get("paged_attention_v_table") is not None),
        has_alibi=bool(rec.get("use_alibi_mask")),
        has_unfuse_fma=bool(rec.get("unfuse_fma")),
        has_block_mask=rec.get("block_mask") is not None,
        has_rng_dump=rec.get("rng_dump") is not None,
        dynamic_scale=dynamic_scale,
        seq_q_trim=seq_q_trim,
        padded=padded,
        thd=thd,
        has_sink=sink_token is not None,
        wants_stats=wants_stats,
        scale=scale,
        device_cc=_device_cc(),
        q_t=q,
        k_t=k,
        v_t=v,
        o_t=o,
        stats_t=(stats if wants_stats else None),
        sink_t=sink_token,
        seq_kv_t=seq_len_kv,
        seq_q_t=(seq_len_q if thd else None),
    )


# Parse cache: one facts extraction per graph, because N registered engines
# each probe the same graph (create_execution_plans, then again at build) and
# the graph walk is the expensive part — capability matching is cheap field
# comparisons. Keyed weakly so graphs can be GC'd; invalidated if the node
# count changes (a graph mutated after probing gets re-parsed).
_FACTS_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def analyze(graph: "cudnn.pygraph") -> Optional[SdpaGraphFacts]:
    """Facts for a single-SDPA graph, or None if the graph is anything else."""
    node = _single_sdpa_node(graph)
    if node is None:
        return None
    try:
        cached = _FACTS_CACHE.get(graph)
    except TypeError:  # non-weakrefable graph objects
        cached = None
    if cached is not None and cached[0] == len(graph.nodes):
        return cached[1]
    facts = _extract_facts(_record_from_node(node))
    try:
        _FACTS_CACHE[graph] = (len(graph.nodes), facts)
    except TypeError:
        pass
    return facts


# ---------------------------------------------------------------------------
# Graph-side runtime helpers shared by SDPA engines
# ---------------------------------------------------------------------------


@dataclass
class SdpaBinding:
    q: Any
    k: Any
    v: Any
    o: Any
    stats: Any = None
    sink_token: Any = None
    seq_len_kv: Any = None
    seq_len_q: Any = None

    def bound_tensors(self) -> list:
        return [t for t in (self.q, self.k, self.v, self.o, self.stats, self.sink_token, self.seq_len_kv, self.seq_len_q) if t is not None]


def _safe_name(t: Any) -> Optional[str]:
    try:
        nm = t.get_name()
    except Exception:  # noqa: BLE001
        return None
    return nm or None


def _safe_uid(t: Any) -> Optional[int]:
    try:
        uid = t.get_uid()
    except Exception:  # noqa: BLE001
        return None
    return uid if isinstance(uid, int) and uid > 0 else None


def resolve_variant_pack(variant_pack: dict, binding: SdpaBinding) -> dict:
    if not isinstance(variant_pack, dict):
        raise TypeError(
            "cudnn.sdpa: compiled plans are called with a variant-pack dict " f"{{cudnn_tensor | uid | name: buffer}}; got {type(variant_pack).__name__}"
        )
    bound = binding.bound_tensors()
    by_obj = {id(t): t for t in bound}

    name_counts: dict = {}
    uid_counts: dict = {}
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

    resolved: dict = {}
    for key, buf in variant_pack.items():
        if id(key) in by_obj:
            t = by_obj[id(key)]
        elif isinstance(key, int) and key in by_uid:
            t = by_uid[key]
        elif isinstance(key, str) and key in by_name:
            t = by_name[key]
        else:
            continue
        resolved[id(t)] = buf
    return resolved


def tensor_desc_from_ir(t: Any, name: str = "") -> "TensorDesc":
    from cudnn.api_base import TensorDesc

    shape = tuple(t.get_dim())
    stride = tuple(t.get_stride())
    dtype = _DTYPE_FROM_CUDNN[t.get_data_type()]
    return TensorDesc(
        dtype=dtype,
        shape=shape,
        stride=stride,
        stride_order=_stride_order(shape, stride),
        device=torch.device("cuda", torch.cuda.current_device()),
        name=name,
    )
