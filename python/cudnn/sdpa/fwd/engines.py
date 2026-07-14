# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FROST SDPA-forward engine registry: capability declarations + spec table.

One registered engine per architecture x phase x geometry, named

    sdpa_fwd_<phase>_sm<arch>_d<dqk>[x<dv>]

(dtype is NOT part of the identity: a cell's engine serves every dtype its
kernel handles — fp16 and bf16 today — via ``Capabilities.dtypes``.)

The shared analyzer (``cudnn.sdpa.graph_analyzer.analyze``) parses the graph
once into :class:`SdpaGraphFacts`; each engine's probe is a field-by-field
match of those facts against its :class:`Capabilities` row below. Adding an engine is
one ``Capabilities``/spec row plus (usually) one kernel template.

An engine is a *lowering strategy*, not a kernel: its ``lower`` hook receives
the parsed facts and returns an executor, and is free to compile one kernel,
pick among several, or chain multiple launches (the THD path already launches
an O-descriptor builder kernel before the main one). Conversely several
engines may share one template. Neither direction is 1:1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import torch

from cudnn.frost import register_engine
from cudnn.frost.dispatch import requested_knobs
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd.config_sm100 import SCHED_NATURAL

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SdpaFwdKnobs:
    """Per-graph tuning request for the SDPA-forward engines.

    This is the operation's knob *vocabulary* — typed fields, no global enum.
    ``None`` means "no preference". Attach to a graph via
    ``graph.set_engine_knobs(SdpaFwdKnobs(...))``; each engine's
    :class:`Capabilities` row advertises the domain it honors, and the probe
    rejects the engine for any request outside that domain (a knob is honored
    or the engine is ineligible — never silently degraded).
    """

    sched_policy: Optional[int] = None  # tile-scheduler policy (SCHED_NATURAL, ...)
    tile_n: Optional[int] = None  # KV tile width
    cga: Optional[int] = None  # cluster size (CTAs cooperating per tile)


@dataclass(frozen=True)
class Capabilities:
    """What one ENGINE can serve — the envelope of graphs (and, later, requested
    tuning knobs) its lowering can honor. An engine spanning several kernels
    declares the union its lowering can actually deliver. Compared
    field-by-field against SdpaGraphFacts in the probe."""

    arch: tuple = (10, 0)
    phase: str = "prefill"
    d_qk: int = 512
    d_v: int = 512
    dtypes: frozenset = frozenset({torch.float16, torch.bfloat16})

    # optional features a graph may request
    bias: bool = False
    dropout: bool = False
    score_mod: bool = False
    paged_kv: bool = False
    alibi: bool = False
    block_mask: bool = False
    rng_dump: bool = False
    dynamic_scale: bool = False
    unfuse_fma: bool = False
    seq_q_trim: bool = False
    right_band_widening: bool = False

    causal: bool = True
    bottom_right: bool = True
    bottom_right_with_swa: bool = False  # kernel gap: BR diagonal excludes SWA
    swa: bool = True
    padded: bool = True
    sink: bool = True
    stats: bool = True
    thd: bool = True
    thd_bottom_right: bool = False  # kernel gap: BR uses global, not per-seq, Q length
    thd_stats: bool = False  # packed LSE output plumbing is a follow-up

    layouts: frozenset = frozenset({"bshd"})
    skv_tile: int = 128  # KV tail rule (waived when padded / causal covers the tail)

    # Tuning-knob domains this engine's lowering honors (see SdpaFwdKnobs).
    sched_policies: frozenset = frozenset({SCHED_NATURAL})
    tile_ns: frozenset = frozenset({128})
    cgas: frozenset = frozenset({2})


def mismatch(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", requested: Optional[SdpaFwdKnobs] = None) -> Optional[str]:
    """First reason this engine cannot serve these facts (and the requested
    tuning knobs, if any), or None if it can serve both.

    Returns a human-readable reason string rather than a bool on purpose:
    with many engine rows, "why was my engine not eligible" is the first
    debugging question, and the strict-select error surfaces this string.
    Knob requests are validated here — in the probe, before any compile —
    because a request an engine cannot honor must make it INELIGIBLE (so a
    different engine can win), never silently degraded to a default.
    """
    if facts.invalid:
        return facts.invalid
    if requested is not None:
        if not isinstance(requested, SdpaFwdKnobs):
            return f"knob request is a {type(requested).__name__}, not SdpaFwdKnobs — wrong operation's vocabulary"
        for value, domain, label in (
            (requested.sched_policy, capabilities.sched_policies, "sched_policy"),
            (requested.tile_n, capabilities.tile_ns, "tile_n"),
            (requested.cga, capabilities.cgas, "cga"),
        ):
            if value is not None and value not in domain:
                return f"requested {label}={value} is outside this engine's domain {sorted(domain)}"
    if facts.device_cc != capabilities.arch:
        return f"requires SM{capabilities.arch[0]}{capabilities.arch[1]}; current device is {facts.device_cc}"
    if (facts.d_qk, facts.d_v) != (capabilities.d_qk, capabilities.d_v):
        return f"serves D_QK={capabilities.d_qk}/D_V={capabilities.d_v}; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
    if facts.dtype not in capabilities.dtypes:
        return f"dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.dtypes)}"
    if not facts.uniform_dtype:
        return "K/V/O dtypes must match Q"
    if "bshd" in capabilities.layouts and not facts.bshd_layout:
        return "Q/K/V/O must be BSHD-physical (stride order 3,1,2,0)"

    for fact, cap, label in (
        (facts.has_bias, capabilities.bias, "bias"),
        (facts.has_dropout, capabilities.dropout, "dropout"),
        (facts.has_score_mod, capabilities.score_mod, "score_mod"),
        (facts.has_paged_kv, capabilities.paged_kv, "paged attention"),
        (facts.has_alibi, capabilities.alibi, "ALiBi"),
        (facts.has_block_mask, capabilities.block_mask, "block_mask"),
        (facts.has_rng_dump, capabilities.rng_dump, "rng_dump"),
        (facts.dynamic_scale, capabilities.dynamic_scale, "tensor attn_scale"),
        (facts.has_unfuse_fma, capabilities.unfuse_fma, "unfuse_fma"),
        (facts.seq_q_trim, capabilities.seq_q_trim, "seq_len_q without padding mask"),
        (facts.right_band_widening, capabilities.right_band_widening, "causal right-band widening"),
        (facts.causal, capabilities.causal, "causal mask"),
        (facts.window_left is not None, capabilities.swa, "sliding window"),
        (facts.padded, capabilities.padded, "padding mask"),
        (facts.has_sink, capabilities.sink, "sink token"),
        (facts.wants_stats, capabilities.stats, "stats output"),
        (facts.thd, capabilities.thd, "THD / ragged"),
    ):
        if fact and not cap:
            return f"graph uses {label}, which this engine does not support"

    if facts.bottom_right:
        if not facts.causal:
            return "bottom-right alignment requires a causal upper bound"
        if not capabilities.bottom_right:
            return "graph uses bottom-right causal, which this kernel does not support"
        if facts.window_left is not None and not capabilities.bottom_right_with_swa:
            return "bottom-right causal combined with a sliding window is not supported"
        if facts.thd and not capabilities.thd_bottom_right:
            return "THD with bottom-right causal is not supported (per-sequence diagonal gap)"
    if facts.thd and facts.wants_stats and not capabilities.thd_stats:
        return "THD with generate_stats is not supported yet"

    if capabilities.skv_tile and facts.s_kv % capabilities.skv_tile != 0:
        causal_covers_tail = facts.causal and (facts.bottom_right or facts.s_q <= facts.s_kv)
        if not (facts.padded or causal_covers_tail):
            return f"S_kv ({facts.s_kv}) must be a multiple of {capabilities.skv_tile} unless a padding mask is given or the causal mask covers the KV tail"
    return None


@dataclass(frozen=True)
class EngineSpec:
    name: str
    capabilities: Capabilities
    # Lowering strategy: facts -> executor. Defaults to the single-template DSL
    # prefill lowering below; a future engine may select between kernels (e.g.
    # decode vs prefill by S_q) or chain several launches under one name.
    lower: "Callable[[EngineSpec, ga.SdpaGraphFacts, Optional[SdpaFwdKnobs]], Any]" = None


def _spec(d: int) -> EngineSpec:
    return EngineSpec(
        name=f"sdpa_fwd_prefill_sm100_d{d}",
        capabilities=Capabilities(arch=(10, 0), phase="prefill", d_qk=d, d_v=d, dtypes=frozenset({torch.float16, torch.bfloat16})),
    )


ENGINE_SPECS = (
    _spec(512),
    _spec(256),
)


def probe(spec: EngineSpec, graph) -> bool:
    facts = ga.analyze(graph)
    if facts is None:
        return False
    reason = mismatch(spec.capabilities, facts, requested_knobs(graph))
    if reason is not None:
        _LOG.debug("cudnn.sdpa: %s ineligible: %s", spec.name, reason)
        return False
    return True


def build(spec: EngineSpec, graph):
    facts = ga.analyze(graph)
    if facts is None:
        raise ValueError("cudnn.sdpa: graph is not a single sdpa() forward node")
    requested = requested_knobs(graph)
    reason = mismatch(spec.capabilities, facts, requested)
    if reason is not None:
        raise ValueError(f"cudnn.sdpa: {spec.name}: {reason}")
    lower = spec.lower or lower_dsl_prefill
    return lower(spec, facts, requested)


def lower_dsl_prefill(spec: EngineSpec, facts: "ga.SdpaGraphFacts", requested: Optional[SdpaFwdKnobs] = None):
    """Default lowering: one DSL prefill template.

    TemplateParams = graph-derived semantics (from facts) + knob choices
    (requested values where given — already validated by the probe — engine
    defaults otherwise).
    """
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl

    sample_q = ga.tensor_desc_from_ir(facts.q_t, name="q")
    sample_k = ga.tensor_desc_from_ir(facts.k_t, name="k")
    sample_v = ga.tensor_desc_from_ir(facts.v_t, name="v")
    sample_o = ga.tensor_desc_from_ir(facts.o_t, name="o")
    lse_shape = (facts.b, facts.h_q, facts.s_q)
    sample_lse = TensorDesc(
        dtype=torch.float32,
        shape=lse_shape,
        stride=TensorDesc._compute_contiguous_stride(lse_shape),
        stride_order=(2, 1, 0),
        device=torch.device("cuda", torch.cuda.current_device()),
        name="lse",
    )

    api = SdpaFwdDsl(
        sample_q=sample_q,
        sample_k=sample_k,
        sample_v=sample_v,
        sample_o=sample_o,
        sample_lse=sample_lse,
        is_causal=facts.causal,
        causal_bottom_right=facts.bottom_right,
        window_size_left=facts.window_left,
        scale_softmax=facts.scale,
        seq_kv_lens_present=facts.padded,
        has_sink=facts.has_sink,
        thd=facts.thd,
    )
    api.check_support()  # raises ValueError / NotImplementedError if unsupported
    api.compile()

    binding = ga.SdpaBinding(
        q=facts.q_t,
        k=facts.k_t,
        v=facts.v_t,
        o=facts.o_t,
        stats=facts.stats_t,
        sink_token=facts.sink_t,
        seq_len_kv=facts.seq_kv_t,
        seq_len_q=facts.seq_q_t,
    )

    def _execute(variant_pack):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        q_buf = resolved[id(binding.q)]
        k_buf = resolved[id(binding.k)]
        v_buf = resolved[id(binding.v)]
        o_buf = resolved[id(binding.o)]
        lse_buf = resolved.get(id(binding.stats)) if binding.stats is not None else None
        if lse_buf is None:
            lse_buf = torch.empty((facts.b, facts.h_q, facts.s_q), dtype=torch.float32, device=q_buf.device)
        sinks_buf = resolved.get(id(binding.sink_token)) if binding.sink_token is not None else None
        seq_kv_buf = resolved.get(id(binding.seq_len_kv)) if binding.seq_len_kv is not None else None
        seq_q_buf = resolved.get(id(binding.seq_len_q)) if binding.seq_len_q is not None else None
        api.execute(
            q_tensor=q_buf,
            k_tensor=k_buf,
            v_tensor=v_buf,
            o_tensor=o_buf,
            lse_tensor=lse_buf.reshape(facts.b, facts.h_q, facts.s_q),
            scale_softmax=facts.scale,
            sinks=sinks_buf,
            seq_kv_lens=seq_kv_buf,
            seq_len_q=seq_q_buf,
        )
        return None

    return _execute


def engine_name(d: int, phase: str = "prefill", arch: str = "sm100") -> str:
    """The registered engine name for a coverage cell (test/user convenience)."""
    return f"sdpa_fwd_{phase}_{arch}_d{d}"


for _s in ENGINE_SPECS:
    register_engine(_s.name, partial(probe, _s), partial(build, _s))

__all__ = ["Capabilities", "EngineSpec", "ENGINE_SPECS", "SdpaFwdKnobs", "engine_name", "mismatch"]
