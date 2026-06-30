"""Kernel-template registry — six-dimension support funnel.

Whether a kernel can run is a function of SIX dimensions:

    gpu architecture · kernel template · tile config (geometry) · graph type ·
    mma type · other graph info

Tile configs are PURE GEOMETRY (see ``tile_config``); the **template** supplies
the execution strategy — ``cta_group`` (1-CTA vs 2-CTA MMA), ``static_sched``
(CLC vs static scheduler), arch family, and which graph type / mainloop it
handles. A single geometry config is therefore run by several templates; the
registry expands each geometry across the accepting templates.

Judging every combination directly is a combinatorial explosion. Instead we run
a **funnel** — cheapest, coarsest dimension first (mirrored by the layers of
:meth:`KernelTemplate.accepts`):

  1. **graph type** — matmul / block_scale_matmul / moe / convolution. Each
     template declares exactly one (``graph_type`` field). Plus arch-family
     match (``sm100`` templates only accept ``sm100`` configs) and the
     mainloop-fusion axis.
  2. **mma type × arch** — the MMA-instruction input shape vs the target GPU's
     SM, via the unified :data:`MMA_TYPE_SUPPORT` table (matmul + block-scale
     merged into one structure here). Graph-type-level: a failure drops every
     template of that graph type.
  3. **tile config** — which catalog geometries a template accepts, by
     PREDICATE (``candidate_configs`` filtering CATALOG) — incl. the template's
     cta_group constraints (2-CTA MMA needs ``cgrp_size_m % 2 == 0`` etc.) and an
     explicit known-bad exclusion (cluster-m=128). Never a hand-maintained list.
  4. **other graph info** — mainloop-fusion op scope + TMA alignment.

Capability checks are **reused** from ``compiler._check_*`` / its tables (lazy
import; ``compiler`` does not import this module) — single source of truth.

This registry DRIVES the compiler's template-file selection via
:func:`select_template`; the compiler passes the chosen template's ``cta_group``
into the geometry's cta_group-taking methods at render time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .fusion_ir import BINARY_OPS, UNARY_OPS, FusionChain
from .tile_config import CATALOG, TileConfig

# Single source of truth for which pointwise ops a mainloop-fusion template can
# transform in SMEM (the canonical op names FusionOp.op uses).
_SUPPORTED_MAINLOOP_OPS: frozenset[str] = frozenset(UNARY_OPS) | frozenset(BINARY_OPS)


# ---------------------------------------------------------------------------
# Dimension 1: graph type
# ---------------------------------------------------------------------------


class GraphType(Enum):
    """The kind of computation the graph expresses. Each kernel template
    supports exactly one. ``MOE`` / ``CONVOLUTION`` are placeholders — no
    template implements them yet."""

    MATMUL = "matmul"
    BLOCK_SCALE_MATMUL = "block_scale_matmul"
    MOE = "moe"
    MOE_BLOCK_SCALE = "moe_block_scale"  # MoE grouped matmul with block-scaled inputs
    CONVOLUTION = "convolution"  # placeholder — no template yet


def classify_graph_type(chain: FusionChain) -> GraphType:
    """Stage-1 classifier: which graph type this chain is."""
    if chain.has_moe and chain.has_block_scale:
        return GraphType.MOE_BLOCK_SCALE
    if chain.has_block_scale:
        return GraphType.BLOCK_SCALE_MATMUL
    if chain.has_moe:
        return GraphType.MOE
    return GraphType.MATMUL


# ---------------------------------------------------------------------------
# Scheduler axis (template strategy, supplied at compile/traversal time)
# ---------------------------------------------------------------------------

CLC = "clc"
STATIC = "static"
SCHEDULERS: tuple[str, ...] = (CLC, STATIC)


# ---------------------------------------------------------------------------
# Dimension 2: mma type × arch (graph-type-level, config/template independent)
# ---------------------------------------------------------------------------


# ===========================================================================
# Unified MMA-type × arch support — the SINGLE source of truth (the old
# compiler `_PIPELINE_DTYPE_ARCH` matmul table and `_BLOCK_SCALE_SUPPORTED`
# per-side case list were merged into this one structure). Indexed by graph
# type; value is `{mma_type_key -> supported arch ranges}`. The key shape
# differs per graph type — matmul = (a, b, accum) dtype combo; block-scale =
# the full 13-field per-side (data, SF dtype, block size, reorder, dequant
# compute/out) + accum — but the lookup is one code path.
# ===========================================================================


def _matmul_mma_type(chain: FusionChain) -> tuple:
    mm = chain.matmul
    return (mm.a_dtype, mm.b_dtype, mm.accum_dtype)


def _block_scale_mma_type(chain: FusionChain) -> tuple:
    mm = chain.matmul
    bs = chain.block_scale
    assert bs is not None
    return (
        mm.a_dtype,
        bs.sf_dtype_a,
        bs.block_size_a,
        bs.sfa_reorder,
        bs.dequant_compute_a,
        bs.dequant_out_a,
        mm.b_dtype,
        bs.sf_dtype_b,
        bs.block_size_b,
        bs.sfb_reorder,
        bs.dequant_compute_b,
        bs.dequant_out_b,
        mm.accum_dtype,
    )


def _bs_key(a: str, sfa: str, b: str, sfb: str, kblk: int) -> tuple:
    """Construct a block-scale mma-type key from the parts that vary (data + SF
    dtypes, K-block). All current cases share reorder F8_128x4, fp32 dequant
    compute/out, fp32 accumulate, A block=(1,kblk) / B block=(kblk,1)."""
    return (a, sfa, (1, kblk), "F8_128x4", "fp32", "fp32", b, sfb, (kblk, 1), "F8_128x4", "fp32", "fp32", "fp32")


_ARCH_SM100 = ((100, 120),)  # sm100 family (Blackwell): 100..119

# Supported block-scale (data, SF dtype, K-block) cases — shared by the plain
# block-scale matmul and the block-scaled MoE grouped matmul.
_BLOCK_SCALE_CASES = {
    _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 16): _ARCH_SM100,  # nvfp4
    _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32): _ARCH_SM100,  # mxfp4
    _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32): _ARCH_SM100,  # mxfp8 e4m3×e4m3
    _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32): _ARCH_SM100,  # mxfp8 e4m3×e5m2
    _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32): _ARCH_SM100,  # mxfp8 e5m2×e4m3
    _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32): _ARCH_SM100,  # mxfp8 e5m2×e5m2
}

# {GraphType: (mma-type-key-fn, {mma_type_key: arch_ranges})}
MMA_TYPE_SUPPORT: dict[GraphType, tuple] = {
    GraphType.MATMUL: (
        _matmul_mma_type,
        {
            ("bf16", "bf16", "fp32"): _ARCH_SM100,
            ("fp16", "fp16", "fp32"): _ARCH_SM100,
            ("int8", "int8", "int32"): ((100, 101), (110, 111)),
            ("fp8_e4m3", "fp8_e4m3", "fp32"): _ARCH_SM100,
            ("fp8_e4m3", "fp8_e5m2", "fp32"): _ARCH_SM100,
            ("fp8_e5m2", "fp8_e4m3", "fp32"): _ARCH_SM100,
            ("fp8_e5m2", "fp8_e5m2", "fp32"): _ARCH_SM100,
        },
    ),
    # MoE grouped matmul shares the plain-matmul (a, b, accum) dtype key shape.
    # BF16 is the validated path; fp16 / fp8 fall out of the same machinery.
    GraphType.MOE: (
        _matmul_mma_type,
        {
            ("bf16", "bf16", "fp32"): _ARCH_SM100,
            ("fp16", "fp16", "fp32"): _ARCH_SM100,
            ("fp8_e4m3", "fp8_e4m3", "fp32"): _ARCH_SM100,
            ("fp8_e4m3", "fp8_e5m2", "fp32"): _ARCH_SM100,
            ("fp8_e5m2", "fp8_e4m3", "fp32"): _ARCH_SM100,
            ("fp8_e5m2", "fp8_e5m2", "fp32"): _ARCH_SM100,
        },
    ),
    GraphType.BLOCK_SCALE_MATMUL: (_block_scale_mma_type, _BLOCK_SCALE_CASES),
    # MoE grouped matmul with block-scaled inputs shares the same block-scale
    # mma-type key + supported cases.
    GraphType.MOE_BLOCK_SCALE: (_block_scale_mma_type, _BLOCK_SCALE_CASES),
}


def mma_arch_reject(chain: FusionChain, graph_type: GraphType) -> str | None:
    """Stage 2: is this graph's MMA type supported on the active GPU arch?
    ``None`` = yes. ONE lookup over :data:`MMA_TYPE_SUPPORT` for every graph
    type. Independent of tile config / cta_group — a failure drops every
    template of this graph type (e.g. int8 on a non-sm100/110 GPU)."""
    from . import compiler as C

    entry = MMA_TYPE_SUPPORT.get(graph_type)
    if entry is None:
        return f"graph type {graph_type.value!r} has no kernel pipeline yet"
    key_fn, table = entry
    key = key_fn(chain)
    ranges = table.get(key)
    if ranges is None:
        if graph_type is GraphType.MATMUL:
            mm = chain.matmul
            return f"the {graph_type.value} pipeline does not support input/acc " f"dtype combo {mm.a_dtype}x{mm.b_dtype}->{mm.accum_dtype}"
        return f"the {graph_type.value} pipeline does not support this " f"configuration: mma type {key}"
    arch = C._current_sm()
    if arch is not None and not any(lo <= arch < hi for lo, hi in ranges):
        spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in ranges)
        return f"the {graph_type.value} pipeline supports this only on {spans}, " f"but the active GPU is sm_{arch}"
    return None


# ---------------------------------------------------------------------------
# Kernel template — owns the execution-strategy axes (cta_group / static / arch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelTemplate:
    """One kernel template. Carries the execution-strategy axes the pure-geometry
    config does NOT: ``arch`` family, ``cta_group`` (1/2), ``static_sched``,
    plus ``graph_type`` and the ``mainloop`` axis. ``accepts`` runs the funnel,
    using ``self.cta_group`` for the geometry's cta_group-dependent gates."""

    file: str  # template filename under kernel_templates/
    arch: str  # arch family ("sm100"); matched to config.arch
    cta_group: int  # 1 or 2 (1-CTA vs 2-CTA MMA)
    static_sched: bool  # no-CLC static scheduler
    graph_type: GraphType  # the single graph type this template supports
    mainloop: bool  # mainloop-fusion variant (transform A/B before MMA)
    # Active-GPU SM range [sm_lo, sm_hi) this template runs on. sm100 templates
    # cover the whole Blackwell family (100..119).
    sm_lo: int = 100
    sm_hi: int = 120
    # Multi-GEMM (parallel matmuls sharing one epilogue): only the 1ctamma CLC
    # template implements it this pass. Other templates reject multi-GEMM chains.
    supports_multi_gemm: bool = False

    @property
    def block_scale(self) -> bool:
        """True iff this template consumes block-scaled (FP4/FP8 + SF) inputs —
        plain block-scale matmul OR block-scaled MoE grouped matmul."""
        return self.graph_type in (GraphType.BLOCK_SCALE_MATMUL, GraphType.MOE_BLOCK_SCALE)

    @property
    def scheduler(self) -> str:
        return STATIC if self.static_sched else CLC

    # -- stage 0: active-GPU SM range (template runs only on [sm_lo, sm_hi)) --

    def arch_active_reject(self) -> str | None:
        """``None`` if the active GPU's SM is in this template's [sm_lo, sm_hi)
        range (or no GPU is visible — render-only / CI). This is the per-TEMPLATE
        arch gate (vs the graph-type-level :func:`mma_arch_reject`)."""
        from . import compiler as C

        sm = C._current_sm()
        if sm is not None and not (self.sm_lo <= sm < self.sm_hi):
            return f"template {self.file} runs only on {self.sm_lo} <= SM < " f"{self.sm_hi}, but the active GPU is sm_{sm}"
        return None

    # -- stage 1: arch / graph-type / mainloop axes --------------------------

    def _axis_reject(self, chain: FusionChain, config: TileConfig, graph_type: GraphType) -> str | None:
        if config.arch != self.arch:
            return f"config arch {config.arch} != template arch {self.arch}"
        if graph_type is not self.graph_type:
            return f"graph_type {graph_type.value} != " f"template graph_type {self.graph_type.value}"
        if chain.has_mainloop_fusion != self.mainloop:
            return f"graph mainloop_fusion={chain.has_mainloop_fusion} != " f"template mainloop={self.mainloop}"
        if chain.is_multi_gemm and not self.supports_multi_gemm:
            return f"template {self.file} does not support multi-GEMM " f"({chain.num_gemms} parallel GEMMs); only the 1ctamma CLC " "template does this pass"
        return None

    # -- stage 3: tile-config gates (this template's cta_group on the geometry)

    def _config_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        # cta_group constraints (moved off TileConfig — they are the template's).
        if self.cta_group == 2:
            if config.cgrp_size_m % 2 != 0:
                return f"2-CTA MMA needs cgrp_size_m % 2 == 0; " f"config has cgrp_size_m={config.cgrp_size_m}"
            if config.cta_tile_n % 2 != 0:
                return f"2-CTA MMA needs cta_tile_n even (pair splits B's N); " f"config has cta_tile_n={config.cta_tile_n}"
            # known-bad: cluster-m=128 (cta_tile_m=64 under 2-CTA MMA) is in
            # CATALOG but miscomputes (~87% wrong) until a working reference
            # lands — excluded from traversal. (select_template still renders it
            # for deliberate single-point probing.)
            if config.cta_tile_m == 64:
                return "cluster-m=128 (cta_tile_m=64 under cta_group=2) not yet " "correctly implemented — excluded from traversal"
        from . import compiler as C

        try:
            if self.block_scale:
                from .tile_config import validate_block_scale_config

                bs = chain.block_scale
                assert bs is not None
                data_elem_bits = 4 if bs.is_fp4 else 8
                cta_k_elems = config.cta_tile_k_bytes * 8 // data_elem_bits
                validate_block_scale_config(config, bs.block_size, cta_k_elems)
            else:
                C._check_dtype_config_compat(chain, config, self.cta_group)
        except (ValueError, NotImplementedError) as e:
            return str(e)
        return None

    # -- stage 4a: per-template capability hook ------------------------------

    def _extra_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        """Template-SPECIFIC constraints beyond the shared gates. Base = none;
        subclasses (mainloop) override. The hook that makes capability truly
        per-template — future variable-MMA / per-template caps attach here."""
        return None

    # -- stage 4: other graph info -------------------------------------------

    def _other_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        extra = self._extra_reject(chain, config)
        if extra is not None:
            return extra
        if self.block_scale:
            return None  # block-scale path is STG-only; no TMA-input alignment gate
        from . import compiler as C

        try:
            C._check_input_alignment(chain)
            C._compute_output_vec_bytes(chain)
        except (ValueError, NotImplementedError) as e:
            return str(e)
        return None

    # -- full accept/reject: the four-stage funnel --------------------------

    def accepts(self, chain: FusionChain, config: TileConfig) -> str | None:
        """``None`` if this template can compile (chain, config); else the first
        stage's rejection reason. Stages cheapest-first (``or`` short-circuits):
        arch/graph-type/mainloop → mma-type×arch → tile-config → other."""
        gt = classify_graph_type(chain)
        return (
            self.arch_active_reject()
            or self._axis_reject(chain, config, gt)
            or mma_arch_reject(chain, gt)
            or self._config_reject(chain, config)
            or self._other_reject(chain, config)
        )

    def candidate_configs(self, chain: FusionChain) -> tuple[TileConfig, ...]:
        """The catalog geometries this template accepts for ``chain`` — derived
        by filtering (predicate), never hand-maintained."""
        return tuple(c for c in CATALOG if self.accepts(chain, c) is None)


class MainloopKernelTemplate(KernelTemplate):
    """A mainloop-fusion template. Owns the supported-mainloop-op contract: every
    pre-MMA pointwise op must be in :data:`_SUPPORTED_MAINLOOP_OPS`. (Can't be
    tripped by constructible input today — FusionOp restricts op to that set —
    but makes the contract live on the template; proven to fire via monkeypatch.)"""

    def _extra_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        for side, ops in (("A", chain.mainloop_a_ops), ("B", chain.mainloop_b_ops)):
            for op in ops:
                if op.op not in _SUPPORTED_MAINLOOP_OPS:
                    return f"{self.file} cannot fuse mainloop op {op.op!r} on " f"operand {side} (supported: unary + scalar-aux binary)"
        return None


# ---------------------------------------------------------------------------
# Registry — one entry per template file (14 today). A single geometry config
# is expanded across these by `candidates`. cta_group / static_sched / mainloop
# live HERE, not on the config.
# ---------------------------------------------------------------------------


def _mm(
    file: str,
    *,
    cta_group: int,
    static: bool,
    mainloop: bool = False,
    graph_type: GraphType = GraphType.MATMUL,
    arch: str = "sm100",
    sm_lo: int = 100,
    sm_hi: int = 120,
    supports_multi_gemm: bool = False,
) -> KernelTemplate:
    cls = MainloopKernelTemplate if mainloop else KernelTemplate
    return cls(
        file=file,
        arch=arch,
        cta_group=cta_group,
        static_sched=static,
        graph_type=graph_type,
        mainloop=mainloop,
        sm_lo=sm_lo,
        sm_hi=sm_hi,
        supports_multi_gemm=supports_multi_gemm,
    )


TEMPLATES: tuple[KernelTemplate, ...] = (
    # plain matmul
    _mm("sm100_matmul_1ctamma.py", cta_group=1, static=False, supports_multi_gemm=True),
    _mm("sm100_matmul_1ctamma_static.py", cta_group=1, static=True, supports_multi_gemm=True),
    _mm("sm100_matmul_2ctamma.py", cta_group=2, static=False, supports_multi_gemm=True),
    _mm("sm100_matmul_2ctamma_static.py", cta_group=2, static=True, supports_multi_gemm=True),
    # block-scaled matmul
    _mm("sm100_block_scale_matmul_1ctamma.py", cta_group=1, static=False, graph_type=GraphType.BLOCK_SCALE_MATMUL, supports_multi_gemm=True),
    _mm("sm100_block_scale_matmul_1ctamma_static.py", cta_group=1, static=True, graph_type=GraphType.BLOCK_SCALE_MATMUL, supports_multi_gemm=True),
    _mm("sm100_block_scale_matmul_2ctamma.py", cta_group=2, static=False, graph_type=GraphType.BLOCK_SCALE_MATMUL, supports_multi_gemm=True),
    _mm("sm100_block_scale_matmul_2ctamma_static.py", cta_group=2, static=True, graph_type=GraphType.BLOCK_SCALE_MATMUL, supports_multi_gemm=True),
    # mainloop-fusion matmul (CLC only — no static / block-scale variant yet)
    _mm("sm100_matmul_mainloop_1ctamma.py", cta_group=1, static=False, mainloop=True),
    _mm("sm100_matmul_mainloop_2ctamma.py", cta_group=2, static=False, mainloop=True),
    # MoE grouped matmul fwd (own grouped persistent scheduler — the static_sched
    # axis is irrelevant here, registered False so default scheduler="clc" selects).
    _mm("sm100_moe_grouped_matmul_fwd_1ctamma.py", cta_group=1, static=False, graph_type=GraphType.MOE, supports_multi_gemm=True),
    _mm("sm100_moe_grouped_matmul_fwd_2ctamma.py", cta_group=2, static=False, graph_type=GraphType.MOE, supports_multi_gemm=True),
    # MoE grouped matmul with block-scaled (FP4/FP8 + SF) inputs.
    _mm("sm100_moe_grouped_block_scale_matmul_fwd_1ctamma.py", cta_group=1, static=False, graph_type=GraphType.MOE_BLOCK_SCALE, supports_multi_gemm=True),
    _mm("sm100_moe_grouped_block_scale_matmul_fwd_2ctamma.py", cta_group=2, static=False, graph_type=GraphType.MOE_BLOCK_SCALE, supports_multi_gemm=True),
)


def select_template(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
    scheduler: str = CLC,
) -> KernelTemplate:
    """The single template that renders (chain, config) under the requested
    execution strategy. ``cta_group`` and ``scheduler`` are the strategy knobs
    the pure-geometry config no longer carries; ``mainloop`` and ``graph_type``
    are derived from the chain. Capability/known-bad gates are NOT applied here —
    single-point JIT renders even known-bad configs for deliberate probing."""
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}; got {scheduler!r}")
    gt = classify_graph_type(chain)
    matches = [
        t
        for t in TEMPLATES
        if t.arch == config.arch
        and t.graph_type is gt
        and t.mainloop == chain.has_mainloop_fusion
        and t.cta_group == cta_group
        and t.static_sched == (scheduler == STATIC)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"no kernel template for graph_type={gt.value}, "
            f"mainloop={chain.has_mainloop_fusion}, cta_group={cta_group}, "
            f"scheduler={scheduler}, arch={config.arch}. "
            "E.g. mainloop fusion has no static or block-scale template variant yet."
        )
    raise ValueError(f"ambiguous template match (registry bug): {[t.file for t in matches]}")


def candidates(chain: FusionChain) -> list[tuple[KernelTemplate, TileConfig]]:
    """Traversal-mode candidate set for ``chain``, via the support funnel.
    Each accepted (template, geometry) is a JIT-able point; one geometry expands
    across the templates that accept it ({1,2}ctamma × {clc,static}, etc.)."""
    gt = classify_graph_type(chain)
    tmpls = [t for t in TEMPLATES if t.graph_type is gt]  # stage 1
    if not tmpls:
        return []
    if mma_arch_reject(chain, gt) is not None:  # stage 2
        return []
    out: list[tuple[KernelTemplate, TileConfig]] = []
    for tmpl in tmpls:
        for cfg in tmpl.candidate_configs(chain):  # stage 3 + 4
            out.append((tmpl, cfg))
    return out


def enumerate_candidates(
    chain: FusionChain,
) -> list[tuple[KernelTemplate, TileConfig]]:
    """Alias for :func:`candidates` (the four-stage funnel)."""
    return candidates(chain)
