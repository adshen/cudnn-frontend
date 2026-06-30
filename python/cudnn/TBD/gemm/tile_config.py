"""Tile config catalog — PURE GEOMETRY for the fused GEMM kernels.

A ``TileConfig`` describes ONLY the tile geometry, dtype-independently:

    cta_tile (per-CTA logical M×N, K in bytes) · per-CTA MMA-instruction tile ·
    cluster shape (cgrp_size) · arch family

It deliberately does **not** carry ``cta_group`` (1-CTA vs 2-CTA MMA),
``static_sched`` (CLC vs static scheduler), or ``ab_stages`` — those are
**execution strategy decided by the kernel template**, not geometry. A single
geometry config is run by different templates (1ctamma / 2ctamma / static /
mainloop); the template imposes cta_group on the geometry, from which the
physical SMEM tile, the hardware MMA shape, B-multicast, and ab_stages are
DERIVED (the cta_group-taking methods below). See ``kernel_registry`` for the
template registry and the support funnel.

## Naming convention

    CONFIG_<arch>_<CTA_M>x<CTA_N>x<K_BYTES>_<MMA_M>x<MMA_N>x<MMA_K_BYTES>_cluster<cgrp_m>x<cgrp_n>

e.g. ``CONFIG_sm100_128x256x128_128x256x32_cluster2x1``. Arch leads (and is a
first-class field) because tile-geometry constraints are arch-specific — an
``sm100`` template only accepts ``sm100`` configs. First triple = per-CTA output
tile (M, N elements; K bytes). Second triple = per-CTA MMA-instruction tile
(M, N elements; K bytes); for every config today MMA == CTA tile in M/N and
MMA-K = 32 bytes, ahead of variable-MMA support. NO ``_Nctamma`` / ``_static``
tokens anymore — those moved to the template. The ``CONFIG_`` prefix keeps the
name a valid Python identifier (pure-digit names can't be module variables).

The kernel symbol the compiler emits is ``<template_stem>_<geometry_name>``,
e.g. ``sm100_block_scale_matmul_2ctamma_static_128x256x128_128x256x32_cluster2x1``
— the cta_group/static come from the template stem, the geometry from the config.

Because K is named in *bytes*, the same config covers every supported dtype:
``K_BYTES = 128`` ⇒ 64 BF16/FP16 elements or 128 FP8 elements per K-tile.

## Geometry axes

  1. **cta_tile_m / cta_tile_n** — per-CTA logical output tile (elements).
     Caps: M ≤ 128, N ≤ 256, K_BYTES ≤ 128 (SWIZZLE_128B). For 2-CTA MMA this
     is still the per-CTA logical tile — the pair-split of B is implicit and
     applied by :meth:`cta_smem_tile_mnk` given the template's cta_group.
  2. **cgrp_size_m / cgrp_size_n** — CTAs per cluster along (M, N). K is never
     split (cgrp_size_k == 1).
  3. **mma_inst_m / mma_inst_n / mma_inst_k_bytes** — per-CTA MMA-instruction
     tile (default = CTA tile M/N, 32-byte K).

## cta_group-derived quantities (template supplies cta_group)

  - **cta_smem_tile_mnk(elem_bytes, cta_group)** = (cta_tile_m,
    cta_tile_n // cta_group, K_elements) — B's N halved under 2-CTA MMA.
  - **mma_inst_mnk(elem_bytes, cta_group)** = (mma_inst_m × cta_group,
    mma_inst_n, K_inst) — hardware MMA M spans the CTA pair.
  - **multicast_b_factor(cta_group)** = cgrp_size_m // cta_group.
  - **max_ab_stages(cta_group, …)** — largest SMEM pipeline depth that fits.
"""

from __future__ import annotations

from dataclasses import dataclass

# sm_100 (B200) per-CTA opt-in SMEM cap = 232448 bytes (227 KB).
# Deduct 1024 bytes reserved by the hardware/system (tensor-core-MMA requirement: all
# data-tile SMEM must start ≥1024 B from the base), plus a small safety margin
# for mbarrier/CLC arrays and 1024-byte alignment padding.
_SM100_SMEM_BUDGET_BYTES = 228 * 1024  # 232448 − 1024 sys − ~3KB margin
_AB_STAGES_CAP = 8  # don't blow past 8 stages even if SMEM permits

# Hardware bounds on the *logical* per-CTA tile (user's model).
_CTA_TILE_M_MAX = 128
_CTA_TILE_N_MAX = 256
_CTA_TILE_K_BYTES_MAX = 128  # SWIZZLE_128B: SMEM row width = 128 bytes

# MMA instruction's K dimension in bytes. sm100 uses the 32-byte s128b
# instruction (= 16 BF16/FP16 elements, = 32 FP8 / 64 FP4 elements).
_MMA_INST_K_BYTES = 32

# Default arch family. Tile-geometry constraints are arch-specific, so the arch
# is part of the config identity (and matched against the template's arch).
_DEFAULT_ARCH = "sm100"


def smem_max_ab_stages(
    cta_tile_m: int,
    cta_tile_n: int,
    cta_tile_k_bytes: int,
    *,
    cta_group: int = 1,
    acc_stages: int = 2,
    extra_smem_bytes: int = 0,
    extra_per_stage_bytes: int = 0,
) -> int:
    """Largest ab_stages that fits in sm_100 per-CTA SMEM.

    Inputs are dtype-independent: tile_m/n are element counts, tile_k is in
    bytes. The SMEM-per-stage formula `(M + N/cta_group) × K_bytes` makes the
    answer the same whether the config is later instantiated as BF16 or FP8.

    ``extra_smem_bytes`` reserves a fixed chunk of SMEM up-front (e.g. for
    the TMA-store epilogue's SMEM-D buffer) — those bytes are subtracted
    from the budget before computing how many AB stages still fit.
    ``extra_per_stage_bytes`` adds to the per-stage cost (e.g. a mixed-input
    mainloop's narrow LOAD buffer, allocated once per AB stage beside the
    wide MMA tile).
    """
    smem_b_n = cta_tile_n // cta_group
    per_stage = (cta_tile_m + smem_b_n) * cta_tile_k_bytes + extra_per_stage_bytes + 2 * 8
    fixed = 2 * acc_stages * 8 + 8
    avail = _SM100_SMEM_BUDGET_BYTES - fixed - extra_smem_bytes
    if avail < per_stage:
        raise ValueError(
            f"tile ({cta_tile_m},{cta_tile_n},K={cta_tile_k_bytes}B) "
            f"cta_group={cta_group} per-stage SMEM {per_stage} bytes exceeds "
            f"the budget (extra_smem_bytes={extra_smem_bytes}) — can't fit "
            f"even 1 stage"
        )
    return min(avail // per_stage, _AB_STAGES_CAP)


@dataclass(frozen=True)
class TileConfig:
    """One pure-geometry tile configuration. Dtype- AND execution-independent.

      cta_tile_m, cta_tile_n   : per-CTA logical (M, N) in elements.
      cta_tile_k_bytes         : K tile in bytes (128 for SWIZZLE_128B).
      cgrp_size_m, cgrp_size_n   : CTAs per cluster along (M, N).
      mma_inst_m, mma_inst_n   : per-CTA MMA-instruction (M, N) in elements.
      mma_inst_k_bytes         : MMA-instruction K in bytes (32 for s128b).
      arch                     : arch family ("sm100"); matched to the template.
      epi_tile_mn, threads_per_cta, acc_stages, tile_swizzle_n : misc geometry.

    NOT here (template strategy, not geometry): cta_group, static_sched,
    ab_stages. K-in-elements, the SMEM tile, the hardware MMA shape, and
    ab_stages are derived per (dtype, cta_group) at render time.
    """

    cta_tile_m: int
    cta_tile_n: int
    cta_tile_k_bytes: int  # K in BYTES (dtype-independent)
    cgrp_size_m: int
    cgrp_size_n: int
    epi_tile_mn: tuple[int, int]  # epilogue subtile (M, 32)
    threads_per_cta: int  # block size (256 = 8-warp warp-spec)
    acc_stages: int = 2  # TMEM accumulator stages (double-buffer)
    tile_swizzle_n: int = 8  # N-direction super-block width (L2 reuse)
    arch: str = _DEFAULT_ARCH  # arch family; matched to the template
    # Per-CTA MMA-instruction tile (None → defaults to the CTA tile in M/N and
    # _MMA_INST_K_BYTES in K, filled in __post_init__). Forward-looking for
    # variable-MMA configs where the MMA tile is smaller than the CTA tile.
    mma_inst_m: int | None = None  # MMA-inst M (elements); None → cta_tile_m
    mma_inst_n: int | None = None  # MMA-inst N (elements); None → cta_tile_n
    mma_inst_k_bytes: int = _MMA_INST_K_BYTES  # MMA-inst K in BYTES (s128b: 32)

    def __post_init__(self) -> None:
        m, n, kb = self.cta_tile_m, self.cta_tile_n, self.cta_tile_k_bytes
        cm, cn = self.cgrp_size_m, self.cgrp_size_n

        # Per-CTA tile hardware bounds.
        if m <= 0 or m > _CTA_TILE_M_MAX or m % 32 != 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: cta_tile_m={m} — must be a " f"positive multiple of 32, ≤ {_CTA_TILE_M_MAX}")
        if n < 32 or n > _CTA_TILE_N_MAX or n % 32 != 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: cta_tile_n={n} — supported range " f"is 32 ≤ N ≤ {_CTA_TILE_N_MAX}, multiple of 32")
        if kb <= 0 or kb > _CTA_TILE_K_BYTES_MAX or kb % _MMA_INST_K_BYTES != 0:
            raise NotImplementedError(
                f"TileConfig {self.name!r}: cta_tile_k_bytes={kb} — must be "
                f"a positive multiple of {_MMA_INST_K_BYTES} (the MMA K-inst "
                f"width in bytes), ≤ {_CTA_TILE_K_BYTES_MAX} (SWIZZLE_128B cap)"
            )

        # CGRP size sanity. (cta_group-specific constraints — e.g. cgrp_size_m
        # even for 2-CTA MMA — now live on the 2ctamma template in the registry.)
        if cm <= 0 or cn <= 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: cgrp_size_mn=({cm},{cn})")

        # MMA-instruction tile: default to the CTA tile (M/N) and the s128b
        # instruction-K (32 bytes). Frozen dataclass → object.__setattr__.
        if self.mma_inst_m is None:
            object.__setattr__(self, "mma_inst_m", m)
        if self.mma_inst_n is None:
            object.__setattr__(self, "mma_inst_n", n)
        mm, mn, mkb = self.mma_inst_m, self.mma_inst_n, self.mma_inst_k_bytes
        if mm <= 0 or m % mm != 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: mma_inst_m={mm} must be positive " f"and divide cta_tile_m={m}")
        if mn <= 0 or n % mn != 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: mma_inst_n={mn} must be positive " f"and divide cta_tile_n={n}")
        if mkb <= 0 or kb % mkb != 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: mma_inst_k_bytes={mkb} must be " f"positive and divide cta_tile_k_bytes={kb}")

    # -- name (derived) ------------------------------------------------------

    @property
    def geometry_name(self) -> str:
        """Geometry token (no ``CONFIG_``/arch prefix) used in the kernel
        symbol: ``<CTA>_<MMA>_cluster<cm>x<cn>``."""
        return (
            f"{self.cta_tile_m}x{self.cta_tile_n}x{self.cta_tile_k_bytes}"
            f"_{self.mma_inst_m}x{self.mma_inst_n}x{self.mma_inst_k_bytes}"
            f"_cluster{self.cgrp_size_m}x{self.cgrp_size_n}"
        )

    @property
    def name(self) -> str:
        """Canonical identifier: ``CONFIG_<arch>_<geometry_name>``. Pure
        geometry — cta_group/static/ab_stages are NOT in the name (they are the
        template's, not the config's)."""
        return f"CONFIG_{self.arch}_{self.geometry_name}"

    # -- dtype-independent shape views --------------------------------------

    @property
    def cta_tile_mn(self) -> tuple[int, int]:
        """Per-CTA logical output tile (M, N) in elements."""
        return (self.cta_tile_m, self.cta_tile_n)

    @property
    def cgrp_size_mn(self) -> tuple[int, int]:
        """CTAs per cluster along (M, N). K is never split (cgrp_size_k == 1)."""
        return (self.cgrp_size_m, self.cgrp_size_n)

    @property
    def cgrp_size_mnk(self) -> tuple[int, int, int]:
        """3-tuple form of cgrp_size — convenience for callers that want a
        cluster_shape-style triple (cgrp_size_k is always 1)."""
        return (self.cgrp_size_m, self.cgrp_size_n, 1)

    @property
    def cgrp_tile_mn(self) -> tuple[int, int]:
        """Cluster aggregate output tile (M, N) in elements."""
        return (self.cta_tile_m * self.cgrp_size_m, self.cta_tile_n * self.cgrp_size_n)

    @property
    def cluster_shape(self) -> tuple[int, int, int]:
        """Alias for cgrp_size_mnk, matching cluster-launch terminology."""
        return self.cgrp_size_mnk

    # -- dtype-dependent shape views (require elem_bytes) -------------------

    def cta_tile_k(self, elem_bytes: int) -> int:
        """K tile in *elements*, given the input dtype's byte width."""
        if self.cta_tile_k_bytes % elem_bytes != 0:
            raise ValueError(f"TileConfig {self.name!r}: cta_tile_k_bytes " f"({self.cta_tile_k_bytes}) is not divisible by " f"elem_bytes={elem_bytes}")
        return self.cta_tile_k_bytes // elem_bytes

    def cta_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Per-CTA logical tile in elements (M, N, K)."""
        return (self.cta_tile_m, self.cta_tile_n, self.cta_tile_k(elem_bytes))

    def cgrp_tile_mnk(self, elem_bytes: int) -> tuple[int, int, int]:
        """Cluster aggregate tile in elements (M, N, K)."""
        m, n = self.cgrp_tile_mn
        return (m, n, self.cta_tile_k(elem_bytes))

    def cta_smem_tile_mnk(self, elem_bytes: int, cta_group: int) -> tuple[int, int, int]:
        """Per-CTA SMEM tile in elements. B's N is halved under 2-CTA MMA, so
        this needs the template's ``cta_group``."""
        return (self.cta_tile_m, self.cta_tile_n // cta_group, self.cta_tile_k(elem_bytes))

    def mma_inst_mnk(self, elem_bytes: int, cta_group: int) -> tuple[int, int, int]:
        """Hardware MMA-instruction shape in elements, given the template's
        ``cta_group``. M spans the CTA pair (``mma_inst_m × cta_group``); K is
        ``mma_inst_k_bytes`` in elements (32 bytes = 16 BF16/FP16, 32 FP8)."""
        k_inst = self.mma_inst_k_bytes // elem_bytes
        return (self.mma_inst_m * cta_group, self.mma_inst_n, k_inst)

    def max_ab_stages(self, cta_group: int, *, extra_smem_bytes: int = 0, extra_per_stage_bytes: int = 0) -> int:
        """Largest SMEM pipeline depth for this geometry under ``cta_group``
        (2-CTA MMA halves B's SMEM N, so it fits more stages)."""
        return smem_max_ab_stages(
            self.cta_tile_m,
            self.cta_tile_n,
            self.cta_tile_k_bytes,
            cta_group=cta_group,
            acc_stages=self.acc_stages,
            extra_smem_bytes=extra_smem_bytes,
            extra_per_stage_bytes=extra_per_stage_bytes,
        )

    # -- multicast model -----------------------------------------------------

    @property
    def multicast_a_factor(self) -> int:
        """# CTAs sharing the same M slice of A (TMA multicast group size).
        Independent of cta_group."""
        return self.cgrp_size_n

    def multicast_b_factor(self, cta_group: int) -> int:
        """# CTAs sharing the same N slice of B (TMA multicast group size).
        For 2-CTA MMA the leading factor of 2 in cgrp_size_m is consumed by the
        MMA pair, so B-multicast only kicks in when cgrp_size_m ≥ 4."""
        return self.cgrp_size_m // cta_group

    @property
    def multicast_a(self) -> bool:
        return self.multicast_a_factor > 1

    def multicast_b(self, cta_group: int) -> bool:
        return self.multicast_b_factor(cta_group) > 1


# ---------------------------------------------------------------------------
# Catalog — pure-geometry enumeration. cta_group / static / mainloop are NOT
# enumerated here; the registry expands each geometry across the templates that
# accept it (1ctamma / 2ctamma × clc / static, etc.).
#
# Geometry axes: cta_m ∈ {128, 64}, cta_n ∈ {256, 128, 64, 32}, K_bytes ∈
# {128, 64}, cluster ∈ the union of the clusters the templates use. (N ∈ {16, 8}
# are valid HW shapes but our __post_init__ rejects cta_tile_n < 32 — needs a
# SHAPE_16X* tcgen05_ld port first.)
#
# 2-CTA templates accept only cgrp_size_m % 2 == 0 (the registry's accept
# predicate); cluster-m=128 (cta_tile_m=64 under cta_group=2) is known-bad and
# excluded from traversal there. 1-CTA templates accept any cluster here.
# ---------------------------------------------------------------------------

_CLUSTERS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (2, 1),
    (2, 2),
    (2, 4),
    (2, 8),
    (4, 1),
    (4, 2),
    (4, 4),
    (8, 1),
    (8, 2),
    (16, 1),
)


def _geom(cta_m: int, cta_n: int, k_bytes: int, cgrp_m: int, cgrp_n: int) -> TileConfig:
    """Build one pure-geometry TileConfig (sm100, 32-byte MMA-instruction K)."""
    return TileConfig(
        cta_tile_m=cta_m,
        cta_tile_n=cta_n,
        cta_tile_k_bytes=k_bytes,
        cgrp_size_m=cgrp_m,
        cgrp_size_n=cgrp_n,
        epi_tile_mn=(cta_m, 32),
        threads_per_cta=256,
        acc_stages=2,
    )


def _build_catalog() -> tuple[TileConfig, ...]:
    cfgs: list[TileConfig] = []
    for cta_m in (128, 64):
        for cta_n in (256, 128, 64, 32):
            for k_bytes in (128, 64):
                for cgrp_m, cgrp_n in _CLUSTERS:
                    cfgs.append(_geom(cta_m, cta_n, k_bytes, cgrp_m, cgrp_n))
    return tuple(cfgs)


CATALOG: tuple[TileConfig, ...] = _build_catalog()


# Expose every catalog entry as a module-level variable matching its canonical
# name — preserves `from .tile_config import CONFIG_sm100_...`.
for _cfg in CATALOG:
    globals()[_cfg.name] = _cfg
del _cfg


def by_name(name: str) -> TileConfig:
    for c in CATALOG:
        if c.name == name:
            return c
    raise KeyError(f"unknown tile config {name!r}; first 5 of {len(CATALOG)}: " f"{[c.name for c in CATALOG[:5]]}...")


DEFAULT_CONFIG: TileConfig = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1")


# ---------------------------------------------------------------------------
# Block-scaled matmul config validation (geometry-only; cta_group lives on the
# template). Template-file selection is in kernel_registry.select_template.
# ---------------------------------------------------------------------------
#
# The scale factors are laid out in the CUDNN_TENSOR_REORDERING_F8_128x4
# (128-row × 4-K) swizzle and copied SMEM→TMEM with the 32x128b.warpx4 utccp
# atom. That atom + the per-block-128 SF layout impose:
#   * cta_tile_m % 128 == 0  (SFA blocks are 128 M-rows wide)
#   * cta_tile_n % 128 == 0  (SFB blocks are 128 N-cols wide)
#   * cta_tile_k (in *elements*) % (4 * block_size) == 0


def validate_block_scale_config(cfg: TileConfig, block_size: int, cta_tile_k_elems: int) -> None:
    """Raise if ``cfg``'s GEOMETRY cannot run a block-scaled matmul.
    ``cta_tile_k_elems`` is the K-tile in *elements* for the data dtype (e.g.
    256 for FP4 at K_BYTES=128, 128 for FP8)."""
    if cfg.cta_tile_m % 128 != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires cta_tile_m % 128 == 0 (SF 128x4 " f"swizzle); config {cfg.name!r} has cta_tile_m={cfg.cta_tile_m}"
        )
    if cfg.cta_tile_n % 128 != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires cta_tile_n % 128 == 0 (SF 128x4 " f"swizzle); config {cfg.name!r} has cta_tile_n={cfg.cta_tile_n}"
        )
    if cfg.cta_tile_k_bytes != 128:
        raise NotImplementedError(f"block-scaled matmul POC requires cta_tile_k_bytes == 128; " f"config {cfg.name!r} has {cfg.cta_tile_k_bytes}")
    if cta_tile_k_elems % (4 * block_size) != 0:
        raise NotImplementedError(
            f"block-scaled matmul requires cta_tile_k (elements) % (4*block_size) " f"== 0; got cta_tile_k={cta_tile_k_elems}, block_size={block_size}"
        )
