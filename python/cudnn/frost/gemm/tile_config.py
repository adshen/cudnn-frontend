"""Tile config catalog — PURE GEOMETRY for the fused GEMM kernels.

A ``TileConfig`` describes ONLY dtype-independent tile geometry (cta tile,
MMA-inst tile, cluster shape, arch). It does NOT carry ``cta_group`` /
``static_sched`` / ``ab_stages`` — those are execution strategy chosen by the
kernel template. K is stored in *bytes*, so one config covers every dtype.
Name: ``CONFIG_<arch>_<CTA_M>x<CTA_N>x<K_BYTES>_<MMA_M>x<MMA_N>x<MMA_K_BYTES>_cluster<cgrp_m>x<cgrp_n>``.
See ``kernel_registry`` for the template registry and the support funnel.
"""

from __future__ import annotations

from dataclasses import dataclass

# sm_100 per-CTA SMEM budget = 232448 cap − 1024 sys reserve − ~3KB margin.
_SM100_SMEM_BUDGET_BYTES = 228 * 1024
_AB_STAGES_CAP = 8  # cap even if SMEM permits more

_CTA_TILE_M_MAX = 128
_CTA_TILE_N_MAX = 256
_CTA_TILE_K_BYTES_MAX = 128  # SWIZZLE_128B: SMEM row width = 128 bytes

# MMA-inst K in bytes. sm100 s128b = 32 B (16 BF16/FP16, 32 FP8, 64 FP4 elems).
_MMA_INST_K_BYTES = 32

# Arch is part of the config identity (matched against the template's arch).
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
    """Largest ab_stages that fits in sm_100 per-CTA SMEM (dtype-independent:
    per-stage SMEM = ``(M + N/cta_group) × K_bytes``).

    ``extra_smem_bytes`` reserves a fixed up-front chunk (e.g. TMA-store SMEM-D
    buffer); ``extra_per_stage_bytes`` adds to the per-stage cost (e.g. a
    mixed-input mainloop's narrow LOAD buffer, one per AB stage).
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
    """One pure-geometry tile config. Dtype- AND execution-independent.

    NOT here (template strategy, not geometry): cta_group, static_sched,
    ab_stages — those, plus K-in-elements, the SMEM tile, and the hardware MMA
    shape, are derived per (dtype, cta_group) at render time.
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
    arch: str = _DEFAULT_ARCH
    # Per-CTA MMA-inst tile (None → CTA tile M/N + _MMA_INST_K_BYTES K, filled in
    # __post_init__). Forward-looking for MMA-tile-smaller-than-CTA-tile configs.
    mma_inst_m: int | None = None
    mma_inst_n: int | None = None
    mma_inst_k_bytes: int = _MMA_INST_K_BYTES

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
        # even for 2-CTA MMA — live on the 2ctamma template in the registry.)
        if cm <= 0 or cn <= 0:
            raise NotImplementedError(f"TileConfig {self.name!r}: cgrp_size_mn=({cm},{cn})")

        # MMA-inst tile defaults to the CTA tile M/N + s128b K. Frozen → setattr.
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

    @property
    def geometry_name(self) -> str:
        """Geometry token (no ``CONFIG_``/arch prefix) used in the kernel symbol."""
        return (
            f"{self.cta_tile_m}x{self.cta_tile_n}x{self.cta_tile_k_bytes}"
            f"_{self.mma_inst_m}x{self.mma_inst_n}x{self.mma_inst_k_bytes}"
            f"_cluster{self.cgrp_size_m}x{self.cgrp_size_n}"
        )

    @property
    def name(self) -> str:
        """Canonical identifier ``CONFIG_<arch>_<geometry_name>`` — pure geometry
        (cta_group/static/ab_stages are the template's, not in the name)."""
        return f"CONFIG_{self.arch}_{self.geometry_name}"

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
        """cgrp_size as a cluster_shape-style triple (cgrp_size_k always 1)."""
        return (self.cgrp_size_m, self.cgrp_size_n, 1)

    @property
    def cgrp_tile_mn(self) -> tuple[int, int]:
        """Cluster aggregate output tile (M, N) in elements."""
        return (self.cta_tile_m * self.cgrp_size_m, self.cta_tile_n * self.cgrp_size_n)

    @property
    def cluster_shape(self) -> tuple[int, int, int]:
        """Alias for cgrp_size_mnk (cluster-launch terminology)."""
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
        """Per-CTA SMEM tile in elements. B's N is halved under 2-CTA MMA (needs
        the template's ``cta_group``)."""
        return (
            self.cta_tile_m,
            self.cta_tile_n // cta_group,
            self.cta_tile_k(elem_bytes),
        )

    def mma_inst_mnk(self, elem_bytes: int, cta_group: int) -> tuple[int, int, int]:
        """Hardware MMA-inst shape in elements. M spans the CTA pair
        (``mma_inst_m × cta_group``); K is ``mma_inst_k_bytes`` in elements."""
        k_inst = self.mma_inst_k_bytes // elem_bytes
        return (self.mma_inst_m * cta_group, self.mma_inst_n, k_inst)

    def max_ab_stages(
        self,
        cta_group: int,
        *,
        extra_smem_bytes: int = 0,
        extra_per_stage_bytes: int = 0,
    ) -> int:
        """Largest SMEM pipeline depth under ``cta_group`` (2-CTA MMA halves B's
        SMEM N, so it fits more stages)."""
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
        """# CTAs sharing the same M slice of A. Independent of cta_group."""
        return self.cgrp_size_n

    def multicast_b_factor(self, cta_group: int) -> int:
        """# CTAs sharing the same N slice of B. Under 2-CTA MMA the leading
        factor of 2 in cgrp_size_m is consumed by the MMA pair, so B-multicast
        only kicks in when cgrp_size_m ≥ 4."""
        return self.cgrp_size_m // cta_group

    @property
    def multicast_a(self) -> bool:
        return self.multicast_a_factor > 1

    def multicast_b(self, cta_group: int) -> bool:
        return self.multicast_b_factor(cta_group) > 1


# ---------------------------------------------------------------------------
# Catalog — pure-geometry enumeration. cta_group / static / mainloop are NOT
# enumerated here; the registry expands each geometry across accepting templates.
# Axes: cta_m ∈ {128,64}, cta_n ∈ {256,128,64,32}, K_bytes ∈ {128,64}, cluster ∈
# _CLUSTERS. N < 32 rejected by __post_init__ (needs a SHAPE_16X* tcgen05_ld
# port). 2-CTA templates accept only cgrp_size_m % 2 == 0 (registry predicate).
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
    """Build one pure-geometry TileConfig (sm100, 32-byte MMA-inst K)."""
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


# Expose each catalog entry as a module-level variable matching its canonical
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


def _floor_pow2(v: int) -> int:
    """Largest power of two <= v (v >= 1)."""
    p = 1
    while p * 2 <= v:
        p *= 2
    return p


def select_config(M: int, N: int, num_gemms: int) -> tuple[TileConfig, int, str]:
    """A simple heuristic to select a tile config for a given M, N and num_gemms."""
    if M <= 64:
        cta_m, cta_group = 64, 1
    elif M <= 128:
        cta_m, cta_group = 128, 1
    else:
        cta_m, cta_group = 128, 2

    x = max(1, num_gemms)
    cta_n_max = max(32, min(256, _floor_pow2(256 // x)))
    if N <= 32:
        want_n = 32
    elif N <= 64:
        want_n = 64
    elif N <= 128:
        want_n = 128
    else:
        want_n = 256
    cta_n = min(want_n, cta_n_max)

    name = f"CONFIG_sm100_{cta_m}x{cta_n}x128_{cta_m}x{cta_n}x32_cluster2x1"
    return by_name(name), cta_group, "clc"


# ---------------------------------------------------------------------------
# Block-scaled matmul config validation (geometry-only; cta_group lives on the
# template). The F8_128x4 SF swizzle + 32x128b.warpx4 utccp atom impose:
# cta_tile_m/n % 128 == 0 and cta_tile_k (elements) % (4*block_size) == 0.
# ---------------------------------------------------------------------------


def validate_block_scale_config(cfg: TileConfig, block_size: int, cta_tile_k_elems: int) -> None:
    """Raise if ``cfg``'s GEOMETRY cannot run a block-scaled matmul.
    ``cta_tile_k_elems`` is the K-tile in *elements* (256 for FP4, 128 for FP8)."""
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
