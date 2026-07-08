"""Unit tests for TileConfig (pure geometry) + guard rails. cta_group /
static_sched / ab_stages are template strategy, not config; cta_group-dependent
views take cta_group as an argument."""

from __future__ import annotations

import re

import pytest

from cudnn.frost.gemm.tile_config import CATALOG, DEFAULT_CONFIG, TileConfig, by_name

pytestmark = pytest.mark.L0


def _mk(N: int) -> TileConfig:
    """A pure-geometry TileConfig with the requested tile N."""
    return TileConfig(
        cta_tile_m=128,
        cta_tile_n=N,
        cta_tile_k_bytes=128,
        cgrp_size_m=1,
        cgrp_size_n=1,
        epi_tile_mn=(128, 32),
        threads_per_cta=256,
    )


def test_default_config_is_in_catalog() -> None:
    assert DEFAULT_CONFIG in CATALOG


def test_catalog_lookup_by_name() -> None:
    assert by_name(DEFAULT_CONFIG.name) is DEFAULT_CONFIG


def test_default_name_is_pure_geometry() -> None:
    assert DEFAULT_CONFIG.name == "CONFIG_sm100_128x256x128_128x256x32_cluster2x1"


def test_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        by_name("does-not-exist")


# --- tile N guard ---


@pytest.mark.parametrize("bad_n", [1, 8, 16, 24, 48, 80])
def test_small_or_unaligned_tile_n_rejected(bad_n: int) -> None:
    with pytest.raises(NotImplementedError, match="cta_tile_n"):
        _mk(bad_n)


@pytest.mark.parametrize("ok_n", [32, 64, 96, 128, 256])
def test_aligned_tile_n_accepted(ok_n: int) -> None:
    assert _mk(ok_n).cta_tile_n == ok_n


def test_all_catalog_configs_pass_guard() -> None:
    for cfg in CATALOG:
        n = cfg.cta_tile_n
        assert n >= 32 and n % 32 == 0, (cfg.name, n)


# --- K-in-bytes ⇒ K-in-elements, with the template's cta_group ---


def test_k_elements_for_bf16() -> None:
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    assert cfg.cta_tile_k_bytes == 128
    assert cfg.cta_tile_k(elem_bytes=2) == 64
    assert cfg.cta_tile_mnk(elem_bytes=2) == (128, 128, 64)
    # cta_group=1: hardware MMA M == cta_tile_m; SMEM N == cta_tile_n
    assert cfg.mma_inst_mnk(elem_bytes=2, cta_group=1) == (128, 128, 16)
    assert cfg.cta_smem_tile_mnk(elem_bytes=2, cta_group=1) == (128, 128, 64)


def test_k_elements_for_fp8() -> None:
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    assert cfg.cta_tile_k(elem_bytes=1) == 128
    assert cfg.mma_inst_mnk(elem_bytes=1, cta_group=1) == (128, 128, 32)


def test_cta_group2_doubles_mma_m_and_halves_smem_n() -> None:
    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1")
    # cta_group=2: hardware MMA spans the pair (M ×2), each CTA holds ½ B's N
    assert cfg.mma_inst_mnk(elem_bytes=2, cta_group=2) == (256, 256, 16)
    assert cfg.cta_smem_tile_mnk(elem_bytes=2, cta_group=2) == (128, 128, 64)
    assert cfg.cta_smem_tile_mnk(elem_bytes=2, cta_group=1) == (128, 256, 64)


def test_max_ab_stages_more_under_cta_group2() -> None:
    """2-CTA MMA halves B's SMEM N, so the same geometry fits more stages."""
    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1")
    assert cfg.max_ab_stages(2) >= cfg.max_ab_stages(1)


# --- name pattern ---


def test_name_matches_canonical_pattern() -> None:
    """Pure-geometry name pattern; no _Nctamma / _static tokens (those are the template's)."""
    pat = re.compile(r"^CONFIG_sm100_\d+x\d+x\d+_\d+x\d+x\d+_cluster\d+x\d+$")
    for cfg in CATALOG:
        assert pat.match(cfg.name), cfg.name
        assert cfg.mma_inst_m == cfg.cta_tile_m
        assert cfg.mma_inst_n == cfg.cta_tile_n
        assert cfg.mma_inst_k_bytes == 32  # sm100 s128b MMA K-inst


# --- derived geometry ---


def test_cgrp_tile_mn() -> None:
    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1")
    assert cfg.cgrp_tile_mn == (256, 256)  # (128*2, 256*1)


def test_multicast_model() -> None:
    cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster4x2")
    assert cfg.multicast_a_factor == 2  # cgrp_size_n
    assert cfg.multicast_a is True
    # B-multicast depends on cta_group: cgrp_size_m // cta_group
    assert cfg.multicast_b_factor(cta_group=1) == 4
    assert cfg.multicast_b_factor(cta_group=2) == 2
    assert cfg.multicast_b(cta_group=2) is True


def test_config_has_no_strategy_fields() -> None:
    """The pure-geometry config must NOT carry execution-strategy axes."""
    cfg = DEFAULT_CONFIG
    for attr in ("cta_group", "static_sched", "ab_stages", "template_file"):
        assert not hasattr(cfg, attr), attr
