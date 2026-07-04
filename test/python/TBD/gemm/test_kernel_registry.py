"""Tests for the six-dimension support funnel (kernel_registry.py).

Tile configs are pure geometry; the template supplies cta_group/static/arch.
Pins template-file uniqueness, the graph-type + mma-type×arch stages, the
2-CTA constraints + cluster-m=128 exclusion, select_template, and that the
staged funnel equals a naive full scan.
"""

from __future__ import annotations

import pytest

from cudnn.TBD.gemm.compiler import _TEMPLATE_DIR
from cudnn.TBD.gemm.fusion_ir import FusionChain, FusionOp, MatmulSpec
from cudnn.TBD.gemm.kernel_registry import (
    CLC,
    STATIC,
    TEMPLATES,
    GraphType,
    MainloopKernelTemplate,
    candidates,
    classify_graph_type,
    enumerate_candidates,
    mma_arch_reject,
    select_template,
)
from cudnn.TBD.gemm.tile_config import CATALOG, DEFAULT_CONFIG, by_name


def _matmul_chain(dtype: str = "bf16") -> FusionChain:
    return FusionChain(
        matmul=MatmulSpec(
            M=4096,
            N=4096,
            K=4096,
            a_major="k",
            b_major="k",
            a_dtype=dtype,
            b_dtype=dtype,
            accum_dtype="fp32",
        ),
        output_dtype="bf16",
    )


def _mainloop_chain(ops_a: list[FusionOp]) -> FusionChain:
    return FusionChain(
        matmul=MatmulSpec(
            M=4096,
            N=4096,
            K=4096,
            a_major="k",
            b_major="k",
            a_dtype="bf16",
            b_dtype="bf16",
            accum_dtype="fp32",
        ),
        output_dtype="bf16",
        mainloop_a_ops=ops_a,
    )


# geometries used below
_G_DEFAULT = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1"  # M128, cgrp_m=2
_G_CLUSTER1X1 = "CONFIG_sm100_128x256x128_128x256x32_cluster1x1"  # cgrp_m=1 (odd)
_G_M64 = "CONFIG_sm100_64x128x128_64x128x32_cluster2x1"  # cluster-m=128
_G_1CTA = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1"


def _t(cta_group: int, static: bool, graph_type=GraphType.MATMUL, mainloop=False):
    return next(t for t in TEMPLATES if t.cta_group == cta_group and t.static_sched == static and t.graph_type is graph_type and t.mainloop == mainloop)


# -- template files / strategy keys ----------------------------------------


def test_all_template_files_exist() -> None:
    for t in TEMPLATES:
        assert (_TEMPLATE_DIR / t.file).is_file(), t.file


def test_template_files_unique() -> None:
    files = [t.file for t in TEMPLATES]
    assert len(files) == len(set(files))


def test_strategy_combo_unique_per_template() -> None:
    combos = [(t.arch, t.cta_group, t.static_sched, t.graph_type, t.mainloop) for t in TEMPLATES]
    assert len(combos) == len(set(combos))


def test_every_template_declares_graph_type_and_arch() -> None:
    for t in TEMPLATES:
        assert isinstance(t.graph_type, GraphType)
        assert t.arch == "sm100"
        assert t.block_scale == (t.graph_type in (GraphType.BLOCK_SCALE_MATMUL, GraphType.MOE_BLOCK_SCALE))


def test_per_arch_active_sm_ranges() -> None:
    """sm100 templates run across the whole Blackwell family [100,120)."""
    expected = {"sm100": (100, 120)}
    for t in TEMPLATES:
        assert (t.sm_lo, t.sm_hi) == expected[t.arch], (t.file, t.sm_lo, t.sm_hi)


def test_template_only_accepts_own_arch_config() -> None:
    """An sm<NNN> template accepts ONLY sm<NNN> configs; an own-arch config
    never rejects on the arch axis."""
    sm100_cfg = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster1x1")
    assert sm100_cfg.arch == "sm100"
    chain = _matmul_chain()
    for t in TEMPLATES:
        own_reason = t._axis_reject(chain, sm100_cfg, t.graph_type)
        assert own_reason is None or "arch" not in own_reason, (t.file, sm100_cfg.name)


# -- dimension 1: graph type ------------------------------------------------


def test_classify_graph_type() -> None:
    assert classify_graph_type(_matmul_chain()) is GraphType.MATMUL
    assert classify_graph_type(_mainloop_chain([FusionOp(op="relu", parent_idx=-1)])) is GraphType.MATMUL


def test_moe_has_templates_conv_is_placeholder() -> None:
    # MoE has 1ctamma + 2ctamma templates; CONVOLUTION is a placeholder (no template).
    assert len([t for t in TEMPLATES if t.graph_type is GraphType.MOE]) == 2
    assert not [t for t in TEMPLATES if t.graph_type is GraphType.CONVOLUTION]


# -- dimension 2: mma type × arch ------------------------------------------


def test_mma_arch_reject_accepts_bf16_fp8() -> None:
    for dt in ("bf16", "fp8_e4m3"):
        assert mma_arch_reject(_matmul_chain(dt), GraphType.MATMUL) is None, dt


def test_mma_arch_reject_rejects_int8_on_this_gpu() -> None:
    reason = mma_arch_reject(_matmul_chain("int8"), GraphType.MATMUL)
    assert reason is not None and "int8" in reason


# -- dimension 3: tile config (cta_group constraints + known-bad) ----------


def test_2ctamma_rejects_odd_cgrp_size_m() -> None:
    chain = _matmul_chain()
    reason = _t(2, False).accepts(chain, by_name(_G_CLUSTER1X1))
    assert reason is not None and "cgrp_size_m" in reason


def test_cluster_m128_excluded_from_traversal_but_renderable() -> None:
    chain = _matmul_chain()
    # excluded from a 2ctamma template's accepted set
    reason = _t(2, False).accepts(chain, by_name(_G_M64))
    assert reason is not None and "cluster-m=128" in reason
    # ... and absent from the traversal candidate set
    cfgs = {(t.file, c.name) for t, c in candidates(chain)}
    assert ("sm100_matmul_2ctamma.py", _G_M64) not in cfgs
    # ... but still renderable for single-point JIT
    assert select_template(chain, by_name(_G_M64), 2, CLC).file == "sm100_matmul_2ctamma.py"


# -- select_template: strategy-driven file choice --------------------------


def test_select_template_by_strategy() -> None:
    chain = _matmul_chain()
    cfg = by_name(_G_DEFAULT)
    assert select_template(chain, cfg, 2, CLC).file == "sm100_matmul_2ctamma.py"
    assert select_template(chain, cfg, 1, CLC).file == "sm100_matmul_1ctamma.py"
    assert select_template(chain, cfg, 1, STATIC).file == "sm100_matmul_1ctamma_static.py"
    assert select_template(chain, cfg, 2, STATIC).file == "sm100_matmul_2ctamma_static.py"


def test_select_template_mainloop() -> None:
    chain = _mainloop_chain([FusionOp(op="relu", parent_idx=-1)])
    cfg = by_name(_G_DEFAULT)
    assert select_template(chain, cfg, 2, CLC).file == "sm100_matmul_mainloop_2ctamma.py"
    assert select_template(chain, cfg, 1, CLC).file == "sm100_matmul_mainloop_1ctamma.py"
    # mainloop has no static variant
    with pytest.raises(ValueError, match="no kernel template"):
        select_template(chain, cfg, 1, STATIC)


def test_select_template_invalid_scheduler() -> None:
    with pytest.raises(ValueError, match="scheduler"):
        select_template(_matmul_chain(), by_name(_G_DEFAULT), 2, "bogus")


# -- traversal funnel -------------------------------------------------------


def test_candidates_exclude_block_scale_and_mainloop_for_plain() -> None:
    pairs = candidates(_matmul_chain())
    assert pairs
    for t, _c in pairs:
        assert not t.block_scale and not t.mainloop


def test_default_config_is_a_candidate() -> None:
    cfgs = {(t.file, c.name) for t, c in candidates(_matmul_chain())}
    assert ("sm100_matmul_2ctamma.py", DEFAULT_CONFIG.name) in cfgs


def test_int8_zero_candidates_on_this_gpu() -> None:
    assert candidates(_matmul_chain("int8")) == []


def test_candidates_funnel_matches_full_scan() -> None:
    for dt in ("bf16", "fp8_e4m3", "int8"):
        chain = _matmul_chain(dt)
        funnel = {(t.file, c.name) for t, c in candidates(chain)}
        full = {(t.file, c.name) for t in TEMPLATES for c in CATALOG if t.accepts(chain, c) is None}
        assert funnel == full, dt


def test_enumerate_candidates_is_candidates() -> None:
    chain = _matmul_chain()
    assert enumerate_candidates(chain) == candidates(chain)


# -- dimension 4: mainloop op-scope hook (per-template) --------------------


def test_mainloop_extra_reject_mechanism_fires(monkeypatch) -> None:
    import cudnn.TBD.gemm.kernel_registry as kr

    chain = _mainloop_chain([FusionOp(op="relu", parent_idx=-1)])
    ml = next(t for t in TEMPLATES if isinstance(t, MainloopKernelTemplate) and t.cta_group == 1)
    assert ml._extra_reject(chain, by_name(_G_1CTA)) is None
    monkeypatch.setattr(kr, "_SUPPORTED_MAINLOOP_OPS", frozenset())
    reason = ml._extra_reject(chain, by_name(_G_1CTA))
    assert reason is not None and "relu" in reason
