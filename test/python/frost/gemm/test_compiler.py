"""Unit tests for compiler-side layout validation and template constants."""

from __future__ import annotations

import pytest

import cudnn.frost.gemm.compiler as compiler
from cudnn.frost.gemm.compiler import (
    _block_quant_cols_per_acc_stage,
    _check_block_quant_supported,
    _check_dtype_config_compat,
    _check_input_alignment,
    _check_supported,
    _render_tile_constants,
    _template_arch_family,
)
from cudnn.frost.gemm.fusion_ir import (
    BlockQuantizeSpec,
    FusionChain,
    FusionOp,
    MatmulSpec,
    MoeSpec,
    gemm_source,
)
from cudnn.frost.gemm.kernel_registry import MMA_TYPE_SUPPORT, GraphType

# Plain-matmul mma-type table (dict half of the unified support entry).
_MATMUL_MMA_TABLE = MMA_TYPE_SUPPORT[GraphType.MATMUL][1]
from cudnn.frost.gemm.tile_config import DEFAULT_CONFIG, by_name


def _chain(
    *,
    M: int = 128,
    N: int = 128,
    K: int = 128,
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
    dtype: str = "bf16",
) -> FusionChain:
    return FusionChain(
        matmul=MatmulSpec(
            M=M,
            N=N,
            K=K,
            a_major=a_major,  # type: ignore[arg-type]
            b_major=b_major,  # type: ignore[arg-type]
            out_major=out_major,  # type: ignore[arg-type]
            a_dtype=dtype,  # type: ignore[arg-type]
            b_dtype=dtype,  # type: ignore[arg-type]
            accum_dtype="fp32",
        ),
        output_dtype="bf16",
    )


def _block_quant_chain(
    *,
    M: int = 128,
    N: int = 128,
    K: int = 128,
    block_size: int = 32,
    out_major: str = "n",
    output_dtype: str = "fp8_e4m3",
    scale_dtype: str = "fp8_e8m0",
    scale_dim: tuple[int, int, int] | None = None,
    scale_reorder: str | None = None,
    mainloop_fusion: bool = False,
    moe: bool = False,
) -> FusionChain:
    return FusionChain(
        matmul=MatmulSpec(
            M=M,
            N=N,
            K=K,
            out_major=out_major,  # type: ignore[arg-type]
        ),
        output_dtype=output_dtype,  # type: ignore[arg-type]
        mainloop_a_ops=[FusionOp("abs")] if mainloop_fusion else [],
        moe=MoeSpec(num_experts=2) if moe else None,
        block_quant=BlockQuantizeSpec(
            source_ref=gemm_source(0),
            block_size=block_size,
            scale_dtype=scale_dtype,  # type: ignore[arg-type]
            scale_dim=scale_dim or (1, M, N // block_size),
            scale_reorder=scale_reorder,
        ),
    )


def _rendered_constants(chain: FusionChain) -> dict[str, str]:
    constants: dict[str, str] = {}
    for line in _render_tile_constants(DEFAULT_CONFIG, chain, 2).splitlines():
        if " = " in line and not line.startswith("#"):
            name, value = line.split(" = ", 1)
            constants[name] = value
    return constants


def test_input_alignment_uses_layout_contiguous_dimension() -> None:
    _check_input_alignment(_chain(M=8, N=8, K=64, a_major="m", b_major="n"))

    with pytest.raises(ValueError, match="A m-major extent 7"):
        _check_input_alignment(_chain(M=7, N=8, K=64, a_major="m"))

    with pytest.raises(ValueError, match="B n-major extent 7"):
        _check_input_alignment(_chain(M=8, N=7, K=64, b_major="n"))


def test_mn_major_config_rejects_unsupported_smem_grouping() -> None:
    cfg = by_name("CONFIG_sm100_64x64x128_64x64x32_cluster2x4")

    with pytest.raises(ValueError, match="cannot use M-major A"):
        _check_dtype_config_compat(_chain(a_major="m", dtype="fp8_e4m3"), cfg, 2)

    with pytest.raises(ValueError, match="cannot use N-major B"):
        _check_dtype_config_compat(_chain(b_major="n"), cfg, 2)


def test_template_arch_family_parses_sm_token() -> None:
    assert _template_arch_family("sm100_matmul_1ctamma.py") == "sm100"
    assert _template_arch_family("sm100_block_scale_matmul_2ctamma.py") == "sm100"
    with pytest.raises(ValueError, match="arch family"):
        _template_arch_family("matmul_1ctamma.py")


def test_gate_accepts_sm100_family(monkeypatch) -> None:
    # bf16 runs on every SM in the sm100 family (100..119).
    for sm in (100, 103, 119):
        monkeypatch.setattr(compiler, "_current_sm", lambda v=sm: v)
        _check_supported(_chain(dtype="bf16"), DEFAULT_CONFIG)


def test_gate_rejects_out_of_range_sm(monkeypatch) -> None:
    # Hopper (sm_90) and post-Blackwell (sm_120+) are out of range.
    for sm in (90, 120, 121):
        monkeypatch.setattr(compiler, "_current_sm", lambda v=sm: v)
        with pytest.raises(NotImplementedError, match=rf"sm_{sm}"):
            _check_supported(_chain(dtype="bf16"), DEFAULT_CONFIG)


def test_gate_dtype_check_runs_without_gpu(monkeypatch) -> None:
    # No device: arch half skipped, but the (pipeline, dtype-combo) half still runs.
    monkeypatch.setattr(compiler, "_current_sm", lambda: None)
    _check_supported(_chain(dtype="bf16"), DEFAULT_CONFIG)
    bad = FusionChain(
        matmul=MatmulSpec(M=128, N=128, K=128, a_dtype="bf16", b_dtype="fp16"),
        output_dtype="bf16",
    )
    with pytest.raises(NotImplementedError, match="does not support input/acc dtype combo"):
        _check_supported(bad, DEFAULT_CONFIG)


def test_block_quant_gate_accepts_supported_layouts() -> None:
    _check_block_quant_supported(_block_quant_chain(), vec_bytes_epi=32, config=DEFAULT_CONFIG, cta_group=2)
    _check_block_quant_supported(
        _block_quant_chain(moe=True),
        vec_bytes_epi=32,
        config=DEFAULT_CONFIG,
        cta_group=2,
    )
    _check_block_quant_supported(
        _block_quant_chain(scale_reorder="F8_128x4"),
        vec_bytes_epi=32,
        config=DEFAULT_CONFIG,
        cta_group=2,
    )
    _check_block_quant_supported(
        _block_quant_chain(
            M=64,
            N=96,
            scale_reorder="F8_128x4",
            scale_dim=(1, 128, 4),
        ),
        vec_bytes_epi=32,
        config=DEFAULT_CONFIG,
        cta_group=2,
    )


def test_block_quant_gate_rejects_unsupported_kernel_families() -> None:
    with pytest.raises(NotImplementedError, match="mainloop fusion"):
        _check_block_quant_supported(
            _block_quant_chain(mainloop_fusion=True),
            vec_bytes_epi=32,
            config=DEFAULT_CONFIG,
            cta_group=2,
        )


def test_block_quant_gate_rejects_unsupported_output_layout_and_block_size() -> None:
    with pytest.raises(NotImplementedError, match="N-major output"):
        _check_block_quant_supported(
            _block_quant_chain(out_major="m"),
            vec_bytes_epi=32,
            config=DEFAULT_CONFIG,
            cta_group=2,
        )

    with pytest.raises(NotImplementedError, match="block_size"):
        _check_block_quant_supported(
            _block_quant_chain(block_size=16),
            vec_bytes_epi=32,
            config=DEFAULT_CONFIG,
            cta_group=2,
        )


def test_block_quant_gate_rejects_partial_cta_quant_blocks() -> None:
    cfg = by_name("CONFIG_sm100_64x32x128_64x32x32_cluster2x1")

    assert _block_quant_cols_per_acc_stage(cfg, cta_group=2) == 16
    with pytest.raises(NotImplementedError, match="whole quantization blocks"):
        _check_block_quant_supported(
            _block_quant_chain(N=32),
            vec_bytes_epi=32,
            config=cfg,
            cta_group=2,
        )


def test_block_quant_gate_rejects_unaligned_f8_128x4_scale_output() -> None:
    with pytest.raises(NotImplementedError, match="F8_128x4"):
        _check_block_quant_supported(
            _block_quant_chain(M=64, N=128, scale_reorder="F8_128x4"),
            vec_bytes_epi=32,
            config=DEFAULT_CONFIG,
            cta_group=2,
        )

    with pytest.raises(NotImplementedError, match="F8_128x4"):
        _check_block_quant_supported(
            _block_quant_chain(M=128, N=96, scale_reorder="F8_128x4"),
            vec_bytes_epi=32,
            config=DEFAULT_CONFIG,
            cta_group=2,
        )


def test_gate_rejects_mismatched_family() -> None:
    # bf16 × fp16 slips past MatmulSpec.__post_init__ (both non-FP8) — the
    # pipeline×dtype table rejects it.
    chain = FusionChain(
        matmul=MatmulSpec(M=128, N=128, K=128, a_dtype="bf16", b_dtype="fp16"),
        output_dtype="bf16",
    )
    with pytest.raises(NotImplementedError, match="does not support input/acc dtype combo"):
        _check_supported(chain, DEFAULT_CONFIG)


def test_gate_supports_disjoint_arch_ranges_per_combo(monkeypatch) -> None:
    # A (pipeline, combo) may run on disjoint SM ranges; accept if SM in ANY.
    monkeypatch.setitem(
        _MATMUL_MMA_TABLE,
        ("bf16", "bf16", "fp32"),
        ((100, 110), (120, 130)),
    )
    for sm in (100, 109, 120, 129):
        monkeypatch.setattr(compiler, "_current_sm", lambda v=sm: v)
        _check_supported(_chain(dtype="bf16"), DEFAULT_CONFIG)
    for sm in (90, 110, 115, 130):
        monkeypatch.setattr(compiler, "_current_sm", lambda v=sm: v)
        with pytest.raises(NotImplementedError, match="100 <= SM < 110 or 120 <= SM < 130"):
            _check_supported(_chain(dtype="bf16"), DEFAULT_CONFIG)


def test_pipeline_dtype_table_covers_all_fp8_pairs() -> None:
    # The plain-matmul gate admits every {E4M3, E5M2} A/B pair (incl. mixed).
    fp8 = ("fp8_e4m3", "fp8_e5m2")
    for a in fp8:
        for b in fp8:
            assert (a, b, "fp32") in _MATMUL_MMA_TABLE
    # Block-scale is a separate graph type with its own per-side key shape.
    assert GraphType.BLOCK_SCALE_MATMUL in MMA_TYPE_SUPPORT
    assert all(len(k) == 3 for k in _MATMUL_MMA_TABLE)


def test_rendered_smem_descriptor_constants_follow_input_layout() -> None:
    k_major = _rendered_constants(_chain())
    assert k_major["mma_a_major"] == "0"
    assert k_major["mma_b_major"] == "0"
    assert k_major["a_smem_desc_leading_byte_offset"] == "16"
    assert k_major["b_smem_desc_leading_byte_offset"] == "16"
    assert k_major["a_smem_k_step_bytes"] == "32"
    assert k_major["b_smem_k_step_bytes"] == "32"
    assert k_major["a_tma_group_elems"] == "1"
    assert k_major["b_tma_group_elems"] == "1"

    mn_major = _rendered_constants(_chain(a_major="m", b_major="n"))
    assert mn_major["mma_a_major"] == "1"
    assert mn_major["mma_b_major"] == "1"
    assert mn_major["a_smem_desc_leading_byte_offset"] == "8192"
    assert mn_major["b_smem_desc_leading_byte_offset"] == "8192"
    assert mn_major["a_smem_desc_stride_byte_offset"] == "1024"
    assert mn_major["b_smem_desc_stride_byte_offset"] == "1024"
    assert mn_major["a_smem_k_step_bytes"] == "2048"
    assert mn_major["b_smem_k_step_bytes"] == "2048"
    assert mn_major["a_tma_group_elems"] == "64"
    assert mn_major["b_tma_group_elems"] == "64"
