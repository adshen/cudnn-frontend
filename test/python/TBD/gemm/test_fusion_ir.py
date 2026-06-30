"""Unit tests for fusion_ir construction and validation."""

from __future__ import annotations

import pytest

from cudnn.TBD.gemm.fusion_ir import (
    BINARY_OPS,
    FusionChain,
    FusionOp,
    MatmulSpec,
    ReductionSpec,
    TensorRef,
    UNARY_OPS,
)


def _mm(M=128, N=128, K=64) -> MatmulSpec:
    return MatmulSpec(M=M, N=N, K=K, batch=1, a_dtype="bf16", b_dtype="bf16", accum_dtype="fp32")


def test_identity_chain() -> None:
    c = FusionChain(matmul=_mm(), output_dtype="bf16")
    assert c.summary().endswith("identity -> bf16]")


def test_unary_chain() -> None:
    c = FusionChain(
        matmul=_mm(),
        ops=[FusionOp("relu"), FusionOp("gelu_tanh")],
        output_dtype="bf16",
    )
    assert "relu -> gelu_tanh" in c.summary()


def test_binary_with_aux_bias() -> None:
    bias = TensorRef(name="bias", dim=(1, 1, 128), stride=(128, 128, 1), dtype="bf16", bcast_mode="per_col")
    c = FusionChain(
        matmul=_mm(),
        aux_tensors=[bias],
        ops=[FusionOp("add", aux="bias"), FusionOp("gelu_tanh")],
        output_dtype="bf16",
    )
    assert c.aux_by_name("bias") is bias
    assert "add(bias) -> gelu_tanh" in c.summary()


def test_rejects_unknown_op() -> None:
    with pytest.raises(ValueError, match="unknown op"):
        FusionOp("matmul_squared")


def test_accepts_supported_epilogue_ops() -> None:
    for op in UNARY_OPS:
        assert FusionOp(op).op == op
    for op in BINARY_OPS:
        assert FusionOp(op, aux="x").op == op


def test_op_compute_dtype_accepts_fp32_and_int32() -> None:
    assert FusionOp("relu").compute_dtype == "fp32"
    assert FusionOp("add", aux="x", compute_dtype="int32").compute_dtype == "int32"
    with pytest.raises(ValueError, match="compute_data_type .* not.*supported"):
        FusionOp("relu", compute_dtype="bf16")


def test_reduction_dtype_accepts_fp32_and_int32_only() -> None:
    assert ReductionSpec("add", source_ref=-1, dim=(1, 1, 1)).dtype == "fp32"
    assert (
        ReductionSpec(
            "add",
            source_ref=-1,
            dim=(1, 1, 1),
            dtype="int32",
            compute_dtype="int32",
        ).compute_dtype
        == "int32"
    )
    with pytest.raises(ValueError, match="reduction output dtype .* not supported"):
        ReductionSpec("add", source_ref=-1, dim=(1, 1, 1), dtype="bf16")
    with pytest.raises(ValueError, match="reduction compute_dtype .* not supported"):
        ReductionSpec(
            "add",
            source_ref=-1,
            dim=(1, 1, 1),
            compute_dtype="bf16",
        )
    with pytest.raises(ValueError, match="must match compute_dtype"):
        ReductionSpec(
            "add",
            source_ref=-1,
            dim=(1, 1, 1),
            dtype="int32",
            compute_dtype="fp32",
        )


def test_op_out_dtype_defaults_fp32_and_accepts_narrow() -> None:
    # Default (legacy) is fp32 = no rounding; a narrow out_dtype is accepted.
    assert FusionOp("relu").out_dtype == "fp32"
    assert FusionOp("relu", out_dtype="bf16").out_dtype == "bf16"
    assert FusionOp("relu", out_dtype="int8").out_dtype == "int8"
    with pytest.raises(ValueError, match="unsupported out_dtype"):
        FusionOp("relu", out_dtype="fp64")  # type: ignore[arg-type]


def test_matmul_out_dtype_default_and_override() -> None:
    assert _mm().out_dtype == "bf16"  # default
    assert MatmulSpec(M=128, N=128, K=64, a_dtype="bf16", b_dtype="bf16", out_dtype="fp16").out_dtype == "fp16"


def test_rejects_unary_with_aux() -> None:
    with pytest.raises(ValueError, match="cannot have aux"):
        FusionOp("relu", aux="bias")


def test_rejects_binary_without_second_operand() -> None:
    with pytest.raises(ValueError, match="aux tensor name or"):
        FusionOp("add")


def test_rejects_binary_with_both_aux_and_parent_idx_b() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        FusionOp("add", aux="x", parent_idx_b=0)


def test_accepts_binary_fan_in() -> None:
    """Phase 4: binary op with parent_idx_b set is fan-in (no aux)."""
    op = FusionOp("add", parent_idx=-1, parent_idx_b=0)
    assert op.aux is None
    assert op.parent_idx_b == 0


def test_rejects_unary_with_parent_idx_b() -> None:
    with pytest.raises(ValueError, match="parent_idx_b"):
        FusionOp("relu", parent_idx_b=0)


def test_rejects_unknown_aux_reference() -> None:
    with pytest.raises(ValueError, match="unknown aux"):
        FusionChain(
            matmul=_mm(),
            aux_tensors=[],
            ops=[FusionOp("add", aux="ghost")],
            output_dtype="bf16",
        )


def test_rejects_duplicate_aux_name() -> None:
    a = TensorRef(name="x", dim=(1,), stride=(1,), dtype="bf16", bcast_mode="scalar")
    b = TensorRef(name="x", dim=(1,), stride=(1,), dtype="bf16", bcast_mode="scalar")
    with pytest.raises(ValueError, match="duplicate names"):
        FusionChain(matmul=_mm(), aux_tensors=[a, b], output_dtype="bf16")


def test_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        TensorRef(name="x", dim=(1,), stride=(1,), dtype="fp64", bcast_mode="scalar")  # type: ignore[arg-type]


def test_accepts_epilogue_integer_and_e8m0_dtypes() -> None:
    for dtype in ("fp8_e8m0", "int8", "uint8", "int32"):
        aux = TensorRef(
            name="x",
            dim=(1,),
            stride=(1,),
            dtype=dtype,  # type: ignore[arg-type]
            bcast_mode="scalar",
        )
        chain = FusionChain(
            matmul=_mm(),
            aux_tensors=[aux],
            ops=[FusionOp("add", aux="x")],
            output_dtype=dtype,  # type: ignore[arg-type]
        )
        assert chain.output_dtype == dtype


def test_rejects_dim_stride_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        TensorRef(name="x", dim=(1, 2), stride=(1,), dtype="bf16", bcast_mode="per_col")


# Dtype family tests


def test_fp8_matmul_accepted() -> None:
    spec = MatmulSpec(M=128, N=128, K=128, a_dtype="fp8_e4m3", b_dtype="fp8_e4m3", accum_dtype="fp32")
    assert spec.a_dtype == "fp8_e4m3"


def test_fp16_matmul_accepted() -> None:
    spec = MatmulSpec(M=128, N=128, K=64, a_dtype="fp16", b_dtype="fp16", accum_dtype="fp32")
    assert spec.a_dtype == "fp16"


def test_matmul_spec_accepts_any_dtype_fields() -> None:
    # MatmulSpec doesn't judge MMA-dtype runnability (no arch info) — it accepts
    # the dtype fields as given; the compiler's gate rejects unrunnable combos
    # (e.g. fp32×fp32, which has no TF32 path) at JIT time.
    spec = MatmulSpec(M=128, N=128, K=64, a_dtype="fp32", b_dtype="fp32", accum_dtype="fp32")
    assert spec.a_dtype == "fp32"


def test_matmul_spec_does_not_validate_mma_dtype_combo() -> None:
    # The IR has no GPU-arch info, so it does NOT judge whether an
    # (a, b, accum) MMA combination is runnable — that's the compiler's
    # arch-aware `_check_supported` gate (see test_compiler.py). MatmulSpec
    # only enforces structural invariants + output-storage dtype. So these
    # combos CONSTRUCT fine here and are rejected later, at JIT time.
    for a, b, acc in [
        ("fp8_e4m3", "bf16", "fp32"),  # mixed fp8 / 16-bit
        ("int8", "bf16", "int32"),  # int8 paired with non-int8
        ("bf16", "bf16", "fp16"),  # non-fp32 accumulate
        ("fp8_e4m3", "fp8_e5m2", "fp32"),  # mixed fp8 variants (actually valid)
        ("int8", "int8", "int32"),  # int8 (actually valid)
    ]:
        spec = MatmulSpec(M=128, N=128, K=128, a_dtype=a, b_dtype=b, accum_dtype=acc)
        assert (spec.a_dtype, spec.b_dtype, spec.accum_dtype) == (a, b, acc)


def test_matmul_spec_still_validates_output_storage_dtype() -> None:
    # Output-storage dtype well-formedness IS checked here (the combo table
    # doesn't cover it; an unknown out_dtype would KeyError at render).
    with pytest.raises(ValueError, match="out_dtype"):
        MatmulSpec(M=128, N=128, K=64, a_dtype="bf16", b_dtype="bf16", accum_dtype="fp32", out_dtype="fp64")  # type: ignore[arg-type]
