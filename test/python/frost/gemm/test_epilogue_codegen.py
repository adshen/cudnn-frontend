"""Snapshot-style tests for epilogue_codegen output strings — the rendered text
is the codegen↔template contract, so pinning it makes drift visible."""

from __future__ import annotations

import pytest

from cudnn.frost.gemm.epilogue_codegen import generate
from cudnn.frost.gemm.fusion_ir import (
    BlockQuantizeSpec,
    FusionChain,
    FusionOp,
    MatmulSpec,
    ReductionSpec,
    TensorRef,
    gemm_source,
)

pytestmark = pytest.mark.L0


def _mm() -> MatmulSpec:
    return MatmulSpec(M=128, N=128, K=64)


def test_identity_chain() -> None:
    out = generate(FusionChain(matmul=_mm(), output_dtype="bf16"))
    assert out.aux_views == "pass"
    # matmul out_dtype defaults to bf16 → accumulator rounded to bf16 (_r_mm) before terminal cast.
    assert out.epilogue == ("_r_mm = (vec_f32).to(cutlass.BFloat16)\n" "vec_out = (_r_mm).to(cutlass.BFloat16)")
    assert out.kernel_params == []
    assert out.host_args == []


@pytest.mark.parametrize(
    "mode,needles",
    (
        (
            "amax",
            (
                "cute.arch.atomic_fmax(gC_tap_0_ptr + ",
                "cute.math.abs(_red_0_src[0])",
                "sign_bit=False",
            ),
        ),
        (
            "max",
            (
                "cute.arch.atomic_fmax(gC_tap_0_ptr + ",
                '_red_0_src[0], sem="relaxed", scope="gpu")',
            ),
        ),
        (
            "min",
            (
                "_red_0_0_bits = (_red_0_src[0]).bitcast(cutlass.Int32)",
                "cute.arch.atomic_max(gC_tap_0_ptr + ",
                "cutlass.Uint32(_red_0_0_bits)",
                "cute.arch.atomic_min(gC_tap_0_ptr + ",
            ),
        ),
    ),
    ids=("amax", "max", "min"),
)
def test_reduction_tap_emits_atomic(mode: str, needles: tuple[str, ...]) -> None:
    chain = FusionChain(
        matmul=_mm(),
        output_dtype="bf16",
        reductions=[
            ReductionSpec(
                mode=mode,
                source_ref=gemm_source(0),
                dim=(1, 1, 1),
                dtype="fp32",
            )
        ],
    )
    out = generate(chain)
    assert out.tap_kernel_params == ["mC_tap_0: cute.Tensor"]
    assert "(1, 1, 1)" in out.tap_compile_fakes[0]
    assert "_red_0_src = (_r_mm).to(cutlass.Float32)" in out.epilogue
    assert "red_stride_m_0" in out.epilogue
    assert "red_stride_n_0" in out.epilogue
    assert "red_stride_l_0" in out.epilogue
    for needle in needles:
        assert needle in out.epilogue
    assert out.epilogue.endswith("vec_out = (_r_mm).to(cutlass.BFloat16)")


@pytest.mark.parametrize(
    "mode,needles",
    (
        (
            "add",
            (
                "_red_0_src = (_r_mm).to(cutlass.Int32)",
                'nvvm.atomicrmw("add", gC_tap_0_ptr + ',
                '_red_0_src[0], mem_order="relaxed", syncscope="gpu")',
            ),
        ),
        (
            "amax",
            (
                "_red_0_src = (_r_mm).to(cutlass.Int32)",
                'nvvm.atomicrmw("max", gC_tap_0_ptr + ',
                'cute.math.abs(_red_0_src[0]), mem_order="relaxed", syncscope="gpu")',
            ),
        ),
        (
            "max",
            (
                "_red_0_src = (_r_mm).to(cutlass.Int32)",
                'nvvm.atomicrmw("max", gC_tap_0_ptr + ',
                '_red_0_src[0], mem_order="relaxed", syncscope="gpu")',
            ),
        ),
        (
            "min",
            (
                "_red_0_src = (_r_mm).to(cutlass.Int32)",
                'nvvm.atomicrmw("min", gC_tap_0_ptr + ',
                '_red_0_src[0], mem_order="relaxed", syncscope="gpu")',
            ),
        ),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_int32_reduction_tap_emits_integer_atomic(mode: str, needles: tuple[str, ...]) -> None:
    chain = FusionChain(
        matmul=MatmulSpec(M=128, N=128, K=64, out_dtype="int32"),
        output_dtype="int32",
        reductions=[
            ReductionSpec(
                mode=mode,
                source_ref=gemm_source(0),
                dim=(1, 1, 1),
                dtype="int32",
                compute_dtype="int32",
            )
        ],
    )
    out = generate(chain)
    for needle in needles:
        assert needle in out.epilogue
    assert "cute.arch.atomic_fmax" not in out.epilogue
    assert "bitcast(cutlass.Int32)" not in out.epilogue
    assert out.epilogue.endswith("vec_out = (_r_mm).to(cutlass.Int32)")


def test_block_quantize_emits_quantized_store_and_scale_tap() -> None:
    chain = FusionChain(
        matmul=MatmulSpec(M=128, N=128, K=64, out_dtype="fp32"),
        output_dtype="fp8_e4m3",
        block_quant=BlockQuantizeSpec(
            source_ref=gemm_source(0),
            block_size=32,
            scale_dtype="fp8_e8m0",
            scale_dim=(1, 128, 4),
        ),
    )
    out = generate(chain, vec_bytes_epi=32, output_elem_bytes=1)

    assert out.tap_kernel_params == ["mC_tap_0: cute.Tensor"]
    assert "cutlass.Float8E8M0FNU" in out.tap_compile_fakes[0]
    assert "(sym_m, (sym_n // 32), 1)" in out.tap_compile_fakes[0]
    assert "_q_src = (vec_f32).to(cutlass.Float32)" in out.epilogue
    assert "_q_amax = cute.math.max(_q_amax, _q_abs[1])" in out.epilogue
    assert "_q_scale = (_q_scale_f32).to(cutlass.Float8E8M0FNU)" in out.epilogue
    assert "_q_scaled = _q_src * cutlass.full_like(_q_src, _q_inv)" in out.epilogue
    assert "cutlass.Float32(-448.0)" in out.epilogue
    assert "cutlass.Float32(448.0)" in out.epilogue
    assert "vec_out = _q_clamped.to(cutlass.Float8E4M3FN)" in out.epilogue
    assert "quant_scale_stride_m" in out.epilogue
    assert "(gC_tap_0_ptr + _q_scale_idx).store(_q_scale, alignment=1)" in out.epilogue


def test_block_quantize_f8_128x4_scale_uses_blocked_index() -> None:
    chain = FusionChain(
        matmul=MatmulSpec(M=128, N=128, K=64, out_dtype="fp32"),
        output_dtype="fp8_e4m3",
        block_quant=BlockQuantizeSpec(
            source_ref=gemm_source(0),
            block_size=32,
            scale_dtype="fp8_e8m0",
            scale_dim=(1, 128, 4),
            scale_reorder="F8_128x4",
        ),
    )
    out = generate(chain, vec_bytes_epi=32, output_elem_bytes=1)

    assert "_q_scale_ncb = ((N // 32) + 3) // 4" in out.epilogue
    assert "((row // 128) * _q_scale_ncb + (_q_scale_col // 4)) * 512" in out.epilogue
    assert "row * quant_scale_stride_m" not in out.epilogue


def test_block_quantize_padded_scale_fake_shape_uses_static_extent() -> None:
    chain = FusionChain(
        matmul=MatmulSpec(M=144, N=320, K=64, out_dtype="fp32"),
        output_dtype="fp8_e4m3",
        block_quant=BlockQuantizeSpec(
            source_ref=gemm_source(0),
            block_size=32,
            scale_dtype="fp8_e8m0",
            scale_dim=(1, 256, 12),
            scale_reorder="F8_128x4",
        ),
    )
    out = generate(chain, vec_bytes_epi=32, output_elem_bytes=1)

    assert "(256, 12, 1)" in out.tap_compile_fakes[0]


def test_relu_chain() -> None:
    out = generate(FusionChain(matmul=_mm(), ops=[FusionOp("relu")], output_dtype="bf16"))
    assert out.aux_views == "pass"
    assert "_r_mm = (vec_f32).to(cutlass.BFloat16)" in out.epilogue
    assert "_c_0_a = (_r_mm).to(cutlass.Float32)" in out.epilogue
    assert "cute.math.max(_c_0_a, cutlass.full_like(_c_0_a, cutlass.Float32(0.0)))" in out.epilogue
    assert out.epilogue.endswith("vec_out = (_op_0).to(cutlass.BFloat16)")


def test_bias_per_row_then_gelu_tanh() -> None:
    bias = TensorRef(name="bias", dim=(128, 1), stride=(1, 1), dtype="bf16", bcast_mode="per_row")
    chain = FusionChain(
        matmul=_mm(),
        aux_tensors=[bias],
        ops=[FusionOp("add", aux="bias"), FusionOp("gelu_tanh")],
        output_dtype="bf16",
    )
    out = generate(chain)
    # `row` is defined by the kernel template (M-aware), not by codegen.
    assert "row = tidx + coord_m" not in out.aux_views
    assert "_aux_bias_ptr = bias.iterator.raw_ptr()" in out.aux_views
    assert "_aux_bias_pre = (_aux_bias_ptr + row).load()" in out.aux_views
    assert "_r_mm = (vec_f32).to(cutlass.BFloat16)" in out.epilogue
    assert "_c_0_a = (_r_mm).to(cutlass.Float32)" in out.epilogue
    assert "_op_0 = _c_0_a + (cutlass.full_like(_c_0_a, _aux_bias_pre.to(cutlass.Float32)))" in out.epilogue
    assert "_c_1_a = (_op_0).to(cutlass.Float32)" in out.epilogue
    # gelu_tanh is whole-vector (tanh via vector exp2/rcp fastmath), no scalar repack.
    assert "cute.math.exp2(" in out.epilogue and "cute.math.rcp(" in out.epilogue
    assert "fastmath=True" in out.epilogue
    assert "approx=True, ftz=True)" in out.epilogue
    assert "vector_from_scalars(" not in out.epilogue
    assert out.epilogue.endswith("vec_out = (_op_1).to(cutlass.BFloat16)")
    assert out.kernel_params == ["bias: cute.Tensor"]
    assert out.host_args == ["bias"]


def test_bias_per_col_uses_per_vector_load() -> None:
    bias = TensorRef(name="bias", dim=(1, 128), stride=(128, 1), dtype="bf16", bcast_mode="per_col")
    chain = FusionChain(
        matmul=_mm(),
        aux_tensors=[bias],
        ops=[FusionOp("add", aux="bias")],
        output_dtype="bf16",
    )
    out = generate(chain)
    # Per-col bias is NOT prefetched — loaded inside the loop (references col_j).
    assert "_aux_bias_pre" not in out.aux_views
    assert "(_aux_bias_ptr + col_j).load(count=vsize" in out.epilogue


def test_rank1_per_col_uses_col_j() -> None:
    bias = TensorRef(name="bias", dim=(128,), stride=(1,), dtype="bf16", bcast_mode="per_col")
    chain = FusionChain(
        matmul=_mm(),
        aux_tensors=[bias],
        ops=[FusionOp("add", aux="bias")],
        output_dtype="bf16",
    )
    out = generate(chain)
    assert "(_aux_bias_ptr + col_j).load(count=vsize" in out.epilogue


def test_rank1_scalar_uses_zero_offset() -> None:
    scale = TensorRef(name="scale", dim=(1,), stride=(1,), dtype="bf16", bcast_mode="scalar")
    chain = FusionChain(
        matmul=_mm(),
        aux_tensors=[scale],
        ops=[FusionOp("add", aux="scale")],
        output_dtype="bf16",
    )
    out = generate(chain)
    assert "_aux_scale_pre = (_aux_scale_ptr + 0).load()" in out.aux_views
    assert "_op_0 = _c_0_a + (cutlass.full_like(_c_0_a, _aux_scale_pre.to(cutlass.Float32)))" in out.epilogue


def test_swish_chain() -> None:
    # swish = x * sigmoid(x): whole-vector exp2 fastmath + rcp approx/ftz (MUFU), no scalar repack.
    out = generate(FusionChain(matmul=_mm(), ops=[FusionOp("swish")], output_dtype="bf16"))
    assert "cute.math.exp2(-_c_0_a * cutlass.full_like(_c_0_a, cutlass.Float32(1.4426950408889634)), fastmath=True)" in out.epilogue
    assert "approx=True, ftz=True)" in out.epilogue
    assert "vector_from_scalars(" not in out.epilogue


@pytest.mark.parametrize(
    "op,needle",
    (
        ("ceil", "_op_0 = cute.math.ceil(_c_0_a)"),
        ("floor", "_op_0 = cute.math.floor(_c_0_a)"),
        ("erf", "_op_0 = cute.math.erf(_c_0_a)"),
        ("log", "_op_0 = cute.math.log(_c_0_a)"),
        ("reciprocal", "_op_0 = cute.math.rcp(_c_0_a, approx=True, ftz=True)"),
        ("rsqrt", "_op_0 = cute.math.rsqrt(_c_0_a)"),
        ("sqrt", "_op_0 = cute.math.sqrt(_c_0_a)"),
    ),
)
def test_additional_unary_codegen(op: str, needle: str) -> None:
    out = generate(FusionChain(matmul=_mm(), ops=[FusionOp(op)], output_dtype="bf16"))
    assert needle in out.epilogue


@pytest.mark.parametrize(
    "op,needle",
    (
        (
            "max",
            "_op_0 = cute.math.max(_c_0_a, (cutlass.full_like(_c_0_a, _aux_aux_pre.to(cutlass.Float32))))",
        ),
        (
            "min",
            "_op_0 = cute.math.min(_c_0_a, (cutlass.full_like(_c_0_a, _aux_aux_pre.to(cutlass.Float32))))",
        ),
        (
            "pow",
            "_op_0 = cute.math.pow(_c_0_a, (cutlass.full_like(_c_0_a, _aux_aux_pre.to(cutlass.Float32))))",
        ),
        ("add_square", "_op_0 = _c_0_a + _sq_aux_0 * _sq_aux_0"),
    ),
)
def test_additional_binary_codegen(op: str, needle: str) -> None:
    aux = TensorRef(name="aux", dim=(1,), stride=(1,), dtype="bf16", bcast_mode="scalar")
    chain = FusionChain(
        matmul=_mm(),
        aux_tensors=[aux],
        ops=[FusionOp(op, aux="aux")],
        output_dtype="bf16",
    )
    out = generate(chain)
    if op == "add_square":
        assert "_sq_aux_0 = (cutlass.full_like(_c_0_a, _aux_aux_pre.to(cutlass.Float32)))" in out.epilogue
    assert needle in out.epilogue


def test_add_square_aux_on_lhs_codegen() -> None:
    aux = TensorRef(name="aux", dim=(1,), stride=(1,), dtype="bf16", bcast_mode="scalar")
    chain = FusionChain(
        matmul=_mm(),
        aux_tensors=[aux],
        ops=[FusionOp("add_square", aux="aux", aux_on_rhs=False)],
        output_dtype="bf16",
    )
    out = generate(chain)
    assert "_sq_aux_0 = (cutlass.full_like(_c_0_a, _aux_aux_pre.to(cutlass.Float32)))" in out.epilogue
    assert "_op_0 = _sq_aux_0 + _c_0_a * _c_0_a" in out.epilogue


def test_sub_with_aux_on_lhs() -> None:
    """aux_on_rhs=False puts the aux on the left of the binary op."""
    aux = TensorRef(name="c", dim=(1,), stride=(1,), dtype="bf16", bcast_mode="scalar")
    chain = FusionChain(
        matmul=_mm(),
        aux_tensors=[aux],
        ops=[FusionOp("sub", aux="c", aux_on_rhs=False)],
        output_dtype="bf16",
    )
    out = generate(chain)
    assert "_op_0 = (cutlass.full_like(_c_0_a, _aux_c_pre.to(cutlass.Float32))) - _c_0_a" in out.epilogue


def test_extra_kernel_params_for_multiple_aux() -> None:
    a1 = TensorRef(name="bias", dim=(1, 128), stride=(128, 1), dtype="bf16", bcast_mode="per_col")
    a2 = TensorRef(name="scale", dim=(1,), stride=(1,), dtype="bf16", bcast_mode="scalar")
    chain = FusionChain(
        matmul=_mm(),
        aux_tensors=[a1, a2],
        ops=[FusionOp("add", aux="bias"), FusionOp("mul", aux="scale")],
        output_dtype="bf16",
    )
    out = generate(chain)
    assert out.kernel_params == ["bias: cute.Tensor", "scale: cute.Tensor"]
    assert out.host_args == ["bias", "scale"]


@pytest.mark.parametrize(
    "dtype,cast_expr",
    [
        ("fp8_e8m0", "(_r_mm).to(cutlass.Float8E8M0FNU)"),
        ("int8", "(_r_mm).to(cutlass.Int8)"),
        ("uint8", "(_r_mm).to(cutlass.Uint8).bitcast(cutlass.Int8)"),
        ("int32", "(_r_mm).to(cutlass.Int32)"),
    ],
)
def test_epilogue_dtype_casts_for_integer_and_e8m0_outputs(dtype, cast_expr) -> None:
    # matmul out_dtype defaults to bf16 → accumulator rounded to bf16 (_r_mm) before terminal cast.
    out = generate(FusionChain(matmul=_mm(), output_dtype=dtype))  # type: ignore[arg-type]
    assert out.epilogue == (f"_r_mm = (vec_f32).to(cutlass.BFloat16)\nvec_out = {cast_expr}")


def test_aux_kernel_param_accepts_integer_dtype() -> None:
    aux = TensorRef(name="scale", dim=(1,), stride=(1,), dtype="int8", bcast_mode="scalar")
    chain = FusionChain(
        matmul=_mm(),
        aux_tensors=[aux],
        ops=[FusionOp("add", aux="scale")],
        output_dtype="bf16",
    )
    out = generate(chain)
    assert "_aux_scale_pre = (_aux_scale_ptr + 0).load()" in out.aux_views


def test_int32_compute_casts_parent_and_aux() -> None:
    aux = TensorRef(name="aux", dim=(1,), stride=(1,), dtype="int32", bcast_mode="scalar")
    chain = FusionChain(
        matmul=MatmulSpec(M=128, N=128, K=64, out_dtype="int32"),
        aux_tensors=[aux],
        ops=[FusionOp("add", aux="aux", compute_dtype="int32")],
        output_dtype="int32",
    )
    out = generate(chain)
    assert "_r_mm = (vec_f32).to(cutlass.Int32)" in out.epilogue
    assert "_c_0_a = (_r_mm).to(cutlass.Int32)" in out.epilogue
    assert "_aux_aux_pre = (_aux_aux_ptr + 0).load()" in out.aux_views
    assert "_op_0 = _c_0_a + (cutlass.full_like(_c_0_a, _aux_aux_pre.to(cutlass.Int32)))" in out.epilogue
    assert out.epilogue.endswith("vec_out = (_op_0).to(cutlass.Int32)")


@pytest.mark.parametrize(
    "dim,stride,bcast,needle",
    (
        ((1, 1, 1), (1, 1, 1), "scalar", "_aux_aux_pre = (_aux_aux_ptr + 0).load()"),
        (
            (2, 1, 1),
            (1, 1, 1),
            "scalar",
            "_aux_aux_pre = (_aux_aux_ptr + tile_l).load()",
        ),
        (
            (1, 128, 1),
            (128, 1, 1),
            "per_row",
            "_aux_aux_pre = (_aux_aux_ptr + row).load()",
        ),
        (
            (2, 128, 1),
            (128, 1, 1),
            "per_row",
            "_aux_aux_pre = (_aux_aux_ptr + tile_l * 128 + row).load()",
        ),
        (
            (1, 1, 128),
            (128, 128, 1),
            "per_col",
            "(_aux_aux_ptr + col_j).load(count=vsize",
        ),
        (
            (2, 1, 128),
            (128, 128, 1),
            "per_col",
            "(_aux_aux_ptr + tile_l * 128 + col_j).load(count=vsize",
        ),
        (
            (1, 128, 128),
            (16384, 128, 1),
            "per_elem",
            "(_aux_aux_ptr + row * 128 + col_j).load(count=vsize",
        ),
        (
            (2, 128, 128),
            (16384, 128, 1),
            "per_elem",
            "(_aux_aux_ptr + tile_l * 16384 + row * 128 + col_j).load(count=vsize",
        ),
    ),
)
def test_rank3_broadcast_indexing_uses_only_non_broadcast_axes(
    dim: tuple[int, int, int],
    stride: tuple[int, int, int],
    bcast: str,
    needle: str,
) -> None:
    aux = TensorRef(
        name="aux",
        dim=dim,
        stride=stride,
        dtype="bf16",
        bcast_mode=bcast,  # type: ignore[arg-type]
    )
    chain = FusionChain(
        matmul=MatmulSpec(M=128, N=128, K=64, batch=2, a_batch=2, b_batch=2),
        aux_tensors=[aux],
        ops=[FusionOp("add", aux="aux")],
        output_dtype="bf16",
    )
    out = generate(chain)
    assert needle in f"{out.aux_views}\n{out.epilogue}"
