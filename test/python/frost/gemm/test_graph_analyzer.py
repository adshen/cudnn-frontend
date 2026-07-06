"""End-to-end tests for graph_analyzer: build pure cuDNN graphs, extract FusionChain."""

from __future__ import annotations

import cudnn
import cudnn.frost.gemm  # noqa: F401  (installs hook)
import pytest

from cudnn.frost.gemm.graph_analyzer import analyze


def _mk_graph() -> cudnn.pygraph:
    return cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )


def _mk_inputs(g: cudnn.pygraph, M: int, N: int, K: int):
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    return A, B


def _rank3_aux_dim_stride(
    batch_extent: int,
    m_extent: int,
    n_extent: int,
) -> tuple[list[int], list[int]]:
    dim = [batch_extent, m_extent, n_extent]
    stride = [m_extent * n_extent, n_extent, 1]
    return dim, stride


def test_matmul_only() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    chain = analyze(g)
    assert chain.matmul.M == 128 and chain.matmul.N == 128 and chain.matmul.K == 64
    assert chain.ops == []
    assert chain.aux_tensors == []
    assert chain.output_dtype == "bf16"
    assert chain.matmul.out_major == "n"


def test_batched_matmul_only() -> None:
    g = _mk_graph()
    A = g.tensor(name="A", dim=[3, 128, 64], stride=[128 * 64, 64, 1])
    B = g.tensor(name="B", dim=[3, 64, 256], stride=[64 * 256, 1, 64])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    chain = analyze(g)
    assert chain.matmul.batch == 3
    assert chain.matmul.M == 128 and chain.matmul.N == 256 and chain.matmul.K == 64
    assert chain.ops == []


def test_batched_matmul_broadcasts_a_batch() -> None:
    g = _mk_graph()
    A = g.tensor(name="A", dim=[1, 128, 64], stride=[128 * 64, 64, 1])
    B = g.tensor(name="B", dim=[3, 64, 256], stride=[64 * 256, 1, 64])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    chain = analyze(g)
    assert chain.matmul.batch == 3
    assert chain.matmul.a_batch == 1
    assert chain.matmul.b_batch == 3
    assert chain.ops == []


def test_batched_matmul_broadcasts_b_batch() -> None:
    g = _mk_graph()
    A = g.tensor(name="A", dim=[3, 128, 64], stride=[128 * 64, 64, 1])
    B = g.tensor(name="B", dim=[1, 64, 256], stride=[64 * 256, 1, 64])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    chain = analyze(g)
    assert chain.matmul.batch == 3
    assert chain.matmul.a_batch == 3
    assert chain.matmul.b_batch == 1
    assert chain.ops == []


@pytest.mark.parametrize(
    "a_major,a_stride,b_major,b_stride",
    (
        ("k", [128 * 64, 64, 1], "k", [64 * 256, 1, 64]),
        ("m", [128 * 64, 1, 128], "k", [64 * 256, 1, 64]),
        ("k", [128 * 64, 64, 1], "n", [64 * 256, 256, 1]),
        ("m", [128 * 64, 1, 128], "n", [64 * 256, 256, 1]),
    ),
    ids=("Ak_Bk", "Am_Bk", "Ak_Bn", "Am_Bn"),
)
def test_input_major_inference(
    a_major: str,
    a_stride: list[int],
    b_major: str,
    b_stride: list[int],
) -> None:
    g = _mk_graph()
    A = g.tensor(name="A", dim=[1, 128, 64], stride=a_stride)
    B = g.tensor(name="B", dim=[1, 64, 256], stride=b_stride)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    chain = analyze(g)
    assert chain.matmul.a_major == a_major
    assert chain.matmul.b_major == b_major


def test_rejects_unsupported_input_major_stride() -> None:
    g = _mk_graph()
    A = g.tensor(name="A", dim=[1, 128, 64], stride=[128 * 64, 2, 3])
    B = g.tensor(name="B", dim=[1, 64, 256], stride=[64 * 256, 1, 64])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    with pytest.raises(ValueError, match="A must be K-major or M-major"):
        analyze(g)


def test_rank2_input_rejected() -> None:
    """Rank-2 matmul operands are no longer supported — use leading dim=1 for un-batched."""
    g = _mk_graph()
    A = g.tensor(name="A", dim=[128, 64], stride=[64, 1])
    B = g.tensor(name="B", dim=[3, 64, 256], stride=[64 * 256, 1, 64])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    with pytest.raises(ValueError, match="must be 3D"):
        analyze(g)


def test_batched_matmul_epilogue_fusion() -> None:
    g = _mk_graph()
    A = g.tensor(name="A", dim=[2, 128, 64], stride=[128 * 64, 64, 1])
    B = g.tensor(name="B", dim=[2, 64, 128], stride=[64 * 128, 1, 64])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True)
    chain = analyze(g)
    assert chain.matmul.batch == 2
    assert [op.op for op in chain.ops] == ["relu"]


def test_epilogue_reduction_tap_from_pointwise_output() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="relu")
    Y.set_output(True)
    R = g.reduction(input=Y, mode=cudnn.reduction_mode.AMAX, name="amax")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    chain = analyze(g)
    assert [op.op for op in chain.ops] == ["relu"]
    assert len(chain.reductions) == 1
    red = chain.reductions[0]
    assert red.mode == "amax"
    assert red.source_ref == 0
    assert red.dim == (1, 1, 1)
    assert red.dtype == "fp32"
    assert [o.source for o in chain.outputs] == ["terminal", "reduction_0"]
    assert chain.outputs[1].is_reduction


def test_block_scale_quantize_terminal_with_scale_output() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    Q, S = g.block_scale_quantize(input=C, block_size=32, name="quant")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    S.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    chain = analyze(g)

    assert chain.output_dtype == "fp8_e4m3"
    assert chain.matmul.out_dtype == "fp32"
    assert chain.block_quant is not None
    assert chain.block_quant.source_ref == -1
    assert chain.block_quant.block_size == 32
    assert chain.block_quant.scale_dtype == "fp8_e8m0"
    assert chain.block_quant.scale_dim == (1, 128, 4)
    assert [o.source for o in chain.outputs] == ["terminal", "block_quant_scale"]
    assert chain.outputs[1].is_quant_scale
    assert chain.outputs[1].dtype == "fp8_e8m0"


def test_moe_block_scale_quantize_terminal_with_scale_output() -> None:
    g = _mk_graph()
    E, S, N, K = 4, 256, 256, 128
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1])
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K])
    fto = g.tensor(
        name="first_token_offset",
        dim=[E, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    C = g.moe_grouped_matmul(
        tok,
        w,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
    )
    Q, S_out = g.block_scale_quantize(input=C, block_size=32, name="quant")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    S_out.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    chain = analyze(g)

    assert chain.has_moe
    assert chain.output_dtype == "fp8_e4m3"
    assert chain.block_quant is not None
    assert chain.block_quant.source_ref == -1
    assert chain.block_quant.block_size == 32
    assert chain.block_quant.scale_dim == (1, S, N // 32)
    assert [o.source for o in chain.outputs] == ["terminal", "block_quant_scale"]


def test_block_scale_quantize_requires_materialized_scale_output() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    Q, _S = g.block_scale_quantize(input=C, block_size=32, name="quant")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)

    with pytest.raises(ValueError, match="scale output must be materialized"):
        analyze(g)


def test_block_scale_quantize_records_scale_reordering() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    Q, S = g.block_scale_quantize(input=C, block_size=32, name="quant")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    S.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    S.set_reordering_type(cudnn.tensor_reordering.F8_128x4)

    chain = analyze(g)

    assert chain.block_quant is not None
    assert chain.block_quant.scale_reorder == "F8_128x4"


def test_block_scale_quantize_accepts_f8_128x4_padded_scale_dim() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 144, 320, 64)
    C = g.matmul(A=A, B=B, name="mm")
    Q, S = g.block_scale_quantize(input=C, block_size=32, name="quant")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    S.set_dim([1, 256, 12]).set_stride([256 * 12, 12, 1])
    S.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    S.set_reordering_type(cudnn.tensor_reordering.F8_128x4)

    chain = analyze(g)

    assert chain.block_quant is not None
    assert chain.block_quant.scale_dim == (1, 256, 12)
    assert chain.outputs[1].dim == (1, 256, 12)


@pytest.mark.parametrize(
    "mode,expected",
    (
        (cudnn.reduction_mode.ADD, "add"),
        (cudnn.reduction_mode.AMAX, "amax"),
        (cudnn.reduction_mode.MAX, "max"),
        (cudnn.reduction_mode.MIN, "min"),
    ),
    ids=("add", "amax", "max", "min"),
)
def test_epilogue_reduction_modes(mode, expected: str) -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    R = g.reduction(input=C, mode=mode, name="red")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    chain = analyze(g)
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == expected


def test_epilogue_reduction_accepts_int32_compute_dtype() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.INT32)
    R = g.reduction(
        input=C,
        mode=cudnn.reduction_mode.ADD,
        name="red_i32",
        compute_data_type=cudnn.data_type.INT32,
    )
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.INT32)

    chain = analyze(g)
    assert chain.matmul.out_dtype == "int32"
    assert chain.output_dtype == "int32"
    assert len(chain.reductions) == 1
    assert chain.reductions[0].dtype == "int32"
    assert chain.reductions[0].compute_dtype == "int32"
    assert chain.outputs[-1].dtype == "int32"


def test_rejects_unsupported_reduction_mode() -> None:
    # Rejected in analyze(), not at g.reduction() time — the recorder stays
    # transparent so non-GEMM graphs with other reduction modes still build.
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    R = g.reduction(input=C, mode=cudnn.reduction_mode.AVG, name="avg")
    R.set_dim([1, 1, 1]).set_stride([1, 1, 1])
    R.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    with pytest.raises(NotImplementedError, match="supported modes"):
        analyze(g)


def test_batched_aux_bcast_inference_per_elem() -> None:
    g = _mk_graph()
    A = g.tensor(name="A", dim=[2, 128, 64], stride=[128 * 64, 64, 1])
    B = g.tensor(name="B", dim=[2, 64, 128], stride=[64 * 128, 1, 64])
    aux = g.tensor(name="aux", dim=[2, 128, 128], stride=[128 * 128, 128, 1])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.add(a=C, b=aux, name="add")
    Y.set_output(True)
    chain = analyze(g)
    assert chain.aux_tensors[0].bcast_mode == "per_elem"
    assert chain.aux_tensors[0].dim == (2, 128, 128)


def test_matmul_relu() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True)
    chain = analyze(g)
    assert [op.op for op in chain.ops] == ["relu"]
    assert chain.aux_tensors == []


def test_matmul_bias_gelu() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    bias = g.tensor(name="bias", dim=[1, 1, 128], stride=[128, 128, 1])
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="b")
    Y = g.gelu_approx_tanh(input=Cb, name="g")
    Y.set_output(True)
    chain = analyze(g)
    assert [op.op for op in chain.ops] == ["add", "gelu_tanh"]
    assert chain.ops[0].aux == "bias"
    assert chain.ops[0].aux_on_rhs is True
    assert len(chain.aux_tensors) == 1
    assert chain.aux_tensors[0].name == "bias"


def test_rejects_zero_matmul() -> None:
    g = _mk_graph()
    A = g.tensor(name="A", dim=[1, 128, 64], stride=[128 * 64, 64, 1])
    Y = g.relu(input=A, name="r")
    Y.set_output(True)
    with pytest.raises(ValueError, match="matmul per graph"):
        analyze(g)


def test_two_independent_matmuls_are_no_epilogue_multi_gemm() -> None:
    """Two same-shape matmuls, separate outputs, no fusion op = no-epilogue
    multi-GEMM. GEMM 0 = terminal, GEMMs >0 = taps."""
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    A2 = g.tensor(name="A2", dim=[1, 128, 64], stride=[128 * 64, 64, 1])
    B2 = g.tensor(name="B2", dim=[1, 64, 128], stride=[64 * 128, 1, 64])
    C1 = g.matmul(A=A, B=B, name="mm1")
    C2 = g.matmul(A=A2, B=B2, name="mm2")
    C1.set_output(True)
    C2.set_output(True)
    chain = analyze(g)
    assert chain.num_gemms == 2 and chain.ops == []
    assert chain.num_a_operands == 2 and chain.num_b_operands == 2
    assert chain.gemm_operands == [(0, 0), (1, 1)]
    assert chain.per_gemm_outputs == ["bf16", "bf16"]
    assert [o.source for o in chain.outputs] == ["terminal", "gemm_1"]


def test_two_matmuls_shared_epilogue_is_multi_gemm() -> None:
    """Two parallel GEMMs feeding one fused op = multi-GEMM (now supported)."""
    g = _mk_graph()
    A = g.tensor(name="A", dim=[1, 128, 128], stride=[128 * 128, 128, 1])
    B0 = g.tensor(name="B0", dim=[1, 128, 128], stride=[128 * 128, 1, 128])
    B1 = g.tensor(name="B1", dim=[1, 128, 128], stride=[128 * 128, 1, 128])
    C0 = g.matmul(A=A, B=B0, name="mm0")
    C1 = g.matmul(A=A, B=B1, name="mm1")
    Y = g.add(a=C0, b=C1, name="add")
    Y.set_output(True)
    chain = analyze(g)
    assert chain.num_gemms == 2
    assert chain.num_a_operands == 1 and chain.num_b_operands == 2
    assert chain.gemm_operands == [(0, 0), (0, 1)]


def test_dag_fan_out_from_matmul() -> None:
    """Matmul output fans out to two branches; the LAST set_output'd is terminal."""
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    Y1 = g.relu(input=C, name="r1")
    Y2 = g.gelu(input=C, name="g1")
    Y1.set_output(True)  # first → slot 1 tap
    Y2.set_output(True)  # last → slot 0 terminal
    chain = analyze(g)
    assert [op.op for op in chain.ops] == ["relu", "gelu"]
    assert chain.ops[0].parent_idx == -1  # relu reads matmul output
    assert chain.ops[1].parent_idx == -1  # gelu reads matmul output
    # gelu is terminal (not a tap), relu is a tap.
    assert chain.ops[1].output_tap is False
    assert chain.ops[0].output_tap is True
    outs = chain.outputs
    assert len(outs) == 2
    assert outs[0].source == "terminal"
    assert outs[1].source == "op_0"


def test_aux_bcast_inference_per_col() -> None:
    """bias dim [1, N] with matmul-out [1, M, N] => per_col broadcast."""
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 256, 64)
    bias = g.tensor(name="bias", dim=[1, 256], stride=[256, 1])
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="b")
    Cb.set_output(True)
    chain = analyze(g)
    assert chain.aux_tensors[0].bcast_mode == "per_col"


def test_rank1_aux_bcast_inference_per_col() -> None:
    """bias dim [N] broadcasts over M and maps to per_col."""
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 256, 64)
    bias = g.tensor(name="bias", dim=[256], stride=[1])
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="b")
    Cb.set_output(True)
    chain = analyze(g)
    assert chain.aux_tensors[0].bcast_mode == "per_col"
    assert chain.aux_tensors[0].dim == (256,)
    assert chain.aux_tensors[0].stride == (1,)


def test_batched_rank1_aux_bcast_inference_per_col() -> None:
    """bias dim [N] broadcasts over batch and M, and maps to per_col."""
    g = _mk_graph()
    A = g.tensor(name="A", dim=[2, 128, 64], stride=[128 * 64, 64, 1])
    B = g.tensor(name="B", dim=[2, 64, 256], stride=[64 * 256, 1, 64])
    bias = g.tensor(name="bias", dim=[256], stride=[1])
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="b")
    Cb.set_output(True)
    chain = analyze(g)
    assert chain.aux_tensors[0].bcast_mode == "per_col"
    assert chain.aux_tensors[0].dim == (256,)
    assert chain.aux_tensors[0].stride == (1,)


def test_rank1_aux_bcast_inference_scalar() -> None:
    """aux dim [1] broadcasts over M and N and maps to scalar."""
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 256, 64)
    scale = g.tensor(name="scale", dim=[1], stride=[1])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.add(a=C, b=scale, name="add")
    Y.set_output(True)
    chain = analyze(g)
    assert chain.aux_tensors[0].bcast_mode == "scalar"
    assert chain.aux_tensors[0].dim == (1,)
    assert chain.aux_tensors[0].stride == (1,)


def test_matmul_output_tap_recorded() -> None:
    """set_output+set_data_type on the matmul output surfaces as
    MatmulSpec.output_tap + out_dtype + a 2-slot chain.outputs."""
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    Y = g.relu(input=C, name="r")
    Y.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    chain = analyze(g)
    assert chain.matmul.output_tap is True
    assert chain.matmul.out_dtype == "bf16"
    assert chain.output_dtype == "fp32"
    outs = chain.outputs
    assert len(outs) == 2
    assert outs[0].source == "terminal" and outs[0].dtype == "fp32"
    assert outs[1].source == "matmul" and outs[1].dtype == "bf16"


def test_intermediate_op_tap_recorded() -> None:
    """Tapping a fusion-op output (after bias, before relu) attaches the tap
    dtype to that FusionOp and adds a chain.outputs slot."""
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    bias = g.tensor(name="bias", dim=[1, 1, 128], stride=[128, 128, 1])
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="b")
    Cb.set_output(True).set_data_type(cudnn.data_type.HALF)  # mid-chain tap → fp16
    Y = g.relu(input=Cb, name="r")
    Y.set_output(True)  # terminal
    chain = analyze(g)
    # first op (bias=add) tapped to fp16.
    assert [op.op for op in chain.ops] == ["add", "relu"]
    assert chain.ops[0].output_tap is True and chain.ops[0].out_dtype == "fp16"
    assert chain.ops[1].output_tap is False
    outs = chain.outputs
    assert len(outs) == 2
    assert outs[0].source == "terminal"
    assert outs[1].source == "op_0" and outs[1].dtype == "fp16"


def test_no_tap_when_matmul_is_terminal() -> None:
    """matmul.set_output(True) with no fusion ops = terminal, no tap emitted."""
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    chain = analyze(g)
    assert chain.matmul.output_tap is False
    assert len(chain.outputs) == 1
    assert chain.outputs[0].source == "terminal"


def test_analyze_raises_when_no_recording() -> None:
    """An unrecorded graph gives a clear error (synthesize the missing-state case)."""
    from cudnn.frost.gemm.graph_analyzer import _GRAPH_STATES

    g = _mk_graph()
    _GRAPH_STATES.pop(g, None)
    with pytest.raises(ValueError, match="no recorded ops"):
        analyze(g)


# Compute / output dtype semantics (input -> compute -> output, per op)


def test_rejects_non_fp32_matmul_graph_compute_dtype() -> None:
    """Matmul accumulation still requires fp32 compute."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.BFLOAT16,  # unsupported
    )
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    R = g.relu(input=C, name="r")
    R.set_output(True)
    with pytest.raises(ValueError, match="compute_data_type .* not.*supported"):
        analyze(g)


def test_epilogue_op_accepts_int32_compute_dtype() -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    aux = g.tensor(
        name="aux",
        dim=[1, 128],
        stride=[128, 1],
        data_type=cudnn.data_type.INT32,
    )
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.add(
        a=C,
        b=aux,
        name="add_i32",
        compute_data_type=cudnn.data_type.INT32,
    )
    Y.set_output(True).set_data_type(cudnn.data_type.INT32)
    chain = analyze(g)
    assert chain.ops[0].compute_dtype == "int32"
    assert chain.aux_tensors[0].dtype == "int32"
    assert chain.output_dtype == "int32"


def test_virtual_intermediate_dtype_is_honored() -> None:
    """A pure-virtual intermediate still rounds to its dtype: bf16
    intermediate_data_type rounds C and the mid op at each step."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.BFLOAT16,  # narrow virtuals
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")  # virtual → bf16
    R = g.relu(input=C, name="r")  # virtual → bf16
    Y = g.gelu_approx_tanh(input=R, name="g")  # terminal
    Y.set_output(True)
    chain = analyze(g)
    assert chain.matmul.out_dtype == "bf16"  # C rounded before relu
    assert chain.ops[0].out_dtype == "bf16"  # relu result rounded before gelu
    assert chain.ops[0].compute_dtype == "fp32"
    # Terminal: no mid-round (the trailing vec_out cast handles output_dtype).
    assert chain.ops[1].out_dtype == "fp32"
    assert chain.output_dtype == "bf16"


def test_explicit_set_data_type_on_virtual_intermediate() -> None:
    """A set_data_type on a non-tapped (virtual) intermediate still forces a
    round, with a FLOAT graph default elsewhere (so only that tensor rounds)."""
    g = _mk_graph()  # intermediate_data_type=FLOAT
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")  # virtual → fp32 (no round)
    R = g.relu(input=C, name="r")
    R.set_data_type(cudnn.data_type.BFLOAT16)  # virtual but explicitly bf16
    Y = g.gelu_approx_tanh(input=R, name="g")
    Y.set_output(True)
    chain = analyze(g)
    assert chain.matmul.out_dtype == "fp32"  # C stays fp32
    assert chain.ops[0].out_dtype == "bf16"  # relu output rounded to bf16
    # R was not set_output -> it's not a materialized tap, just a rounded virtual.
    assert [(o.source, o.dtype) for o in chain.taps] == []


@pytest.mark.parametrize(
    "cudnn_dtype,expected",
    (
        (cudnn.data_type.FP8_E8M0, "fp8_e8m0"),
        (cudnn.data_type.INT8, "int8"),
        (cudnn.data_type.UINT8, "uint8"),
        (cudnn.data_type.INT32, "int32"),
    ),
)
def test_aux_dtype_mapping_for_epilogue_inputs(cudnn_dtype, expected: str) -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    aux = g.tensor(
        name="aux",
        dim=[1, 1],
        stride=[1, 1],
        data_type=cudnn_dtype,
    )
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.add(a=C, b=aux, name="add")
    Y.set_output(True)
    chain = analyze(g)
    assert chain.aux_tensors[0].dtype == expected


@pytest.mark.parametrize(
    "cudnn_dtype,expected",
    (
        (cudnn.data_type.FP8_E8M0, "fp8_e8m0"),
        (cudnn.data_type.INT8, "int8"),
        (cudnn.data_type.UINT8, "uint8"),
        (cudnn.data_type.INT32, "int32"),
    ),
)
def test_output_dtype_mapping_for_epilogue_outputs(cudnn_dtype, expected: str) -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True).set_data_type(cudnn_dtype)
    chain = analyze(g)
    assert chain.output_dtype == expected


@pytest.mark.parametrize(
    "cudnn_name,ir_name",
    (
        ("ceil", "ceil"),
        ("floor", "floor"),
        ("erf", "erf"),
        ("log", "log"),
        ("reciprocal", "reciprocal"),
        ("rsqrt", "rsqrt"),
        ("sqrt", "sqrt"),
    ),
)
def test_analyzer_records_additional_unary_epilogue_ops(
    cudnn_name: str,
    ir_name: str,
) -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    C = g.matmul(A=A, B=B, name="mm")
    Y = getattr(g, cudnn_name)(input=C, name=cudnn_name)
    Y.set_output(True)
    chain = analyze(g)
    assert [op.op for op in chain.ops] == [ir_name]


@pytest.mark.parametrize(
    "cudnn_name,ir_name",
    (
        ("max", "max"),
        ("min", "min"),
        ("pow", "pow"),
        ("add_square", "add_square"),
    ),
)
def test_analyzer_records_additional_binary_epilogue_ops(
    cudnn_name: str,
    ir_name: str,
) -> None:
    g = _mk_graph()
    A, B = _mk_inputs(g, 128, 128, 64)
    aux = g.tensor(name="aux", dim=[1, 128], stride=[128, 1])
    C = g.matmul(A=A, B=B, name="mm")
    if cudnn_name in {"max", "min", "pow"}:
        Y = getattr(g, cudnn_name)(input0=C, input1=aux, name=cudnn_name)
    else:
        Y = getattr(g, cudnn_name)(a=C, b=aux, name=cudnn_name)
    Y.set_output(True)
    chain = analyze(g)
    assert chain.ops[0].op == ir_name
    assert chain.ops[0].aux == "aux"
    assert chain.ops[0].aux_on_rhs is True


@pytest.mark.parametrize(
    "batch_extent,m_extent,n_extent,expected_bcast",
    (
        (1, 1, 1, "scalar"),
        (2, 1, 1, "scalar"),
        (1, 128, 1, "per_row"),
        (2, 128, 1, "per_row"),
        (1, 1, 128, "per_col"),
        (2, 1, 128, "per_col"),
        (1, 128, 128, "per_elem"),
        (2, 128, 128, "per_elem"),
    ),
)
@pytest.mark.parametrize("aux_on_rhs", (True, False), ids=("rhs", "lhs"))
def test_rank3_aux_supports_all_batch_m_n_broadcast_patterns(
    aux_on_rhs: bool,
    batch_extent: int,
    m_extent: int,
    n_extent: int,
    expected_bcast: str,
) -> None:
    g = _mk_graph()
    batch, M, N, K = 2, 128, 128, 64
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K])
    aux_dim, aux_stride = _rank3_aux_dim_stride(batch_extent, m_extent, n_extent)
    aux = g.tensor(name="aux", dim=aux_dim, stride=aux_stride)
    C = g.matmul(A=A, B=B, name="mm")
    if aux_on_rhs:
        Y = g.sub(a=C, b=aux, name="sub")
    else:
        Y = g.sub(a=aux, b=C, name="sub")
    Y.set_output(True)

    chain = analyze(g)
    assert chain.aux_tensors[0].dim == tuple(aux_dim)
    assert chain.aux_tensors[0].stride == tuple(aux_stride)
    assert chain.aux_tensors[0].bcast_mode == expected_bcast
    assert chain.ops[0].aux_on_rhs is aux_on_rhs
