"""Tiny IR describing a matmul + linear pointwise-epilogue fusion.

This is the only contract between `graph_analyzer.py` (which builds it from a
cuDNN frontend graph) and `epilogue_codegen.py` (which lowers it into a cute DSL
snippet that lives inside the generated kernel's epilogue warps).

POC scope: one matmul + a DAG of pointwise ops with both fan-out and fan-in.
Each binary op has one of two shapes:
  - ``aux=<name>`` + ``parent_idx`` set → one in-chain operand + one
    graph-input aux tensor (the original form).
  - ``parent_idx_b=<i>`` + ``parent_idx`` set, ``aux=None`` → two in-chain
    operands (fan-in: both inputs are prior op results / matmul output).
``aux`` and ``parent_idx_b`` are mutually exclusive.

Multi-output: the **terminal** of the chain is always materialized; any
intermediate point (the raw matmul output and / or any fusion-op output) can
be tapped as an extra GMEM output by calling
``.set_output(True).set_data_type(<tap_dtype>)`` on the cuDNN graph side.
Pointwise taps are full `(batch, M, N)` rank-3 tensors; reduction taps keep
rank 3 with the reduced dimensions set to 1; block-quant scale taps use
`(batch, M, ceil_div(N, block_size))`. ``FusionChain.outputs`` lists
every materialized slot in canonical order (terminal first, then taps in chain
order); callers of ``CompiledFusedGemm`` pass runtime output tensors in this
same order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Dtype sets are split by role:
#   * SUPPORTED_DTYPES: epilogue aux/output storage dtypes.
#   * COMPUTE_DTYPES: pointwise compute dtypes accepted from cuDNN graph attrs.
# ---------------------------------------------------------------------------

Dtype = Literal[
    "bf16",
    "fp16",
    "fp32",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_e8m0",
    "fp4_e2m1",
    "int8",
    "uint8",
    "int32",
    "int64",
]
SUPPORTED_DTYPES: tuple[Dtype, ...] = (
    "bf16",
    "fp16",
    "fp32",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_e8m0",
    "fp4_e2m1",
    "int8",
    "uint8",
    "int32",
)
COMPUTE_DTYPES: tuple[Dtype, ...] = ("fp32", "int32")
AMajor = Literal["k", "m"]
BMajor = Literal["k", "n"]
OutMajor = Literal["n", "m"]

# Block-scaled-matmul scale-factor dtypes (the descale tensors SFA/SFB):
#   * nvfp4  : FP4 (e2m1) data, FP8 E4M3 scale, block_size 16
#   * mxfp4  : FP4 (e2m1) data, FP8 E8M0 scale, block_size 32
#   * mxfp8  : FP8 (e4m3/e5m2) data, FP8 E8M0 scale, block_size 32
BLOCK_SCALE_SF_DTYPES: tuple[Dtype, ...] = ("fp8_e4m3", "fp8_e8m0")

# How an aux tensor broadcasts onto the (M, N) output tile.
#   scalar    — single value
#   per_row   — vector of length M (broadcast across N)
#   per_col   — vector of length N (broadcast across M)
#   per_elem  — full (M, N) matrix
BroadcastMode = Literal["scalar", "per_row", "per_col", "per_elem"]
ReductionMode = Literal["add", "amax", "max", "min"]
REDUCTION_DTYPES: tuple[Dtype, ...] = ("fp32", "int32")

# Unary pointwise ops: take only the running accumulator.
UNARY_OPS: tuple[str, ...] = (
    "identity",
    "relu",
    "gelu",  # exact gelu via erf
    "gelu_tanh",  # gelu approx-tanh
    "swish",
    "sigmoid",
    "tanh",
    "exp",
    "abs",
    "neg",
    "cos",
    "sin",
    "ceil",
    "floor",
    "erf",
    "log",
    "reciprocal",
    "rsqrt",
    "sqrt",
)

# Binary pointwise ops: take running accumulator + one aux tensor.
# `aux_on_rhs=True` means op(acc, aux); False means op(aux, acc) — matters for
# non-commutative ops (sub, div).
BINARY_OPS: tuple[str, ...] = (
    "add",
    "mul",
    "sub",
    "div",
    "max",
    "min",
    "pow",
    "add_square",
)

ALL_OPS: tuple[str, ...] = UNARY_OPS + BINARY_OPS

# Ops that map 0 -> 0 regardless of any other operand, i.e. the output is 0
# whenever the *in-chain* input is 0, for ALL possible aux values. This is the
# single source of truth for the mainloop K-OOB fix: when a fused operand's
# whole op chain is zero-preserving, the TMA K-tail zero-fill stays 0 through
# the transform, so it contributes 0 to the MMA and no OOB masking is needed.
#
# An op qualifies ONLY if `f(0, *) == 0` unconditionally:
#   unary : identity/relu/abs/neg/tanh/sin/gelu/gelu_tanh/swish plus
#           ceil/floor/erf/sqrt  (f(0)=0)
#   binary: mul only  (0*aux == aux*0 == 0)
# Everything else is NON-zero-preserving (sigmoid/exp/cos: f(0)!=0; add/sub:
# 0+aux=aux; div: aux/0=inf so it's not unconditionally safe).
#
# IMPORTANT: a newly-added pointwise op defaults to NON-zero-preserving unless
# it is added here. That default is the safe one — forgetting to register a
# genuinely zero-preserving op only costs an unnecessary (still correct) OOB
# mask; the reverse (wrongly assuming zero-preserving) would silently miscompute.
ZERO_PRESERVING_OPS: frozenset[str] = frozenset(
    {
        "identity",
        "relu",
        "gelu",
        "gelu_tanh",
        "swish",
        "tanh",
        "abs",
        "neg",
        "sin",
        "ceil",
        "floor",
        "erf",
        "sqrt",
        "mul",
    }
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainOutput:
    """One materialized GMEM output produced by the kernel.

    ``source`` describes which point in the chain the value is taken from:
      - ``"matmul"``    — fp32 accumulator before any fusion op
      - ``"op_<i>"``    — the i'th fusion op's result (0-indexed)
      - ``"terminal"``  — final value after the full epilogue chain (always present)
      - ``"reduction_<i>"`` — the i'th epilogue reduction output
      - ``"block_quant_scale"`` — scale side-output from block-scale quantize

    ``dtype`` is the on-disk dtype: each output gets its own cast from the
    running fp32 value to its declared dtype.
    """

    source: str  # "matmul" | "terminal" | "op_<i>" | "reduction_<i>"
    dtype: Dtype
    dim: "tuple[int, int, int] | None" = None
    is_reduction: bool = False
    is_quant_scale: bool = False
    quant_block_size: int | None = None


@dataclass(frozen=True)
class ReductionSpec:
    """A materialized reduction side-output from the epilogue.

    ``source_ref`` names the producer being reduced using the same reference
    scheme as :class:`FusionOp` parents: a GEMM output is negative, a pointwise
    op output is non-negative. ``dim`` is the public `(B, M, N)` output shape;
    each dimension is either the full matmul extent or 1, and dimensions with
    1 are reduced.
    """

    mode: ReductionMode
    source_ref: int
    dim: tuple[int, int, int]
    dtype: Dtype = "fp32"
    compute_dtype: Dtype = "fp32"
    grouped_by_moe: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("add", "amax", "max", "min"):
            raise ValueError(f"unsupported reduction mode {self.mode!r}")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported reduction output dtype {self.dtype!r}; " f"expected one of {SUPPORTED_DTYPES}")
        if self.dtype not in REDUCTION_DTYPES:
            raise ValueError(f"reduction output dtype {self.dtype!r} is not supported; " f"expected one of {REDUCTION_DTYPES}")
        if self.compute_dtype not in REDUCTION_DTYPES:
            raise ValueError(f"reduction compute_dtype {self.compute_dtype!r} is not supported; " f"expected one of {REDUCTION_DTYPES}")
        if self.dtype != self.compute_dtype:
            raise ValueError(f"reduction output dtype {self.dtype!r} must match compute_dtype " f"{self.compute_dtype!r} for direct atomic reduction")


@dataclass(frozen=True)
class BlockQuantizeSpec:
    """Terminal block-scale quantize in the epilogue.

    This models cuDNN ``block_scale_quantize(input, block_size, axis)`` for the
    row/N-axis case used by GEMM epilogues: every contiguous N-block produces
    one scale value and one block of quantized output elements. The quantized
    output is the terminal chain output; ``scale_dim`` describes the scale
    side-output in public `(B, scale_M, scale_N)` order. For compact scale
    output this is `(B, M, N / block_size)`. F8_128x4 scale output may be
    padded to the layout's 128-row x 4-column block.
    """

    source_ref: int
    block_size: int
    axis: int = -1
    transpose: bool = False
    scale_dtype: Dtype = "fp8_e8m0"
    scale_dim: tuple[int, int, int] | None = None
    scale_reorder: str | None = None
    compute_dtype: Dtype = "fp32"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block quantize block_size must be positive; got {self.block_size}")
        if self.axis not in (-1, 2):
            raise NotImplementedError(f"block quantize supports only the last/N axis in cudnn.TBD.gemm; got axis={self.axis}")
        if self.transpose:
            raise NotImplementedError("block quantize transpose=True is not supported")
        if self.scale_dtype not in ("fp8_e8m0", "fp8_e4m3"):
            raise ValueError(f"block quantize scale dtype {self.scale_dtype!r} is not supported; " "expected fp8_e8m0 or fp8_e4m3")
        if self.scale_reorder not in (None, "F8_128x4"):
            raise ValueError(f"block quantize scale reordering {self.scale_reorder!r} is not supported; " "expected None or F8_128x4")
        if self.compute_dtype != "fp32":
            raise ValueError(f"block quantize compute_dtype {self.compute_dtype!r} is not supported; " "expected fp32")


@dataclass(frozen=True)
class TensorRef:
    """An auxiliary tensor input to the epilogue (e.g., bias, alpha, scale)."""

    name: str
    dim: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Dtype
    bcast_mode: BroadcastMode

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported aux dtype {self.dtype!r}; expected one of {SUPPORTED_DTYPES}")
        if len(self.dim) != len(self.stride):
            raise ValueError(f"dim {self.dim} and stride {self.stride} length mismatch")


# ---------------------------------------------------------------------------
# Producing-operation references (the "where does this op's input come from?"
# encoding). A pointwise op's input is a reference to the operation that
# produced it — uniformly, whether that operation is a GEMM (an MMA) or a prior
# pointwise op. The op reads that operation's output register. This mirrors
# cuDNN, which tracks each pointwise input back to its producing op rather than
# trying to attribute a (possibly already-mixed) value to one GEMM.
#
# A reference is a single int (``FusionOp.parent_idx`` / ``parent_idx_b``):
#   * ``>= 0``  → a pointwise op result, ``ops[ref]``.
#   * ``< 0``   → a GEMM output; the GEMM index is ``gemm_index(ref) = -1 - ref``
#                 (so -1 = GEMM 0, -2 = GEMM 1, ...). For a single-GEMM chain
#                 the only GEMM ref is -1, identical to the legacy "matmul
#                 output" sentinel.
#   * ``None``  → (parent_idx only) "auto": the previous op, or GEMM 0 for op 0.
#                 (parent_idx_b only) → no second in-chain operand.
# This is why there is no separate "which GEMM" field: a reference names exactly
# one producing operation, GEMM or pointwise. Once two GEMMs are merged by an
# op, downstream inputs reference that op (>=0) — they neither can nor need to
# attribute the value back to an individual GEMM.


def gemm_source(g: int) -> int:
    """Encode 'this input is GEMM ``g``'s output' as a parent reference."""
    return -1 - g


def is_gemm_source(ref: int) -> bool:
    """True iff a parent reference names a GEMM output (vs a pointwise op)."""
    return ref < 0


def gemm_index(ref: int) -> int:
    """The GEMM index named by a (gemm) parent reference (inverse of gemm_source)."""
    return -1 - ref


@dataclass(frozen=True)
class FusionOp:
    """A single pointwise op in the epilogue chain.

    For unary ops, `aux` is None. For binary ops, `aux` references one of the
    `aux_tensors` in the enclosing FusionChain by name.

    ``output_tap``, if True, materializes this op's result to a separate GMEM
    buffer (at the op's ``out_dtype``) *before* the next op runs.

    ``parent_idx`` / ``parent_idx_b`` are **producing-operation references** (see
    the module-level note above): ``>= 0`` names a prior pointwise op
    (``ops[ref]``); ``< 0`` names a GEMM output (``gemm_index(ref)``); ``None``
    means "auto" for ``parent_idx`` (previous op, or GEMM 0 for op 0) and "no
    second operand" for ``parent_idx_b``. Fan-out = several ops sharing one
    parent ref; fan-in = a binary op whose ``parent_idx`` AND ``parent_idx_b``
    are both in-chain (``aux`` must be None then — they are mutually exclusive).
    ``aux_on_rhs`` controls which side the second operand sits on.
    """

    op: str
    aux: str | None = None  # TensorRef.name, or None for unary / fan-in
    aux_on_rhs: bool = True  # for binary ops; True = op(parent_idx, other)
    output_tap: bool = False  # materialize this op's result (at out_dtype)
    parent_idx: int | None = None  # producing-op ref (None = auto)
    parent_idx_b: int | None = None  # fan-in: second producing-op ref (None = absent)
    # ``compute_dtype`` is the precision the op's math runs in (cuDNN's per-op
    # ``compute_data_type``). POC accepts fp32 and int32 as requested, but does
    # not enforce a per-op semantic matrix: transcendental ops such as exp,
    # log, sqrt, rsqrt, sigmoid, gelu, and swish should use fp32 unless the
    # caller intentionally wants GEMM/codegen-defined integer behavior.
    compute_dtype: Dtype = "fp32"
    # ``out_dtype`` is the declared data_type of this op's output (virtual or
    # materialized) tensor. The running value is rounded to this dtype before
    # the next op consumes it — so a narrow ``out_dtype`` loses precision on
    # purpose, matching cuDNN's tensor-dtype semantics even for virtual
    # tensors. ``fp32`` (default) means "no rounding" (the legacy behavior).
    out_dtype: Dtype = "fp32"

    def __post_init__(self) -> None:
        if self.op not in ALL_OPS:
            raise ValueError(f"unknown op {self.op!r}; expected one of {ALL_OPS}")
        if self.compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(f"op {self.op!r}: compute_data_type {self.compute_dtype!r} is not " f"supported — expected one of {COMPUTE_DTYPES}.")
        if self.out_dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"op {self.op!r}: unsupported out_dtype {self.out_dtype!r}; " f"expected one of {SUPPORTED_DTYPES}")
        if self.op in UNARY_OPS and self.aux is not None:
            raise ValueError(f"unary op {self.op!r} cannot have aux input")
        if self.op in UNARY_OPS and self.parent_idx_b is not None:
            raise ValueError(f"unary op {self.op!r} cannot have parent_idx_b")
        if self.op in BINARY_OPS and self.aux is None and self.parent_idx_b is None:
            raise ValueError(f"binary op {self.op!r} requires either an aux tensor name or " f"a parent_idx_b pointing at another producing operation")
        if self.aux is not None and self.parent_idx_b is not None:
            raise ValueError(f"binary op {self.op!r}: aux and parent_idx_b are mutually exclusive")

    def resolved_parent_idx(self, position: int) -> int:
        """Resolve the ``None`` ("auto") ``parent_idx`` to a concrete producing-op
        reference given this op's position in ``ops``: the previous op
        (``position - 1``) for ``position > 0``, else GEMM 0 (``gemm_source(0)``,
        i.e. -1). A non-None ``parent_idx`` is returned as-is."""
        if self.parent_idx is not None:
            return self.parent_idx
        return position - 1 if position > 0 else gemm_source(0)


@dataclass(frozen=True)
class MatmulSpec:
    """Shape + dtype of the anchor matmul. All matmuls are rank-3:
    A=(batch_A, M, K), B=(batch_B, K, N), out=(batch, M, N), where
    batch=max(batch_A, batch_B) and each operand batch is either the
    output batch or 1 (broadcast).

    ``output_tap``, if True, requests an extra GMEM output that materializes the
    matmul result (the accumulator rounded to ``out_dtype``) *before* any fusion
    op runs. The compiler emits a parallel STG path in the epilogue."""

    M: int
    N: int
    K: int
    batch: int = 1
    a_batch: int = 1
    b_batch: int = 1
    a_major: AMajor = "k"
    b_major: BMajor = "k"
    a_dtype: Dtype = "bf16"
    b_dtype: Dtype = "bf16"
    accum_dtype: Dtype = "fp32"
    output_tap: bool = False  # materialize the matmul result (at out_dtype)
    out_dtype: Dtype = "bf16"
    out_major: OutMajor = "n"

    def __post_init__(self) -> None:
        if self.a_major not in ("k", "m"):
            raise ValueError(f"a_major must be 'k' or 'm' (got {self.a_major!r})")
        if self.b_major not in ("k", "n"):
            raise ValueError(f"b_major must be 'k' or 'n' (got {self.b_major!r})")
        if self.out_major not in ("n", "m"):
            raise ValueError(f"out_major must be 'n' or 'm' (got {self.out_major!r})")
        if self.batch < 1 or self.a_batch < 1 or self.b_batch < 1:
            raise ValueError("batch dimensions must be positive")
        if self.a_batch not in (1, self.batch) or self.b_batch not in (1, self.batch):
            raise ValueError(
                f"matmul batches must either match output batch={self.batch} "
                f"or be broadcast batch=1; got a_batch={self.a_batch}, "
                f"b_batch={self.b_batch}"
            )
        if max(self.a_batch, self.b_batch) != self.batch:
            raise ValueError(f"output batch must be max(a_batch, b_batch); got " f"batch={self.batch}, a_batch={self.a_batch}, " f"b_batch={self.b_batch}")
        # NOTE: which (a_dtype, b_dtype, accum_dtype) MMA combinations are
        # actually runnable is NOT validated here — it depends on the target GPU
        # architecture and the kernel pipeline, neither of which the IR knows.
        # That check lives in the compiler (`_check_supported` against the
        # arch-aware `_PIPELINE_DTYPE_ARCH` table), the single source of truth.
        # The IR only validates structural invariants (above) and output-storage
        # dtype well-formedness (below), which the combo table doesn't cover.
        if self.out_dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported matmul out_dtype {self.out_dtype!r}; " f"expected one of {SUPPORTED_DTYPES}")


@dataclass(frozen=True)
class BlockScaleSpec:
    """A block-scaled matmul: ``C = (descale_a ⊙ A) @ (descale_b ⊙ B)`` where A
    and B are narrow (FP4 / FP8) and each is dequantized by a per-block scale
    factor along K *inside* the MMA (``tcgen05.mma.kind::mx*.block_scale``).

    Detected by the analyzer (``graph_analyzer``) by a purely STRUCTURAL match:
    if the matmul's A and/or B operand is the output of a
    ``block_scale_dequantize`` node, the dequant(s) + matmul are folded into one
    block-scale matmul. Three shapes are matched — ``dequant(A) @ B``,
    ``A @ dequant(B)``, ``dequant(A) @ dequant(B)`` — so ``sfa`` / ``sfb`` are
    each present only for the side(s) that were dequantized (``None`` otherwise).
    The analyzer applies NO dtype/block-size/arch rules; which combinations are
    actually runnable is decided at compile time.

    Combinations the compiler currently runs (both sides scaled):
      * nvfp4 : data=fp4_e2m1, scale=fp8_e4m3, block_size=16
      * mxfp4 : data=fp4_e2m1, scale=fp8_e8m0, block_size=32
      * mxfp8 : data=fp8_e4m3/e5m2, scale=fp8_e8m0, block_size=32

    Like the A/B data operands (described by :class:`MatmulSpec` + passed
    positionally), the SF tensors are NOT stored as ``TensorRef``s — they're
    runtime-positional (``compiled(a, b, c, (M,N,K), sfa, sfb)``) and fully
    described here by per-side scalars: ``sf_dtype_*`` (dtype), ``block_size_*``
    (block dims), ``sfa/sfb_reorder`` (layout). Their logical dims are derivable
    from M/N/K/block_size, and the kernel's SF TMA descriptors encode the
    blocked layout directly. SFA / SFB are passed at runtime in the
    ``CUDNN_TENSOR_REORDERING_F8_128x4`` swizzled layout (128-row × 4-K blocked)
    — see the example's ``to_blocked`` helper.
    """

    a_dtype: Dtype  # packed data dtype of A (mirror of matmul.a_dtype)
    b_dtype: Dtype  # packed data dtype of B
    # ---- Per-side scale info (A and B kept SEPARATE so an asymmetric block-
    # scale matmul — different block size / SF dtype / layout per operand —
    # can be represented in future). A side that was NOT block-scale-dequantized
    # has all of its fields None.
    # `block_size_*` is stored 2D (A=[non_K, K], B=[K, non_K]). Only 1D scaling
    # along K is supported today; the 2D form leaves room for 2D block scaling.
    block_size_a: "tuple[int, ...] | None" = None
    block_size_b: "tuple[int, ...] | None" = None
    sf_dtype_a: "Dtype | None" = None  # A's scale-factor dtype (None if A not scaled)
    sf_dtype_b: "Dtype | None" = None
    # SF reorder layout per side (cuDNN reordering name, e.g. "F8_128x4"); None
    # = default/NONE. Recorded for the compile-stage support check.
    sfa_reorder: "str | None" = None
    sfb_reorder: "str | None" = None
    # Each block_scale_dequantize op's compute (math) dtype and output dtype.
    # The dequant OUTPUT dtype is the MMA's logical input type for that operand.
    # None for a side that wasn't dequantized. Recorded for the compile-stage
    # support check (cuDNN requires the dequant math precision to be FLOAT).
    dequant_compute_a: "Dtype | None" = None
    dequant_compute_b: "Dtype | None" = None
    dequant_out_a: "Dtype | None" = None  # = MMA input type for A
    dequant_out_b: "Dtype | None" = None  # = MMA input type for B
    # NOTE: matmul-level dtypes (accumulate, output, output-tap) are NOT
    # duplicated here — read them from chain.matmul / chain.output_dtype. Only
    # input-side info (packed a/b dtype, SF, block, reorder, dequant) lives here,
    # since block-scale has both a real packed dtype and a virtual dequant-output
    # dtype that a single shared field couldn't disambiguate.

    def __post_init__(self) -> None:
        # Structural sanity ONLY. No dtype / block-size / family / arch
        # runnability rules here — the analyzer pattern-matches structurally and
        # the compile stage decides which combos actually run.
        if self.sf_dtype_a is None and self.sf_dtype_b is None:
            raise ValueError("block_scale spec has neither operand dequantized " "(at least one of A / B must be block-scaled)")

    @property
    def both_sided(self) -> bool:
        """True iff BOTH operands were dequantized (the runnable case today).
        A side is dequantized iff its scale-factor dtype is set."""
        return self.sf_dtype_a is not None and self.sf_dtype_b is not None

    # --- Convenience scalars for the symmetric both-sided path the compiler
    # currently runs. (For a future asymmetric matmul these collapse the per-
    # side info to a single value — only valid when the two sides agree, which
    # the compile-stage support check enforces.)
    @property
    def block_size(self) -> int:
        """K-block size as a single int. A is [non_K, K] (K = last dim); B is
        [K, non_K] (K = first dim)."""
        if self.block_size_a is not None:
            return int(self.block_size_a[-1])
        if self.block_size_b is not None:
            return int(self.block_size_b[0])
        return 1

    @property
    def sf_dtype(self) -> "Dtype | None":
        return self.sf_dtype_a if self.sf_dtype_a is not None else self.sf_dtype_b

    @property
    def combo(self) -> str:
        """One of 'nvfp4', 'mxfp4', 'mxfp8' — the dtype/block family."""
        if self.a_dtype == "fp4_e2m1":
            return "nvfp4" if self.sf_dtype == "fp8_e4m3" else "mxfp4"
        return "mxfp8"

    @property
    def is_fp4(self) -> bool:
        return self.a_dtype == "fp4_e2m1"

    @property
    def mma_block_scale_kind(self) -> str:
        """GEMM ``nvvm.MMABlockScaleKind`` member name."""
        return "MXF4NVF4" if self.is_fp4 else "MXF8F6F4"

    @property
    def scale_vec_size(self) -> str:
        """GEMM ``nvvm.Tcgen05MMABlockScale`` member name. block16→X4 scale-vec,
        block32→X2 (the MMA reads 1 SF per ``block_size`` K-elements)."""
        return "BLOCK16" if self.block_size == 16 else "BLOCK32"

    @property
    def sf_scale_format(self) -> int:
        """``Tcgen05MxInstrDesc`` ``scale_format`` field: 0 for E4M3 scale
        (nvfp4), 1 for E8M0 scale (mx)."""
        return 0 if self.sf_dtype == "fp8_e4m3" else 1


@dataclass(frozen=True)
class MoeSpec:
    """A MoE grouped matmul forward pass.

    ``out[first_token_offset[g] : first_token_offset[g+1]] =
        token[that range] @ weight[g % num_experts].T`` for each routed group g.

    It is a batched matmul where every expert owns a variable, runtime-determined
    M-range of the flat token tensor (the boundaries live in an INT32
    ``first_token_offset`` tensor passed at runtime, NOT baked here). The anchor
    :class:`MatmulSpec` carries the dims: ``M`` = total tokens (S), ``K`` = hidden
    size (H), ``N`` = weight size; ``a_batch=1`` (one flat token plane),
    ``b_batch=num_experts`` (the weight tensor is indexed per group).

    Set by the analyzer when the graph's single op is ``moe_grouped_matmul``.
    When non-None the compiler routes to the ``sm100_moe_grouped_matmul_fwd_*``
    template, which runs a grouped persistent scheduler + per-group A TMA
    descriptor replacement. POC scope: ``mode == "none"`` only (gather / scatter
    are rejected at analysis time)."""

    num_experts: int  # E — also the number of routed groups
    mode: str = "none"  # "none" only in the POC
    # dtype of the first_token_offset tensor — INT32 or INT64 (cuDNN accepts
    # both). Baked into the kernel at JIT time; the scheduler casts reads to
    # Int32 internally (token counts fit in i32), so the math is dtype-agnostic.
    offset_dtype: Dtype = "int32"

    def __post_init__(self) -> None:
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be positive; got {self.num_experts}")
        if self.mode != "none":
            raise ValueError(f"MoE grouped matmul mode {self.mode!r} is out of POC scope; " "only 'none' is supported (gather / scatter rejected)")
        if self.offset_dtype not in ("int32", "int64"):
            raise ValueError(f"first_token_offset dtype must be int32 or int64; " f"got {self.offset_dtype!r}")


@dataclass
class FusionChain:
    """Full description of a matmul + pointwise-epilogue fusion (linear chain
    or fan-out DAG)."""

    matmul: MatmulSpec
    aux_tensors: list[TensorRef] = field(default_factory=list)
    ops: list[FusionOp] = field(default_factory=list)
    output_dtype: Dtype = "bf16"
    # ---- Multi-GEMM (parallel matmuls sharing one epilogue) ----------------
    # K parallel GEMMs (K = num_gemms) all share the SAME shape / layout / dtype
    # (described by ``matmul``) but may use different A / B operands, and their
    # outputs feed the same pointwise-epilogue DAG (e.g. silu(A@B0)*(A@B1)).
    # Operands are deduplicated: ``num_a_operands`` / ``num_b_operands`` are the
    # counts of DISTINCT A / B tensors; ``gemm_operands[g] = (a_idx, b_idx)``
    # picks GEMM g's operands from those distinct pools. A pointwise op names
    # which GEMM output it reads via a negative ``parent_idx`` (``gemm_source(g)``
    # = -1 - g); see the producing-operation-reference note on FusionOp.
    # Single-GEMM default (1 distinct A, 1 distinct B, one (0,0) GEMM) is the
    # legacy path (the only GEMM ref is -1).
    num_a_operands: int = 1
    num_b_operands: int = 1
    gemm_operands: list[tuple[int, int]] = field(default_factory=lambda: [(0, 0)])
    # ---- Mainloop fusion (Phase 6) ----------------------------------------
    # Unary pointwise ops applied to the **A** operand *before* the MMA reads
    # it — i.e. ``A' = op_k(... op_0(A))`` and ``C = A' @ B``. These run on the
    # dedicated mainloop-fusion warps (warps 8..11 in the 12-warp template),
    # which read the freshly-TMA'd A tile out of SMEM, transform it in
    # registers (fp32 compute), and write it back in place before the MMA
    # consumes it. POC scope: a *linear chain of unary ops on A only* (no aux,
    # no binary, no B-side fusion). Empty list ⇒ no mainloop fusion ⇒ the
    # compiler picks the ordinary 8-warp template.
    mainloop_a_ops: list[FusionOp] = field(default_factory=list)
    # Same as ``mainloop_a_ops`` but for the **B** operand: ``B' = op(B)`` and
    # ``C = A @ B'`` (or both A and B transformed). Each CTA's mainloop warps
    # transform their own B SMEM tile in place.
    mainloop_b_ops: list[FusionOp] = field(default_factory=list)
    # Mixed-input mainloop: when the fused operand is LOADED at a narrower dtype
    # than the MMA reads (e.g. int8 A -> identity -> bf16 MMA), this holds the
    # narrow LOAD dtype while ``matmul.a_dtype`` / ``matmul.b_dtype`` hold the
    # (wider) MMA dtype. ``None`` ⇒ no cast (load dtype == MMA dtype, the
    # ordinary dtype-preserving mainloop). The mainloop warps stage the cast:
    # TMA loads the narrow tile, the warps widen it into the MMA SMEM tile.
    mainloop_a_load_dtype: Dtype | None = None
    mainloop_b_load_dtype: Dtype | None = None
    # Position of the terminal op in `ops`. Sentinel ``-2`` (default) means
    # "auto": for linear chains it resolves to ``len(ops) - 1``; for the
    # matmul-only case (no ops) it resolves to ``-1`` (matmul itself is the
    # terminal). The analyzer sets it explicitly when the DAG has multiple
    # branches and the terminal is not simply the last op in the list.
    terminal_op_idx: int = -2
    # ---- No-epilogue multi-GEMM (parallel matmuls, each materialized) -------
    # When set (length == num_gemms), there are NO fusion ops: every GEMM's
    # accumulator is stored DIRECTLY to its own GMEM output (cast to the dtype
    # here). GEMM 0 is the terminal (slot 0); GEMMs >0 are taps (slots 1..).
    # This is the "K parallel GEMMs, same shape, no shared epilogue" case
    # (e.g. the DualBlockScaleMatmul benchmark). None ⇒ a fused epilogue (ops).
    per_gemm_outputs: "list[Dtype] | None" = None
    # ---- Block-scaled matmul (FP4 / FP8 + per-block scale factors) ---------
    # Set by the analyzer when the matmul's A and B operands are produced by
    # ``block_scale_dequantize`` nodes. When non-None the compiler routes to the
    # ``sm100_block_scale_matmul_*`` template, which loads the
    # scale factors via TMA, copies them SMEM→TMEM (utccp), and runs the
    # block-scaled MMA. The epilogue DAG (``ops``) still runs on the fp32
    # accumulator as usual. None ⇒ ordinary (non-scaled) matmul.
    block_scale: "BlockScaleSpec | None" = None
    # ---- MoE grouped matmul (per-group variable M-range) -------------------
    # Set by the analyzer when the graph's single op is ``moe_grouped_matmul``.
    # When non-None the compiler routes to the ``sm100_moe_grouped_matmul_fwd_*``
    # template. None ⇒ ordinary matmul.
    moe: "MoeSpec | None" = None
    # ---- Epilogue reductions ----------------------------------------------
    # Materialized reduction side-outputs. The terminal full-size output is
    # still slot 0; reductions are extra output slots, initialized by the
    # runtime wrapper and updated atomically from the epilogue loop.
    reductions: list[ReductionSpec] = field(default_factory=list)
    # ---- Terminal block-scale quantize -------------------------------------
    # When set, the terminal full-size output is the quantized tensor and an
    # extra scale-factor output is materialized from the epilogue.
    block_quant: "BlockQuantizeSpec | None" = None

    def __post_init__(self) -> None:
        if self.output_dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported output dtype {self.output_dtype!r}")
        # Multi-GEMM structural invariants.
        if self.num_a_operands < 1 or self.num_b_operands < 1:
            raise ValueError("num_a_operands / num_b_operands must be positive")
        if not self.gemm_operands:
            raise ValueError("gemm_operands must list at least one GEMM")
        for g, (ai, bi) in enumerate(self.gemm_operands):
            if not (0 <= ai < self.num_a_operands) or not (0 <= bi < self.num_b_operands):
                raise ValueError(f"gemm_operands[{g}]=({ai},{bi}) out of range for " f"{self.num_a_operands} A / {self.num_b_operands} B operands")
        names = {t.name for t in self.aux_tensors}
        if len(names) != len(self.aux_tensors):
            raise ValueError("aux_tensors contain duplicate names")
        for op in self.ops:
            if op.aux is not None and op.aux not in names:
                raise ValueError(f"op {op.op!r} references unknown aux {op.aux!r}")
        for red in self.reductions:
            if is_gemm_source(red.source_ref):
                g = gemm_index(red.source_ref)
                if g >= self.num_gemms:
                    raise ValueError(f"reduction source GEMM {g} out of range for " f"{self.num_gemms} GEMM(s)")
            elif not (0 <= red.source_ref < len(self.ops)):
                raise ValueError(f"reduction source op index {red.source_ref} out of range " f"for {len(self.ops)} op(s)")
        if self.block_quant is not None:
            q = self.block_quant
            if is_gemm_source(q.source_ref):
                g = gemm_index(q.source_ref)
                if g >= self.num_gemms:
                    raise ValueError(f"block quantize source GEMM {g} out of range for " f"{self.num_gemms} GEMM(s)")
            elif not (0 <= q.source_ref < len(self.ops)):
                raise ValueError(f"block quantize source op index {q.source_ref} out of range " f"for {len(self.ops)} op(s)")
            if self.output_dtype not in ("fp8_e4m3", "fp8_e5m2"):
                raise ValueError(f"block quantize output dtype {self.output_dtype!r} is not " "supported; expected fp8_e4m3 or fp8_e5m2")
        # Mainloop fusion (POC): a straight chain applied to the operand (op i
        # reads op i-1's result; op 0 reads the raw A / B tile). Unary ops, or
        # binary ops with a single SCALAR graph-input aux (e.g. A * alpha) —
        # scalar so the transform stays element-wise / swizzle-agnostic. No
        # fan-in (parent_idx_b); per-row / per-col / per-elem aux is out of
        # scope (would need swizzle-aware SMEM indexing).
        for label, oplist in (("mainloop_a_ops", self.mainloop_a_ops), ("mainloop_b_ops", self.mainloop_b_ops)):
            for op in oplist:
                if op.parent_idx_b is not None:
                    raise ValueError(f"{label} op {op.op!r} cannot have parent_idx_b (no fan-in)")
                if op.op in BINARY_OPS and op.aux is None:
                    raise ValueError(f"{label} binary op {op.op!r} needs a scalar aux tensor")
                if op.aux is not None:
                    if op.aux not in names:
                        raise ValueError(f"{label} op {op.op!r} references unknown aux {op.aux!r}")
                    ref = self.aux_by_name(op.aux)
                    if ref.bcast_mode != "scalar":
                        raise ValueError(
                            f"{label} op {op.op!r} aux {op.aux!r} must broadcast as a "
                            f"scalar (got {ref.bcast_mode!r}); per-row/col/elem mainloop "
                            "aux is out of POC scope"
                        )

    @property
    def num_gemms(self) -> int:
        """Number of parallel GEMMs feeding the shared epilogue (1 = legacy)."""
        return len(self.gemm_operands)

    @property
    def is_multi_gemm(self) -> bool:
        """True iff more than one parallel GEMM feeds the epilogue."""
        return self.num_gemms > 1

    @property
    def has_block_scale(self) -> bool:
        """True iff this is a block-scaled (FP4/FP8 + per-block SF) matmul."""
        return self.block_scale is not None

    @property
    def has_moe(self) -> bool:
        """True iff this is a MoE grouped matmul."""
        return self.moe is not None

    @property
    def has_mainloop_fusion_a(self) -> bool:
        """True iff a pointwise op is applied to A before the MMA."""
        return bool(self.mainloop_a_ops)

    @property
    def has_mainloop_fusion_b(self) -> bool:
        """True iff a pointwise op is applied to B before the MMA."""
        return bool(self.mainloop_b_ops)

    @property
    def has_mainloop_fusion(self) -> bool:
        """True iff any operand has a mainloop (pre-MMA) pointwise op."""
        return self.has_mainloop_fusion_a or self.has_mainloop_fusion_b

    @property
    def mainloop_a_cast(self) -> bool:
        """True iff A is loaded narrow and widened to the MMA dtype in the mainloop."""
        return self.mainloop_a_load_dtype is not None

    @property
    def mainloop_b_cast(self) -> bool:
        """True iff B is loaded narrow and widened to the MMA dtype in the mainloop."""
        return self.mainloop_b_load_dtype is not None

    @property
    def resolved_terminal_idx(self) -> int:
        """Resolve the ``terminal_op_idx`` sentinel. Returns -1 when the
        matmul itself is the terminal, otherwise a concrete index in
        ``self.ops``."""
        if self.terminal_op_idx == -2:
            return len(self.ops) - 1 if self.ops else -1
        return self.terminal_op_idx

    def aux_by_name(self, name: str) -> TensorRef:
        for t in self.aux_tensors:
            if t.name == name:
                return t
        raise KeyError(name)

    @property
    def outputs(self) -> list["ChainOutput"]:
        """GMEM outputs the kernel materializes, in canonical slot order:
        slot 0 is always the terminal output; subsequent slots are taps in
        chain order — matmul first (if tapped), each fusion op with
        ``output_tap``, then reduction side-outputs. A pointwise tap's dtype is
        the producer's ``out_dtype``. Callers of :class:`CompiledFusedGemm`
        must pass runtime output tensors in this order."""
        if self.per_gemm_outputs is not None:
            # No-epilogue multi-GEMM: GEMM 0 = terminal, GEMMs >0 = taps.
            outs = [ChainOutput(source="terminal", dtype=self.per_gemm_outputs[0])]
            for g in range(1, len(self.per_gemm_outputs)):
                outs.append(ChainOutput(source=f"gemm_{g}", dtype=self.per_gemm_outputs[g]))
            return outs
        outs: list[ChainOutput] = [
            ChainOutput(source="terminal", dtype=self.output_dtype),
        ]
        if self.matmul.output_tap:
            outs.append(ChainOutput(source="matmul", dtype=self.matmul.out_dtype))
        for i, op in enumerate(self.ops):
            if op.output_tap:
                outs.append(ChainOutput(source=f"op_{i}", dtype=op.out_dtype))
        for i, red in enumerate(self.reductions):
            outs.append(
                ChainOutput(
                    source=f"reduction_{i}",
                    dtype=red.dtype,
                    dim=red.dim,
                    is_reduction=True,
                )
            )
        if self.block_quant is not None:
            outs.append(
                ChainOutput(
                    source="block_quant_scale",
                    dtype=self.block_quant.scale_dtype,
                    dim=self.block_quant.scale_dim,
                    is_quant_scale=True,
                    quant_block_size=self.block_quant.block_size,
                )
            )
        return outs

    @property
    def taps(self) -> list["ChainOutput"]:
        """Just the non-terminal outputs (chain.outputs[1:])."""
        return self.outputs[1:]

    def summary(self) -> str:
        """One-line human-readable summary, useful in logs and error messages."""
        m = self.matmul
        batch = f"batch={m.batch}"
        if m.a_batch != m.batch or m.b_batch != m.batch:
            batch += f" A_batch={m.a_batch} B_batch={m.b_batch}"
        chain = " -> ".join(op.op if op.aux is None else f"{op.op}({op.aux})" for op in self.ops) or "identity"
        reductions = ""
        if self.reductions:
            red = ", ".join(f"{r.mode}{r.dim}" for r in self.reductions)
            reductions = f" | reductions: {red}"
        quant = ""
        if self.block_quant is not None:
            q = self.block_quant
            quant = f" | block_quant: block{q.block_size}->{q.scale_dtype}"
        mainloop = ""
        if self.mainloop_a_ops:
            ml = " -> ".join(op.op for op in self.mainloop_a_ops)
            cast = f" [{self.mainloop_a_load_dtype}->{self.matmul.a_dtype}]" if self.mainloop_a_cast else ""
            mainloop += f"mainloop A: {ml}(A){cast} | "
        if self.mainloop_b_ops:
            ml = " -> ".join(op.op for op in self.mainloop_b_ops)
            cast = f" [{self.mainloop_b_load_dtype}->{self.matmul.b_dtype}]" if self.mainloop_b_cast else ""
            mainloop += f"mainloop B: {ml}(B){cast} | "
        return (
            f"FusionChain[matmul {m.M}x{m.N}x{m.K} {batch} "
            f"A{m.a_major}/B{m.b_major} "
            f"{m.a_dtype}*{m.b_dtype}->{m.accum_dtype} | {mainloop}"
            f"epilogue: {chain} -> {self.output_dtype}{reductions}{quant}]"
        )
