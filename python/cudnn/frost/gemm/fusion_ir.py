"""Tiny IR describing a matmul + pointwise-epilogue fusion — the contract
between `graph_analyzer.py` (builds it from a cuDNN graph) and
`epilogue_codegen.py` (lowers it to a cute DSL epilogue snippet).

A binary op is either ``aux`` + ``parent_idx`` (one in-chain operand + a
graph-input aux) or ``parent_idx_b`` + ``parent_idx`` (two in-chain operands,
fan-in); ``aux`` and ``parent_idx_b`` are mutually exclusive. The terminal is
always materialized; any intermediate can be tapped as an extra GMEM output via
``.set_output(True).set_data_type(...)``. ``FusionChain.outputs`` lists slots in
canonical order (terminal first, then taps) — the same order callers pass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# SUPPORTED_DTYPES = aux/output storage dtypes; COMPUTE_DTYPES = pointwise
# compute dtypes accepted from cuDNN graph attrs.
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

# Block-scaled-matmul scale-factor dtypes (SFA/SFB): nvfp4 = FP4 data + E4M3
# scale block16; mxfp4 = FP4 + E8M0 block32; mxfp8 = FP8 + E8M0 block32.
BLOCK_SCALE_SF_DTYPES: tuple[Dtype, ...] = ("fp8_e4m3", "fp8_e8m0")

# Aux broadcast onto the (M, N) tile: scalar / per_row (len M) / per_col
# (len N) / per_elem (full M×N).
BroadcastMode = Literal["scalar", "per_row", "per_col", "per_elem"]
ReductionMode = Literal["add", "amax", "max", "min"]
REDUCTION_DTYPES: tuple[Dtype, ...] = ("fp32", "int32")

# Unary pointwise ops (take only the running accumulator).
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

# Binary pointwise ops (running accumulator + one aux). `aux_on_rhs=True` means
# op(acc, aux); False means op(aux, acc) — matters for sub/div.
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

# Ops with `f(0, *) == 0` unconditionally. Single source of truth for the
# mainloop K-OOB fix: a fully zero-preserving fused chain keeps the TMA K-tail
# zero-fill at 0, so no OOB masking is needed. Qualifies only if unconditional:
# unary f(0)=0 ops + binary `mul`. sigmoid/exp/cos (f(0)!=0), add/sub, div
# (aux/0=inf) do NOT. A new op defaults to NON-zero-preserving unless listed
# here — the safe default (forgetting only costs an unneeded mask; wrongly
# assuming it would silently miscompute).
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


@dataclass(frozen=True)
class ChainOutput:
    """One materialized GMEM output. ``source`` = where the value is taken from:
    "matmul" (fp32 accumulator) / "op_<i>" / "terminal" / "reduction_<i>" /
    "block_quant_scale". ``dtype`` is the on-disk dtype (each output casts from
    the running fp32 value)."""

    source: str  # "matmul" | "terminal" | "op_<i>" | "reduction_<i>" | "block_quant_scale"
    dtype: Dtype
    dim: "tuple[int, int, int] | None" = None
    is_reduction: bool = False
    is_quant_scale: bool = False
    quant_block_size: int | None = None


@dataclass(frozen=True)
class ReductionSpec:
    """A materialized reduction side-output. ``source_ref`` names the reduced
    producer (FusionOp parent scheme: GEMM output negative, op output >=0).
    ``dim`` is public `(B, M, N)`; extent-1 dims are the reduced ones."""

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
    """Terminal block-scale quantize (cuDNN ``block_scale_quantize`` for the
    row/N-axis case): each contiguous N-block yields one scale + one block of
    quantized output. Quantized output is the terminal; ``scale_dim`` is the
    scale side-output in `(B, scale_M, scale_N)` order (compact =
    `(B, M, N/block_size)`; F8_128x4 may pad to 128-row × 4-col blocks)."""

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
            raise NotImplementedError(f"block quantize supports only the last/N axis in cudnn.frost.gemm; got axis={self.axis}")
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


# Producing-operation references — "where does this op's input come from?".
# A reference (``FusionOp.parent_idx`` / ``parent_idx_b``) is one int:
#   * ``>= 0`` → a pointwise op result, ``ops[ref]``.
#   * ``< 0``  → a GEMM output; ``gemm_index(ref) = -1 - ref`` (-1 = GEMM 0,
#                -2 = GEMM 1, ...). Single-GEMM's only ref is -1 (the legacy
#                "matmul output" sentinel).
#   * ``None`` → parent_idx: auto (prev op, or GEMM 0 for op 0); parent_idx_b:
#                no second in-chain operand.
# No separate "which GEMM" field: a ref names one producing op (GEMM or
# pointwise), mirroring cuDNN. Once two GEMMs merge in an op, downstream refs
# point at that op (>=0).


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
    """A single pointwise op in the epilogue chain. Unary → ``aux`` None; binary
    → ``aux`` names an enclosing-chain aux tensor. ``output_tap`` materializes
    this op's result (at ``out_dtype``) before the next op runs.
    ``parent_idx`` / ``parent_idx_b`` are producing-operation references (see the
    module note). Fan-out = ops sharing one parent ref; fan-in = a binary op
    with both parent refs in-chain (``aux`` must be None — mutually exclusive).
    ``aux_on_rhs`` picks the second operand's side."""

    op: str
    aux: str | None = None  # TensorRef.name, or None for unary / fan-in
    aux_on_rhs: bool = True  # for binary ops; True = op(parent_idx, other)
    output_tap: bool = False  # materialize this op's result (at out_dtype)
    parent_idx: int | None = None  # producing-op ref (None = auto)
    parent_idx_b: int | None = None  # fan-in: second producing-op ref (None = absent)
    # Op math precision (cuDNN per-op ``compute_data_type``); fp32 or int32.
    # Transcendentals (exp/log/sqrt/rsqrt/sigmoid/gelu/swish) should use fp32.
    compute_dtype: Dtype = "fp32"
    # Declared output dtype of this op's (virtual or materialized) tensor. The
    # running value is rounded to it before the next op — a narrow dtype loses
    # precision on purpose (matching cuDNN even for virtual tensors). fp32
    # (default) = no rounding.
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
        """Resolve an auto (``None``) ``parent_idx`` given the op's position:
        previous op for position > 0, else GEMM 0 (-1). Non-None returned as-is."""
        if self.parent_idx is not None:
            return self.parent_idx
        return position - 1 if position > 0 else gemm_source(0)


@dataclass(frozen=True)
class MatmulSpec:
    """Shape + dtype of the anchor matmul. Rank-3: A=(batch_A, M, K),
    B=(batch_B, K, N), out=(batch, M, N), batch=max(batch_A, batch_B); each
    operand batch is the output batch or 1 (broadcast). ``output_tap`` requests
    an extra GMEM output of the matmul result (accumulator rounded to
    ``out_dtype``) before any fusion op runs."""

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
        # MMA (a, b, accum) runnability is NOT validated here (depends on arch +
        # pipeline, which the IR doesn't know) — the compiler checks it against
        # the arch-aware table. IR only validates structural + output-storage dtype.
        if self.out_dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported matmul out_dtype {self.out_dtype!r}; " f"expected one of {SUPPORTED_DTYPES}")


@dataclass(frozen=True)
class BlockScaleSpec:
    """A block-scaled matmul: ``C = (descale_a ⊙ A) @ (descale_b ⊙ B)`` — narrow
    (FP4/FP8) A/B each dequantized by a per-block K scale inside the MMA
    (``tcgen05.mma.kind::mx*.block_scale``).

    Detected by a purely STRUCTURAL match in the analyzer: if A and/or B is the
    output of a ``block_scale_dequantize`` node, dequant(s) + matmul fold into
    one block-scale matmul (three shapes: dequant(A)@B, A@dequant(B),
    dequant(A)@dequant(B); ``sfa``/``sfb`` present only for the scaled side(s)).
    No dtype/block/arch rules here — runnability is decided at compile time.
    Currently runs (both sides): nvfp4 (fp4+e4m3, block16), mxfp4 (fp4+e8m0,
    block32), mxfp8 (fp8+e8m0, block32).

    SF tensors are runtime-positional (not ``TensorRef``s), fully described here
    by per-side scalars; their logical dims derive from M/N/K/block_size. Passed
    at runtime in the ``F8_128x4`` swizzled layout (128-row × 4-K blocked)."""

    a_dtype: Dtype  # packed data dtype of A (mirror of matmul.a_dtype)
    b_dtype: Dtype  # packed data dtype of B
    # Per-side scale info, kept SEPARATE for future asymmetric block-scale. A
    # non-dequantized side has all None fields. `block_size_*` is 2D (A=[non_K,
    # K], B=[K, non_K]); only 1D K-scaling used today (2D form is headroom).
    block_size_a: "tuple[int, ...] | None" = None
    block_size_b: "tuple[int, ...] | None" = None
    sf_dtype_a: "Dtype | None" = None  # A's scale-factor dtype (None if A not scaled)
    sf_dtype_b: "Dtype | None" = None
    # SF reorder layout per side (cuDNN name, e.g. "F8_128x4"; None = NONE).
    sfa_reorder: "str | None" = None
    sfb_reorder: "str | None" = None
    # Each dequant op's compute + output dtype (dequant OUTPUT = the MMA's input
    # type for that operand). None for a non-dequantized side. Recorded for the
    # compile-stage check (cuDNN requires dequant math precision = FLOAT).
    dequant_compute_a: "Dtype | None" = None
    dequant_compute_b: "Dtype | None" = None
    dequant_out_a: "Dtype | None" = None  # = MMA input type for A
    dequant_out_b: "Dtype | None" = None  # = MMA input type for B
    # matmul-level dtypes (accum/output/tap) are NOT duplicated here — read
    # chain.matmul / chain.output_dtype. Only input-side info lives here.

    def __post_init__(self) -> None:
        # Structural sanity ONLY — no dtype/block/family/arch rules (analyzer is
        # structural; the compile stage decides which combos run).
        if self.sf_dtype_a is None and self.sf_dtype_b is None:
            raise ValueError("block_scale spec has neither operand dequantized " "(at least one of A / B must be block-scaled)")

    @property
    def both_sided(self) -> bool:
        """True iff BOTH operands were dequantized (a side is dequantized iff
        its scale-factor dtype is set) — the runnable case today."""
        return self.sf_dtype_a is not None and self.sf_dtype_b is not None

    # Convenience scalars for the symmetric both-sided path (only valid when the
    # two sides agree, which the compile-stage check enforces).
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
    """A MoE grouped matmul forward pass: per routed group g,
    ``out[fto[g]:fto[g+1]] = token[range] @ weight[g % num_experts].T``. A
    batched matmul where each expert owns a runtime M-range of the flat token
    tensor (boundaries in a runtime ``first_token_offset``, not baked). Anchor
    MatmulSpec dims: M=total tokens, K=hidden, N=weight; a_batch=1,
    b_batch=num_experts. Compiler routes to ``sm100_moe_grouped_matmul_fwd_*``
    (grouped persistent scheduler + per-group A TMA descriptor replacement).
    POC scope: ``mode == "none"`` only (gather/scatter rejected)."""

    num_experts: int  # E — also the number of routed groups
    mode: str = "none"  # "none" only in the POC
    # first_token_offset dtype (INT32 or INT64; cuDNN accepts both). Baked at JIT
    # time; the scheduler casts reads to Int32 so the math is dtype-agnostic.
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
    # Multi-GEMM: num_gemms parallel GEMMs share the SAME shape/layout/dtype
    # (``matmul``) but may use different A/B operands, all feeding one epilogue
    # DAG (e.g. silu(A@B0)*(A@B1)). Operands are deduped: num_a/b_operands count
    # DISTINCT tensors; ``gemm_operands[g] = (a_idx, b_idx)`` picks from them. An
    # op names which GEMM output it reads via a negative parent_idx (gemm_source).
    # Single-GEMM default (1 A, 1 B, one (0,0) GEMM) is the legacy path (ref -1).
    num_a_operands: int = 1
    num_b_operands: int = 1
    gemm_operands: list[tuple[int, int]] = field(default_factory=lambda: [(0, 0)])
    # Mainloop fusion: unary ops on A applied BEFORE the MMA (``C = op(A) @ B``),
    # run on the dedicated mainloop warps (8..11 in the 12-warp template) that
    # transform the TMA'd A SMEM tile in place. POC scope: linear unary chain on
    # A only. Empty ⇒ no fusion ⇒ ordinary 8-warp template.
    mainloop_a_ops: list[FusionOp] = field(default_factory=list)
    # Same for B (``C = A @ op(B)``); each CTA transforms its own B tile in place.
    mainloop_b_ops: list[FusionOp] = field(default_factory=list)
    # Mixed-input mainloop: narrow LOAD dtype when the fused operand is loaded
    # narrower than the MMA reads (e.g. int8 A -> bf16 MMA); ``matmul.a/b_dtype``
    # hold the wider MMA dtype. None ⇒ no cast (load == MMA). The mainloop warps
    # stage the cast (TMA loads narrow, warps widen into the MMA SMEM tile).
    mainloop_a_load_dtype: Dtype | None = None
    mainloop_b_load_dtype: Dtype | None = None
    # Terminal op position in `ops`. Sentinel -2 = auto (len(ops)-1, or -1 =
    # matmul when no ops). The analyzer sets it explicitly for multi-branch DAGs.
    terminal_op_idx: int = -2
    # No-epilogue multi-GEMM: when set (len == num_gemms), NO fusion ops — each
    # GEMM's accumulator stores directly to its own output (cast to this dtype).
    # GEMM 0 = terminal (slot 0); GEMMs >0 = taps. None ⇒ fused epilogue.
    per_gemm_outputs: "list[Dtype] | None" = None
    # Block-scaled matmul: set when A/B are produced by block_scale_dequantize
    # nodes. Routes to ``sm100_block_scale_matmul_*`` (TMA-loads SF, SMEM→TMEM
    # utccp, block-scaled MMA); the epilogue DAG still runs on the fp32
    # accumulator. None ⇒ ordinary matmul.
    block_scale: "BlockScaleSpec | None" = None
    # MoE grouped matmul: set when the graph's single op is moe_grouped_matmul.
    # Routes to ``sm100_moe_grouped_matmul_fwd_*``. None ⇒ ordinary matmul.
    moe: "MoeSpec | None" = None
    # Materialized reduction side-outputs (extra slots after the terminal),
    # initialized by the runtime wrapper and updated atomically in the epilogue.
    reductions: list[ReductionSpec] = field(default_factory=list)
    # Terminal block-scale quantize: terminal output is the quantized tensor +
    # an extra scale-factor output materialized from the epilogue.
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
        # Mainloop fusion (POC): a straight op chain. Unary, or binary with a
        # single SCALAR aux (e.g. A*alpha) — scalar keeps it swizzle-agnostic.
        # No fan-in; per-row/col/elem aux is out of scope (needs swizzle-aware
        # SMEM indexing).
        for label, oplist in (
            ("mainloop_a_ops", self.mainloop_a_ops),
            ("mainloop_b_ops", self.mainloop_b_ops),
        ):
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
        """Resolve the ``terminal_op_idx`` sentinel: -1 = matmul is the
        terminal, else a concrete index in ``self.ops``."""
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
        """GMEM outputs in canonical slot order: slot 0 = terminal; then taps in
        chain order (matmul first if tapped, each output_tap op, then
        reductions). Callers must pass runtime output tensors in this order."""
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
        """One-line human-readable summary for logs / error messages."""
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
