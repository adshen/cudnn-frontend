"""Codegen: FusionChain -> string snippets that fill the kernel template's
hook slots (aux_views + per-vector epilogue). compiler.py merges them via
string replacement at the `# FUSION_HOOK:*` markers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from .fusion_ir import (
    BlockQuantizeSpec,
    Dtype,
    FusionChain,
    FusionOp,
    ReductionSpec,
    TensorRef,
    gemm_index,
    is_gemm_source,
)


@dataclass(frozen=True)
class EpilogueSnippets:
    aux_views: str  # inserted at INJECT_AUX_VIEWS
    epilogue: str  # inserted at INJECT_EPILOGUE
    kernel_params: list[str]  # extra kernel-signature param decls
    host_args: list[str]  # extra host-side arg names for .launch
    # Tap plumbing: one entry per non-terminal output in `chain.taps` order,
    # numbered 0..N-1 (templates reference mC_tap_<i> / c_tap_<i> / etc.).
    tap_kernel_params: list[str] = field(default_factory=list)
    tap_host_params: list[str] = field(default_factory=list)
    tap_host_pass: list[str] = field(default_factory=list)
    tap_compile_fakes: list[str] = field(default_factory=list)
    tap_compile_pass: list[str] = field(default_factory=list)
    tap_ptr_binds: list[str] = field(default_factory=list)
    tap_constants: list[str] = field(default_factory=list)  # vec_bytes_tap_<i> assignments
    # Mainloop-fusion transforms (INJECT_MAINLOOP_A/B in the 12-warp templates):
    # given ml_vec_<a|b>, compute the op chain in fp32 and define ml_out_<a|b>
    # (cast back) which the template stores in place. "pass" = no fusion.
    mainloop_transform_a: str = "pass"
    mainloop_transform_b: str = "pass"


def _aux_ptr_var(name: str) -> str:
    return f"_aux_{name}_ptr"


def _aux_prefetch_var(name: str) -> str:
    return f"_aux_{name}_pre"


def _compute_cast(var: str, dtype: Dtype, tag: str) -> tuple[list[str], str]:
    """Cast a running vector/scalar to the op's compute dtype."""
    new = f"_c_{tag}"
    return [f"{new} = ({var}).to({DTYPE_TO_CUTLASS[dtype]})"], new


def _compute_literal(dtype: Dtype, value: float | int) -> str:
    if dtype == "int32":
        return f"cutlass.Int32({int(value)})"
    return f"cutlass.Float32({float(value)})"


# Aux load expressions (string forms used inside the inner loop)


def _index_term(var: str, stride: int) -> str:
    if stride == 1:
        return var
    return f"{var} * {stride}"


def _aux_index_expr(aux: TensorRef) -> str:
    """Linear element offset for the current epilogue location. Extent-1
    dims are broadcast and don't contribute to the offset."""
    if len(aux.dim) == 1:
        axes = ((aux.dim[0], aux.stride[0], "col_j"),)
    elif len(aux.dim) == 2:
        axes = (
            (aux.dim[0], aux.stride[0], "row"),
            (aux.dim[1], aux.stride[1], "col_j"),
        )
    elif len(aux.dim) == 3:
        axes = (
            (aux.dim[0], aux.stride[0], "tile_l"),
            (aux.dim[1], aux.stride[1], "row"),
            (aux.dim[2], aux.stride[2], "col_j"),
        )
    else:
        raise ValueError(f"unsupported aux rank {len(aux.dim)} for {aux.name!r}")

    terms = [_index_term(var, stride) for dim, stride, var in axes if dim != 1]
    return " + ".join(terms) if terms else "0"


def _aux_load_expr(aux: TensorRef, compute_dtype: Dtype, like_var: str) -> str:
    """Expression yielding aux value(s) as a length-`vsize` vector in the op's
    compute dtype, matching ``like_var``'s dtype."""
    idx = _aux_index_expr(aux)
    cast = DTYPE_TO_CUTLASS[compute_dtype]
    if aux.bcast_mode == "scalar":
        # scalar prefetched in aux_views, broadcast to vec
        return f"cutlass.full_like({like_var}, {_aux_prefetch_var(aux.name)}.to({cast}))"
    if aux.bcast_mode == "per_row":
        # per-row scalar prefetched in aux_views, broadcast to vec
        return f"cutlass.full_like({like_var}, {_aux_prefetch_var(aux.name)}.to({cast}))"
    if aux.bcast_mode == "per_col":
        # vector load of vsize elements from aux_ptr[col_j]
        return f"({_aux_ptr_var(aux.name)} + {idx}).load(count=vsize, " f"alignment=VEC_BYTES).to({cast})"
    if aux.bcast_mode == "per_elem":
        # vector load of vsize elements from aux_ptr[row*N + col_j]
        return f"({_aux_ptr_var(aux.name)} + {idx}).load(count=vsize, " f"alignment=VEC_BYTES).to({cast})"
    raise AssertionError(f"unknown bcast_mode {aux.bcast_mode!r}")


# Per-op emitter

# exp(y) = exp2(y * LOG2E); tanh(x) = 1 - 2/(exp2(2x*LOG2E) + 1).
_LOG2E = "cutlass.Float32(1.4426950408889634)"
_TWO_LOG2E = "cutlass.Float32(2.8853900817779268)"


def _tanh_expr(v: str) -> str:
    """Vector tanh via exp2/rcp: 1 - 2/(exp2(2x*log2e) + 1). Native
    ``cute.math.tanh`` aborts under vector lowering; exp2/rcp with fastmath
    lower fine and saturate cleanly (exp2→inf ⇒ tanh→1; exp2→0 ⇒ tanh→-1)."""
    one = f"cutlass.full_like({v}, cutlass.Float32(1.0))"
    two = f"cutlass.full_like({v}, cutlass.Float32(2.0))"
    e2 = f"cute.math.exp2({v} * cutlass.full_like({v}, {_TWO_LOG2E}), fastmath=True)"
    return f"({one} - {two} * cute.math.rcp({e2} + {one}, approx=True, ftz=True))"


def _emit_op(
    op: FusionOp,
    prev: str,
    idx: int,
    aux_loads: dict[str, str],
    other_in_chain: str | None = None,
) -> tuple[list[str], str]:
    """Emit lines computing this op, given the previous-step var name.
    ``other_in_chain`` = second operand var for fan-in binary ops (else None).
    Returns (new_lines, new_var_name)."""
    new = f"_op_{idx}"

    if op.op == "identity":
        return [], prev

    if op.op == "relu":
        return [f"{new} = cute.math.max({prev}, cutlass.full_like({prev}, {_compute_literal(op.compute_dtype, 0)}))"], new

    if op.op == "tanh":
        return [f"{new} = {_tanh_expr(prev)}"], new

    if op.op == "exp":
        # exp(x) = exp2(x * log2e); vector exp2.approx (MUFU).
        return [f"{new} = cute.math.exp2({prev} * cutlass.full_like({prev}, {_LOG2E}), fastmath=True)"], new

    if op.op == "abs":
        return [f"{new} = cute.math.abs({prev})"], new

    if op.op == "neg":
        return [f"{new} = -{prev}"], new

    if op.op == "cos":
        return [f"{new} = cute.math.cos({prev})"], new

    if op.op == "sin":
        return [f"{new} = cute.math.sin({prev})"], new

    if op.op == "ceil":
        return [f"{new} = cute.math.ceil({prev})"], new

    if op.op == "floor":
        return [f"{new} = cute.math.floor({prev})"], new

    if op.op == "erf":
        return [f"{new} = cute.math.erf({prev})"], new

    if op.op == "log":
        return [f"{new} = cute.math.log({prev})"], new

    if op.op == "reciprocal":
        return [f"{new} = cute.math.rcp({prev}, approx=True, ftz=True)"], new

    if op.op == "rsqrt":
        return [f"{new} = cute.math.rsqrt({prev})"], new

    if op.op == "sqrt":
        return [f"{new} = cute.math.sqrt({prev})"], new

    if op.op == "sigmoid":
        # sigmoid(x) = 1/(1+exp(-x)) — vector exp2.approx + rcp.approx (MUFU).
        return [
            f"{new} = cute.math.rcp(cutlass.full_like({prev}, cutlass.Float32(1.0)) + "
            f"cute.math.exp2(-{prev} * cutlass.full_like({prev}, {_LOG2E}), fastmath=True), approx=True, ftz=True)"
        ], new

    if op.op == "swish":
        # swish/SiLU = x * sigmoid(x).
        return [
            f"{new} = {prev} * cute.math.rcp(cutlass.full_like({prev}, cutlass.Float32(1.0)) + "
            f"cute.math.exp2(-{prev} * cutlass.full_like({prev}, {_LOG2E}), fastmath=True), approx=True, ftz=True)"
        ], new

    if op.op == "gelu_tanh":
        # 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))); native vector tanh.approx.
        return [
            f"_g_x{idx} = {prev}",
            f"_g_inner{idx} = cutlass.full_like(_g_x{idx}, cutlass.Float32(0.7978845608028654)) * "
            f"(_g_x{idx} + cutlass.full_like(_g_x{idx}, cutlass.Float32(0.044715)) * _g_x{idx} * _g_x{idx} * _g_x{idx})",
            f"_g_tanh{idx} = {_tanh_expr(f'_g_inner{idx}')}",
            f"{new} = cutlass.full_like(_g_x{idx}, cutlass.Float32(0.5)) * _g_x{idx} * (cutlass.full_like(_g_x{idx}, cutlass.Float32(1.0)) + _g_tanh{idx})",
        ], new

    if op.op == "gelu":
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        return [
            f"_e_x{idx} = {prev}",
            f"_e_erf{idx} = cute.math.erf(_e_x{idx} * cutlass.full_like(_e_x{idx}, {_compute_literal(op.compute_dtype, 0.7071067811865475)}))",
            f"{new} = cutlass.full_like(_e_x{idx}, {_compute_literal(op.compute_dtype, 0.5)}) * _e_x{idx} * "
            f"(cutlass.full_like(_e_x{idx}, {_compute_literal(op.compute_dtype, 1)}) + _e_erf{idx})",
        ], new

    if op.op in {"add", "mul", "sub", "div", "max", "min", "pow", "add_square"}:
        py_op = {"add": "+", "mul": "*", "sub": "-", "div": "/"}.get(op.op)
        if op.parent_idx_b is not None:
            # fan-in: second operand is another in-chain op result
            assert other_in_chain is not None
            if op.op == "max":
                return [f"{new} = cute.math.max({prev}, {other_in_chain})"], new
            if op.op == "min":
                return [f"{new} = cute.math.min({prev}, {other_in_chain})"], new
            if op.op == "pow":
                if op.aux_on_rhs:
                    return [f"{new} = cute.math.pow({prev}, {other_in_chain})"], new
                return [f"{new} = cute.math.pow({other_in_chain}, {prev})"], new
            if op.op == "add_square":
                if op.aux_on_rhs:
                    return [f"{new} = {prev} + {other_in_chain} * {other_in_chain}"], new
                return [f"{new} = {other_in_chain} + {prev} * {prev}"], new
            assert py_op is not None
            if op.aux_on_rhs:
                return [f"{new} = {prev} {py_op} {other_in_chain}"], new
            return [f"{new} = {other_in_chain} {py_op} {prev}"], new
        assert op.aux is not None
        aux_expr = aux_loads[op.aux]
        if op.op == "max":
            return [f"{new} = cute.math.max({prev}, ({aux_expr}))"], new
        if op.op == "min":
            return [f"{new} = cute.math.min({prev}, ({aux_expr}))"], new
        if op.op == "pow":
            if op.aux_on_rhs:
                return [f"{new} = cute.math.pow({prev}, ({aux_expr}))"], new
            return [f"{new} = cute.math.pow(({aux_expr}), {prev})"], new
        if op.op == "add_square":
            if op.aux_on_rhs:
                return [
                    f"_sq_aux_{idx} = ({aux_expr})",
                    f"{new} = {prev} + _sq_aux_{idx} * _sq_aux_{idx}",
                ], new
            return [
                f"_sq_aux_{idx} = ({aux_expr})",
                f"{new} = _sq_aux_{idx} + {prev} * {prev}",
            ], new
        assert py_op is not None
        if op.aux_on_rhs:
            return [f"{new} = {prev} {py_op} ({aux_expr})"], new
        return [f"{new} = ({aux_expr}) {py_op} {prev}"], new

    raise AssertionError(f"unhandled op {op.op!r}")


# Top-level codegen


def _emit_round(var: str, out_dtype: Dtype, tag: str) -> tuple[list[str], str]:
    """Round a register value to a tensor's declared dtype (no-op for fp32)."""
    if out_dtype == "fp32":
        return [], var
    new = f"_r_{tag}"
    return [f"{new} = ({var}).to({DTYPE_TO_CUTLASS[out_dtype]})"], new


def _store_cast_expr(var: str, dtype: Dtype) -> str:
    if dtype == "uint8":
        # GEMM exposes uint8 tensor raw pointers as Int8; bitcast preserves the byte payload.
        return f"({var}).to(cutlass.Uint8).bitcast(cutlass.Int8)"
    return f"({var}).to({DTYPE_TO_CUTLASS[dtype]})"


def _emit_tap_store(tap_idx: int, source_var: str, tap_dtype: Dtype) -> list[str]:
    """Store one tap vector. M-major TMA taps scatter the 16x256b fragment;
    all other paths use the regular `linear_idx` vector store."""
    tap_var = f"_tap_{tap_idx}"
    return [
        f"{tap_var} = {_store_cast_expr(source_var, tap_dtype)}",
        f"if cutlass.const_expr(cd_out_is_m_major and use_tma_store_epi):",
        f"    for _tr_{tap_idx} in cutlass.range_constexpr(16):",
        f"        _tm_{tap_idx} = coord_m + warp_idx * 32 + (lane // 4)" f" + 8 * ((_tr_{tap_idx} >> 1) & 1) + 16 * _h",
        f"        _tn_{tap_idx} = col + 2 * (lane % 4)" f" + (_tr_{tap_idx} & 1) + 8 * (_tr_{tap_idx} >> 2)",
        f"        if _tm_{tap_idx} < M and _tn_{tap_idx} < N:",
        f"            (gC_tap_{tap_idx}_ptr + tile_l * c_stride_l"
        f" + _tm_{tap_idx} * c_stride_m + _tn_{tap_idx} * c_stride_n).store("
        f"{tap_var}[_tr_{tap_idx} : _tr_{tap_idx} + 1], alignment=VEC_BYTES_TAP_{tap_idx})",
        f"else:",
        f"    (gC_tap_{tap_idx}_ptr + linear_idx).store(" f"{tap_var}, alignment=VEC_BYTES_TAP_{tap_idx})",
    ]


def _reduction_output_offset_expr(red_idx: int, red: ReductionSpec, value_idx: str) -> str:
    """Runtime-stride offset for a reduction output. Outputs are permuted to
    internal `(M, N, B)` order; the runtime wrapper passes matching strides."""
    b_extent, m_extent, n_extent = red.dim
    if red.grouped_by_moe:
        l = "cutlass.Int64(group_idx)"
    else:
        l = "cutlass.Int64(0)" if b_extent == 1 else "tile_l"
    m = "cutlass.Int64(0)" if m_extent == 1 else "row"
    n = "cutlass.Int64(0)" if n_extent == 1 else f"(col_j + {value_idx})"
    return f"(({m}) * red_stride_m_{red_idx} + " f"({n}) * red_stride_n_{red_idx} + " f"({l}) * red_stride_l_{red_idx})"


def _emit_reduction_atomic(
    tap_idx: int,
    red_idx: int,
    red: ReductionSpec,
    source_var: str,
) -> list[str]:
    src = f"_red_{red_idx}_src"
    lines = [f"{src} = ({source_var}).to({DTYPE_TO_CUTLASS[red.compute_dtype]})"]
    for i in range(32):
        # const_expr(i < vsize) keeps emitted code valid for every vsize, no dynamic index
        lines.append(f"if cutlass.const_expr({i} < vsize):")
        val = f"{src}[{i}]"
        offset = _reduction_output_offset_expr(red_idx, red, str(i))
        ptr = f"gC_tap_{tap_idx}_ptr + {offset}"
        if red.compute_dtype == "int32":
            if red.mode == "amax":
                val = f"cute.math.abs({val})"
                op = "max"
            elif red.mode in {"add", "max", "min"}:
                op = red.mode
            else:
                raise AssertionError(f"unhandled int32 reduction mode {red.mode!r}")
            lines.append(f'    nvvm.atomicrmw("{op}", {ptr}, ' f'{val}, mem_order="relaxed", syncscope="gpu")')
            continue
        if red.mode == "amax":
            lines.append(f"    cute.arch.atomic_fmax({ptr}, " f'cute.math.abs({val}), sign_bit=False, sem="relaxed", scope="gpu")')
            continue
        if red.mode == "max":
            lines.append(f'    cute.arch.atomic_fmax({ptr}, {val}, sem="relaxed", scope="gpu")')
            continue
        if red.mode == "min":
            bits = f"_red_{red_idx}_{i}_bits"
            lines.extend(
                [
                    f"    {bits} = ({val}).bitcast(cutlass.Int32)",
                    f"    if {bits} < cutlass.Int32(0):",
                    f"        cute.arch.atomic_max({ptr}, cutlass.Uint32({bits}), " f'sem="relaxed", scope="gpu")',
                    "    else:",
                    f"        cute.arch.atomic_min({ptr}, {bits}, " f'sem="relaxed", scope="gpu")',
                ]
            )
            continue
        else:
            assert red.mode == "add"
            op = "add"
        lines.append(f'    nvvm.atomicrmw("{op}", {ptr}, ' f'{val}, mem_order="relaxed", syncscope="gpu")')
    return lines


def _quant_output_max(dtype: Dtype) -> str:
    if dtype == "fp8_e4m3":
        return "cutlass.Float32(448.0)"
    if dtype == "fp8_e5m2":
        return "cutlass.Float32(57344.0)"
    raise ValueError(f"block quantize output dtype {dtype!r} is not supported by codegen")


def _quant_output_min(dtype: Dtype) -> str:
    if dtype == "fp8_e4m3":
        return "cutlass.Float32(-448.0)"
    if dtype == "fp8_e5m2":
        return "cutlass.Float32(-57344.0)"
    raise ValueError(f"block quantize output dtype {dtype!r} is not supported by codegen")


def _emit_block_quant(
    quant: BlockQuantizeSpec,
    source_var: str,
    output_dtype: Dtype,
    tap_idx: int,
    batch_index_expr: str = "tile_l",
) -> list[str]:
    """Emit row/N-axis block quantize for one epilogue vector. The compiler
    gates this so `vsize == block_size` — one scale value per vector chunk."""
    lines: list[str] = [
        f"_q_src = ({source_var}).to(cutlass.Float32)",
        "_q_abs = cute.math.abs(_q_src)",
        "_q_amax = _q_abs[0]",
    ]
    for i in range(1, 32):
        lines.append(f"if cutlass.const_expr({i} < vsize):")
        lines.append(f"    _q_amax = cute.math.max(_q_amax, _q_abs[{i}])")
    scale_dtype = DTYPE_TO_CUTLASS[quant.scale_dtype]
    lines.extend(
        [
            f"_q_scale_f32 = _q_amax * cute.arch.rcp_approx({_quant_output_max(output_dtype)})",
            f"_q_scale = (_q_scale_f32).to({scale_dtype})",
            "_q_scale_up = (_q_scale).to(cutlass.Float32)",
            "_q_inv = cutlass.Float32(0.0)",
            "if _q_scale_up > cutlass.Float32(0.0):",
            "    _q_inv = cute.arch.rcp_approx(_q_scale_up)",
            "_q_scaled = _q_src * cutlass.full_like(_q_src, _q_inv)",
            (
                f"_q_clamped = cute.math.min(cute.math.max(_q_scaled, "
                f"cutlass.full_like(_q_scaled, {_quant_output_min(output_dtype)})), "
                f"cutlass.full_like(_q_scaled, {_quant_output_max(output_dtype)}))"
            ),
            f"vec_out = _q_clamped.to({DTYPE_TO_CUTLASS[output_dtype]})",
        ]
    )
    lines.append(f"_q_scale_col = col_j // {quant.block_size}")
    if quant.scale_reorder == "F8_128x4":
        lines.extend(
            [
                f"_q_scale_ncb = ((N // {quant.block_size}) + 3) // 4",
                (
                    f"_q_scale_idx = {batch_index_expr} * quant_scale_stride_l + "
                    "((row // 128) * _q_scale_ncb + (_q_scale_col // 4)) * 512 + "
                    "(row % 32) * 16 + ((row % 128) // 32) * 4 + (_q_scale_col % 4)"
                ),
            ]
        )
    else:
        lines.append(f"_q_scale_idx = {batch_index_expr} * quant_scale_stride_l + " "row * quant_scale_stride_m + _q_scale_col * quant_scale_stride_n")
    lines.append(f"(gC_tap_{tap_idx}_ptr + _q_scale_idx).store(_q_scale, alignment=1)")
    return lines


def _tap_fake_shape(tap, chain: FusionChain | None = None) -> str:
    if tap.is_quant_scale:
        if chain is None or chain.block_quant is None:
            raise AssertionError("quant scale tap requires FusionChain context")
        q = chain.block_quant
        b, m, n = q.scale_dim or (
            chain.matmul.batch,
            chain.matmul.M,
            chain.matmul.N // q.block_size,
        )
        logical_n = chain.matmul.N // q.block_size
        m_expr = "1" if m == 1 else ("sym_m" if m == chain.matmul.M else str(m))
        n_expr = "1" if n == 1 else (f"(sym_n // {q.block_size})" if n == logical_n else str(n))
        l_expr = "1" if b == 1 else ("sym_l" if b == chain.matmul.batch else str(b))
        return f"({m_expr}, {n_expr}, {l_expr})"
    if tap.dim is None:
        return "(sym_m, sym_n, sym_l)"
    b, m, n = tap.dim
    m_expr = "1" if m == 1 else "sym_m"
    n_expr = "1" if n == 1 else "sym_n"
    l_expr = "1" if b == 1 else "sym_l"
    if tap.is_reduction and chain is not None:
        red_idx = int(tap.source.rsplit("_", 1)[1])
        if chain.reductions[red_idx].grouped_by_moe:
            l_expr = "sym_g"
    return f"({m_expr}, {n_expr}, {l_expr})"


# Per-op temp-var index base for mainloop transforms, kept distinct per operand
# and far from the epilogue's 0-based indices so snippets sharing one JIT
# function never collide on a var name (cute type-checks the whole body).
_MAINLOOP_IDX_BASE = {"a": 900, "b": 800}


# Mainloop identity-cast fast paths (more intrinsics can be added).
_MAINLOOP_IDENTITY_CAST_INTRINSICS: dict[tuple[Dtype, Dtype], str] = {
    ("int8", "bf16"): "cute.arch.cvt_i8_bf16_intrinsic",
}


def generate_mainloop(chain: FusionChain, operand: str = "a") -> str:
    """Emit the mainloop-fusion transform snippet for one operand
    (INJECT_MAINLOOP_A/B). Contract: the template loaded ``ml_vec_<operand>``
    from SMEM; this defines ``ml_out_<operand>`` = the op chain applied in fp32,
    cast back to the operand dtype, which the template stores in place."""
    ops = chain.mainloop_a_ops if operand == "a" else chain.mainloop_b_ops
    if not ops:
        return "pass"
    # dtype-preserving: result rounded back to the operand's own dtype in place
    src_dtype = chain.matmul.a_dtype if operand == "a" else chain.matmul.b_dtype
    ab_dtype = DTYPE_TO_CUTLASS[src_dtype]
    vec_var = f"ml_vec_{operand}"
    out_var = f"ml_out_{operand}"
    f32_var = f"ml_f32_{operand}"
    base = _MAINLOOP_IDX_BASE[operand]
    load_dtype = chain.mainloop_a_load_dtype if operand == "a" else chain.mainloop_b_load_dtype
    identity_cast_intrinsic = (
        _MAINLOOP_IDENTITY_CAST_INTRINSICS.get((load_dtype, src_dtype)) if len(ops) == 1 and ops[0].op == "identity" and load_dtype is not None else None
    )
    if identity_cast_intrinsic is not None:
        cvt = f"{identity_cast_intrinsic}({vec_var}.ir_value(), ml_vec_elems)"
        return f"{out_var} = cutlass.Vector({cvt}, dtype={ab_dtype})"
    # Scalar-aux loads for binary mainloop ops: broadcast the scalar (loaded
    # from its GMEM ptr — a kernel param) to a fp32 vector. Scalar only.
    aux_loads: dict[str, str] = {}
    for op in ops:
        if op.aux is not None:
            ptr = f"{op.aux}.iterator.raw_ptr()"
            aux_loads[op.aux] = f"cutlass.full_like({f32_var}, ({ptr} + 0).load().to(cutlass.Float32))"
    lines: list[str] = [f"{f32_var} = {vec_var}.to(cutlass.Float32)"]
    result_var: dict[int, str] = {-1: f32_var}
    for i, op in enumerate(ops):
        parent = op.resolved_parent_idx(i)
        parent_var = result_var[parent]
        emit_lines, cur = _emit_op(op, parent_var, base + i, aux_loads)
        lines.extend(emit_lines)
        result_var[i] = cur
    terminal_var = result_var[len(ops) - 1]
    # int -> fp8 fold workaround (foot-gun #3): int-loaded + fp8 MMA folds the
    # int->fp32 into the fp32->fp8 narrowing → invalid direct int->fp8 cast
    # (NaN). Break the def-use chain with `+ 0.0` (a two-step .to() does NOT
    # help — must be an arithmetic op).
    if load_dtype in ("int8", "uint8") and src_dtype in ("fp8_e4m3", "fp8_e5m2"):
        terminal_var = f"({terminal_var} + cutlass.full_like({terminal_var}, 0.0))"
    lines.append(f"{out_var} = ({terminal_var}).to({ab_dtype})")
    return "\n".join(lines)


def generate(
    chain: FusionChain,
    *,
    vec_bytes_epi: int = 32,
    output_elem_bytes: int = 2,
) -> EpilogueSnippets:
    """Produce the two hook-site snippets, the extra kernel param list, and
    all per-tap plumbing. ``vec_bytes_epi`` / ``output_elem_bytes`` (from the
    compiler) fix the inner-loop chunk size: each tap stores
    ``vsize = vec_bytes_epi // output_elem_bytes`` elements per chunk."""
    # aux_views snippet. `row` is defined by the template just before this hook
    # (M-aware: differs for MMA_M=64 vs MMA_M>=128) — we just consume it.
    aux_lines: list[str] = []
    for aux in chain.aux_tensors:
        aux_lines.append(f"{_aux_ptr_var(aux.name)} = {aux.name}.iterator.raw_ptr()")
        if aux.bcast_mode == "scalar":
            aux_lines.append(f"{_aux_prefetch_var(aux.name)} = " f"({_aux_ptr_var(aux.name)} + {_aux_index_expr(aux)}).load()")
        elif aux.bcast_mode == "per_row":
            aux_lines.append(f"{_aux_prefetch_var(aux.name)} = " f"({_aux_ptr_var(aux.name)} + {_aux_index_expr(aux)}).load()")
        # per_col / per_elem load inside the inner loop.

    aux_views = "\n".join(aux_lines) if aux_lines else "pass"

    # epilogue snippet (interleaves op chain with tap stores)
    body_lines: list[str] = []
    normal_tap_count = sum(not tap.is_reduction and not tap.is_quant_scale for tap in chain.taps)
    tap_idx = 0  # numbered 0..N-1 in `chain.taps` order

    # Per-op result var name lookup (handles `identity` pass-throughs).
    result_var: dict[int, str] = {}

    # Round each GEMM's fp32 accumulator to the matmul out_dtype before any op
    # reads it (no-op when fp32). GEMM 0 binds legacy ``vec_f32`` (so every
    # non-multi-GEMM template is unchanged); GEMMs >0 bind ``vec_f32_<g>``.
    gemm_var: dict[int, str] = {}
    for g in range(chain.num_gemms):
        src = "vec_f32" if g == 0 else f"vec_f32_{g}"
        tag = "mm" if g == 0 else f"mm{g}"
        round_lines, var = _emit_round(src, chain.matmul.out_dtype, tag)
        body_lines.extend(round_lines)
        gemm_var[g] = var

    def _parent_value(ref: int) -> str:
        """Var name for an op input: a GEMM output (ref < 0) or a prior op."""
        if is_gemm_source(ref):
            return gemm_var[gemm_index(ref)]
        return result_var[ref]

    # Matmul-output tap (single-GEMM only) — fires before any op runs.
    if chain.matmul.output_tap and chain.num_gemms == 1:
        body_lines.extend(_emit_tap_store(tap_idx, gemm_var[0], chain.matmul.out_dtype))
        tap_idx += 1

    # No-epilogue multi-GEMM: GEMMs >0 store to their own outputs (taps);
    # GEMM 0 is the terminal. No pointwise ops run.
    if chain.per_gemm_outputs is not None:
        for g in range(1, chain.num_gemms):
            body_lines.extend(_emit_tap_store(tap_idx, gemm_var[g], chain.per_gemm_outputs[g]))
            tap_idx += 1

    terminal_idx = chain.resolved_terminal_idx
    for i, op in enumerate(chain.ops):
        parent = op.resolved_parent_idx(i)
        parent_raw = _parent_value(parent)
        cast_lines, parent_var = _compute_cast(parent_raw, op.compute_dtype, f"{i}_a")
        body_lines.extend(cast_lines)
        aux_loads = {aux.name: _aux_load_expr(aux, op.compute_dtype, parent_var) for aux in chain.aux_tensors}
        other_in_chain = _parent_value(op.parent_idx_b) if op.parent_idx_b is not None else None
        if other_in_chain is not None:
            cast_lines, other_in_chain = _compute_cast(other_in_chain, op.compute_dtype, f"{i}_b")
            body_lines.extend(cast_lines)
        lines, cur = _emit_op(op, parent_var, i, aux_loads, other_in_chain)
        body_lines.extend(lines)
        # Round to the op's out_dtype (no-op for fp32) so the next op + tap see it.
        round_lines, cur = _emit_round(cur, op.out_dtype, str(i))
        body_lines.extend(round_lines)
        result_var[i] = cur
        if op.output_tap:
            body_lines.extend(_emit_tap_store(tap_idx, cur, op.out_dtype))
            tap_idx += 1

    assert tap_idx == normal_tap_count
    for red_idx, red in enumerate(chain.reductions):
        red_source = _parent_value(red.source_ref)
        body_lines.extend(_emit_reduction_atomic(normal_tap_count + red_idx, red_idx, red, red_source))

    # Terminal cast → `vec_out`. GEMM-output terminal (ref < 0) → gemm_var;
    # else an op result.
    terminal_var = gemm_var[gemm_index(terminal_idx)] if is_gemm_source(terminal_idx) else result_var[terminal_idx]
    if chain.block_quant is not None:
        quant_tap_idx = next(i for i, tap in enumerate(chain.taps) if tap.is_quant_scale)
        body_lines.extend(
            _emit_block_quant(
                chain.block_quant,
                terminal_var,
                chain.output_dtype,
                quant_tap_idx,
                "0" if chain.has_moe else "tile_l",
            )
        )
    else:
        body_lines.append(f"vec_out = {_store_cast_expr(terminal_var, chain.output_dtype)}")

    epilogue = "\n".join(body_lines)

    # aux kernel params
    kernel_params = [f"{aux.name}: cute.Tensor" for aux in chain.aux_tensors]
    host_args = [aux.name for aux in chain.aux_tensors]

    # tap plumbing (one entry per non-terminal output)
    taps = chain.taps
    tap_kernel_params = [f"mC_tap_{i}: cute.Tensor" for i in range(len(taps))]
    tap_host_params = [f"c_tap_{i}: cute.Tensor" for i in range(len(taps))]
    tap_host_pass = [f"c_tap_{i}" for i in range(len(taps))]
    tap_compile_fakes: list[str] = []
    for i, tap in enumerate(taps):
        tap_compile_fakes.append(
            f"fake_c_tap_{i} = make_fake_compact_tensor(\n"
            f"    {DTYPE_TO_CUTLASS[tap.dtype]},\n"
            f"    {_tap_fake_shape(tap, chain)},\n"
            f"    stride_order=(1, 0, 2),\n"
            f"    assumed_align=16,\n"
            f")"
        )
    tap_compile_pass = [f"fake_c_tap_{i}" for i in range(len(taps))]
    tap_ptr_binds: list[str] = []
    for i in range(len(taps)):
        tap_ptr_binds.append(f"gC_tap_{i}_ptr = mC_tap_{i}.iterator.raw_ptr()")
        tap_ptr_binds.append(f"VEC_BYTES_TAP_{i} = vec_bytes_tap_{i}")
    # vsize is shared; each tap's bytes-per-chunk scales with its element width.
    vsize = vec_bytes_epi // output_elem_bytes
    tap_constants = [f"vec_bytes_tap_{i} = {vsize * DTYPE_BYTES[tap.dtype]}" for i, tap in enumerate(taps)]

    mainloop_transform_a = generate_mainloop(chain, "a")
    mainloop_transform_b = generate_mainloop(chain, "b")

    return EpilogueSnippets(
        aux_views=aux_views,
        epilogue=epilogue,
        mainloop_transform_a=mainloop_transform_a,
        mainloop_transform_b=mainloop_transform_b,
        kernel_params=kernel_params,
        host_args=host_args,
        tap_kernel_params=tap_kernel_params,
        tap_host_params=tap_host_params,
        tap_host_pass=tap_host_pass,
        tap_compile_fakes=tap_compile_fakes,
        tap_compile_pass=tap_compile_pass,
        tap_ptr_binds=tap_ptr_binds,
        tap_constants=tap_constants,
    )
