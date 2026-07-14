# SPDX-License-Identifier: LicenseRef-NvidiaProprietary


import cutlass
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import tcgen05_ld_red, Tcgen05LdStShape
import cutlass.cute as cute
from cutlass._mlir.dialects import arith, vector
from cutlass._mlir.extras import types as T_
from cutlass._mlir import ir as _ir

from .regtile import RegTile, vec_concat


@cute.jit
def tmem_load_max_reduction(tmem_addr, num: cutlass.Constexpr = 64):
    return tcgen05_ld_red(
        Tcgen05LdStShape.SHAPE_32X32B,
        tmem_addr,
        num=num,
        red_op="max",
        type_="f32",
    )


def row_max_reduction(vec):
    n = int(vec.shape[0])
    elems = [vec[i] for i in range(n)]
    while len(elems) > 1:
        nxt = []
        for i in range(0, len(elems), 3):
            grp = elems[i : i + 3]
            acc = grp[0]
            for g in grp[1:]:
                acc = cute.math.max(acc, g, ftz=True)
            nxt.append(acc)
        elems = nxt
    return elems[0]


def row_reduction_pair(vec):
    n = int(vec.shape[0])
    assert n % 2 == 0, f"row_reduction_pair: N={n} must be even"
    half = n // 2
    paired_ty = _ir.VectorType.get([half, 2], T_.f32())
    paired = vector.shape_cast(paired_ty, vec.ir_value())

    acc = vector.extract(paired, dynamic_position=[], static_position=[0])
    for i in range(1, half):
        pair = vector.extract(paired, dynamic_position=[], static_position=[i])
        acc = arith.addf(acc, pair)
    return cutlass.Vector(acc, dtype=cutlass.Float32)


def tmem_load_max_reduction_x64(tmem_addr):
    return tmem_load_max_reduction(tmem_addr, num=64)


def row_max_reduction_64(vec64):
    return row_max_reduction(vec64)


def row_reduction_pair_64(vec64):
    return row_reduction_pair(vec64)


def tmem_load_tile(tmem_addr, num_elems: int, ld_num: int = 64) -> RegTile:
    assert num_elems % ld_num == 0, f"tmem_load_tile: num_elems={num_elems} must be a multiple of " f"ld_num={ld_num}"
    chunks = [
        nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(tmem_addr + cutlass.Int32(i * ld_num), cutlass.Float32),
            num=ld_num,
        )
        for i in range(num_elems // ld_num)
    ]
    return RegTile(vec_concat(chunks))


def tmem_load_max_reduction_tile(tmem_addr, num_elems: int):
    assert num_elems % 64 == 0, f"tmem_load_max_reduction_tile: num_elems={num_elems} must be a multiple of 64"
    raw_results = [
        tcgen05_ld_red(
            Tcgen05LdStShape.SHAPE_32X32B,
            tmem_addr + cutlass.Int32(i * 64),
            num=64,
            red_op="max",
            type_="f32",
        )
        for i in range(num_elems // 64)
    ]
    data_chunks = [cutlass.Vector.from_elements(tuple(res[:64]), cutlass.Int32).bitcast(cutlass.Float32) for res in raw_results]
    max_scalars = [cutlass.Vector.from_elements((res[64],), cutlass.Int32).bitcast(cutlass.Float32)[0] for res in raw_results]
    final_max = max_scalars[0]
    for m in max_scalars[1:]:
        final_max = cute.math.max(final_max, m)
    return RegTile(vec_concat(data_chunks)), final_max


@cute.jit
def fp32_to_fp16(lo, hi, *, dtype=cutlass.Float16):
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"fp32_to_fp16: dtype must be Float16 or BFloat16, got {dtype}")
    tag = "f16" if cutlass.const_expr(dtype == cutlass.Float16) else "bf16"
    return inline_ptx(
        f"cvt.rn.{tag}x2.f32 $0, $2, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


def fp32_to_fp8_pack(values, *, dtype_tag: str):
    assert len(values) == 16, f"fp32_to_fp8_pack: expected 16 input values, got {len(values)}"
    assert dtype_tag in ("e4m3", "e5m2"), f"fp32_to_fp8_pack: dtype_tag must be 'e4m3' or 'e5m2', got {dtype_tag!r}"

    u0, u1, u2, u3 = inline_ptx(
        "{ .reg .b16 lo, hi;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $5,  $4;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $7,  $6;\n"
        "mov.b32 $0, {lo, hi};\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $9,  $8;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $11, $10;\n"
        "mov.b32 $1, {lo, hi};\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $13, $12;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $15, $14;\n"
        "mov.b32 $2, {lo, hi};\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $17, $16;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $19, $18;\n"
        "mov.b32 $3, {lo, hi}; }",
        write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
        read_only_args=list(values),
    )
    return cutlass.Vector.from_elements((u0, u1, u2, u3), cutlass.Int32)


def vec_scale_pair(vec, scalar, N):
    assert N % 2 == 0, f"vec_scale_pair: N={N} must be even"
    pair_ty = _ir.VectorType.get([2], T_.f32())
    paired_ty = _ir.VectorType.get([N // 2, 2], T_.f32())
    flat_ty = _ir.VectorType.get([N], T_.f32())

    scalar_pair = vector.broadcast(pair_ty, scalar.ir_value())

    paired_in = vector.shape_cast(paired_ty, vec.ir_value())
    result = paired_in
    for i in range(N // 2):
        pair = vector.extract(paired_in, dynamic_position=[], static_position=[i])
        scaled = nvvm.mul_packed_f32x2(pair, scalar_pair, rnd=nvvm.FPRoundingMode.RN)
        result = vector.insert(
            scaled,
            result,
            dynamic_position=[],
            static_position=[i],
        )
    return cutlass.Vector(vector.shape_cast(flat_ty, result), dtype=cutlass.Float32)
