# SPDX-License-Identifier: LicenseRef-NvidiaProprietary


import cutlass
import cutlass.cute as cute


@cute.jit
def swizzle_xor_128b(row, col_elem, *, elem_bytes: cutlass.Constexpr[int] = 2):
    chunk_elems = 16 // elem_bytes
    chunk_idx = col_elem // chunk_elems
    in_chunk = col_elem % chunk_elems
    swz_chunk = chunk_idx ^ (row & 7)
    return swz_chunk * chunk_elems + in_chunk


@cute.jit
def swizzle_lin_128b(lin, *, row_stride_log2: cutlass.Constexpr[int], elem_bytes: cutlass.Constexpr[int] = 2):
    chunk_log2 = cutlass.const_expr((16 // elem_bytes).bit_length() - 1)
    shift = cutlass.const_expr(row_stride_log2 - chunk_log2)
    mask = cutlass.const_expr(0x7 << chunk_log2)
    return lin ^ ((lin >> shift) & mask)
