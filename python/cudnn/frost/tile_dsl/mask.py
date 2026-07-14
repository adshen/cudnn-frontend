# SPDX-License-Identifier: LicenseRef-NvidiaProprietary


import cutlass
from cutlass._mlir.dialects import arith

MASK_NONE = 0
MASK_PADDED = 1 << 0
MASK_CAUSAL = 1 << 1
MASK_SWA = 1 << 2


_NEG_INF_BITS = -3.4028235e38


def apply_mask_chunk(
    reg_S,
    q_abs,
    kv_col_base,
    seq_kv_len,
    swa_window: int,
    mask_flags: int,
    N: int = 64,
    causal_bottom_right: int = 0,
    causal_diag=None,
):
    if cutlass.const_expr(mask_flags == MASK_NONE):
        return reg_S

    neg_inf = cutlass.Float32(_NEG_INF_BITS)
    q_minus_w = q_abs - cutlass.Int32(swa_window) if (mask_flags & MASK_SWA) else None
    if cutlass.const_expr((mask_flags & MASK_CAUSAL) and causal_bottom_right):
        q_caus_lim = q_abs + causal_diag
    else:
        q_caus_lim = q_abs

    elems = []
    for i in range(N):
        kv_abs = kv_col_base + cutlass.Int32(i)
        masked = None
        if cutlass.const_expr(mask_flags & MASK_PADDED):
            term = kv_abs >= seq_kv_len
            masked = term if masked is None else (masked | term)
        if cutlass.const_expr(mask_flags & MASK_CAUSAL):
            term = kv_abs > q_caus_lim
            masked = term if masked is None else (masked | term)
        if cutlass.const_expr(mask_flags & MASK_SWA):
            term = kv_abs < q_minus_w
            masked = term if masked is None else (masked | term)
        val = cutlass.Float32(
            arith.select(
                masked.ir_value(),
                neg_inf.ir_value(),
                reg_S[i].ir_value(),
            )
        )
        elems.append(val)
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)
