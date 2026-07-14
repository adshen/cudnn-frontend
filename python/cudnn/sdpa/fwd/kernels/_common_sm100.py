# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.scheduler import SCHED_NATURAL
from cudnn.frost.tile_dsl.mask import MASK_CAUSAL, MASK_PADDED, MASK_SWA
from cudnn.frost.tile_dsl.barrier import MBarrier, Producer, Scope


class Bars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object

    mb_bmm1_done: object
    mb_bmm2_done: object
    mb_bmm2_ready: object

    mb_stat_full: object
    mb_stat_empty: object

    mb_o_full: object
    mb_o_empty: object

    mb_tmem_dealloc: object
    mb_empty_mainloop: object

    mb_q_o_alias: object


class D256Bars(NamedTuple):
    mb_q_full: object
    mb_q_o_alias: object
    mb_tmastg_go: object

    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object

    mb_bmm1_done: object
    mb_bmm2_done: object
    mb_bmm2_ready: object

    mb_stat_full: object
    mb_stat_empty: object

    mb_o_full: object
    mb_o_empty: object

    mb_empty_mainloop: object
    mb_tmem_dealloc: object


def make_d256_bars(CFG, *, N_O_CHUNKS: int) -> D256Bars:
    SOFTMAX_LANES_TOTAL = CFG.SOFTMAX_LANES * CFG.CTA_MMA
    CORR_LANES_TOTAL = CFG.CORR_LANES * CFG.CTA_MMA
    SOFTMAX_PLUS_CORR_TOTAL = SOFTMAX_LANES_TOTAL + CORR_LANES_TOTAL
    KV_EMPTY_ARRIVERS = (CFG.CGA_M // CFG.CTA_MMA) + CFG.CGA_N - 1
    N_BMM2_CHUNKS = CFG.N_BMM2_CHUNKS

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return D256Bars(
        mb_q_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_o_alias=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        mb_tmastg_go=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        mb_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_bmm1_done=MBarrier(_alloc(2), stages=2, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(2), stages=2, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(2 * N_BMM2_CHUNKS),
            stages=2 * N_BMM2_CHUNKS,
            init_count=tuple(SOFTMAX_PLUS_CORR_TOTAL if (s % N_BMM2_CHUNKS) == 0 else SOFTMAX_LANES_TOTAL for s in range(2 * N_BMM2_CHUNKS)),
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_stat_full=MBarrier(_alloc(1), stages=1, init_count=CFG.SOFTMAX_LANES, producer=Producer.THREAD),
        mb_stat_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_full=MBarrier(_alloc(N_O_CHUNKS), stages=N_O_CHUNKS, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.THREAD),
    )


def make_classic_bars(CFG) -> Bars:
    SOFTMAX_PLUS_CORR_TOTAL = CFG.SOFTMAX_LANES * 2 * CFG.CTA_MMA
    SOFTMAX_LANES_TOTAL = CFG.SOFTMAX_LANES * CFG.CTA_MMA
    CORR_LANES_TOTAL = CFG.CORR_LANES * CFG.CTA_MMA
    KV_EMPTY_ARRIVERS = (CFG.CGA_M // CFG.CTA_MMA) + CFG.CGA_N - 1
    N_BMM2_CHUNKS = CFG.N_BMM2_CHUNKS

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        mb_q_full=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_bmm1_done=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(CFG.TILES_Q * N_BMM2_CHUNKS),
            stages=CFG.TILES_Q * N_BMM2_CHUNKS,
            init_count=tuple(SOFTMAX_PLUS_CORR_TOTAL if (s % N_BMM2_CHUNKS) == 0 else SOFTMAX_LANES_TOTAL for s in range(CFG.TILES_Q * N_BMM2_CHUNKS)),
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_stat_full=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.SOFTMAX_LANES, producer=Producer.THREAD),
        mb_stat_empty=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_full=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_empty=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.THREAD),
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_q_o_alias=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
    )


class KvLoopBounds(NamedTuple):
    left: object
    unmasked_lo: object
    unmasked_hi: object
    right: object


def _div_up(a, b):
    return (a + cutlass.Int32(b - 1)) // cutlass.Int32(b)


def compute_kv_loop_bounds(
    q_row_coord,
    seqlen_q,
    seq_kv_len,
    swa_window: int,
    mask_flags: int,
    tile_n: int,
    cga_tile_m: int,
    causal_bottom_right: bool = False,
) -> KvLoopBounds:
    left = cutlass.Int32(0)
    right = _div_up(seq_kv_len, tile_n)

    if cutlass.const_expr(causal_bottom_right):
        causal_diag = seq_kv_len - seqlen_q
    else:
        causal_diag = cutlass.Int32(0)

    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        kv_hi_caus = _div_up(q_row_coord + cutlass.Int32(cga_tile_m) + causal_diag, tile_n)
        right = cute.math.min(right, kv_hi_caus)

    if cutlass.const_expr(mask_flags & MASK_SWA):
        cond = q_row_coord > cutlass.Int32(swa_window)
        delta = q_row_coord - cutlass.Int32(swa_window)
        kv_lo_swa = cutlass.Int32(
            arith.select(
                cond.ir_value(),
                (delta // cutlass.Int32(tile_n)).ir_value(),
                cutlass.Int32(0).ir_value(),
            )
        )
        left = cute.math.max(left, kv_lo_swa)

    unmasked_hi = right
    if cutlass.const_expr(mask_flags & MASK_PADDED):
        unaligned = (seq_kv_len % cutlass.Int32(tile_n)) != cutlass.Int32(0)
        lo_pad = cutlass.Int32(
            arith.select(
                unaligned.ir_value(),
                (right - cutlass.Int32(1)).ir_value(),
                right.ir_value(),
            )
        )
        unmasked_hi = cute.math.min(unmasked_hi, lo_pad)
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        lo_caus = (q_row_coord + causal_diag) // cutlass.Int32(tile_n)
        unmasked_hi = cute.math.min(unmasked_hi, lo_caus)
    unmasked_hi = cute.math.max(unmasked_hi, left)

    unmasked_lo = left
    if cutlass.const_expr(mask_flags & MASK_SWA):
        anchor = q_row_coord + cutlass.Int32(cga_tile_m - 1 - swa_window)
        swa_unmasked_lo = _div_up(anchor, tile_n)
        cond = anchor > cutlass.Int32(0)
        swa_unmasked_lo = cutlass.Int32(
            arith.select(
                cond.ir_value(),
                swa_unmasked_lo.ir_value(),
                cutlass.Int32(0).ir_value(),
            )
        )
        unmasked_lo = cute.math.max(unmasked_lo, swa_unmasked_lo)

    unmasked_lo = cute.math.min(unmasked_lo, unmasked_hi)

    return KvLoopBounds(
        left=left,
        unmasked_lo=unmasked_lo,
        unmasked_hi=unmasked_hi,
        right=right,
    )


@cute.jit
def decode_linear_tile_lpt(linear, q_h, batch, q_tiles):
    hb = q_h * batch
    row_rank = linear // hb
    within = linear % hb
    row = (q_tiles - cutlass.Int32(1)) - row_rank
    head = within % q_h
    batch_idx = within // q_h
    return row, head, batch_idx


class SdpaHelpers(NamedTuple):
    decode_initial: object
    decode_payload: object
    bounds_for_tile: object
    resolve_seqlen_kv: object
    thd_decode: object
    dispatch_decode_initial: object
    dispatch_decode_payload: object
    thd_tma_offsets: object
    thd_sf_tile_bases: object


def make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units: bool = False) -> SdpaHelpers:
    cga_tile_m = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

    if CFG.SCHEDULER_POLICY == SCHED_NATURAL:
        if CFG.SPLIT_PIPELINE == 1:

            @cute.jit
            def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch):
                blocked_row = (bidx // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
                return blocked_row, bidy, bidz

            @cute.jit
            def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch):
                blocked_row = (t0 // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
                head = t1 & cutlass.Int32(0xFFFF)
                batch = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
                return blocked_row, head, batch

        else:

            @cute.jit
            def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch):
                return bidx, bidy, bidz

            @cute.jit
            def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch):
                head = t1 & cutlass.Int32(0xFFFF)
                batch = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
                return t0 + cta_in_pair, head, batch

    else:

        @cute.jit
        def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch):
            linear = bidx // cutlass.Int32(CFG.CGA_M)
            q_tiles = n_q_supers // cutlass.Int32(CFG.CTA_MMA) if lpt_q_tiles_in_cga_units else n_q_supers
            row, head, batch = decode_linear_tile_lpt(linear, n_qh, n_batch, q_tiles)
            return row * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair, head, batch

        @cute.jit
        def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch):
            linear = t0 // cutlass.Int32(CFG.CGA_M)
            q_tiles = n_q_supers // cutlass.Int32(CFG.CTA_MMA) if lpt_q_tiles_in_cga_units else n_q_supers
            row, head, batch = decode_linear_tile_lpt(linear, n_qh, n_batch, q_tiles)
            return row * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair, head, batch

    @cute.jit
    def _bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair):
        cga_base_super = q_super_idx - cta_in_pair
        q_row_coord = cga_base_super * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
        return compute_kv_loop_bounds(
            q_row_coord,
            seqlen_q,
            seqlen_kv,
            CFG.SWA_WINDOW,
            CFG.MASK_FLAGS,
            CFG.TILE_N,
            cga_tile_m,
            causal_bottom_right=bool(CFG.CAUSAL_BOTTOM_RIGHT),
        )

    @cute.jit
    def _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, scalar_seqlen_kv):
        if cutlass.const_expr(CFG.SEQ_KV_LENS_PRESENT == 1):
            arr = cutlass.make_array_view(seq_kv_lens_tensor)
            return cutlass.Int32(arr[batch_idx])
        return scalar_seqlen_kv

    _thd_on = int(getattr(CFG, "THD_VARLEN", 0))

    @cute.jit
    def _thd_decode(linear_cta, seq_kv_lens_t, n_batch, n_qh, cta_in_pair):
        u = linear_cta // cutlass.Int32(CFG.CGA_M)
        cu = cutlass.make_array_view(seq_kv_lens_t)
        cuq0 = n_batch
        acc = cutlass.Int32(0)
        f_batch = cutlass.Int32(0)
        f_head = cutlass.Int32(0)
        f_qc = cutlass.Int32(0)
        done = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            s_i = cutlass.Int32(cu[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[cuq0 + b])
            cb = (s_i + cutlass.Int32(cga_tile_m - 1)) // cutlass.Int32(cga_tile_m)
            units_b = cb * n_qh
            in_rng = (done == cutlass.Int32(0)) & (u < acc + units_b)
            local = u - acc
            f_batch = cutlass.Int32(arith.select(in_rng.ir_value(), b.ir_value(), f_batch.ir_value()))
            f_head = cutlass.Int32(arith.select(in_rng.ir_value(), (local // cb).ir_value(), f_head.ir_value()))
            f_qc = cutlass.Int32(arith.select(in_rng.ir_value(), (local % cb).ir_value(), f_qc.ir_value()))
            done = cutlass.Int32(arith.select(in_rng.ir_value(), cutlass.Int32(1).ir_value(), done.ir_value()))
            acc = acc + units_b
        q_super = f_qc * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
        return q_super, f_head, f_batch

    @cute.jit
    def _dispatch_decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t):
        if cutlass.const_expr(_thd_on):
            return _thd_decode(bidx, seq_kv_lens_t, n_batch, n_qh, cta_in_pair)
        return _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch)

    @cute.jit
    def _dispatch_decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t):
        if cutlass.const_expr(_thd_on):
            return _thd_decode(t0, seq_kv_lens_t, n_batch, n_qh, cta_in_pair)
        return _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch)

    @cute.jit
    def _thd_tma_offsets(seq_kv_lens_t, batch_idx, n_batch):
        if cutlass.const_expr(_thd_on):
            cu = cutlass.make_array_view(seq_kv_lens_t)
            q_off = cutlass.Int32(cu[n_batch + batch_idx])
            kv_off = cutlass.Int32(cu[cutlass.Int32(2) * n_batch + cutlass.Int32(1) + batch_idx])
            return q_off, kv_off, cutlass.Int32(0)
        return cutlass.Int32(0), cutlass.Int32(0), batch_idx

    @cute.jit
    def _thd_sf_tile_bases(seq_kv_lens_t, batch_idx, n_batch):
        if cutlass.const_expr(_thd_on):
            cu = cutlass.make_array_view(seq_kv_lens_t)
            q0 = n_batch
            k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
            sfq = cutlass.Int32(0)
            sfk = cutlass.Int32(0)
            for b in cutlass.range(0, batch_idx, 1, unroll=1):
                s_q = cutlass.Int32(cu[q0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[q0 + b])
                s_kv = cutlass.Int32(cu[k0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[k0 + b])
                sfq = sfq + (s_q + cutlass.Int32(CFG.TILE_M - 1)) // cutlass.Int32(CFG.TILE_M)
                sfk = sfk + (s_kv + cutlass.Int32(CFG.TILE_N - 1)) // cutlass.Int32(CFG.TILE_N)
            return cute.arch.make_warp_uniform(sfq), cute.arch.make_warp_uniform(sfk)
        return cutlass.Int32(0), cutlass.Int32(0)

    return SdpaHelpers(
        decode_initial=_decode_initial,
        decode_payload=_decode_payload,
        bounds_for_tile=_bounds_for_tile,
        resolve_seqlen_kv=_resolve_seqlen_kv,
        thd_decode=_thd_decode,
        dispatch_decode_initial=_dispatch_decode_initial,
        dispatch_decode_payload=_dispatch_decode_payload,
        thd_tma_offsets=_thd_tma_offsets,
        thd_sf_tile_bases=_thd_sf_tile_bases,
    )
