"""sm100 CTA_1 MoE grouped matmul fwd: single-CTA MMA + grouped persistent scheduler.

Per routed group g, ``out[first_token_offset[g] : first_token_offset[g+1]] =
token[range] @ weight[g % num_experts].T`` (token = A ``(1, S, K)``, weight = B
``(num_experts, K, N)``, out ``(1, S, N)``). A fixed cluster count is launched and
strides through tiles; each tile maps to a routed group via a warp-shuffle search
over ``first_token_offset``. A's TMA descriptor is patched in SMEM per routed-group
change. STG epilogue only.

Warp layout (8 warps × 32 = 256 threads/CTA):
  warps 0–3 : epilogue (warp 0 also allocates TMEM)  — setmaxnreg.inc 216
  warp  4   : MMA driver (reads is_valid from the ring)  — setmaxnreg.dec 40
  warp  5   : TMA producer (ring-consume + A descriptor patch + coord_expert as B's batch coord)  — setmaxnreg.dec 40
  warp  6   : grouped persistent scheduler (per-group range search → SMEM ring)  — setmaxnreg.dec 40
  warp  7   : unused donor — setmaxnreg.dec 40, idle to dealloc barrier
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch

import cutlass.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor

_TENSOR_MAP_QWORDS = 16


@cute.jit
def _copy_tensormap_to_workspace(src_desc_ptr, dst_i64_ptr) -> None:
    """Copy the 128-byte A tensormap into ``dst_i64_ptr`` (seeds the SMEM copy)."""
    src_words = cute.make_ptr(cutlass.Int64, src_desc_ptr.toint(), mem_space=cute.AddressSpace.generic)
    for i in cutlass.range_constexpr(_TENSOR_MAP_QWORDS):
        (dst_i64_ptr + i).store((src_words + i).load())


@cute.jit
def _replace_tensormap_global_address(desc_ptr, new_address) -> None:
    nvvm.tensormap_replace(
        nvvm.TensormapField.GLOBAL_ADDRESS,
        desc_ptr,
        new_value=cutlass.Int64(new_address),
    )


@cute.jit
def _replace_tensormap_global_dim_1(desc_ptr, new_dim) -> None:
    nvvm.tensormap_replace(
        nvvm.TensormapField.GLOBAL_DIM,
        desc_ptr,
        new_value=cutlass.Int32(new_dim),
        ord=1,
    )


@cute.jit
def _fence_tensormap_release() -> None:
    nvvm.fence_proxy_release(
        nvvm.MemScope.GPU,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )


@cute.jit
def _fence_tensormap_acquire(desc_ptr) -> None:
    nvvm.fence_proxy_acquire(
        nvvm.MemScope.GPU,
        desc_ptr,
        _TENSOR_MAP_QWORDS * 8,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )


# @@INJECT_TILE_CONSTANTS@@


SCHED_STAGES = 2
SCHED_SLOT_WORDS = 8

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path
EPI_SMEM_STAGES = 2

# Named barrier id for cross-warp sync of the 4 epilogue warps
EPI_SYNC_BAR_ID = 1

# Named barrier id for the TMEM-alloc handoff
TMEM_ALLOC_BARRIER_ID = 2


@cute.kernel
def _kernel(
    # @@INJECT_KERNEL_AB_DESC_PARAMS@@
    tma_c_desc: cutlass.GridConstant[_tma.TensorMap],
    mC_mnl: cute.Tensor,
    # @@INJECT_MOE_KERNEL_MA_PARAMS@@
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    # @@INJECT_KERNEL_TAP_PARAMS@@
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    num_experts: cutlass.Int32,
    num_groups: cutlass.Int32,
    a_stride_m: cutlass.Int64,
    c_stride_m: cutlass.Int64,
    # @@INJECT_KERNEL_REDUCTION_STRIDE_PARAMS@@
    # @@INJECT_KERNEL_AUX_PARAMS@@
) -> None:
    # @@INJECT_AB_DESC_LISTS@@
    # @@INJECT_MOE_MA_LIST@@

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
    num_epilogue_warps = 4
    epi_reg_count = 232
    prod_reg_count = 24

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]
    gridx = cute.arch.grid_dim()[0]
    gridy = cute.arch.grid_dim()[1]

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    m_rank = cta_rank_in_cluster % cluster_m
    n_rank = cta_rank_in_cluster // cluster_m

    cluster_linear_init = bidx // cluster_m

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

    a_pattern = 0
    for n_idx in cutlass.range_constexpr(cluster_n):
        a_pattern = a_pattern | (1 << (n_idx * cluster_m))
    b_pattern = (1 << cluster_m) - 1

    if cutlass.const_expr(multicast_a):
        tma_mcast_mask_a = cutlass.Int16(a_pattern) << m_rank
    else:
        tma_mcast_mask_a = cutlass.Int16(1) << cta_rank_in_cluster
    if cutlass.const_expr(multicast_b):
        tma_mcast_mask_b = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
    else:
        tma_mcast_mask_b = cutlass.Int16(1) << cta_rank_in_cluster

    a_part_arrive = cutlass.Int16(a_pattern) << m_rank
    b_part_arrive = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
    ab_empty_arrive_mask = a_part_arrive | b_part_arrive

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    sched_storage = cutlass.Array(
        cutlass.Int32,
        SCHED_STAGES * SCHED_SLOT_WORDS,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    sched_full_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sched_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)

    sA_elems = cta_tile_mnk[0] * cta_tile_mnk[2]
    sB_elems = cta_tile_mnk[1] * cta_tile_mnk[2]
    smem_a_list = [
        cutlass.Array(
            ab_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            ab_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    tma_a_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            _TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_a_operands)
    ]

    # @@TMA_STORE_ONLY:BEGIN@@
    epi_subtile_elems = cta_tile_mnk[0] * epi_tile_mn[1]
    smem_d_ptr = cutlass.Array(
        cd_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    # @@TMA_STORE_ONLY:END@@

    ab_empty_count = cluster_m + cluster_n - 1
    sched_empty_count = 1 + 1 + num_epilogue_warps
    if warp_idx == 0:
        if nvvm.elect_sync():
            for i in range(ab_stages):
                nvvm.mbarrier_init(ab_full_mbar_ptr + i, 1)
                nvvm.mbarrier_init(ab_empty_mbar_ptr + i, ab_empty_count)
            for i in range(acc_stages):
                nvvm.mbarrier_init(acc_full_mbar_ptr + i, 1)
                nvvm.mbarrier_init(acc_empty_mbar_ptr + i, num_epilogue_warps)
            for i in range(SCHED_STAGES):
                nvvm.mbarrier_init(sched_full_mbar_ptr + i, 1)
                nvvm.mbarrier_init(sched_empty_mbar_ptr + i, sched_empty_count)
    nvvm.fence_mbarrier_init()
    if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
        nvvm.barrier_cluster_arrive_relaxed()
        nvvm.barrier_cluster_wait()
    else:
        nvvm.barrier_cta_sync(0)

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    num_tma_copy_bytes = num_a_operands * sA_bytes + num_b_operands * sB_bytes

    idesc = cutlass.primitives.Tcgen05InstrDesc.build(
        a_dtype=mma_a_dtype,
        b_dtype=mma_b_dtype,
        c_dtype=mma_c_dtype,
        n_dim=cta_tile_mnk[1],
        m_dim=cta_tile_mnk[0],
        a_major=mma_a_major,
        b_major=mma_b_major,
    )

    cols_per_acc_stage = cta_tile_mnk[1]
    acc_region_cols = num_gemms * cols_per_acc_stage
    num_tmem_cols = acc_stages * acc_region_cols
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    gC_ptr = mC_mnl.iterator.raw_ptr()

    # @@INJECT_TAP_PTRS@@

    VEC_BYTES = vec_bytes_epi
    vsize = (VEC_BYTES * 8) // mC_mnl.dtype.width

    M = m
    N = n
    clusters_along_n = cute.ceil_div(cutlass.Int32(N), cgrp_tile_mnk[1])
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    num_k_blocks = cta_tile_mnk[2] // mma_inst_shape_mnk[2]
    first_token_arr = cutlass.make_array_view(first_token_offset)

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        gemm_s = cutlass.Int32(M)
        sched_stage = cutlass.Int32(0)
        sched_empty_phase = cutlass.Int32(1)
        linear_idx = cutlass.Int32(cluster_linear_init)
        start_linear_idx = cutlass.Int32(0)
        total_tiles = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)
        cached_next_end = cutlass.Int32(0)
        if lane + 1 < num_groups:
            cached_next_end = cutlass.Int32(first_token_arr[lane + 1])
        else:
            cached_next_end = gemm_s
        tile_lower_bound = nvvm.shfl_sync(full_warp_mask, cached_next_end, 1, shfl_up_clamp, nvvm.Shfl.UP)
        cached_next_begin = cutlass.Int32(0)
        if lane != 0:
            cached_next_begin = tile_lower_bound

        while is_tile_valid != 0:
            group_begin = cached_next_begin
            group_end = cached_next_end

            if linear_idx >= start_linear_idx + total_tiles:
                group_idx += lane
                is_search_live = cutlass.Int32(1)
                while is_search_live != 0:
                    cached_group_begin = cached_next_begin
                    cached_group_end = cached_next_end
                    tile_start_idx = nvvm.shfl_sync(
                        full_warp_mask,
                        cached_next_end,
                        31,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    next_end_group = group_idx + 32 + 1
                    if next_end_group < num_groups:
                        cached_next_end = cutlass.Int32(first_token_arr[next_end_group])
                    else:
                        cached_next_end = gemm_s
                    tile_lower_bound = nvvm.shfl_sync(
                        full_warp_mask,
                        cached_next_end,
                        1,
                        shfl_up_clamp,
                        nvvm.Shfl.UP,
                    )
                    if lane != 0:
                        cached_next_begin = tile_lower_bound
                    else:
                        cached_next_begin = tile_start_idx

                    group_m = cached_group_end - cached_group_begin
                    total_tiles = cute.ceil_div(group_m, cgrp_tile_mnk[0]) * clusters_along_n
                    prefix_tiles = total_tiles
                    for delta in (1, 2, 4, 8, 16):
                        prefix_delta = nvvm.shfl_sync(
                            full_warp_mask,
                            prefix_tiles,
                            delta,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        if lane >= delta:
                            prefix_tiles += prefix_delta
                    start_linear_idx += prefix_tiles - total_tiles
                    thread_succeed = nvvm.vote_sync(
                        full_warp_mask,
                        linear_idx < start_linear_idx + total_tiles,
                        nvvm.VoteSync.BALLOT,
                    )
                    if thread_succeed != 0:
                        winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(thread_succeed)).to(cutlass.Int32)
                        group_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            group_idx,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        start_linear_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            start_linear_idx,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        total_tiles = nvvm.shfl_sync(
                            full_warp_mask,
                            total_tiles,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        tile_start_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            cached_group_begin,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        group_end_idx = group_idx + lane + 1
                        if group_end_idx < num_groups:
                            cached_next_end = cutlass.Int32(first_token_arr[group_end_idx])
                        else:
                            cached_next_end = gemm_s
                        tile_lower_bound = nvvm.shfl_sync(
                            full_warp_mask,
                            cached_next_end,
                            1,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        if lane != 0:
                            cached_next_begin = tile_lower_bound
                        else:
                            cached_next_begin = tile_start_idx
                        group_begin = cached_next_begin
                        group_end = cached_next_end
                        is_search_live = cutlass.Int32(0)
                    else:
                        group_idx += 32
                        first_lane_group = nvvm.shfl_sync(
                            full_warp_mask,
                            group_idx,
                            0,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        if first_lane_group >= num_groups:
                            is_tile_valid = cutlass.Int32(0)
                            is_search_live = cutlass.Int32(0)
                        else:
                            next_start_linear_idx = start_linear_idx + total_tiles
                            start_linear_idx = nvvm.shfl_sync(
                                full_warp_mask,
                                next_start_linear_idx,
                                31,
                                shfl_idx_clamp,
                                nvvm.Shfl.IDX,
                            )

            coord_expert = cutlass.Int32(0)
            cluster_tile_m = cutlass.Int32(0)
            coord_n = cutlass.Int32(0)
            if is_tile_valid != 0:
                local_linear_idx = linear_idx - start_linear_idx
                cluster_tile_m = local_linear_idx // clusters_along_n
                coord_n = local_linear_idx % clusters_along_n
                coord_expert = group_idx % num_experts
                linear_idx += grid_num_clusters

            while not nvvm.mbarrier_try_wait_parity(
                sched_empty_mbar_ptr + sched_stage,
                sched_empty_phase,
                time_limit=10_000_000,
            ):
                pass
            if lane == 0:
                slot = sched_storage + sched_stage * SCHED_SLOT_WORDS
                (slot + 0).store(coord_expert)
                (slot + 1).store(cluster_tile_m)
                (slot + 2).store(coord_n)
                (slot + 3).store(is_tile_valid)
                (slot + 4).store(group_begin)
                (slot + 5).store(group_end)
                (slot + 7).store(group_idx)
                nvvm.mbarrier_arrive(sched_full_mbar_ptr + sched_stage)

            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_empty_phase = sched_empty_phase ^ 1

    if warp_idx == tma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            if nvvm.elect_sync():
                nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        elect_one = nvvm.elect_sync()

        lane = tidx % 32
        block_linear = bidx + bidy * gridx
        cta_desc_base_list = [a_tma_workspace.iterator.raw_ptr() + (block_linear * num_a_operands + _ai) * _TENSOR_MAP_QWORDS for _ai in range(num_a_operands)]
        a_desc_tma_ptr_list = [
            cute.make_ptr(
                cutlass.Int64,
                cta_desc_base_list[_ai].toint(),
                mem_space=cute.AddressSpace.generic,
            )
            for _ai in range(num_a_operands)
        ]
        previous_group_begin = cutlass.Int32(-1)
        if elect_one:
            for _ai in cutlass.range_constexpr(num_a_operands):
                _copy_tensormap_to_workspace(tma_a_descs[_ai].get_ptr(), tma_a_desc_smem_list[_ai])
        nvvm.bar_warp_sync(0xFFFFFFFF)

        while is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr + sched_stage,
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            slot = sched_storage + sched_stage * SCHED_SLOT_WORDS
            coord_expert = (slot + 0).load()
            tile_m = (slot + 1).load()
            tile_n = (slot + 2).load()
            is_valid = (slot + 3).load()
            group_begin = (slot + 4).load()
            group_end = (slot + 5).load()
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr + sched_stage)
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_group = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                coord_n_per_cta = tile_n * cgrp_tile_mnk[1] + n_rank * cta_tile_mnk[1]

                if group_begin != previous_group_begin:
                    previous_group_begin = group_begin
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        _fence_tensormap_acquire(a_desc_tma_ptr_list[_ai])
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        if elect_one:
                            row_base = mA_list[_ai].iterator.raw_ptr() + group_begin * a_stride_m
                            _replace_tensormap_global_address(tma_a_desc_smem_list[_ai], row_base.toint())
                            _replace_tensormap_global_dim_1(tma_a_desc_smem_list[_ai], group_end - group_begin)
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        if lane < _TENSOR_MAP_QWORDS:
                            (cta_desc_base_list[_ai] + lane).store((tma_a_desc_smem_list[_ai] + lane).load())
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        _fence_tensormap_release()

                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        ab_empty_mbar_ptr + stage,
                        ab_empty_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    coord_k = k_tile_idx * cta_tile_mnk[2]
                    if nvvm.elect_sync():
                        nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr + stage, num_tma_copy_bytes)

                    a_issue = (not multicast_a) or (n_rank == 0)
                    if a_issue:
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            sA_stage = smem_a_list[_ai] + sA_elems * stage
                            if nvvm.elect_sync():
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_stage,
                                    a_desc_tma_ptr_list[_ai],
                                    (coord_k, coord_m_group, cutlass.Int32(0)),
                                    ab_full_mbar_ptr + stage,
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_1,
                                )
                    b_issue = (not multicast_b) or (m_rank == 0)
                    if b_issue:
                        for _bj in cutlass.range_constexpr(num_b_operands):
                            sB_stage = smem_b_list[_bj] + sB_elems * stage
                            if nvvm.elect_sync():
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_stage,
                                    tma_b_descs[_bj].get_ptr(),
                                    (coord_k, coord_n_per_cta, coord_expert),
                                    ab_full_mbar_ptr + stage,
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=nvvm.CTAGroup.CTA_1,
                                )
                    ab_iter += 1

        tail_stage = ab_iter % ab_stages
        tail_phase = ab_empty_phase_bit
        if tail_stage == 0 and ab_iter != 0:
            tail_phase = tail_phase ^ 1
        for _ in range(ab_stages - 1):
            tail_stage = tail_stage + 1
            if tail_stage == ab_stages:
                tail_stage = cutlass.Int32(0)
                tail_phase = tail_phase ^ 1
        if nvvm.elect_sync():
            while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr + tail_stage, tail_phase, time_limit=10_000_000):
                pass

    if warp_idx == mma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.tcgen05_alloc(tmem_ptr_i32, num_tmem_cols, group=nvvm.CTAGroup.CTA_1)
        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.barrier_cta_arrive(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        ab_full_phase_bit = cutlass.Int32(0)
        ab_iter = cutlass.Int32(0)
        acc_empty_phase_bit = cutlass.Int32(1)
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        acc_stage = cutlass.Int32(0)
        while is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr + sched_stage,
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            is_valid = (sched_storage + sched_stage * SCHED_SLOT_WORDS + 3).load()
            if nvvm.elect_sync():
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr + sched_stage)
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                acc_stage = tile_iter % acc_stages
                if acc_stage == 0 and tile_iter != 0:
                    acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr + acc_stage,
                    acc_empty_phase_bit,
                    time_limit=10_000_000,
                ):
                    pass

                acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                tmem_addr_gemms = [
                    cutlass.inttoptr(
                        (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage),
                        6,
                        cutlass.Int32,
                    )
                    for g in range(num_gemms)
                ]

                scale_d = cutlass.Boolean(False)
                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_full_phase_bit = ab_full_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        ab_full_mbar_ptr + stage,
                        ab_full_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        for g in cutlass.range_constexpr(num_gemms):
                            sA_stage = smem_a_list[gemm_a_idx[g]] + sA_elems * stage
                            sB_stage = smem_b_list[gemm_b_idx[g]] + sB_elems * stage
                            desc_a = cutlass.primitives.Tcgen05SmemDesc.build(
                                start_address=sA_stage,
                                leading_byte_offset=a_smem_desc_leading_byte_offset,
                                stride_byte_offset=a_smem_desc_stride_byte_offset,
                                layout=ab_smem_swizzle,
                            ).advance_start_address(a_smem_k_step_bytes * k_block_idx)
                            desc_b = cutlass.primitives.Tcgen05SmemDesc.build(
                                start_address=sB_stage,
                                leading_byte_offset=b_smem_desc_leading_byte_offset,
                                stride_byte_offset=b_smem_desc_stride_byte_offset,
                                layout=ab_smem_swizzle,
                            ).advance_start_address(b_smem_k_step_bytes * k_block_idx)
                            if nvvm.elect_sync():
                                nvvm.tcgen05_mma(
                                    mma_kind,
                                    nvvm.CTAGroup.CTA_1,
                                    tmem_addr_gemms[g],
                                    desc_a,
                                    desc_b,
                                    idesc,
                                    scale_d,
                                )
                        scale_d = cutlass.Boolean(True)

                    if nvvm.elect_sync():
                        nvvm.tcgen05_commit(
                            ab_empty_mbar_ptr + stage,
                            multicast_mask=ab_empty_arrive_mask,
                            group=nvvm.CTAGroup.CTA_1,
                        )
                    ab_iter += 1

                if nvvm.elect_sync():
                    nvvm.tcgen05_commit(
                        acc_full_mbar_ptr + acc_stage,
                        group=nvvm.CTAGroup.CTA_1,
                    )
                tile_iter += 1

        if cutlass.const_expr(USE_PDL):
            if nvvm.elect_sync():
                nvvm.griddepcontrol("launch_dependents")

        if tile_iter != 0:
            tail_stage = acc_stage
            tail_phase = acc_empty_phase_bit
            if nvvm.elect_sync():
                for _ in range(acc_stages):
                    tail_stage = tail_stage + 1
                    if tail_stage == acc_stages:
                        tail_stage = cutlass.Int32(0)
                        tail_phase = tail_phase ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        acc_empty_mbar_ptr + tail_stage,
                        tail_phase,
                        time_limit=10_000_000,
                    ):
                        pass

        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
        nvvm.tcgen05_dealloc(alloc_ptr, num_tmem_cols, group=nvvm.CTAGroup.CTA_1)

    if warp_idx < num_epilogue_warps:
        nvvm.setmaxregister(epi_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        nvvm.barrier_cta_sync(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        tile_iter = cutlass.Int32(0)
        acc_full_phase_bit = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)

        if cutlass.const_expr(cta_tile_mnk[0] == 64):
            row_id_with_warp_offset = base_row_id
        else:
            row_id_with_warp_offset = base_row_id + warp_idx * 32

        subtile_cnt = cute.ceil_div(cta_tile_mnk[1], 32)
        t2r_inst_repx = epi_tile_mn[1]
        shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        lane = tidx % 32
        # @@TMA_STORE_ONLY:BEGIN@@
        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)
        # @@TMA_STORE_ONLY:END@@

        while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr + sched_stage, sched_full_phase, time_limit=10_000_000):
            pass
        _slot = sched_storage + sched_stage * SCHED_SLOT_WORDS
        tile_m = (_slot + 1).load()
        tile_n = (_slot + 2).load()
        is_valid = (_slot + 3).load()
        group_begin = (_slot + 4).load()
        group_end = (_slot + 5).load()
        group_idx = (_slot + 7).load()
        if nvvm.elect_sync():
            nvvm.mbarrier_arrive(sched_empty_mbar_ptr + sched_stage)
        sched_stage += 1
        if sched_stage == SCHED_STAGES:
            sched_stage = cutlass.Int32(0)
            sched_full_phase = sched_full_phase ^ 1

        while is_valid != 0:
            coord_m = group_begin + tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
            coord_n = tile_n * cgrp_tile_mnk[1] + n_rank * cta_tile_mnk[1]

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr + acc_stage, acc_full_phase_bit, time_limit=10_000_000):
                pass

            acc_base_col = base_col_id_root + acc_stage * acc_region_cols
            tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (acc_base_col + g * cols_per_acc_stage) for g in range(num_gemms)]

            if cutlass.const_expr(cta_tile_mnk[0] == 64):
                row = coord_m + warp_idx * 16 + lane
                row_active = lane < 16
            else:
                row = coord_m + tidx
                row_active = True

            # @@INJECT_AUX_VIEWS@@

            for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                subtile_col_offset = subtile_idx * 32
                c_rmem_vecs = []
                for g in cutlass.range_constexpr(num_gemms):
                    subtile_tmem_addr = tmem_col_addr_gemms[g] + subtile_col_offset
                    tmem = cutlass.inttoptr(subtile_tmem_addr, 6, mma_c_dtype)
                    _cv = nvvm.tcgen05_ld(shape, tmem, num=t2r_inst_repx)
                    if cutlass.const_expr(acc_widen_to_fp32):
                        _accf = _cv.to(cutlass.Float32)
                        _cv = _accf + cutlass.full_like(_accf, 0.0)
                    c_rmem_vecs.append(_cv)
                c_rmem_vec = c_rmem_vecs[0]

                if cutlass.const_expr(subtile_idx == subtile_cnt - 1):
                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    if nvvm.elect_sync():
                        nvvm.mbarrier_arrive(acc_empty_mbar_ptr + acc_stage)

                col = coord_n + subtile_col_offset

                # @@TMA_STORE_ONLY:BEGIN@@
                epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                smem_subtile_ptr = smem_d_ptr + epi_stage_idx * epi_subtile_elems
                smem_thr_ptr = smem_subtile_ptr + tidx * t2r_inst_repx

                vec_f32 = c_rmem_vec
                col_j = col
                linear_idx = row * c_stride_m + col_j

                # @@INJECT_EPILOGUE@@

                smem_thr_ptr.data_ptr().store_swizzled(vec_out, alignment=64, swizzle=cutlass.Swizzle(2, 4, 3))

                cute.arch.fence_view_async_shared()
                nvvm.barrier_cta_sync(
                    barrier_id=EPI_SYNC_BAR_ID,
                    thread_count=num_epilogue_warps * 32,
                )

                if warp_idx == 0:
                    if nvvm.elect_sync():
                        nvvm.cp_async_bulk_tensor_global_shared_cta(
                            tma_c_desc.get_ptr(),
                            smem_subtile_ptr,
                            (col, coord_m, tile_l),
                        )
                        nvvm.cp_async_bulk_commit_group()
                    nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)

                nvvm.barrier_cta_sync(
                    barrier_id=EPI_SYNC_BAR_ID,
                    thread_count=num_epilogue_warps * 32,
                )
                # @@TMA_STORE_ONLY:END@@

                # @@STG_ONLY:BEGIN@@
                if row_active and row < group_end:
                    for j in cutlass.range_constexpr(t2r_inst_repx // vsize):
                        col_j = col + j * vsize
                        if col_j + vsize <= N:
                            vec_f32 = c_rmem_vec[j * vsize : (j + 1) * vsize]

                            # @@INJECT_STG_VEC_BINDINGS@@

                            linear_idx = row * c_stride_m + col_j

                            # @@INJECT_EPILOGUE@@

                            (gC_ptr + linear_idx).store(vec_out, alignment=VEC_BYTES)
                # @@STG_ONLY:END@@

            tile_iter += 1

            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr + sched_stage,
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            _slot = sched_storage + sched_stage * SCHED_SLOT_WORDS
            tile_m = (_slot + 1).load()
            tile_n = (_slot + 2).load()
            is_valid = (_slot + 3).load()
            group_begin = (_slot + 4).load()
            group_end = (_slot + 5).load()
            group_idx = (_slot + 7).load()
            if nvvm.elect_sync():
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr + sched_stage)
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)


@cute.jit
def _host(
    # @@INJECT_HOST_AB_PARAMS@@
    c: cute.Tensor,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    # @@INJECT_HOST_TAP_PARAMS@@
    problem_size: tuple,
    # @@INJECT_HOST_AUX_PARAMS@@
) -> None:
    # @@INJECT_HOST_AB_LISTS@@
    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    num_experts = problem_size[3]
    num_groups = problem_size[4]
    a_stride_m = problem_size[5]
    a_stride_k = problem_size[6]
    a_stride_l = problem_size[7]
    b_stride_n = problem_size[8]
    b_stride_k = problem_size[9]
    b_stride_l = problem_size[10]
    c_stride_m = problem_size[11]
    c_stride_l = problem_size[13]
    # @@INJECT_HOST_REDUCTION_STRIDES@@

    tma_a_desc_list = []
    for _a_op in _a_operands:
        tma_a_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_a_op.iterator.toint(),
                dtype=ab_tma_dtype,
                global_dims=[k_sym, m, 1],
                global_strides=[
                    a_stride_m * ab_dtype.width // 128,
                    a_stride_l * ab_dtype.width // 128,
                ],
                box_dims=[cta_tile_mnk[2], cta_tile_mnk[0], 1],
                swizzle=ab_tma_swizzle,
            )
        )
    tma_b_desc_list = []
    for _b_op in _b_operands:
        tma_b_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_b_op.iterator.toint(),
                dtype=ab_tma_dtype,
                global_dims=[k_sym, n, num_experts],
                global_strides=[
                    b_stride_n * ab_dtype.width // 128,
                    b_stride_l * ab_dtype.width // 128,
                ],
                box_dims=[cta_tile_mnk[2], cta_tile_mnk[1], 1],
                swizzle=ab_tma_swizzle,
            )
        )
    tma_c_desc = _tma.create_tensor_map_tiled(
        global_address=c.iterator.toint(),
        dtype=cd_tma_dtype,
        global_dims=[n, m, 1],
        global_strides=[
            c_stride_m * cd_dtype.width // 128,
            c_stride_l * cd_dtype.width // 128,
        ],
        box_dims=[epi_tile_mn[1], cta_tile_mnk[0], 1],
        swizzle=_tma.TensorMapSwizzle.none,
    )

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_shape = (grid_num_clusters * cluster_m, cluster_n, 1)
    _kernel(
        # @@INJECT_HOST_KERNEL_DESC_PASS@@
        tma_c_desc,
        c,
        # @@INJECT_MOE_HOST_MA_PASS@@
        first_token_offset,
        a_tma_workspace,
        # @@INJECT_HOST_TAP_PASS@@
        problem_size[0],
        problem_size[1],
        problem_size[2],
        cutlass.Int32(num_experts),
        cutlass.Int32(num_groups),
        problem_size[5],
        problem_size[11],
        # @@INJECT_HOST_REDUCTION_STRIDE_PASS@@
        # @@INJECT_HOST_AUX_PASS@@
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnk,
        use_pdl=USE_PDL,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    ab_stride_elems = 16 // (ab_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    sym_k = cute.sym_int64(divisibility=cta_tile_mnk[2])
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()

    def _make_fake_a():
        return make_fake_compact_tensor(
            ab_dtype,
            (sym_m, sym_k, 1),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            ab_dtype,
            (sym_n, sym_k, sym_e),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

    fake_c = make_fake_compact_tensor(
        cd_dtype,
        (sym_m, sym_n, 1),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    fake_first_token_offset = make_fake_compact_tensor(
        offset_cutlass_dtype,
        (sym_g,),
        stride_order=(0,),
        assumed_align=offset_cutlass_dtype.width // 8,
    )
    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_ctas = grid_num_clusters * cluster_m * cluster_n
    fake_a_tma_workspace = make_fake_compact_tensor(
        cutlass.Int64,
        (grid_ctas * num_a_operands * 16,),
        stride_order=(0,),
        assumed_align=128,
    )
    sym_a_stride_m = cute.sym_int64(divisibility=ab_stride_elems)
    sym_a_stride_k = cute.sym_int64(divisibility=ab_stride_elems)
    sym_a_stride_l = cute.sym_int64(divisibility=ab_stride_elems)
    sym_b_stride_n = cute.sym_int64(divisibility=ab_stride_elems)
    sym_b_stride_k = cute.sym_int64(divisibility=ab_stride_elems)
    sym_b_stride_l = cute.sym_int64(divisibility=ab_stride_elems)
    sym_c_stride_m = cute.sym_int64(divisibility=out_vec_elems)
    sym_c_stride_n = cute.sym_int64()
    sym_c_stride_l = cute.sym_int64(divisibility=out_vec_elems)
    # @@INJECT_COMPILE_REDUCTION_STRIDE_DECLS@@
    # @@INJECT_COMPILE_AB_FAKES@@
    # @@INJECT_COMPILE_TAP_FAKES@@
    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_e,
        sym_g,
        sym_a_stride_m,
        sym_a_stride_k,
        sym_a_stride_l,
        sym_b_stride_n,
        sym_b_stride_k,
        sym_b_stride_l,
        sym_c_stride_m,
        sym_c_stride_n,
        sym_c_stride_l,
        # @@INJECT_COMPILE_REDUCTION_STRIDE_SYMBOLS@@
    )
    # @@INJECT_COMPILE_AUX_FAKES@@
    return cute.compile(
        _host,
        # @@INJECT_COMPILE_AB_PASS@@
        fake_c,
        fake_first_token_offset,
        fake_a_tma_workspace,
        # @@INJECT_COMPILE_TAP_PASS@@
        problem_size,
        # @@INJECT_COMPILE_AUX_PASS@@
    )
