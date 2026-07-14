# SPDX-License-Identifier: LicenseRef-NvidiaProprietary


from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute


@cute.jit
def tmem_alloc(tmem_ptr_i32, num_cols: int, cta_group_kind, is_exclusive: bool = True):
    nvvm.tcgen05_alloc(
        tmem_ptr_i32,
        cutlass.Int32(num_cols),
        is_exclusive=is_exclusive,
        group=cta_group_kind,
    )
    nvvm.tcgen05_relinquish_alloc_permit(group=cta_group_kind)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def tmem_dealloc(tmem_ptr_i32, num_cols: int, cta_group_kind):
    tmem_ptr_for_dealloc = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    nvvm.tcgen05_dealloc(
        tmem_ptr_for_dealloc,
        cutlass.Int32(num_cols),
        group=cta_group_kind,
    )
