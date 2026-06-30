"""Example 15: mixed-input mainloop — a fused operand LOADED at a narrower
dtype than the MMA reads.

`after = identity(A_int8)` with `after.set_data_type(bf16)` feeding the matmul
is detected as mainloop fusion whose LOAD dtype (int8) differs from the MMA
dtype (bf16). The analyzer records `matmul.a_dtype = bf16` (what tcgen05.mma
reads) and `mainloop_a_load_dtype = int8` (the TMA load). The mainloop warps
stage the widen: TMA loads the int8 tile into a narrow SMEM buffer, the warps
cast it (via the op chain) into the wide bf16 MMA SMEM tile before the MMA.

This is the int8-weight / bf16-activation mixed GEMM (`int8 A @ bf16 B`). The
cast is `identity`, so it is lossless; any unary op chain works the same way
(e.g. `(A_int8 * scale) -> bf16`). Both operands can be cast independently.
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (installs hook)
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph


def _run(M: int, N: int, K: int) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # A is stored as int8; identity casts it to bf16 before the MMA.
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.INT8)
    Ai = g.identity(input=A, name="pw_in_mainloop0")
    Ai.set_data_type(cudnn.data_type.BFLOAT16)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])  # bf16
    C = g.matmul(A=Ai, B=B, name="mm")
    C.set_output(True)

    compiled = jit_from_cudnn_graph(g)  # picks a mainloop template
    print(f"[15] {compiled.chain.summary()}")
    assert compiled.chain.mainloop_a_cast
    assert compiled.chain.matmul.a_dtype == "bf16"
    assert compiled.chain.mainloop_a_load_dtype == "int8"

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-4, 4).to(torch.int8).cuda()
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-4, 4).to(torch.bfloat16).cuda()
    c = torch.empty(1, M, N, dtype=torch.bfloat16).cuda()

    compiled({A: a, B: b, C: c})
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[15] PASS  int8(A) -> bf16 @ bf16(B)  M={M} N={N} K={K}")


def main() -> None:
    _run(512, 512, 512)
    _run(2048, 2048, 2048)


if __name__ == "__main__":
    main()
