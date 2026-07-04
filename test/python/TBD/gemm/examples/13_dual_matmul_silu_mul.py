"""Example 13: multi-GEMM — dual matmul + SwiGLU (pure cuDNN frontend API).

Two parallel GEMMs share the A operand and feed one fused epilogue:
    out = silu(A @ B0) * (A @ B1) * scale
A is loaded once (deduped); two tcgen05 MMAs feed two TMEM accumulators that the
shared epilogue reads. Multi-GEMM lives in the 1-CTA-MMA template (cta_group=1).
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (installs hook)
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph


def main(M: int = 256, N: int = 256, K: int = 128) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B0 = g.tensor(name="B0", dim=[1, K, N], stride=[K * N, 1, K])
    B1 = g.tensor(name="B1", dim=[1, K, N], stride=[K * N, 1, K])
    scale = g.tensor(name="scale", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)

    C0 = g.matmul(A=A, B=B0, name="mm0")
    C1 = g.matmul(A=A, B=B1, name="mm1")  # shares A
    S0 = g.swish(input=C0, name="silu")
    MU = g.mul(a=S0, b=C1, name="mul")
    DQ = g.mul(a=MU, b=scale, name="dequant")
    DQ.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    compiled = jit_from_cudnn_graph(g, cta_group=1)

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b0 = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b1 = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    scale_t = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)

    compiled({A: a, B0: b0, B1: b1, scale: scale_t, DQ: c})
    torch.cuda.synchronize()

    mm0 = torch.einsum("bmk,bnk->bmn", a.float(), b0.float())
    mm1 = torch.einsum("bmk,bnk->bmn", a.float(), b1.float())
    ref = (torch.nn.functional.silu(mm0) * mm1 * 0.5).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[13] PASS  M={M} N={N} K={K}")


if __name__ == "__main__":
    main()
