"""Example 13: multi-GEMM — DualMatmulSiluMulDequant (pure cuDNN frontend API).

Two parallel GEMMs share the A operand and feed one fused epilogue:

    C0 = A @ B0
    C1 = A @ B1
    out = silu(C0) * C1 * scale        # SwiGLU-style gated FFN block

Both GEMMs share the same shape / layout; only B differs. The kernel loads A
once (deduped) and runs two ``tcgen05`` MMAs into two TMEM accumulators, then the
epilogue reads both into the shared op chain. Multi-GEMM is implemented in the
1-CTA-MMA CLC template, so this passes ``cta_group=1``.
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

    C0 = g.matmul(A=A, B=B0, name="mm0")  # gemm 0: A @ B0
    C1 = g.matmul(A=A, B=B1, name="mm1")  # gemm 1: A @ B1 (shares A)
    S0 = g.swish(input=C0, name="silu")  # silu(C0)
    MU = g.mul(a=S0, b=C1, name="mul")  # silu(C0) * C1
    DQ = g.mul(a=MU, b=scale, name="dequant")  # * scale
    DQ.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    compiled = jit_from_cudnn_graph(g, cta_group=1)
    print(f"[13] {compiled.chain.summary()}")
    print(f"[13] num_gemms={compiled.chain.num_gemms} " f"distinct A={compiled.chain.num_a_operands} B={compiled.chain.num_b_operands}")
    print(f"[13] generated kernel: {compiled.generated_path}")

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b0 = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b1 = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    scale_t = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)

    # Multi-GEMM call: a list of per-GEMM (a, b) pairs. A is shared, so it
    # appears in both pairs; the kernel loads it once.
    compiled({A: a, B0: b0, B1: b1, scale: scale_t, DQ: c})
    torch.cuda.synchronize()

    mm0 = torch.einsum("bmk,bnk->bmn", a.float(), b0.float())
    mm1 = torch.einsum("bmk,bnk->bmn", a.float(), b1.float())
    ref = (torch.nn.functional.silu(mm0) * mm1 * 0.5).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[13] PASS  M={M} N={N} K={K}")


if __name__ == "__main__":
    main()
