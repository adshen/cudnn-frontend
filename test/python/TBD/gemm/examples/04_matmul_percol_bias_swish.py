"""Example 04: matmul -> bias (per-col) -> swish (pure cuDNN frontend API)."""

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
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1])
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="b")
    Y = g.swish(input=Cb, name="sw")
    Y.set_output(True)

    compiled = jit_from_cudnn_graph(g)
    print(f"[04] {compiled.chain.summary()}")
    print(f"[04] generated kernel: {compiled.generated_path}")

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    bias_t = torch.randn(1, 1, N, device="cuda", dtype=torch.bfloat16)

    compiled({A: a, B: b, bias: bias_t, Y: c})
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref = torch.nn.functional.silu(mm + bias_t.to(torch.float32)).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[04] PASS  M={M} N={N} K={K}")


if __name__ == "__main__":
    main()
