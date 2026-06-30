"""Example 03: matmul -> bias (per-row) -> gelu_approx_tanh (pure cuDNN frontend API)."""

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
    bias = g.tensor(name="bias", dim=[1, M, 1], stride=[M, 1, 1])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    Cb = g.bias(input=C, bias=bias, name="b")
    Y = g.gelu_approx_tanh(input=Cb, name="g")
    Y.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = jit_from_cudnn_graph(g)
    print(f"[03] {compiled.chain.summary()}")
    print(f"[03] generated kernel: {compiled.generated_path}")

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    # Two GMEM outputs (chain.outputs order):
    #   slot 0 = terminal (Y, FP32 after gelu)
    #   slot 1 = matmul tap (C, BF16 raw matmul output)
    c_term = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    bias_t = torch.randn(1, M, 1, device="cuda", dtype=torch.bfloat16)

    compiled({A: a, B: b, bias: bias_t, Y: c_term, C: c_tap})
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_term = torch.nn.functional.gelu(mm + bias_t.to(torch.float32), approximate="tanh")
    ref_tap = mm.to(torch.bfloat16)
    torch.testing.assert_close(c_term, ref_term, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_tap, ref_tap, atol=1e-1, rtol=1e-2)
    print(f"[03] PASS  M={M} N={N} K={K}")


if __name__ == "__main__":
    main()
