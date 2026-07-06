"""Example 06: FP8 (E4M3) matmul -> FP16 output.

FP8 E4M3 inputs, FP32 accumulate, FP16 output. K is stored in bytes, so the
same tile config serves BF16/FP16/FP8 — no FP8-specific catalog entry.
"""

from __future__ import annotations

import cudnn
import cudnn.frost.gemm  # noqa: F401
import torch

from cudnn.frost.gemm.compiler import jit_from_cudnn_graph
from cudnn.frost.gemm.tile_config import CONFIG_sm100_128x128x128_128x128x32_cluster1x1


def main(M: int = 256, N: int = 256, K: int = 256) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    # Downcast FP32 accumulator to FP16 output (else the chain stores FP8).
    C.set_data_type(cudnn.data_type.HALF)

    compiled = jit_from_cudnn_graph(g, config=CONFIG_sm100_128x128x128_128x128x32_cluster1x1, cta_group=1)

    torch.manual_seed(0)
    # Small-integer FP8 values keep the FP32 reference bit-exact comparable.
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.float16, device="cuda")

    compiled({A: a, B: b, C: c})
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[06] PASS  M={M} N={N} K={K}")


if __name__ == "__main__":
    main()
