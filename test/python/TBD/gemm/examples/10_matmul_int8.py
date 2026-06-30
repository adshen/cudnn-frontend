"""Example 10: INT8 matmul → FP32 output (integer tensor-core MMA).

Demonstrates the integer GEMM path:
  - inputs A, B are signed INT8 (1 byte/elem)
  - accumulator is INT32 (sm100 tcgen05 `.kind::i8`)
  - output C is FP32 (the int32 accumulator is widened to fp32 in the epilogue)

The matmul's `compute_data_type` must be INT32 for INT8 inputs (the analyzer
rejects anything else). Like FP8, K is 128 elements per tile for K_BYTES=128, so
the same dtype-agnostic tile configs apply — no INT8-specific catalog entry.

Output dtype can be FP32 (shown here), BF16, FP16, FP8, or INT32 (raw
accumulator — passed through without an fp32 round-trip so it stays exact past
2**24). INT8 runs on SM 100 or SM 110 only.

Usage:

    source active_tbd.sh
    python cudnn.TBD.gemm/examples/10_matmul_int8.py
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.tile_config import CONFIG_sm100_128x128x128_128x128x32_cluster1x1


def main(M: int = 256, N: int = 256, K: int = 256) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.INT8,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.INT32,  # int32 accumulate
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    C.set_data_type(cudnn.data_type.FLOAT)  # widen int32 acc -> fp32 out

    compiled = jit_from_cudnn_graph(g, config=CONFIG_sm100_128x128x128_128x128x32_cluster1x1, cta_group=1)
    print(f"[10] {compiled.chain.summary()}")
    print(f"[10] generated kernel: {compiled.generated_path}")

    torch.manual_seed(0)
    a = torch.randint(-40, 40, (1, M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-40, 40, (1, N, K), dtype=torch.int8, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.float32, device="cuda")

    compiled({A: a, B: b, C: c})
    torch.cuda.synchronize()

    # fp32 reference is exact: |acc| <= K*40*40 = 409600, an integer that fits
    # exactly in fp32.
    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    torch.testing.assert_close(c, ref, atol=0, rtol=0)
    print(f"[10] PASS  M={M} N={N} K={K}  (bit-exact int8xint8->int32->fp32)")


if __name__ == "__main__":
    main()
