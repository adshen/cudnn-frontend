"""Example 06: FP8 (E4M3) matmul → FP16 output.

Demonstrates an FP8-input GEMM path:
  - inputs A, B are FP8 E4M3 (1 byte/elem)
  - accumulator is FP32 (sm100 tcgen05 F8F6F4 supports fp32 accum)
  - output C is FP16 (user downcasts via Y.set_data_type(HALF))

Because TileConfig stores K in **bytes** (SWIZZLE_128B → K_BYTES=128), the
same config works for BF16/FP16 inputs (64 K-elements per tile) and FP8
inputs (128 K-elements per tile). No FP8-specific tile catalog is needed.

Usage:

    source active_tbd.sh
    python cudnn.TBD.gemm/examples/06_matmul_fp8.py
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.tile_config import CONFIG_sm100_128x128x128_128x128x32_cluster1x1


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
    # Downcast FP32 accumulator to FP16 for the materialized output. Without
    # this the chain would try to store FP8 (rarely useful, no precision).
    C.set_data_type(cudnn.data_type.HALF)

    compiled = jit_from_cudnn_graph(g, config=CONFIG_sm100_128x128x128_128x128x32_cluster1x1, cta_group=1)
    print(f"[06] {compiled.chain.summary()}")
    print(f"[06] generated kernel: {compiled.generated_path}")

    torch.manual_seed(0)
    # Small-integer FP8 values to keep the FP32 reference bit-exact comparable.
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
