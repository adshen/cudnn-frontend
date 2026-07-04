"""Example 10: INT8 matmul -> FP32 output (integer tensor-core MMA).

Signed INT8 inputs, INT32 accumulate (compute_data_type MUST be INT32 for INT8
— analyzer rejects otherwise), FP32 output (int32 widened in the epilogue). Runs
on SM 100 or SM 110 only. Output may also be BF16/FP16/FP8/INT32 (INT32 stays
exact past 2**24). Same dtype-agnostic tile configs as FP8.
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

    torch.manual_seed(0)
    a = torch.randint(-40, 40, (1, M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-40, 40, (1, N, K), dtype=torch.int8, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.float32, device="cuda")

    compiled({A: a, B: b, C: c})
    torch.cuda.synchronize()

    # fp32 reference is exact: |acc| <= K*40*40 = 409600 fits exactly in fp32.
    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float())
    torch.testing.assert_close(c, ref, atol=0, rtol=0)
    print(f"[10] PASS  M={M} N={N} K={K}  (bit-exact int8xint8->int32->fp32)")


if __name__ == "__main__":
    main()
