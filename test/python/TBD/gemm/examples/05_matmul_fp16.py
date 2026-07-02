"""Example 05: FP16 matmul + relu.

Same path as 02_matmul_relu.py, but the cudnn graph is built with
io_data_type=HALF so all inputs and the output are FP16. Exercises the
dtype-injection path through TileConfig rendering — the generated kernel
should have `mma_a_dtype = cutlass.Float16` and friends.

Usage:

    source active_tbd.sh
    python cudnn.TBD.gemm/examples/05_matmul_fp16.py
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (registers TBD_eng0 + installs hook)
import torch


def main(M: int = 256, N: int = 256, K: int = 128) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.select_engines(["TBD_eng0"])
    g.check_support()
    g.build_plans()

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.float16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.float16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.float16, device="cuda")

    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, Y: c}, workspace)
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref = torch.relu(ref).to(torch.float16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[05] PASS  M={M} N={N} K={K}")


if __name__ == "__main__":
    main()
