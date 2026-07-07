"""Example 02: matmul -> relu fusion (pure cuDNN frontend API)."""

from __future__ import annotations

import cudnn
import os

os.environ.setdefault("NV_CUDNN_FE_ENABLE_FROST_ENGINES", "1")

import cudnn.frost.gemm  # noqa: F401  (registers frost_gemm_eng0 + installs hook)
import torch


def main(M: int = 256, N: int = 256, K: int = 128) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
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
    g.select_engines(["frost_gemm_eng0"])
    g.check_support()
    g.build_plans()

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")

    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, Y: c}, workspace)
    torch.cuda.synchronize()

    ref = torch.relu(torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[02] PASS  M={M} N={N} K={K}")


if __name__ == "__main__":
    main()
