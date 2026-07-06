"""Example 08: mainloop fusion with cos on BOTH operands — cos(A) @ cos(B).

cos(0)=1, so a partial last K-tile is a trap: TMA zero-fills OOB K elements but
the transform turns them into 1, contributing 1*1 to every accumulator. Fix: the
mainloop zeros A's OOB K elements (swizzle-aware, sub-K-block granular). Shapes
cover K%16==0 and K%16==8 (partial OOB K-block).
"""

from __future__ import annotations

import cudnn
import cudnn.frost.gemm  # noqa: F401  (registers frost_eng0 + installs hook)
import torch


def _run(M: int, N: int, K: int) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    Ac = g.cos(input=A, name="cosA").set_data_type(cudnn.data_type.BFLOAT16)
    Bc = g.cos(input=B, name="cosB").set_data_type(cudnn.data_type.BFLOAT16)
    C = g.matmul(A=Ac, B=Bc, name="mm")
    C.set_output(True)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.select_engines(["frost_eng0"])
    g.check_support()
    g.build_plans()

    torch.manual_seed(0)
    a = (torch.rand(1, M, K, device="cuda") * 6.0 - 3.0).to(torch.bfloat16)
    b = (torch.rand(1, N, K, device="cuda") * 6.0 - 3.0).to(torch.bfloat16)
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, workspace)
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", torch.cos(a.float()), torch.cos(b.float())).to(torch.bfloat16)
    diff = (c.float() - ref.float()).abs()
    print(f"[08] cos(A)@cos(B)  M={M} N={N} K={K} (K%16={K % 16})  " f"max|diff|={diff.max().item():.4f}  bad(>0.5)={int((diff > 0.5).sum())}")
    torch.testing.assert_close(c, ref, atol=6e-1, rtol=2e-2)


def main() -> None:
    _run(240, 272, 288)  # K%16 == 0
    _run(240, 272, 264)  # K%16 == 8, partial OOB K-block
    print("[08] PASS  cos(A) @ cos(B)")


if __name__ == "__main__":
    main()
