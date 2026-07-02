"""Example 08: mainloop fusion with cos on BOTH operands — cos(A) @ cos(B).

cos has f(0) != 0, so a partial last K-tile (K not a multiple of the K-tile)
is a correctness trap: the TMA zero-fills the out-of-bounds K elements, but the
mainloop transform turns those zeros into cos(0)=1, which would then contribute
1*1 to every accumulator. The mainloop fixes this by zeroing A's OOB K elements
before the MMA reads them. It reads/writes A with the SAME XOR swizzle the MMA
expects (load_swizzled / store_swizzled), so the loop index is the *logical*
(m, k) and the OOB test is just `global_k >= K` — no manual de-swizzle, and it
works at sub-K-block granularity (handles K=264 where K%16 != 0).

Shapes exercised: M=240 (M-tail), N=272 (N%16==0 but not tile-aligned), and
K in {288 (K%16==0), 264 (K%16==8, partial OOB K-block)}.
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (registers TBD_eng0 + installs hook)
import torch


def _run(M: int, N: int, K: int) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    Ac = g.cos(input=A, name="cosA")  # mainloop fusion on A
    Bc = g.cos(input=B, name="cosB")  # mainloop fusion on B
    C = g.matmul(A=Ac, B=Bc, name="mm")
    C.set_output(True)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.select_engines(["TBD_eng0"])
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
    _run(240, 272, 264)  # K%16 == 8 — partial OOB K-block
    print("[08] PASS  cos(A) @ cos(B)")


if __name__ == "__main__":
    main()
