"""Example 15: mixed-input mainloop — a fused operand LOADED at a narrower dtype
than the MMA reads (int8-weight / bf16-activation GEMM: int8 A @ bf16 B).

identity(A_int8).set_data_type(bf16) feeding the matmul is mainloop fusion whose
LOAD dtype (int8) differs from the MMA dtype (bf16). The mainloop warps stage the
widen: TMA loads the int8 tile into a narrow SMEM buffer, then casts it into the
wide bf16 MMA tile before the MMA. Any unary op chain works; both operands can be
cast independently.
"""

from __future__ import annotations

import cudnn
import os

os.environ.setdefault("NV_CUDNN_FE_ENABLE_FROST_ENGINES", "1")

import cudnn.frost.gemm  # noqa: F401  (registers frost_gemm_eng0 + installs hook)
import torch


def _run(M: int, N: int, K: int) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # A is stored as int8; identity casts it to bf16 before the MMA.
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.INT8)
    Ai = g.identity(input=A, name="pw_in_mainloop0")
    Ai.set_data_type(cudnn.data_type.BFLOAT16)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])  # bf16 (io_data_type)
    C = g.matmul(A=Ai, B=B, name="mm")
    C.set_output(True)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.select_engines(["frost_gemm_eng0"])
    g.check_support()
    g.build_plans()

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-4, 4).to(torch.int8).cuda()
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-4, 4).to(torch.bfloat16).cuda()
    c = torch.empty(1, M, N, dtype=torch.bfloat16).cuda()

    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, workspace)
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.float(), b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[15] PASS  int8(A) -> bf16 @ bf16(B)  M={M} N={N} K={K}")


def main() -> None:
    _run(512, 512, 512)
    _run(2048, 2048, 2048)


if __name__ == "__main__":
    main()
