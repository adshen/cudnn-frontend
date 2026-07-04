"""Example 07: mainloop fusion — unary ops on A and/or B before the MMA.

A unary op feeding a matmul operand runs on dedicated mainloop warps that
transform the operand tile in SMEM before the MMA (not in the epilogue); the
12-warp mainloop template is picked automatically. Cases: A-only, B-only, both,
plus scalar-aux binary (A*alpha) @ (B*beta).
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (registers TBD_eng0 + installs hook)
import torch

_T = {"abs": torch.abs, "relu": torch.relu, "none": lambda x: x}


def _run(aop: str, bop: str, M: int, N: int, K: int) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    Ai = getattr(g, aop)(input=A, name="aop").set_data_type(cudnn.data_type.BFLOAT16) if aop != "none" else A
    Bi = getattr(g, bop)(input=B, name="bop").set_data_type(cudnn.data_type.BFLOAT16) if bop != "none" else B
    C = g.matmul(A=Ai, B=Bi, name="mm")
    C.set_output(True)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.select_engines(["TBD_eng0"])
    g.check_support()
    g.build_plans()

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")

    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, workspace)
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", _T[aop](a.float()), _T[bop](b.float())).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[07] PASS  A={aop} B={bop}  M={M} N={N} K={K}")


def _run_scaled(M: int, N: int, K: int, av: float = 2.0, bv: float = 0.5) -> None:
    """Scalar-aux mainloop fusion: (A * alpha) @ (B * beta)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    alpha = g.tensor(name="alpha", dim=[1, 1, 1], stride=[1, 1, 1])
    beta = g.tensor(name="beta", dim=[1, 1, 1], stride=[1, 1, 1])
    As = g.mul(a=A, b=alpha, name="sA")
    As.set_data_type(cudnn.data_type.BFLOAT16)
    Bs = g.mul(a=B, b=beta, name="sB")
    Bs.set_data_type(cudnn.data_type.BFLOAT16)
    C = g.matmul(A=As, B=Bs, name="mm")
    C.set_output(True)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.select_engines(["TBD_eng0"])
    g.check_support()
    g.build_plans()

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    auxmap = {
        "alpha": torch.full((1, 1, 1), av, dtype=torch.bfloat16, device="cuda"),
        "beta": torch.full((1, 1, 1), bv, dtype=torch.bfloat16, device="cuda"),
    }
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c, **auxmap}, workspace)
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.float() * av, b.float() * bv).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=2e-1, rtol=2e-2)
    print(f"[07] PASS  (A*{av}) @ (B*{bv})  M={M} N={N} K={K}")


def main(M: int = 512, N: int = 512, K: int = 256) -> None:
    _run("abs", "none", M, N, K)  # A only
    _run("none", "relu", M, N, K)  # B only
    _run("abs", "relu", M, N, K)  # both
    _run_scaled(M, N, K)  # scalar-aux binary


if __name__ == "__main__":
    main()
