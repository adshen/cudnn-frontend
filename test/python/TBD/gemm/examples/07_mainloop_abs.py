"""Example 07: mainloop fusion — unary ops on A and/or B before the MMA.

A unary op feeding the matmul's A or B operand is detected as *mainloop
fusion*: it runs on dedicated mainloop warps that transform the operand tile
in SMEM before the MMA reads it, rather than in the epilogue. The compiler
picks the 12-warp mainloop template automatically. This example runs three
cases: A-only (`abs(A) @ B`), B-only (`A @ relu(B)`), and both
(`abs(A) @ relu(B)`).
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (installs hook)
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph

_T = {"abs": torch.abs, "relu": torch.relu, "none": lambda x: x}


def _run(aop: str, bop: str, M: int, N: int, K: int) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    Ai = getattr(g, aop)(input=A, name="aop") if aop != "none" else A
    Bi = getattr(g, bop)(input=B, name="bop") if bop != "none" else B
    C = g.matmul(A=Ai, B=Bi, name="mm")
    C.set_output(True)

    compiled = jit_from_cudnn_graph(g)
    print(f"[07] {compiled.chain.summary()}")
    assert compiled.chain.has_mainloop_fusion

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")

    compiled({A: a, B: b, C: c})
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
    Bs = g.mul(a=B, b=beta, name="sB")
    C = g.matmul(A=As, B=Bs, name="mm")
    C.set_output(True)

    compiled = jit_from_cudnn_graph(g)
    print(f"[07] {compiled.chain.summary()}  aux={compiled.aux_names}")

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    auxmap = {
        "alpha": torch.full((1, 1, 1), av, dtype=torch.bfloat16, device="cuda"),
        "beta": torch.full((1, 1, 1), bv, dtype=torch.bfloat16, device="cuda"),
    }
    compiled({A: a, B: b, C: c, **auxmap})
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.float() * av, b.float() * bv).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=2e-1, rtol=2e-2)
    print(f"[07] PASS  (A*{av}) @ (B*{bv})  M={M} N={N} K={K}")


def main(M: int = 512, N: int = 512, K: int = 256) -> None:
    _run("abs", "none", M, N, K)  # mainloop fusion on A only
    _run("none", "relu", M, N, K)  # mainloop fusion on B only
    _run("abs", "relu", M, N, K)  # mainloop fusion on both A and B
    _run_scaled(M, N, K)  # scalar-aux binary: (A*alpha) @ (B*beta)


if __name__ == "__main__":
    main()
