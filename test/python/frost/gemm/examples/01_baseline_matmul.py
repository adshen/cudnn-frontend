"""Example 01: baseline matmul (no fusion) — FROST engine vs native cuDNN.

FROST is a named engine (``frost_gemm_eng0``) in ``heur_mode.A``'s plan list, not a
heuristic mode. Builds the same graph twice: select frost_gemm_eng0 (OSS JIT GEMM) vs
deselect it (native cuDNN); asserts both match torch.
"""

from __future__ import annotations

import cudnn
import os

os.environ.setdefault("NV_CUDNN_FE_ENABLE_FROST_ENGINES", "1")

import cudnn.frost.gemm  # noqa: F401  (registers frost_gemm_eng0 + installs hook)
import torch


def _build_matmul_graph(M: int, N: int, K: int):
    """Build the baseline bf16 matmul graph -> (g, A, B, C)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    return g, A, B, C


def main(M: int = 256, N: int = 256, K: int = 128) -> None:
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")

    # FROST engine: select frost_gemm_eng0 out of heur_mode.A's plan list
    g_frost, A, B, C = _build_matmul_graph(M, N, K)
    g_frost.validate()
    g_frost.build_operation_graph()
    g_frost.create_execution_plans([cudnn.heur_mode.A])
    g_frost.select_engines(["frost_gemm_eng0"])
    g_frost.check_support()
    g_frost.build_plans()

    c_frost = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    ws_frost = torch.empty(max(g_frost.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g_frost.execute({A: a, B: b, C: c_frost}, ws_frost)

    # native cuDNN: deselect frost_gemm_eng0
    g_ref, A_ref, B_ref, C_ref = _build_matmul_graph(M, N, K)
    g_ref.validate()
    g_ref.build_operation_graph()
    g_ref.create_execution_plans([cudnn.heur_mode.A])
    g_ref.deselect_engines(["frost_gemm_eng0"])
    g_ref.check_support()
    g_ref.build_plans()

    c_ref = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    ws_ref = torch.empty(max(g_ref.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g_ref.execute({A_ref: a, B_ref: b, C_ref: c_ref}, ws_ref)

    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)).to(torch.bfloat16)
    torch.testing.assert_close(c_frost, c_ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_frost, ref, atol=1e-1, rtol=1e-2)
    print(f"[01] PASS  M={M} N={N} K={K}  (FROST == cuDNN native == torch)")


if __name__ == "__main__":
    main()
