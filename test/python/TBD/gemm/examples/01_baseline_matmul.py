"""Example 01: baseline matmul (no fusion) — TBD engine vs native cuDNN.

Builds the *same* matmul graph twice with the **pure cuDNN frontend API** and
the *same* heuristic (``heur_mode.A``). TBD is not a heuristic mode — it is a
named engine (``TBD_eng0``) appended to the plan list that ``heur_mode.A``
produces. The two graphs differ only in which engine they pick:

  * ``g_tbd`` calls ``select_engines(["TBD_eng0"])`` → the OSS JIT GEMM engine,
  * ``g_ref`` calls ``deselect_engines(["TBD_eng0"])`` → legacy native cuDNN.

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])   # cuDNN engines + TBD_eng0
    g.select_engines(["TBD_eng0"])  /  g.deselect_engines(["TBD_eng0"])
    g.check_support()
    g.build_plans()
    g.execute(variant_pack, workspace)

The two GPU outputs are asserted to match (and both against torch.matmul), so
the example doubles as a cross-check that the TBD engine reproduces cuDNN.

Usage:

    source active_tbd.sh
    python cudnn.TBD.gemm/examples/01_baseline_matmul.py
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (registers the TBD_eng0 engine + installs the lifecycle hook)
import torch


def _build_matmul_graph(M: int, N: int, K: int):
    """Build the baseline bf16 matmul graph. Returns ``(g, A, B, C)``."""
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

    # --- TBD engine: select TBD_eng0 out of heur_mode.A's plan list -----------
    g_tbd, A, B, C = _build_matmul_graph(M, N, K)
    g_tbd.validate()
    g_tbd.build_operation_graph()
    g_tbd.create_execution_plans([cudnn.heur_mode.A])
    g_tbd.select_engines(["TBD_eng0"])
    g_tbd.check_support()
    g_tbd.build_plans()

    c_tbd = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    ws_tbd = torch.empty(max(g_tbd.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g_tbd.execute({A: a, B: b, C: c_tbd}, ws_tbd)

    # --- native cuDNN: deselect TBD_eng0 → legacy behavior --------------------
    g_ref, A_ref, B_ref, C_ref = _build_matmul_graph(M, N, K)
    g_ref.validate()
    g_ref.build_operation_graph()
    g_ref.create_execution_plans([cudnn.heur_mode.A])
    g_ref.deselect_engines(["TBD_eng0"])
    g_ref.check_support()
    g_ref.build_plans()

    c_ref = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    ws_ref = torch.empty(max(g_ref.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g_ref.execute({A_ref: a, B_ref: b, C_ref: c_ref}, ws_ref)

    torch.cuda.synchronize()

    # The TBD engine must reproduce native cuDNN (and both match torch).
    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32)).to(torch.bfloat16)
    torch.testing.assert_close(c_tbd, c_ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_tbd, ref, atol=1e-1, rtol=1e-2)
    print(f"[01] PASS  M={M} N={N} K={K}  (TBD == cuDNN native == torch)")


if __name__ == "__main__":
    main()
