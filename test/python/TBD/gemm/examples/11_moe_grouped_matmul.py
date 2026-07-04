"""Example 11: MoE grouped matmul forward (mode=NONE).

Each routed group g computes
    out[first_token_offset[g] : first_token_offset[g+1]] =
        token[that range] @ weight[g].T
Compared against a torch group-loop reference.
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (installs hook)
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.tile_config import CATALOG


def _offsets(group_sizes: list[int], S: int) -> torch.Tensor:
    starts, cur = [], 0
    for g in group_sizes:
        starts.append(cur)
        cur += g
    assert cur == S, f"group sizes sum to {cur}, expected S={S}"
    return torch.tensor(starts, dtype=torch.int32, device="cuda")


def _torch_ref(token, weight, offsets, S, N, E) -> torch.Tensor:
    out = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    starts = offsets.tolist()
    for g in range(E):
        b = starts[g]
        e = starts[g + 1] if g + 1 < E else S
        if b == e:
            continue
        out[b:e] = token[0, b:e].float() @ weight[g].float().T
    return out.to(torch.bfloat16)


def main() -> None:
    # E experts; uneven (incl. empty) token groups summing to S.
    E, N, K = 8, 256, 128
    group_sizes = [64, 0, 200, 128, 100, 12, 196, 68]
    S = sum(group_sizes)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(
        name="token",
        dim=[1, S, K],
        stride=[S * K, K, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    # weight [E, K, N] K-major (== (E, N, K) row-major in memory)
    w = g.tensor(
        name="weight",
        dim=[E, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.BFLOAT16,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[E, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    out = g.moe_grouped_matmul(
        tok,
        w,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
    )
    out.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    cfg = next(c for c in CATALOG if c.name == "CONFIG_sm100_128x256x128_128x256x32_cluster2x1")
    compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=2)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = _offsets(group_sizes, S)

    compiled({tok: token, w: weight, fto: offsets, out: output})
    torch.cuda.synchronize()

    ref = _torch_ref(token, weight, offsets, S, N, E)
    torch.testing.assert_close(output[0], ref, atol=1e-1, rtol=1e-2)
    print(f"[11] PASS  E={E} S={S} N={N} K={K}  groups={group_sizes}")


if __name__ == "__main__":
    main()
