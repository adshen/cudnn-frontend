"""Example 12: MoE grouped block-scale matmul forward (NVFP4, mode=NONE).

Folds block_scale_dequantize(token) + block_scale_dequantize(weight) feeding
moe_grouped_matmul into one fused kernel (NVFP4: FP4 E2M1 + FP8 E4M3 scale,
K-block 16). Group g (expert g % E) computes
    out[fto[g] : fto[g+1]] = dequant(token[range]) @ dequant(weight[g % E]).T
Group sizes are arbitrary: SFA is reordered + padded to 128 rows PER GROUP and
concatenated, and the scheduler tracks each group's start SF-block.
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (installs hook)
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.tile_config import CATALOG

# E2M1 (FP4) 4-bit code -> value lookup (low nibble first within a byte).
_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(x: torch.Tensor) -> torch.Tensor:
    """(rows, cols) SF matrix -> 128x4-blocked (F8_128x4) layout, flat."""
    rows, cols = x.shape
    nrb, ncb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _unpack_fp4(u8: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """(..., Kp) uint8 -> (..., 2*Kp) fp32, low nibble first."""
    lo = lut[(u8 & 0xF).long()]
    hi = lut[(u8 >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def main(
    S: int = 1024,
    N: int = 256,
    K: int = 512,
    E: int = 2,
    block_size: int = 16,
    offsets_list: list[int] | None = None,
) -> None:
    dev = "cuda"
    torch.manual_seed(0)
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    # 4 routed groups over E=2 experts (BxE > E). Any group sizes work.
    if offsets_list is None:
        offsets_list = [0, 256, 384, 512]
    num_groups = len(offsets_list)

    # Packed FP4 data, K-major: token (1, S, K/2) bytes; weight (E, N, K/2) bytes.
    tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
    w_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # token [1,S,K] FP4; weight [E,K,N] FP4 K-major (== (E,N,K) row-major memory).
    tok = g.tensor(
        name="token",
        dim=[1, S, K],
        stride=[S * K, K, 1],
        data_type=cudnn.data_type.FP4_E2M1,
    )
    w = g.tensor(
        name="weight",
        dim=[E, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.FP4_E2M1,
    )
    SFA = g.tensor(
        name="SFA",
        dim=[1, S, sf_k],
        stride=[S * sf_k, sf_k, 1],
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[E, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[num_groups, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size])
    w_d = g.block_scale_dequantize(input=w, descale=SFB, block_size=[block_size, 1])
    out = g.moe_grouped_matmul(
        tok_d,
        w_d,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
    )
    out.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    cfg = next(c for c in CATALOG if c.name == "CONFIG_sm100_128x256x128_128x256x32_cluster2x1")
    compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=2)

    # Runtime: packed FP4 (x2), SF in F8_128x4 blocked layout. SFA reordered +
    # padded to 128 rows PER GROUP then concatenated; SFB is per-expert.
    tok_f4 = tok_u8.view(torch.float4_e2m1fn_x2)
    w_f4 = w_u8.view(torch.float4_e2m1fn_x2)
    sfa_parts = []
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        sfa_parts.append(to_blocked(sfa_log[b:e]))
    sfa_blk = torch.cat(sfa_parts).view(1, -1, 1)
    sfb_blk = torch.cat([to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)

    compiled({tok: tok_f4, w: w_f4, SFA: sfa_blk, SFB: sfb_blk, fto: offsets, out: output})
    torch.cuda.synchronize()

    tok_deq = _unpack_fp4(tok_u8, lut).view(S, K) * sfa_log.float().repeat_interleave(block_size, 1)
    w_deq = _unpack_fp4(w_u8, lut).view(E, N, K) * sfb_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    starts = offsets_list
    for gi in range(num_groups):
        b = starts[gi]
        e = starts[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ref[b:e] = tok_deq[b:e] @ w_deq[gi % E].T
    ref = ref.to(torch.bfloat16)

    torch.testing.assert_close(output[0], ref, atol=1e-1, rtol=1e-2)
    print(f"[12] PASS  S={S} N={N} K={K} E={E} groups={offsets_list} " f"block_size={block_size}")


if __name__ == "__main__":
    main()
