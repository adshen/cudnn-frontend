"""Example 09: NVFP4 block-scaled matmul -> FP16 output.

Demonstrates the block-scaled-matmul path: the inputs A, B are FP4 (E2M1,
packed 2-per-byte) and each is dequantized by a *per-block* scale factor along
K **inside** the MMA (``tcgen05.mma.kind::mxf4nvf4.block_scale``). NVFP4 uses an
FP8 E4M3 scale with a K-block size of 16.

The user builds a pure cuDNN-frontend graph: two ``block_scale_dequantize``
nodes feeding a ``matmul``. ``cudnn.TBD.gemm`` pattern-matches that shape (exactly
like cuDNN's ``pattern_match_block_scale_matmul`` pass) and JITs the block-scale
kernel.

Scale factors are passed in the ``CUDNN_TENSOR_REORDERING_F8_128x4`` swizzled
layout — see ``to_blocked`` below.

    source active_tbd.sh
    python cudnn.TBD.gemm/examples/09_block_scale_nvfp4.py
"""

from __future__ import annotations

import cudnn
import cudnn.TBD.gemm  # noqa: F401
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.tile_config import CONFIG_sm100_128x256x128_128x256x32_cluster1x1

# E2M1 (FP4) 4-bit code -> value lookup table (low nibble first within a byte).
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(x: torch.Tensor) -> torch.Tensor:
    """Convert a (rows, cols) scale-factor matrix to the 128x4 blocked
    (CUDNN_TENSOR_REORDERING_F8_128x4) layout, returned flat."""
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


def main(M: int = 256, N: int = 256, K: int = 512, block_size: int = 16) -> None:
    dev = "cuda"
    torch.manual_seed(0)
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    # Packed FP4 data, K-major (K contiguous): A is (1, M, K/2) bytes, B is
    # (1, N, K/2) bytes (B is stored (N, K) so the matmul reads it K-major).
    a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)

    # Scale factors (E4M3), small positive integers.
    sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.FP4_E2M1)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.FP4_E2M1)
    SFA = g.tensor(
        name="SFA", dim=[1, M, sf_k], stride=[M * sf_k, sf_k, 1], data_type=cudnn.data_type.FP8_E4M3, reordering_type=cudnn.tensor_reordering.F8_128x4
    )
    SFB = g.tensor(
        name="SFB", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=cudnn.data_type.FP8_E4M3, reordering_type=cudnn.tensor_reordering.F8_128x4
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block_size])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.HALF)

    compiled = jit_from_cudnn_graph(g, config=CONFIG_sm100_128x256x128_128x256x32_cluster1x1, cta_group=1)

    # Runtime: A/B as packed FP4 (view as float4_e2m1fn_x2), SF in blocked layout.
    a_f4 = a_u8.view(torch.float4_e2m1fn_x2)
    b_f4 = b_u8.view(torch.float4_e2m1fn_x2)
    sfa_blk = to_blocked(sfa_log).view(1, M, sf_k)
    sfb_blk = to_blocked(sfb_log).view(1, N, sf_k)
    c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    compiled({A: a_f4, B: b_f4, SFA: sfa_blk, SFB: sfb_blk, C: c})
    torch.cuda.synchronize()

    # Torch reference: dequantize then matmul.
    a_deq = _unpack_fp4(a_u8, lut).view(M, K) * sfa_log.float().repeat_interleave(block_size, 1)
    b_deq = _unpack_fp4(b_u8, lut).view(N, K) * sfb_log.float().repeat_interleave(block_size, 1)
    ref = (a_deq @ b_deq.t()).to(torch.float16)

    torch.testing.assert_close(c[0], ref, atol=1e-1, rtol=1e-2)
    print(f"[09] PASS  M={M} N={N} K={K} block_size={block_size}")


if __name__ == "__main__":
    main()
