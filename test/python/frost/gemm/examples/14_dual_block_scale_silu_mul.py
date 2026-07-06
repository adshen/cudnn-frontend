"""Example 14: block-scale multi-GEMM — dual nvfp4 matmul + SwiGLU.

Two parallel block-scaled (nvfp4) GEMMs share the dequantized A operand:
    out = silu(dequant(A) @ dequant(B0)) * (dequant(A) @ dequant(B1))
The single block_scale_dequantize(A) is matched into BOTH GEMMs (one distinct A
operand + SFA, loaded once). cta_n capped at 128: dual accumulators + SF region
must fit 512 TMEM cols (2*128 + SF <= 512). 1-CTA-MMA template (cta_group=1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cudnn
import cudnn.frost.gemm  # noqa: F401  (installs hook)
import torch

from cudnn.frost.gemm.compiler import jit_from_cudnn_graph
from cudnn.frost.gemm.tile_config import by_name

# Reuse the SF-blocking + fp4-unpack helpers from the test module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_block_scale import _to_blocked, _unpack_fp4, _E2M1  # noqa: E402


def main(M: int = 256, N: int = 128, K: int = 512) -> None:
    bs, sf_k = 16, K // 16
    fp4, e4m3 = cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3
    rk = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=fp4)
    SFA = g.tensor(name="SFA", dim=[1, M, sf_k], stride=[M * sf_k, sf_k, 1], data_type=e4m3, **rk)
    B0 = g.tensor(name="B0", dim=[1, K, N], stride=[K * N, 1, K], data_type=fp4)
    SFB0 = g.tensor(name="SFB0", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=e4m3, **rk)
    B1 = g.tensor(name="B1", dim=[1, K, N], stride=[K * N, 1, K], data_type=fp4)
    SFB1 = g.tensor(name="SFB1", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=e4m3, **rk)

    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, bs])  # shared by both GEMMs
    B0d = g.block_scale_dequantize(input=B0, descale=SFB0, block_size=[bs, 1])
    B1d = g.block_scale_dequantize(input=B1, descale=SFB1, block_size=[bs, 1])
    C0 = g.matmul(A=Ad, B=B0d, name="mm0")
    C1 = g.matmul(A=Ad, B=B1d, name="mm1")
    Y = g.mul(a=g.swish(input=C0, name="silu"), b=C1, name="mul")
    Y.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=1, scheduler="clc")

    dev = "cuda"
    torch.manual_seed(0)
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    def _mk(rows):
        u8 = torch.randint(0, 256, (1, rows, K // 2), dtype=torch.uint8, device=dev)
        return u8.view(torch.float4_e2m1fn_x2), _unpack_fp4(u8, lut).view(rows, K)

    a_rt, a_deq = _mk(M)
    b0_rt, b0_deq = _mk(N)
    b1_rt, b1_deq = _mk(N)
    sfa = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfb0 = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfb1 = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfa_b = _to_blocked(sfa).view(1, M, sf_k)

    c = torch.zeros(1, M, N, dtype=torch.float32, device=dev)
    compiled(
        {
            A: a_rt,
            SFA: sfa_b,
            B0: b0_rt,
            SFB0: _to_blocked(sfb0).view(1, N, sf_k),
            B1: b1_rt,
            SFB1: _to_blocked(sfb1).view(1, N, sf_k),
            Y: c,
        }
    )
    torch.cuda.synchronize()

    a_s = a_deq * sfa.float().repeat_interleave(bs, 1)
    b0_s = b0_deq * sfb0.float().repeat_interleave(bs, 1)
    b1_s = b1_deq * sfb1.float().repeat_interleave(bs, 1)
    ref = torch.nn.functional.silu(a_s @ b0_s.t()) * (a_s @ b1_s.t())
    # matmul exact (nvfp4), but swish uses cuDNN's fast __expf/__fdividef → ~1e-3 rel.
    torch.testing.assert_close(c[0], ref, rtol=2e-2, atol=2e-1)
    print(f"[14] PASS  M={M} N={N} K={K}")


if __name__ == "__main__":
    main()
