"""
Shared dtype conversion tables for the GEMM engine (single source of truth).
"""

from __future__ import annotations

from typing import Any

import cudnn

from .fusion_ir import Dtype

# internal dtype -> cute-DSL / cutlass type name (same string serves both the
# `.to(<type>)` DSL casts and the `cutlass.<Type>` enum args). FP4 is packed
# 2-per-byte as Float4E2M1FNx2.
DTYPE_TO_CUTLASS: dict[Dtype, str] = {
    "bf16": "cutlass.BFloat16",
    "fp16": "cutlass.Float16",
    "fp32": "cutlass.Float32",
    "int8": "cutlass.Int8",
    "fp8_e4m3": "cutlass.Float8E4M3FN",
    "fp8_e5m2": "cutlass.Float8E5M2",
    "fp8_e8m0": "cutlass.Float8E8M0FNU",
    "fp4_e2m1": "cutlass.Float4E2M1FNx2",
    "uint8": "cutlass.Uint8",
    "int32": "cutlass.Int32",
    "int64": "cutlass.Int64",
}

# internal dtype -> element size in bytes (FP4 packs 2/byte; counted as 1 here).
DTYPE_BYTES: dict[Dtype, int] = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "int8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp8_e8m0": 1,
    "fp4_e2m1": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
}

# input dtype -> tcgen05 MMA kind.
DTYPE_TO_MMA_KIND: dict[Dtype, str] = {
    "bf16": "nvvm.Tcgen05MMAKind.F16",
    "fp16": "nvvm.Tcgen05MMAKind.F16",
    "fp8_e4m3": "nvvm.Tcgen05MMAKind.F8F6F4",
    "fp8_e5m2": "nvvm.Tcgen05MMAKind.F8F6F4",
    "int8": "nvvm.Tcgen05MMAKind.INT8",
}

# cudnn.data_type <-> internal dtype.
DTYPE_FROM_CUDNN: dict[Any, Dtype] = {
    cudnn.data_type.BFLOAT16: "bf16",
    cudnn.data_type.HALF: "fp16",
    cudnn.data_type.FLOAT: "fp32",
    cudnn.data_type.INT8: "int8",
    cudnn.data_type.FP8_E4M3: "fp8_e4m3",
    cudnn.data_type.FP8_E5M2: "fp8_e5m2",
    cudnn.data_type.FP8_E8M0: "fp8_e8m0",
    cudnn.data_type.FP4_E2M1: "fp4_e2m1",
    cudnn.data_type.UINT8: "uint8",
    cudnn.data_type.INT32: "int32",
    cudnn.data_type.INT64: "int64",
}

CUDNN_FROM_DTYPE: dict[Dtype, Any] = {v: k for k, v in DTYPE_FROM_CUDNN.items()}
