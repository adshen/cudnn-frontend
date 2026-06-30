"""End-to-end smoke tests: build cudnn graph, jit via cudnn.TBD.gemm, verify vs torch.

Requires a CUDA device. Skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


_EXAMPLES = Path(__file__).resolve().parent / "examples"


def _load_example(name: str):
    path = _EXAMPLES / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "example",
    [
        "01_baseline_matmul.py",
        "02_matmul_relu.py",
        "03_matmul_bias_gelu.py",
        "04_matmul_percol_bias_swish.py",
    ],
)
def test_example_runs_and_passes(example: str) -> None:
    mod = _load_example(example)
    # Each example exposes main(M=, N=, K=) and asserts internally.
    mod.main(M=256, N=256, K=128)
