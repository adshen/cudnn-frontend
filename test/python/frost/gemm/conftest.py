"""Make ``cudnn.frost.gemm`` importable: append the source ``python/cudnn`` dir
to the installed ``cudnn`` package's ``__path__`` (the wheel lacks the ``FROST``
subtree). Unnecessary once the engine ships in the built frontend package."""

from __future__ import annotations

import os
from pathlib import Path

# FROST engines are off by default; the frost test suite exercises them, so enable.
os.environ["NV_CUDNN_FE_ENABLE_FROST_ENGINES"] = "1"

import cudnn

_SRC_CUDNN = Path(__file__).resolve().parents[4] / "python" / "cudnn"
if _SRC_CUDNN.is_dir() and str(_SRC_CUDNN) not in cudnn.__path__:
    cudnn.__path__.append(str(_SRC_CUDNN))
