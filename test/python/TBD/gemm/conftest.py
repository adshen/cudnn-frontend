"""Make ``cudnn.TBD.gemm`` importable: append the source ``python/cudnn`` dir
to the installed ``cudnn`` package's ``__path__`` (the wheel lacks the ``TBD``
subtree). Unnecessary once the engine ships in the built frontend package."""

from __future__ import annotations

from pathlib import Path

import cudnn

_SRC_CUDNN = Path(__file__).resolve().parents[4] / "python" / "cudnn"
if _SRC_CUDNN.is_dir() and str(_SRC_CUDNN) not in cudnn.__path__:
    cudnn.__path__.append(str(_SRC_CUDNN))
