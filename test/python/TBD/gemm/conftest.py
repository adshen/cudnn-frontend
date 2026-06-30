"""Make ``cudnn.TBD.gemm`` importable for this test tree.

The GEMM engine lives in the cudnn-frontend source tree at
``python/cudnn/TBD/gemm``, but ``import cudnn`` resolves to the *installed*
cuDNN frontend wheel (site-packages), which does not carry the ``TBD`` subtree.
Append the source ``python/cudnn`` directory to the installed ``cudnn``
package's ``__path__`` so submodule lookup finds ``cudnn.TBD.gemm`` there.

Once the engine ships as part of the built cuDNN frontend package this shim is
unnecessary (the subpackage is then on the installed ``__path__`` already).
"""

from __future__ import annotations

from pathlib import Path

import cudnn

_SRC_CUDNN = Path(__file__).resolve().parents[4] / "python" / "cudnn"
if _SRC_CUDNN.is_dir() and str(_SRC_CUDNN) not in cudnn.__path__:
    cudnn.__path__.append(str(_SRC_CUDNN))
