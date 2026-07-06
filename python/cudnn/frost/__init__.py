"""cudnn.frost: staging namespace for OSS engines reachable via the cuDNN frontend.

Importing this installs the shared FROST-engine lifecycle patch on
``cudnn.pygraph``. Op engines (e.g. ``cudnn.frost.gemm``) register their builder
via :func:`register_engine`; the user pins one via ``select_engines``.
"""

from .heuristics import (
    engine_names,
    install_lifecycle_patches,
    is_frost_engine,
    register_engine,
)

install_lifecycle_patches()

__all__ = [
    "register_engine",
    "engine_names",
    "is_frost_engine",
    "install_lifecycle_patches",
]
