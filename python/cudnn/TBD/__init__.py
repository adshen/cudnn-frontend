"""cudnn.TBD: staging namespace for OSS engines reachable via the cuDNN frontend.

Importing this installs the shared TBD-engine lifecycle patch on
``cudnn.pygraph``. TBD engines are appended (by name, e.g. ``TBD_eng0``) to the
plan list produced by the native cuDNN heuristics; the user pins one via
``graph.select_engines(["TBD_eng0"])`` or drops it via ``deselect_engines``.
Op engines (e.g. ``cudnn.TBD.gemm``) register their builder via
:func:`register_engine`.
"""

from .heuristics import (
    engine_names,
    install_lifecycle_patches,
    is_tbd_engine,
    register_engine,
)

install_lifecycle_patches()

__all__ = ["register_engine", "engine_names", "is_tbd_engine", "install_lifecycle_patches"]
