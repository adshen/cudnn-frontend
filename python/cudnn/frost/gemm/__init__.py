"""cudnn.frost.gemm: JIT fused sm100 GEMM kernels from cuDNN graphs via cute DSL.

User code uses the pure cuDNN frontend API; the analyzer reads the Python-native
``cudnn.pygraph`` IR (``graph.nodes``) directly, and graphs are passed to
:func:`cudnn.frost.gemm.compiler.jit_from_cudnn_graph`.
"""

from cudnn.frost import register_engine

from .graph_analyzer import build_gemm_plan, probe_gemm_plan

# GEMM engine name; selected via graph.select_engines(["frost_gemm_eng0"]).
ENGINE_NAME = "frost_gemm_eng0"
register_engine(ENGINE_NAME, probe_gemm_plan, build_gemm_plan)

__all__ = ["ENGINE_NAME"]
