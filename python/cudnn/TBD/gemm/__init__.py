"""cudnn.TBD.gemm: JIT fused sm100 GEMM kernels from cuDNN graphs via cute DSL.

Importing this monkey-patches ``cudnn.pygraph`` to record op chains, so user code
uses the pure cuDNN frontend API and passes graphs to
:func:`cudnn.TBD.gemm.compiler.jit_from_cudnn_graph`.
"""

from cudnn.TBD import register_engine

from .graph_analyzer import build_gemm_plan, install_recorder, probe_gemm_plan

install_recorder()

# GEMM engine name; selected via graph.select_engines(["TBD_eng0"]).
ENGINE_NAME = "TBD_eng0"
register_engine(ENGINE_NAME, probe_gemm_plan, build_gemm_plan)

__all__ = ["install_recorder", "ENGINE_NAME"]
