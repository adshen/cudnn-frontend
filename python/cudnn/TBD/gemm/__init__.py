"""cudnn.TBD.gemm: dynamically generate fused sm100 GEMM kernels from cuDNN graphs via cute DSL.

Importing this package monkey-patches ``cudnn.pygraph`` to record op chains
on every graph instance, so user code can use the **pure cuDNN frontend API**
to build graphs and pass them to :func:`cudnn.TBD.gemm.compiler.jit_from_cudnn_graph`.
"""

from cudnn.TBD import register_engine

from .graph_analyzer import build_gemm_plan, install_recorder, probe_gemm_plan

install_recorder()

# The GEMM engine is named "TBD_eng0"; it is appended to the plan list produced
# by the native cuDNN heuristics (e.g. heur_mode.A) and selected by name via
# graph.select_engines(["TBD_eng0"]). Future TBD engines take TBD_eng1, ...
ENGINE_NAME = "TBD_eng0"
register_engine(ENGINE_NAME, probe_gemm_plan, build_gemm_plan)

__all__ = ["install_recorder", "ENGINE_NAME"]
