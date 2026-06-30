"""cudnn.TBD.gemm: dynamically generate fused sm100 GEMM kernels from cuDNN graphs via cute DSL.

Importing this package monkey-patches ``cudnn.pygraph`` to record op chains
on every graph instance, so user code can use the **pure cuDNN frontend API**
to build graphs and pass them to :func:`cudnn.TBD.gemm.compiler.jit_from_cudnn_graph`.
"""

from .graph_analyzer import install_recorder

install_recorder()
