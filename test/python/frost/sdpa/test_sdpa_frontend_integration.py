"""Frontend integration for the FROST DSL SDPA engines as opt-in plan-list entries appended to cuDNN's native plans."""

from __future__ import annotations

import pytest
import torch

import cudnn
import cudnn.sdpa  # noqa: F401 — registers the FROST DSL engines
from cudnn.frost import engine_names, is_frost_engine
from cudnn.frost.dispatch import _get_plan_state

from cudnn.sdpa.fwd.engines import engine_name

_FROST = engine_name(512)  # matches the D=512 graphs below
_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(torch.cuda.current_device()) == (10, 0)


def _dsl_deps_available() -> bool:
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


_SM100 = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100 (Blackwell) GPU")
_SM100_DSL = pytest.mark.skipif(
    not (_is_sm100() and _dsl_deps_available()),
    reason="needs an SM100 (Blackwell) GPU with cutlass installed",
)

# The default pytest.ini addopts is `-m L0`; mark the whole module so it runs.
pytestmark = pytest.mark.L0

B, H, S, D = 2, 8, 256, 512


def _build_causal_sdpa(dtype=cudnn.data_type.HALF, d=D):
    dims = (B, H, S, d)
    strides = (S * H * d, d, H * d, 1)
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor(dim=dims, stride=strides, data_type=dtype, name="q")
    k = g.tensor(dim=dims, stride=strides, data_type=dtype, name="k")
    v = g.tensor(dim=dims, stride=strides, data_type=dtype, name="v")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / (d**0.5), is_inference=True, use_causal_mask=True)
    o.set_output(True).set_dim(dims).set_stride(strides)
    return g, q, k, v, o


def test_engine_registered():
    assert _FROST in engine_names()
    assert is_frost_engine(_FROST)
    assert not is_frost_engine("A")
    assert not is_frost_engine(cudnn.heur_mode.A)


@_GPU
def test_ineligible_graph_lists_no_dsl_engine():
    """d=128 validates on any GPU, so no SM100 needed to check the probe rejects it."""
    g, q, k, v, o = _build_causal_sdpa(d=128)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    state = _get_plan_state(g)
    assert _FROST not in state["eligible"]


@_SM100
def test_eligible_graph_lists_matching_dsl_engine():
    g, q, k, v, o = _build_causal_sdpa()
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    state = _get_plan_state(g)
    assert _FROST in state["eligible"]
    with_frost = g.get_execution_plan_count()
    g.deselect_engines([_FROST])
    without_frost = g.get_execution_plan_count()
    assert with_frost == without_frost + 1


@_SM100
def test_default_and_deselect_run_native():
    for deselect in (False, True):
        g, q, k, v, o = _build_causal_sdpa()
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        if deselect:
            g.deselect_engines([_FROST])
        state = _get_plan_state(g)
        assert state["selected"] is None


@_SM100_DSL
def test_select_dsl_engine_runs_and_matches_torch():
    q_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    k_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    v_gpu = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    o_gpu = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)

    g, q, k, v, o = _build_causal_sdpa()
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.select_engines([_FROST])
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == 0

    ws = torch.empty(1, device="cuda", dtype=torch.uint8)
    g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu}, ws)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(
        q_gpu,
        k_gpu,
        v_gpu,
        is_causal=True,
        scale=1.0 / (D**0.5),
    )
    torch.testing.assert_close(o_gpu, ref, atol=5e-2, rtol=3e-2)


@_SM100
def test_knob_request_gates_selection():
    """An unsupported knob request makes the selected engine ineligible: strict
    select raises at check_support instead of silently degrading the knob."""
    g, q, k, v, o = _build_causal_sdpa()
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    g.set_engine_knobs(SdpaFwdKnobs(sched_policy=1))  # no engine advertises LPT yet
    g.select_engines([_FROST])
    with pytest.raises(ValueError, match="not eligible"):
        g.check_support()
    # Honoring the domain restores eligibility.
    g.set_engine_knobs(SdpaFwdKnobs(sched_policy=0))
    g.check_support()


@_SM100
def test_no_magic_import_required():
    """Env var + engine name is the whole opt-in: a fresh process that only
    imports cudnn (never cudnn.sdpa / cudnn.frost.gemm) can still select and
    run a FROST engine — the dispatch lazily imports the engine manifest."""
    import subprocess
    import sys

    code = (
        "import torch, cudnn\n"
        "assert hasattr(cudnn.pygraph, 'set_engine_knobs')\n"
        "b,h,s,d = 1,2,256,512\n"
        "q_gpu = torch.randn(b,s,h,d, device='cuda', dtype=torch.float16).transpose(1,2)\n"
        "k_gpu, v_gpu, o_gpu = q_gpu.clone(), q_gpu.clone(), torch.empty_like(q_gpu)\n"
        "g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF,\n"
        "                  intermediate_data_type=cudnn.data_type.FLOAT,\n"
        "                  compute_data_type=cudnn.data_type.FLOAT)\n"
        "q = g.tensor_like(q_gpu); k = g.tensor_like(k_gpu); v = g.tensor_like(v_gpu)\n"
        "o, _ = g.sdpa(name='s', q=q, k=k, v=v, attn_scale=0.08, generate_stats=False, use_causal_mask=True)\n"
        "o.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())\n"
        "g.validate(); g.build_operation_graph(); g.create_execution_plans([cudnn.heur_mode.A])\n"
        "g.select_engines(['sdpa_fwd_prefill_sm100_d512'])\n"
        "g.check_support()\n"
        "print('ELIGIBLE-WITHOUT-IMPORT')\n"
    )
    import os

    env = dict(os.environ, NV_CUDNN_FE_ENABLE_FROST_ENGINES="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert "ELIGIBLE-WITHOUT-IMPORT" in out.stdout, out.stderr[-2000:]
