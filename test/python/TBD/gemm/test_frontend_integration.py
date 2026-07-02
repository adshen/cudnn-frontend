"""Frontend integration for the TBD engine as a named plan-list entry.

TBD is not a heuristics mode: the GEMM engine ``TBD_eng0`` is appended to the
plan list produced by the native cuDNN heuristics (e.g. ``heur_mode.A``). These
tests exercise:

  * ``create_execution_plans([A]) + select_engines(["TBD_eng0"])`` → the TBD
    engine runs and matches torch,
  * the default (no select) and ``deselect_engines(["TBD_eng0"])`` → native
    cuDNN runs,
  * the plan-count grows by one when TBD is eligible,
  * a TBD-ineligible graph (fp32 matmul) does not list ``TBD_eng0``,
  * the ``cudnn.wrapper.Graph`` path with ``heuristics=[A]`` + engine select.
"""

from __future__ import annotations

import pytest
import torch

import cudnn
import cudnn.TBD.gemm  # noqa: F401  (registers TBD_eng0 + installs lifecycle patches)
from cudnn.TBD import engine_names, is_tbd_engine
from cudnn.TBD.heuristics import _get_plan_state

_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

M, N, K = 256, 256, 128
_TBD = "TBD_eng0"


def _build_matmul_bias_relu():
    """A bf16 matmul + per-col bias + relu graph (recorded). Returns
    ``(g, A, B, bias, Y)``."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1], data_type=cudnn.data_type.BFLOAT16)
    C = g.matmul(A=A, B=B, name="mm")
    Cb = g.bias(input=C, bias=bias, name="bs")
    Y = g.relu(input=Cb, name="r")
    Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g, A, B, bias, Y


def _operands():
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(torch.bfloat16).cuda()
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(torch.bfloat16).cuda()
    bias_t = torch.randn(1, 1, N, dtype=torch.bfloat16).cuda()
    ref = torch.relu(torch.einsum("bmk,bnk->bmn", a.float(), b.float()) + bias_t.float()).to(torch.bfloat16)
    return a, b, bias_t, ref


def _run_native(g):
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    return g


def test_engine_registered():
    assert _TBD in engine_names()
    assert is_tbd_engine(_TBD)
    assert not is_tbd_engine("A")
    assert not is_tbd_engine(cudnn.heur_mode.A)


@_GPU
def test_select_tbd_engine_runs_tbd():
    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    _run_native(g)
    # TBD_eng0 is appended after cuDNN's engines.
    state = _get_plan_state(g)
    assert _TBD in state["eligible"]
    g.select_engines([_TBD])
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == 0  # TBD owns its workspace
    ws = torch.empty(1, device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


@_GPU
def test_default_and_deselect_run_native():
    """With no select (default) and with TBD deselected, native cuDNN runs."""
    a, b, bias_t, ref = _operands()
    for deselect in (False, True):
        g, A, B, bias, Y = _build_matmul_bias_relu()
        _run_native(g)
        if deselect:
            g.deselect_engines([_TBD])
        g.check_support()
        g.build_plans()
        ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
        y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
        g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


@_GPU
def test_plan_count_grows_with_tbd():
    g, _A, _B, _bias, _Y = _build_matmul_bias_relu()
    _run_native(g)
    state = _get_plan_state(g)
    assert _TBD in state["eligible"]
    with_tbd = g.get_execution_plan_count()
    g.deselect_engines([_TBD])
    without_tbd = g.get_execution_plan_count()
    assert with_tbd == without_tbd + 1


@_GPU
def test_build_convenience_then_select():
    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    g.build([cudnn.heur_mode.A])
    g.select_engines([_TBD])
    g.build_plans()
    ws = torch.empty(1, device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


def test_ineligible_graph_not_listed():
    """A fp32 matmul (no fp32 MMA path) is not eligible → TBD_eng0 not listed."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, 64, 64], stride=[64 * 64, 64, 1], data_type=cudnn.data_type.FLOAT)
    B = g.tensor(name="B", dim=[1, 64, 64], stride=[64 * 64, 1, 64], data_type=cudnn.data_type.FLOAT)
    C = g.matmul(A=A, B=B, name="mm", compute_data_type=cudnn.data_type.FLOAT)
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    state = _get_plan_state(g)
    assert _TBD not in state["eligible"]
    # Selecting an ineligible engine is a no-op — no TBD entry was added, so the
    # plan count is just cuDNN's (selecting doesn't make an ineligible engine live).
    count_before = g.get_execution_plan_count()
    g.select_engines([_TBD])
    assert g.get_execution_plan_count() == count_before


@_GPU
def test_wrapper_graph_path():
    """cudnn.wrapper.Graph(heuristics=[A]) builds cuDNN + TBD_eng0; select TBD."""
    from cudnn.wrapper import Graph

    a, b, bias_t, ref = _operands()
    with Graph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        heuristics=[cudnn.heur_mode.A],
        handle="auto",
    ) as g:
        A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
        B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
        bias = g.tensor(name="bias", dim=[1, 1, N], stride=[N, N, 1], data_type=cudnn.data_type.BFLOAT16)
        C = g.matmul(A=A, B=B, name="mm")
        Cb = g.bias(input=C, bias=bias, name="bs")
        Y = g.relu(input=Cb, name="r")
        Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g({A: a, B: b, bias: bias_t, Y: y})
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)
