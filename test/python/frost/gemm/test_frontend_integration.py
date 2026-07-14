"""Frontend integration: the GEMM engine ``frost_gemm_eng0`` is appended to the plan
list from native cuDNN heuristics (not a heuristics mode). Exercises engine
select/deselect, plan-count growth, ineligible graphs, and the wrapper.Graph path."""

from __future__ import annotations

import pytest
import torch

from gemm_test_utils import requires_sm100

import cudnn
import cudnn.frost.gemm  # noqa: F401  (registers frost_gemm_eng0 + installs lifecycle patches)
from cudnn.frost import engine_names, is_frost_engine
from cudnn.frost.dispatch import _get_plan_state

pytestmark = pytest.mark.L0

_GPU = requires_sm100

M, N, K = 256, 256, 128
_FROST = "frost_gemm_eng0"


def _build_matmul_bias_relu():
    """A recorded bf16 matmul + per-col bias + relu graph → (g, A, B, bias, Y)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=[M * K, K, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.BFLOAT16,
    )
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
    assert _FROST in engine_names()
    assert is_frost_engine(_FROST)
    assert not is_frost_engine("A")
    assert not is_frost_engine(cudnn.heur_mode.A)


def test_env_flag_gates_frost(monkeypatch):
    """NV_CUDNN_FE_ENABLE_FROST_ENGINES=0 removes FROST from the plan list entirely;
    =1 restores it (the frost test suite runs with it enabled via conftest)."""
    from cudnn.frost import frost_engines_enabled

    monkeypatch.setenv("NV_CUDNN_FE_ENABLE_FROST_ENGINES", "0")
    assert frost_engines_enabled() is False
    g, *_ = _build_matmul_bias_relu()
    _run_native(g)
    assert _get_plan_state(g)["eligible"] == []  # disabled → never eligible

    monkeypatch.setenv("NV_CUDNN_FE_ENABLE_FROST_ENGINES", "1")
    assert frost_engines_enabled() is True
    g2, *_ = _build_matmul_bias_relu()
    _run_native(g2)
    assert _FROST in _get_plan_state(g2)["eligible"]  # enabled → back in the list


@_GPU
def test_select_frost_engine_runs_frost():
    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    _run_native(g)
    # frost_gemm_eng0 is appended after cuDNN's engines.
    state = _get_plan_state(g)
    assert _FROST in state["eligible"]
    g.select_engines([_FROST])
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == 0  # FROST owns its own workspace
    ws = torch.empty(1, device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


@_GPU
@pytest.mark.parametrize("deselect", [False, True], ids=["default", "deselected"])
def test_default_and_deselect_run_native(deselect):
    """No select (default) and FROST deselected both run native cuDNN."""
    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    _run_native(g)
    if deselect:
        g.deselect_engines([_FROST])
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


@_GPU
def test_plan_count_grows_with_frost():
    g, _A, _B, _bias, _Y = _build_matmul_bias_relu()
    _run_native(g)
    state = _get_plan_state(g)
    assert _FROST in state["eligible"]
    with_frost = g.get_execution_plan_count()
    g.deselect_engines([_FROST])
    without_frost = g.get_execution_plan_count()
    assert with_frost == without_frost + 1


@_GPU
def test_build_convenience_then_select():
    a, b, bias_t, ref = _operands()
    g, A, B, bias, Y = _build_matmul_bias_relu()
    g.build([cudnn.heur_mode.A])
    g.select_engines([_FROST])
    g.build_plans()
    ws = torch.empty(1, device="cuda", dtype=torch.uint8)
    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g.execute({A: a, B: b, bias: bias_t, Y: y}, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)


def test_ineligible_graph_not_listed():
    """fp32 matmul (no fp32 MMA path) is not eligible → frost_gemm_eng0 not listed."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, 64, 64],
        stride=[64 * 64, 64, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    B = g.tensor(
        name="B",
        dim=[1, 64, 64],
        stride=[64 * 64, 1, 64],
        data_type=cudnn.data_type.FLOAT,
    )
    C = g.matmul(A=A, B=B, name="mm", compute_data_type=cudnn.data_type.FLOAT)
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    state = _get_plan_state(g)
    assert _FROST not in state["eligible"]
    # Selecting an ineligible engine is a no-op — plan count stays cuDNN's.
    count_before = g.get_execution_plan_count()
    g.select_engines([_FROST])
    assert g.get_execution_plan_count() == count_before


def test_probe_exception_marks_ineligible(monkeypatch, caplog):
    """A FROST engine whose probe() raises is treated as ineligible without
    aborting eligibility for the others — a probe must never break the native
    path."""
    import logging

    from cudnn.frost import dispatch as _h

    def _raising_probe(_g):
        raise RuntimeError("probe boom")

    fake_engines = {
        "frost_probe_raise": (_raising_probe, lambda _g: None),
        "frost_probe_ok": (lambda _g: True, lambda _g: None),
    }
    monkeypatch.setattr(_h, "_ENGINES", fake_engines)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    with caplog.at_level(logging.DEBUG, logger="cudnn.frost.dispatch"):
        _h._probe_and_append(g)  # must not raise despite one probe throwing

    state = _h._get_plan_state(g)
    assert "frost_probe_raise" not in state["eligible"]  # raising probe → ineligible
    assert "frost_probe_ok" in state["eligible"]  # loop still evaluated the rest
    assert any("raised" in r.getMessage() for r in caplog.records)


@_GPU
def test_wrapper_graph_path():
    """wrapper.Graph(heuristics=[A]) builds cuDNN + frost_gemm_eng0; select FROST."""
    from cudnn.wrapper import Graph

    a, b, bias_t, ref = _operands()
    with Graph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        heuristics=[cudnn.heur_mode.A],
        handle="auto",
    ) as g:
        A = g.tensor(
            name="A",
            dim=[1, M, K],
            stride=[M * K, K, 1],
            data_type=cudnn.data_type.BFLOAT16,
        )
        B = g.tensor(
            name="B",
            dim=[1, K, N],
            stride=[K * N, 1, K],
            data_type=cudnn.data_type.BFLOAT16,
        )
        bias = g.tensor(
            name="bias",
            dim=[1, 1, N],
            stride=[N, N, 1],
            data_type=cudnn.data_type.BFLOAT16,
        )
        C = g.matmul(A=A, B=B, name="mm")
        Cb = g.bias(input=C, bias=bias, name="bs")
        Y = g.relu(input=Cb, name="r")
        Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    y = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    g({A: a, B: b, bias: bias_t, Y: y})
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref, atol=1e-1, rtol=1e-2)
