"""Shared TBD engine dispatch for the ``cudnn.TBD`` OSS engines.

TBD engines are named engines appended (by name, e.g. GEMM = ``TBD_eng0``) to the
plan list produced by the native cuDNN heuristics; the default stays cuDNN and
the user pins TBD via ``select_engines(["TBD_eng0"])`` (drop via
``deselect_engines``). Implemented as a named-engine registry + per-graph shadow
plan-state layered over the native ``cudnn.pygraph`` lifecycle. Pure Python — the
C++ side is untouched.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any, Callable

import cudnn

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named-engine registry
# ---------------------------------------------------------------------------

# Each registered engine: name -> (probe, build).
#   probe(graph) -> bool     cheap eligibility (analyze + gates), NO compile
#   build(graph) -> callable expensive JIT; returns compiled(variant_pack)
_ENGINES: "dict[str, tuple[Callable[[Any], bool], Callable[[Any], Any]]]" = {}


def register_engine(name: str, probe: Callable[[Any], bool], build: Callable[[Any], Any]) -> None:
    """Register a named TBD engine (idempotent). ``name`` (``TBD_eng<N>``) is what
    ``select_engines`` / ``deselect_engines`` match."""
    _ENGINES[name] = (probe, build)


def engine_names() -> "list[str]":
    """Registered TBD engine names, in registration order."""
    return list(_ENGINES)


def is_tbd_engine(name: Any) -> bool:
    """True if ``name`` is a registered TBD engine name."""
    return isinstance(name, str) and name in _ENGINES


# ---------------------------------------------------------------------------
# Per-graph shadow plan-state (independent of cuDNN's native plan list)
# ---------------------------------------------------------------------------

_PLAN_STATES: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


def _plan_state(graph: cudnn.pygraph) -> dict:
    state = _PLAN_STATES.get(graph)
    if state is None:
        state = {
            "eligible": [],  # TBD engine names whose probe passed (appended order)
            "selected": None,  # name pinned via select_engines, else None
            "barred": set(),  # names removed via deselect_engines
            "compiled": {},  # name -> compiled plan (filled lazily)
        }
        _PLAN_STATES[graph] = state
    return state


def _get_plan_state(graph: cudnn.pygraph) -> "dict | None":
    return _PLAN_STATES.get(graph)


def _active_tbd(graph: cudnn.pygraph) -> "str | None":
    """The TBD engine name that should run, or None (→ cuDNN). TBD runs only when
    explicitly selected."""
    state = _get_plan_state(graph)
    if state is None:
        return None
    sel = state["selected"]
    if sel is not None and sel in state["eligible"] and sel not in state["barred"]:
        return sel
    return None


def _live_tbd_engines(state: dict) -> "list[str]":
    """Eligible-and-not-barred TBD engines, in appended order."""
    return [n for n in state["eligible"] if n not in state["barred"]]


def _probe_and_append(graph: cudnn.pygraph) -> None:
    """Probe every registered TBD engine and record the eligible ones (no compile)."""
    state = _plan_state(graph)
    state["eligible"] = []
    for name, (probe, _build) in _ENGINES.items():
        try:
            ok = probe(graph)
        except Exception:  # noqa: BLE001  — a probe must never break the native path
            _LOG.debug(
                "cudnn.TBD: probe for %s raised; treating as ineligible",
                name,
                exc_info=True,
            )
            ok = False
        if ok:
            state["eligible"].append(name)


def _compile_selected(graph: cudnn.pygraph) -> Any:
    """JIT-compile the selected TBD engine (lazy) and cache it. Returns the plan."""
    state = _plan_state(graph)
    name = state["selected"]
    if name in state["compiled"]:
        return state["compiled"][name]
    _probe, build = _ENGINES[name]
    compiled = build(graph)  # the expensive cute.compile — may raise on rejection
    state["compiled"][name] = compiled
    return compiled


# ---------------------------------------------------------------------------
# cudnn.pygraph lifecycle patches — layer TBD over the native path
# ---------------------------------------------------------------------------

_INSTALLED = False
_ORIGINALS: dict[str, Any] = {}


def _patched_create_execution_plans(self, *args, **kwargs):
    # Native cuDNN builds its plans first. Tolerate a native rejection so an
    # eligible TBD engine is still offered; if neither can, re-raise cuDNN's error.
    native_exc = None
    result = None
    try:
        result = _ORIGINALS["create_execution_plans"](self, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        native_exc = exc
    # Append eligible TBD engines after cuDNN's (cheap probe, no compile).
    _probe_and_append(self)
    if native_exc is not None and not _plan_state(self)["eligible"]:
        raise native_exc
    return result


def _patched_build(self, *args, **kwargs):
    # build([...]) = validate + build_operation_graph + create_execution_plans +
    # check_support + build_plans. Run the native prologue, then probe TBD.
    native_exc = None
    result = None
    try:
        result = _ORIGINALS["build"](self, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        native_exc = exc
    _probe_and_append(self)
    if native_exc is not None and not _plan_state(self)["eligible"]:
        raise native_exc
    return result


def _patched_select_engines(self, engine_names):
    """Select engines by name. TBD names pin the TBD engine; non-TBD select is
    best-effort (cuDNN keeps its own candidate — no C++ hook)."""
    if isinstance(engine_names, str):
        engine_names = [engine_names]
    state = _plan_state(self)
    tbd = [n for n in engine_names if is_tbd_engine(n)]
    if tbd:
        # Pin the first named TBD engine (one per graph in practice).
        state["selected"] = tbd[0]
    return None


def _patched_deselect_engines(self, engine_names):
    if isinstance(engine_names, str):
        engine_names = [engine_names]
    state = _plan_state(self)
    tbd = [n for n in engine_names if is_tbd_engine(n)]
    for n in tbd:
        state["barred"].add(n)
        if state["selected"] == n:
            state["selected"] = None
    rest = [n for n in engine_names if not is_tbd_engine(n)]
    if rest:
        return _ORIGINALS["deselect_engines"](self, rest)
    return self


def _patched_check_support(self, *args, **kwargs):
    # TBD eligibility already decided by the probe; native check for cuDNN plans.
    if _active_tbd(self) is not None:
        return None
    return _ORIGINALS["check_support"](self, *args, **kwargs)


def _patched_build_plans(self, *args, **kwargs):
    # Selected TBD engine → lazy JIT-compile now; else native build.
    if _active_tbd(self) is not None:
        _compile_selected(self)
        return None
    return _ORIGINALS["build_plans"](self, *args, **kwargs)


def _patched_execute(self, *args, **kwargs):
    if _active_tbd(self) is not None:
        compiled = _compile_selected(self)  # cached if build_plans already ran
        variant_pack = args[0] if args else kwargs.get("tensor_to_device_buffer")
        # TBD kernel takes only the variant pack (infers sizes from buffer shapes,
        # needs no cuDNN workspace / handle).
        return compiled(variant_pack)
    return _ORIGINALS["execute"](self, *args, **kwargs)


def _patched_get_workspace_size(self, *args, **kwargs):
    if _active_tbd(self) is not None:
        return 0  # the TBD kernel manages its own workspace
    return _ORIGINALS["get_workspace_size"](self, *args, **kwargs)


def _patched_get_execution_plan_count(self, *args, **kwargs):
    native = _ORIGINALS["get_execution_plan_count"](self, *args, **kwargs)
    state = _get_plan_state(self)
    if state is None:
        return native
    return native + len(_live_tbd_engines(state))


def install_lifecycle_patches() -> None:
    """Monkey-patch the ``cudnn.pygraph`` lifecycle to layer TBD engines over the
    native plan list. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    patches = {
        "create_execution_plans": _patched_create_execution_plans,
        "build": _patched_build,
        "select_engines": _patched_select_engines,
        "deselect_engines": _patched_deselect_engines,
        "check_support": _patched_check_support,
        "build_plans": _patched_build_plans,
        "execute": _patched_execute,
        "get_workspace_size": _patched_get_workspace_size,
        "get_execution_plan_count": _patched_get_execution_plan_count,
    }
    for name, patched in patches.items():
        # select_engines has no C++ original — getattr default None handles it.
        _ORIGINALS[name] = getattr(cudnn.pygraph, name, None)
        setattr(cudnn.pygraph, name, patched)
    _INSTALLED = True
