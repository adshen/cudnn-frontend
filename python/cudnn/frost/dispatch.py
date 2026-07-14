"""Shared FROST engine DISPATCH: registry, pygraph lifecycle patches, strict
select, knob-request plumbing. Deliberately no ranking intelligence — real
heuristics are per operation and live next to the op's facts/knobs types when
they earn their existence (see "Heuristics are per operation" in README.md).


FROST engines are named engines appended (by name, e.g. GEMM = ``frost_gemm_eng0``) to the
plan list produced by the native cuDNN heuristics; the default stays cuDNN and
the user pins FROST via ``select_engines(["frost_gemm_eng0"])`` (drop via
``deselect_engines``). Implemented as a named-engine registry + per-graph shadow
plan-state layered over the native ``cudnn.pygraph`` lifecycle. Pure Python — the
C++ side is untouched.

FROST engines are OFF by default; set ``NV_CUDNN_FE_ENABLE_FROST_ENGINES=1`` to
enable them (see :func:`frost_engines_enabled`).
"""

from __future__ import annotations

import logging
import os
import weakref
from typing import Any, Callable

import cudnn

_LOG = logging.getLogger(__name__)

# FROST engines are OFF by default; opt in via this env var.
_ENABLE_ENV = "NV_CUDNN_FE_ENABLE_FROST_ENGINES"

# The in-tree OPSETS: one module per operation family (op + pass), imported
# lazily at first probe when FROST is enabled. An opset module owns one
# operation's engines end to end — its analyzer's pattern match is the
# graph->opset membership test, and importing it runs the register_engine()
# calls for every engine it serves (see "Opsets" in frost/README.md). This
# tuple is a manifest, not a registry of behavior; the env var stays the ONE
# user-facing opt-in — users never need to know which module registers which
# engine name. Out-of-tree opsets can still register by importing their own
# module before building plans (or via entry points, if that ever lands).
_OPSET_MODULES = (
    "cudnn.frost.gemm",
    "cudnn.sdpa.fwd.engines",
)
_OPSETS_LOADED = False


def _ensure_opsets_loaded() -> None:
    """Lazily import the in-tree opset modules (once) when FROST is enabled.

    Deferred to first probe so that `import cudnn` never pays the opsets'
    import cost (torch / CuTe DSL) for users who never enable FROST."""
    global _OPSETS_LOADED
    if _OPSETS_LOADED or not frost_engines_enabled():
        return
    _OPSETS_LOADED = True
    import importlib

    for mod in _OPSET_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001 — one opset's import failure must not kill the rest
            _LOG.warning("cudnn.frost: failed to import opset module %s", mod, exc_info=True)


def frost_engines_enabled() -> bool:
    """Whether FROST engines are enabled (env ``NV_CUDNN_FE_ENABLE_FROST_ENGINES``;
    default off). Read live so it can be toggled per process / test."""
    return os.environ.get(_ENABLE_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Named-engine registry
# ---------------------------------------------------------------------------

# Each registered engine: name -> (probe, build).
#   probe(graph) -> bool     cheap eligibility (analyze + gates), NO compile
#   build(graph) -> callable expensive JIT; returns compiled(variant_pack)
_ENGINES: "dict[str, tuple[Callable[[Any], bool], Callable[[Any], Any]]]" = {}


def register_engine(name: str, probe: Callable[[Any], bool], build: Callable[[Any], Any]) -> None:
    """Register a named FROST engine (idempotent). ``name`` is what
    ``select_engines`` / ``deselect_engines`` match (e.g. ``frost_gemm_eng0``,
    ``sdpa_fwd_prefill_sm100_d512`` — naming is the op's choice; see the op's
    engines module and python/cudnn/frost/README.md)."""
    _ENGINES[name] = (probe, build)


def engine_names() -> "list[str]":
    """Registered FROST engine names, in registration order (loads the
    in-tree opset modules first if FROST is enabled)."""
    _ensure_opsets_loaded()
    return list(_ENGINES)


def is_frost_engine(name: Any) -> bool:
    """True if ``name`` is a registered FROST engine name."""
    return isinstance(name, str) and name in _ENGINES


# ---------------------------------------------------------------------------
# Per-graph shadow plan-state (independent of cuDNN's native plan list)
# ---------------------------------------------------------------------------

_PLAN_STATES: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


def _plan_state(graph: cudnn.pygraph) -> dict:
    state = _PLAN_STATES.get(graph)
    if state is None:
        state = {
            "eligible": [],  # FROST engine names whose probe passed (appended order)
            "selected": None,  # name pinned via select_engines, else None
            "barred": set(),  # names removed via deselect_engines
            "compiled": {},  # name -> compiled plan (filled lazily)
            "knobs": None,  # per-graph tuning request (op-specific dataclass), see set_engine_knobs
            "probed": False,  # whether _probe_and_append has run (re-probe trigger for knob changes)
        }
        _PLAN_STATES[graph] = state
    return state


def _get_plan_state(graph: cudnn.pygraph) -> "dict | None":
    return _PLAN_STATES.get(graph)


def _active_frost(graph: cudnn.pygraph) -> "str | None":
    """The FROST engine name that should run, or None (→ cuDNN). FROST runs only when
    explicitly selected."""
    state = _get_plan_state(graph)
    if state is None:
        return None
    sel = state["selected"]
    if sel is not None and sel in state["eligible"] and sel not in state["barred"]:
        return sel
    return None


def _live_frost_engines(state: dict) -> "list[str]":
    """Eligible-and-not-barred FROST engines, in appended order."""
    return [n for n in state["eligible"] if n not in state["barred"]]


def _probe_and_append(graph: cudnn.pygraph) -> None:
    """Probe every registered FROST engine and record the eligible ones (no compile)."""
    state = _plan_state(graph)
    state["eligible"] = []
    state["probed"] = True
    if not frost_engines_enabled():
        return
    _ensure_opsets_loaded()
    for name, (probe, _build) in _ENGINES.items():
        try:
            ok = probe(graph)
        except Exception:  # noqa: BLE001  — a probe must never break the native path
            _LOG.debug(
                "cudnn.frost: probe for %s raised; treating as ineligible",
                name,
                exc_info=True,
            )
            ok = False
        if ok:
            state["eligible"].append(name)


def _compile_selected(graph: cudnn.pygraph) -> Any:
    """JIT-compile the selected FROST engine (lazy) and cache it. Returns the plan."""
    state = _plan_state(graph)
    name = state["selected"]
    if name in state["compiled"]:
        return state["compiled"][name]
    _probe, build = _ENGINES[name]
    compiled = build(graph)  # the expensive cute.compile — may raise on rejection
    state["compiled"][name] = compiled
    return compiled


# ---------------------------------------------------------------------------
# cudnn.pygraph lifecycle patches — layer FROST over the native path
# ---------------------------------------------------------------------------

_INSTALLED = False
_ORIGINALS: dict[str, Any] = {}


def _patched_create_execution_plans(self, *args, **kwargs):
    # Native cuDNN builds its plans first. Tolerate a native rejection so an
    # eligible FROST engine is still offered; if neither can, re-raise cuDNN's error.
    native_exc = None
    result = None
    try:
        result = _ORIGINALS["create_execution_plans"](self, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        native_exc = exc
    # Append eligible FROST engines after cuDNN's (cheap probe, no compile).
    _probe_and_append(self)
    if native_exc is not None and not _plan_state(self)["eligible"]:
        raise native_exc
    return result


def _patched_build(self, *args, **kwargs):
    # build([...]) = validate + build_operation_graph + create_execution_plans +
    # check_support + build_plans. Run the native prologue, then probe FROST.
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
    """Select engines by name. FROST names pin the FROST engine; non-FROST select is
    best-effort (cuDNN keeps its own candidate — no C++ hook)."""
    if isinstance(engine_names, str):
        engine_names = [engine_names]
    # A user may select before create_execution_plans has probed; make sure
    # the in-tree opsets are loaded so their engine names are recognized.
    _ensure_opsets_loaded()
    state = _plan_state(self)
    frost = [n for n in engine_names if is_frost_engine(n)]
    if frost:
        # Pin the first named FROST engine (one per graph in practice).
        state["selected"] = frost[0]
    return None


def _patched_deselect_engines(self, engine_names):
    if isinstance(engine_names, str):
        engine_names = [engine_names]
    state = _plan_state(self)
    frost = [n for n in engine_names if is_frost_engine(n)]
    for n in frost:
        state["barred"].add(n)
        if state["selected"] == n:
            state["selected"] = None
    rest = [n for n in engine_names if not is_frost_engine(n)]
    if rest:
        return _ORIGINALS["deselect_engines"](self, rest)
    return self


def _raise_if_selected_but_ineligible(graph: cudnn.pygraph) -> None:
    """A selected FROST engine must run — never silently fall back to native.

    ``select_engines([<frost name>])`` records the pin; if the probe deemed the
    graph ineligible (or the engine was deselected afterwards), running the
    native path instead would be a silent false positive, so fail loudly.
    """
    state = _get_plan_state(graph)
    if state is None:
        return
    sel = state["selected"]
    if sel is not None and _active_frost(graph) is None:
        raise ValueError(
            f"cudnn.frost: engine {sel!r} was selected via select_engines() but "
            f"is not eligible for this graph (probe rejected it or it was "
            f"deselected); eligible FROST engines: {_live_frost_engines(state)}"
        )


def _patched_set_engine_knobs(self, knobs):
    """Attach a per-graph tuning request (an op-specific knobs dataclass, e.g.
    ``cudnn.sdpa.fwd.SdpaFwdKnobs``). Knob requests change eligibility — an
    engine that cannot honor a requested value is ineligible — so if the
    engines were already probed, they are re-probed against the new request.
    There is deliberately no global knob enum: each operation defines its own
    typed vocabulary, each engine advertises the domains it honors."""
    state = _plan_state(self)
    state["knobs"] = knobs
    state["compiled"] = {}  # a different request may lower differently
    # Re-probe on the "probed" flag, NOT on a non-empty eligible list: if a
    # previous request made every engine ineligible, eligible is [] but a new
    # (honorable) request must still restore eligibility.
    if state["probed"]:
        _probe_and_append(self)
    return self


def requested_knobs(graph: cudnn.pygraph):
    """The tuning request attached via ``set_engine_knobs``, or None."""
    state = _get_plan_state(graph)
    return None if state is None else state.get("knobs")


def _patched_check_support(self, *args, **kwargs):
    # FROST eligibility already decided by the probe; native check for cuDNN plans.
    _raise_if_selected_but_ineligible(self)
    if _active_frost(self) is not None:
        return None
    return _ORIGINALS["check_support"](self, *args, **kwargs)


def _patched_build_plans(self, *args, **kwargs):
    # Selected FROST engine → lazy JIT-compile now; else native build.
    _raise_if_selected_but_ineligible(self)
    if _active_frost(self) is not None:
        _compile_selected(self)
        return None
    return _ORIGINALS["build_plans"](self, *args, **kwargs)


def _patched_execute(self, *args, **kwargs):
    _raise_if_selected_but_ineligible(self)
    if _active_frost(self) is not None:
        compiled = _compile_selected(self)  # cached if build_plans already ran
        variant_pack = args[0] if args else kwargs.get("tensor_to_device_buffer")
        # FROST kernel takes only the variant pack (infers sizes from buffer shapes,
        # needs no cuDNN workspace / handle).
        return compiled(variant_pack)
    return _ORIGINALS["execute"](self, *args, **kwargs)


def _patched_get_workspace_size(self, *args, **kwargs):
    if _active_frost(self) is not None:
        return 0  # the FROST kernel manages its own workspace
    return _ORIGINALS["get_workspace_size"](self, *args, **kwargs)


def _patched_get_execution_plan_count(self, *args, **kwargs):
    native = _ORIGINALS["get_execution_plan_count"](self, *args, **kwargs)
    state = _get_plan_state(self)
    if state is None:
        return native
    return native + len(_live_frost_engines(state))


def install_lifecycle_patches() -> None:
    """Monkey-patch the ``cudnn.pygraph`` lifecycle to layer FROST engines over the
    native plan list. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    patches = {
        "create_execution_plans": _patched_create_execution_plans,
        "build": _patched_build,
        "select_engines": _patched_select_engines,
        "deselect_engines": _patched_deselect_engines,
        "set_engine_knobs": _patched_set_engine_knobs,
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
