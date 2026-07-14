# FROST engines -- design and contributor guide

FROST engines are opt-in, pure-Python execution engines for `cudnn.pygraph`
graphs, JIT-compiled with the CuTe DSL. The native cuDNN path is always the
default; a FROST engine runs only when the user pins it:

```python
import cudnn                          # nothing else: engines self-register lazily

g = cudnn.pygraph(...)
o, _ = g.sdpa(q=q, k=k, v=v, ...)
g.validate()
g.build_operation_graph()
g.create_execution_plans([cudnn.heur_mode.A])
g.select_engines(["sdpa_fwd_prefill_sm100_d512"])   # strict: raises later if ineligible
g.check_support()
g.build_plans()
g.execute({q: q_buf, k: k_buf, v: v_buf, o: o_buf}, workspace)
```

FROST engines are OFF by default; set `NV_CUDNN_FE_ENABLE_FROST_ENGINES=1` to
enable them. That is the whole opt-in and the only environment variable in
the FROST stack; everything else travels as explicit, typed Python
parameters. Users never import op modules to make engine names work:
`import cudnn` installs the (inert) dispatch, and the first probe lazily
imports the in-tree opset modules (`frost/dispatch.py _OPSET_MODULES`) --
deferred so plain `import cudnn` never pays the CuTe-DSL import cost.
Importing `cudnn.sdpa.fwd` explicitly is still how you reach the op's
symbols (`SdpaFwdKnobs`, `engine_name`).

This document is the contract. If you are an agent or a human adding an
engine, a kernel, a knob, or an op: read "The rules" at the bottom first,
then the section for the layer you are touching.


## Where things live

Engine code lives with its operation under `python/cudnn/<op>/`. The frost
directory holds only the shared framework -- no op code, ever.

```
python/cudnn/
  frost/                        framework ONLY
    dispatch.py                 engine registry + pygraph lifecycle patches +
                                strict select + knob plumbing (no ranking
                                logic -- see "Heuristics are per operation")
    template_loader.py          (path, TemplateParams) -> uniquely-named
                                kernel module
    tile_dsl/                   shared CuTe-DSL primitives (barriers, TMA,
                                MMA, softmax pieces, masks). Import by dotted
                                path: cudnn.frost.tile_dsl.<module>

  sdpa/                         ALL sdpa engines, every arch / pass / phase
    graph_analyzer.py           facts extraction: graph.nodes -> SdpaGraphFacts,
                                engine-agnostic, parsed once per graph, cached;
                                plus the shared variant-pack binding helpers
    fwd/
      api.py                    existing experimental engine (handwritten cute)
      api_dsl.py                DSL adapter (APIBase). Arch-free filename:
                                APIs differ by PASS (fwd vs bwd), never by
                                sm version or head dim
      engines.py                Capabilities + EngineSpec table + SdpaFwdKnobs
                                + probe/build
      config_sm100.py           TemplateParams + per-geometry Cfg + raising
                                validation
      kernels/
        prefill_d256_f16_sm100.py     naming: <phase>_d<dim>_<dtype-family>_sm<arch>.py
        prefill_d512_f16_sm100.py
        _common_sm100.py
        thd_sm100.py
    bwd/                        future: same shape, its own api_dsl.py
```

Two levels under the pass directory, always. The coverage axes (arch, phase,
head dim, dtype family) are encoded in filenames and engine names, never in
directory depth. A new reader should be able to list `sdpa/fwd/kernels/` and
see the whole coverage matrix on one screen.

As a layer stack (each layer talks only to its neighbors):

```
+----------------------------------------------------------------+
| user code             pygraph build, select_engines,           |
|                       set_engine_knobs, execute                |
+----------------------------------------------------------------+
| frost/dispatch.py     registry, lifecycle patches, strict      |
|                       select, knob plumbing, lazy opset load   |
+----------------------------------------------------------------+
| opset modules         one per op + pass (cudnn/sdpa/fwd, ...): |
|                       facts analyzer, Capabilities/EngineSpec, |
|                       knobs vocabulary, lower(), APIBase       |
|                       adapter, TemplateParams -> Cfg           |
+----------------------------------------------------------------+
| kernel templates      CuTe-DSL sources; frost/template_loader  |
|                       makes one specialized module per         |
|                       TemplateParams                           |
+----------------------------------------------------------------+
```


## The flow, end to end

What each quickstart line actually does, across the layers:

```
user code                      frost/dispatch.py               opset (e.g. cudnn/sdpa/fwd)
---------                      -----------------               ---------------------------
import cudnn                   lifecycle patches installed
                               (inert unless env var set)

g.sdpa(...)                    (nothing -- graph.nodes
                               records the op natively)

g.create_execution_plans([A])  native cuDNN plans first;
                               lazy-load _OPSET_MODULES;
                               probe every engine  ---------> analyze(graph) -> facts
                                                              (parsed once, cached)
                                                              mismatch(capabilities,
                                                                facts, requested_knobs)
                               eligible = [names ...]

g.set_engine_knobs(K)          store request in plan state;
  (optional)                   re-probe  -------------------> mismatch(..., K) again

g.select_engines([name])       pin the name (strict)

g.check_support()              raise if the pinned engine
                               is not eligible

g.build_plans()                build(spec, graph)  ---------> lower(spec, facts, knobs):
                                                                TemplateParams =
                                                                  facts-derived semantics
                                                                  + knob choices
                                                                load_template(path, params)
                                                                  -> specialized module
                                                                module.compile(shapes)
                               executor cached per graph          (CuTe JIT)

g.execute(vp, workspace)       route to executor  ----------> resolve variant pack
                                                              -> kernel launch
```

And the same flow as data (which record feeds which decision):

```
                static, per engine          per graph, cached
                Capabilities                SdpaGraphFacts <--analyze-- graph.nodes
                       \                       /
                        v                     v
  SdpaFwdKnobs ------> mismatch(capabilities, facts, knobs)
  (optional request)        |
                            +-- reason string --> engine ineligible (probe False /
                            |                     strict select raises)
                            +-- None (eligible) --> lower(spec, facts, knobs)
                                                        |
                                                        v
                                             TemplateParams (frozen)
                                                        |
                                          load_template(path, params)
                                                        |
                                                        v
                                    specialized kernel module (one per params)
                                                        |
                                             module.compile(shapes)
                                                        |
                                                        v
                                     executor(variant_pack) -> kernel launch
```

Reading the two diagrams together: everything left of `lower()` is cheap and
compile-free (facts extraction plus field comparisons -- safe to run on every
graph); everything right of it is the expensive JIT, reached only for the one
engine the user pinned. The records travel one way: facts and knob requests
feed eligibility; eligibility plus the engine's defaults produce
TemplateParams; TemplateParams produces exactly one specialized module.


## Eligibility: Facts, Capabilities, Knobs, TemplateParams

Four records with distinct jobs:

| record | describes | one entry is | lifetime |
|---|---|---|---|
| `SdpaGraphFacts` | what the graph asks for | a single concrete value ("this graph is bf16") | per graph, runtime |
| `Capabilities` | what one ENGINE can serve | an acceptance set or rule ("dtypes = {fp16, bf16}", "skv % 128") | static, per engine row |
| knobs (requested) | tuning the user / heuristics wants forced | one requested value per knob | per graph request |
| `TemplateParams` | how one template instance compiles | one chosen value per compile-time switch | per compiled template |

**Facts** (`graph_analyzer.analyze`) describe what the graph asks for:
geometry, dtype, masks, sink/THD/stats, requested features (bias, dropout,
paged KV, ...). Extraction never judges supportedness. A malformed graph
(K/V shape mismatch, padding mask without seq_len_kv, ...) sets
`facts.invalid` and every engine refuses. Parsed once per graph, cached.

**Capabilities** (`fwd/engines.py`) declare, in the same vocabulary, the
envelope one ENGINE can serve -- including the tuning-knob domains it honors.
An engine is a lowering strategy that may span several kernels (and several
engines may share a kernel); its row declares what its lowering can actually
deliver.

**The probe checks both facts and requested knobs against capabilities:**

```
eligible = mismatch(capabilities, facts)            # graph semantics
       and mismatch(capabilities, requested_knobs)  # tuning requests
```

Both checks run before any compile, field by field, returning the first
human-readable reason the engine cannot serve the request (or None). Engines
with different feature envelopes (one has paged-KV, another has sinks) are
just different rows; there is no shared if-ladder that must know about every
engine.

**TemplateParams is the output of a successful match, not an input.** After
the probe passes, the engine's `lower` hook assembles the frozen record the
loader injects into the kernel template: graph-derived semantics plus knob
choices (requested values where given, engine defaults otherwise).
`config._validate_params` / `make_cfg_*` re-validate that record and raise
`ValueError`, but that is a backstop: reaching it means a `Capabilities` row
is dishonest, not that a user did something wrong.


### Feature interactions: the box and the notches

Field-wise capabilities describe an axis-aligned box in feature space (the
product of per-axis sets: dtypes x masks x layouts x ...). Real support
surfaces are almost boxes, with NOTCHES cut where feature conjunctions break
(bottom-right + SWA; THD + stats; a future mxfp8 + SWA + dropout). Both are
expressible, with one discipline separating them:

- The box is pure data: per-axis fields on `Capabilities`. Covers most of
  the surface; adding an engine is writing a row, not logic.
- A notch is a rule in `mismatch()` gated by a conjunction flag on the row
  (e.g. `bottom_right_with_swa: bool`). The matcher encodes the SHAPE of the
  interaction once; each engine's row supplies the VERDICT. When a future
  kernel supports the conjunction, flip its flag -- never edit the matcher.
  This is what keeps interaction checks from regressing into a per-engine
  if-ladder: shared code may know about kinds of interactions, never about
  specific engines.
- Escape hatch for a truly one-engine oddity (if one ever exists): an
  optional `extra_mismatch(facts, requested) -> reason | None` hook on its
  `EngineSpec` -- engine-local code, same reason-string contract, still runs
  in the probe so the "ValueError past probe() is a capabilities bug" rule
  holds. Not built until a constraint needs it.


## The knob request channel (no global enum)

The C++ backend forces one global `knobType_t` enum on every engine of every
op -- an ABI constraint, not a design ideal. Here knobs are scoped at three
levels, with no shared vocabulary at all:

- **Vocabulary per operation.** Each op defines a typed, frozen dataclass:
  `cudnn.sdpa.fwd.SdpaFwdKnobs(sched_policy=None, tile_n=None, cga=None)`,
  where `None` means "no preference". SDPA's knobs cannot collide with
  GEMM's; fields have real types instead of enum-plus-int64.
- **Domains per engine.** Each `Capabilities` row advertises the values its
  lowering honors: `sched_policies = {NATURAL}`, `tile_ns = {128}`,
  `cgas = {2}`. Two engines of the same op may honor different subsets.
- **Requests per graph.** `graph.set_engine_knobs(SdpaFwdKnobs(...))` -- a
  `cudnn.pygraph` method installed by the frost lifecycle patch. The request
  lives in the graph's plan state, and setting it re-probes already-probed
  engines. Two graphs in one process can request different tunings of the
  same engine; each distinct `TemplateParams` compiles into its own module.

The probe fetches the request via `cudnn.frost.dispatch.requested_knobs`
and validates it inside the same `mismatch()` that checks the facts. A value
outside the engine's domain makes the engine ineligible with a readable
reason; a wrong-op knob object is rejected outright. Combined with strict
`select_engines`, pinning an engine that cannot honor the request raises at
`check_support`.

**A knob is honored or the engine is ineligible -- never silently degraded.**
If a kernel cannot run the requested scheduler policy, the answer is "this
engine cannot serve this request", not "ran with a different policy".

Generic discoverability survives without the enum: knob domains are ordinary
dataclass fields on `Capabilities`, so "list every engine and the knobs it
honors" is a `dataclasses.fields()` walk over the registry.


## Engines are lowering strategies, not kernels

The engine-to-kernel mapping is many-to-many by design:

- One engine serves several dtypes through one template: the d512 engine
  lowers fp16 and bf16 graphs to `prefill_d512_f16_sm100.py` with different
  TemplateParams.
- One engine can drive several kernels: `EngineSpec.lower` is a hook
  `(spec, facts, requested_knobs) -> executor`. The default
  (`lower_dsl_prefill`) compiles one template, but an engine may pick between
  kernels (decode vs prefill by S_q) or chain launches -- the THD path
  already runs an O-descriptor builder kernel before the main one.


## Heuristics are per operation (future seam)

`frost/dispatch.py` is exactly what its name says: registry, lifecycle
patches, strict select, knob-request plumbing -- no ranking intelligence.
Eligible engines are listed in registration order and the user pins one
explicitly.

Real heuristics (the analogue of the C++ per-op heur files such as
`jit_engine_heur_sdpa.cpp`) belong next to the op's facts and knobs types,
because ranking knowledge is op-specific: what makes one SDPA engine beat
another (seqlen regime, GQA ratio, causal fraction) is meaningless for GEMM.
The contract, for when a cell has more than one engine or a knob domain has
more than one value:

- Home: `cudnn/<op>/<pass>/heuristics.py`.
- Signature: `rank(facts, eligible_specs, device) -> ordered list of
  (EngineSpec, proposed <Op>Knobs)` -- order the eligible engines and propose
  knob values for each, mirroring the C++ heur returning ordered engine
  configs with knob choices.
- Dispatch keeps only the protocol: probe for eligibility, ask the op's
  ranker for order and proposals, list native cuDNN plans first.
- Knob precedence: **user request > heuristic proposal > engine default** --
  with the same rule at every level: a proposal outside the engine's
  `Capabilities` domain is a heuristic bug; a user request outside it makes
  the engine ineligible.

Do not build the ranker before it has observable behavior to test (today it
would be the identity function). The seam is documented here so the contract
is fixed before anyone needs it.


## Opsets: graph-to-operation mapping is implicit and multi-valued

The C++ backend maps each graph to exactly one opset (a central pattern
matcher with precedence), and engines register against opsets. There is no
opset enum here, deliberately -- the same reasoning as the knob enum:

- **In code, an opset is one module** (op + pass granularity):
  `cudnn.sdpa.fwd.engines` is the SDPA-forward opset, `cudnn.frost.gemm` the
  GEMM one. Importing it registers every engine it serves; the in-tree list
  lives in `frost/dispatch._OPSET_MODULES` and loads lazily at first probe.
- **Each op's analyzer IS its pattern matcher.** `analyze()` returning facts
  means "this graph is an instance of my operation"; returning None means
  "not mine". Membership is a predicate the op owns, not an entry in a
  central table someone must maintain. The per-op facts cache keeps this
  cheap: N engines of an op share one parse, and a graph of some other op
  costs one failed node-type check.
- **Multiple ops may claim one graph.** Nothing prevents two ops' analyzers
  from both matching (a matmul graph claimed by gemm engines and by a future
  fused-epilogue op's engines). This is coherent because a FROST engine's
  `build` executes the ENTIRE graph: a claim is a complete alternative
  execution strategy, never a partition -- so claims compete, they cannot
  conflict. All eligible engines land in the same flat list; strict
  `select_engines` resolves the competition today.
- **Ranking composes in two levels.** Within an op: the op's ranker (see
  "Heuristics are per operation"). Across ops: do NOT resolve by matcher
  precedence (that is the opset enum reborn); instead rankers return a
  common currency -- estimated cost -- and dispatch sorts the union of all
  ops' candidates by it, native cuDNN plans first. Cross-op order is
  cosmetic until auto-selection exists, so none of this is built yet; the
  shape (flat registry, whole-graph claims, per-op rankers with comparable
  scores) is what must not be broken.


## Engine naming

```
sdpa_<pass>_<phase>_sm<arch>_d<dqk>[x<dv>]

sdpa_fwd_prefill_sm100_d512
sdpa_fwd_prefill_sm100_d256
sdpa_fwd_decode_sm100_d256          (future)
sdpa_bwd_sm100_d128                 (future)
```

- dtype is deliberately NOT part of the name: one cell's engine serves every
  dtype its kernel handles (fp16 and bf16 today, via `Capabilities.dtypes`),
  so users do not switch engine strings when they flip precision.
- `d<dqk>x<dv>` only when the two head dims differ.
- No version counters. If a genuinely distinct second engine ever serves the
  same cell, give it a descriptive variant suffix (e.g. `_cga4`), not a
  number.
- `select_engines` is strict: if the probe rejected the graph (or the knob
  request), `check_support` / `build_plans` / `execute` raise instead of
  silently running native cuDNN.
- `cudnn.sdpa.fwd.engines.engine_name(d)` computes names programmatically.


## Kernel templates and TemplateParams

Compile-time kernel parameters travel as a frozen `TemplateParams` dataclass
(graph-derived semantics + chosen knob values). The shared loader
(`frost/template_loader.py`) executes each kernel template into a
uniquely-named module per `(path, params)`, injecting them as the module
global `FROST_TEMPLATE_PARAMS` before the body runs. Multiple parameter sets
coexist in one process; nothing is reloaded, nothing is popped from
`sys.modules`, no `sys.path` entries are added, no environment variables are
read or written.

A kernel template:

- imports shared primitives by dotted path
  (`from cudnn.frost.tile_dsl.mma import mma_ss`) -- never via `sys.path`
  manipulation or bare top-level names;
- builds its config once at import:
  `PARAMS = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())` then
  `CFG, _TMA = make_cfg_d<dim>(PARAMS)` -- the plain-import default keeps
  `python kernels/<file>.py` (standalone benchmark) working;
- treats `CFG.*` fields as compile-time constants (`cutlass.const_expr`,
  `cutlass.range_constexpr`) so each parameter set traces to specialized code;
- exposes `compile(b, qh, kh, sq, skv) -> callable` with an `@lru_cache`
  (per-shape cache; the per-parameter split already happened at module load).

Asserts:

- Anything derived from user input raises `ValueError` inside `make_cfg_*` --
  never a module-level `assert` (stripped under `python -O`, and an
  import-time crash is undebuggable from the frontend).
- Hardware-invariant geometry checks in a template use a raising helper (see
  `_require` in `prefill_d512_f16_sm100.py`) and must be unreachable for any
  parameter set the engine's capabilities admit.
- Never `assert api.check_support()` -- it raises on failure and the assert
  is stripped under `-O`; call it plainly.


## Adding coverage, cheapest first

1. **New dtype an existing template already handles**: add it to the row's
   `Capabilities.dtypes`. **New geometry**: one new `EngineSpec` row.
2. **New knob**: add the field to the op's knobs dataclass (`None` default),
   the domain to `Capabilities`, the check line to `mismatch()`, and the
   merge in `lower`. Update the gating tests.
3. **New kernel (decode, fp8, new arch)**: write the template in
   `<op>/<pass>/kernels/` following the naming grammar; add its `make_cfg_*`
   to the config module (a new `config_sm90.py` for a new arch); add the spec
   row with an honest `Capabilities`. `api_dsl.py` and `graph_analyzer.py`
   should not need changes.
4. **New pass (bwd)**: new `<op>/bwd/` with its own `api_dsl.py` (the tensor
   contract differs), reusing the shared analyzer facts and the loader.
5. **New op**: new `python/cudnn/<op>/` with the same layers (analyzer facts,
   engines/capabilities, kernels); register via
   `cudnn.frost.register_engine`.


## The rules (read before changing anything)

1. **Facts never judge; capabilities never parse.** The analyzer describes,
   the engine rows accept or reject. If you find yourself writing an
   if-ladder that knows about specific kernels inside shared code, stop --
   that logic belongs in a `Capabilities` row.
2. **Every kernel constraint appears in its engine's `Capabilities`.** A
   `ValueError` escaping a kernel template or `make_cfg_*` after
   `probe() == True` is a bug in the capabilities row, and there must be a
   probe test for the constraint (accept AND reject).
3. **A knob is honored or the engine is ineligible.** Never substitute,
   never silently degrade. Precedence when heuristics land:
   user request > heuristic proposal > engine default.
4. **No global vocabularies.** Knob dataclasses are per operation; no shared
   enum, no shared registry of knob meanings.
5. **No environment variables** for configuration (the single opt-in
   `NV_CUDNN_FE_ENABLE_FROST_ENGINES` excepted). Parameters travel as typed
   dataclasses through the loader.
6. **No monkey-patching of op builders and no import side effects** beyond
   `register_engine` in opset modules (the lifecycle patches are the
   framework's job, installed once by `cudnn.frost`). The pygraph IR
   (`graph.nodes`) already records everything; read it after the graph is
   built.
7. **No module-level asserts on anything a user could trip.** Raise
   `ValueError` in validation functions; keep `assert` for programmer
   invariants inside tile_dsl at most.
8. **Two directory levels under the pass, maximum.** Coverage axes go in
   filenames and engine names, never in directory depth.
9. **Identifiers are keyed by op geometry, not model names.** `make_cfg_d512`,
   not `make_cfg_dsv4`; model provenance goes in comments only.
10. **Tests gate everything.** New capabilities field -> accept and reject
    probe tests. New knob -> gating tests (in-domain, out-of-domain, wrong
    vocabulary). Mark test modules `pytest.mark.L0` -- the default pytest
    addopts is `-m L0` and unmarked tests silently never run. Run
    `ci/run_style_check_diff.sh --apply` (black, 160 cols) before pushing.
11. **Keep this document true.** If code and this contract disagree and you
    change the code, change this file in the same commit.
