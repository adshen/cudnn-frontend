"""Functional-verification driver for CI and agent-callable correctness gates.

One entrypoint that runs the right gate subset per tier with consistent env and
emits both a pretty and a machine-readable summary. Tiers: ``smoke`` (unit
tests, <30s), ``quick`` (+ sweeps + examples, <5min), ``full`` (+ full 384-config
matmul sweep, ~30min). Exit code 0 = clean, 1 = real failure (incl. strict
xpass), 2 = harness broke. Usage: ``python -m cudnn.frost.gemm.verify [--tier ...]
[--json] [--list] [-k EXPR]``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

# ---------------------------------------------------------------------------
# Layout — find project root by walking up from this file
# ---------------------------------------------------------------------------


_THIS_FILE = Path(__file__).resolve()
# parents[4] = cudnn-frontend root; tests/examples under test/python/frost/gemm/.
_REPO_ROOT = _THIS_FILE.parents[4]
_TESTS_DIR = _REPO_ROOT / "test" / "python" / "frost" / "gemm"
_EXAMPLES_DIR = _TESTS_DIR / "examples"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GateResult:
    """Outcome of one named gate (one pytest run or one example).

    ``xfailed`` = expected failures (known-issue, not regressions); ``xpassed``
    (strict unexpected pass) is a real failure and also appears in ``failures``.
    """

    name: str
    duration_s: float
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    failures: list[dict] = dataclasses.field(default_factory=list)
    error: str | None = None  # set when the gate itself crashed

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped + self.xfailed + self.xpassed

    @property
    def is_clean(self) -> bool:
        return self.failed == 0 and self.xpassed == 0 and self.error is None


@dataclasses.dataclass
class TierResult:
    tier: str
    duration_s: float
    gates: list[GateResult]

    @property
    def is_clean(self) -> bool:
        return all(g.is_clean for g in self.gates)

    @property
    def exit_code(self) -> int:
        if any(g.error for g in self.gates):
            return 2
        return 0 if self.is_clean else 1

    def to_dict(self) -> dict:
        totals = {
            "passed": sum(g.passed for g in self.gates),
            "failed": sum(g.failed for g in self.gates),
            "skipped": sum(g.skipped for g in self.gates),
            "xfailed": sum(g.xfailed for g in self.gates),
            "xpassed": sum(g.xpassed for g in self.gates),
        }
        return {
            "tier": self.tier,
            "duration_s": round(self.duration_s, 2),
            "exit_code": self.exit_code,
            "is_clean": self.is_clean,
            "totals": totals,
            "gates": [
                {
                    "name": g.name,
                    "duration_s": round(g.duration_s, 2),
                    "passed": g.passed,
                    "failed": g.failed,
                    "skipped": g.skipped,
                    "xfailed": g.xfailed,
                    "xpassed": g.xpassed,
                    "is_clean": g.is_clean,
                    "error": g.error,
                    "failures": g.failures,
                }
                for g in self.gates
            ],
        }


# ---------------------------------------------------------------------------
# Pytest invocation + JUnit XML parsing
# ---------------------------------------------------------------------------


def _parse_junit(xml_path: Path) -> tuple[int, int, int, int, int, list[dict]]:
    """Return (passed, failed, skipped, xfailed, xpassed, failures).

    JUnit encodes xfail via ``skipped type=pytest.xfail``; a strict xpass is
    reported as a ``failure`` (→ counted as failed above).
    """
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()
    testcases = root.findall(".//testcase")
    passed = failed = skipped = xfailed = xpassed = 0
    failures: list[dict] = []
    for tc in testcases:
        skip = tc.find("skipped")
        fail = tc.find("failure")
        err = tc.find("error")
        cls = tc.get("classname", "")
        name = tc.get("name", "")
        case_id = f"{cls}::{name}" if cls else name
        if err is not None:
            failed += 1
            failures.append(
                {
                    "id": case_id,
                    "kind": "error",
                    "msg": (err.get("message") or "").splitlines()[0][:300],
                }
            )
            continue
        if skip is not None:
            stype = skip.get("type", "")
            if stype.endswith("xfail"):
                xfailed += 1  # strict xpass is a `failure`, handled above
            else:
                skipped += 1
            continue
        if fail is not None:
            # Regular failure or strict xpass — either way a regression.
            failed += 1
            msg = (fail.get("message") or "").splitlines()[0][:300]
            failures.append({"id": case_id, "kind": "fail", "msg": msg})
            continue
        passed += 1
    return passed, failed, skipped, xfailed, xpassed, failures


def _run_pytest(
    name: str,
    targets: Iterable[str],
    extra_args: Iterable[str] = (),
    env_overrides: dict[str, str] | None = None,
    cwd: Path = _REPO_ROOT,
) -> GateResult:
    """Run pytest on the given targets, parse JUnit XML, return a GateResult."""
    junit_path = Path(f"/tmp/cudnn_gemm_verify_{os.getpid()}_{name}.junit.xml")
    if junit_path.exists():
        junit_path.unlink()

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    argv = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        f"--junitxml={junit_path}",
        "-q",
        "--tb=line",
        "--no-header",
        *extra_args,
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        return GateResult(
            name=name,
            duration_s=time.monotonic() - start,
            error=f"failed to launch pytest: {type(e).__name__}: {e}",
        )
    duration = time.monotonic() - start

    if not junit_path.exists():
        # pytest crashed before emitting XML (e.g. collection error).
        tail = (proc.stderr or proc.stdout or "").splitlines()[-20:]
        return GateResult(
            name=name,
            duration_s=duration,
            error="pytest produced no JUnit XML; tail:\n" + "\n".join(tail),
        )

    passed, failed, skipped, xfailed, xpassed, failures = _parse_junit(junit_path)
    junit_path.unlink()
    return GateResult(
        name=name,
        duration_s=duration,
        passed=passed,
        failed=failed,
        skipped=skipped,
        xfailed=xfailed,
        xpassed=xpassed,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Examples — each is a standalone script that asserts correctness internally
# ---------------------------------------------------------------------------


def _run_examples() -> GateResult:
    """Each example is `assert_close`-self-checking; a non-zero exit is a fail."""
    start = time.monotonic()
    failures: list[dict] = []
    passed = 0
    examples = sorted(_EXAMPLES_DIR.glob("0?_*.py"))
    if not examples:
        return GateResult(
            name="examples",
            duration_s=0,
            error=f"no examples found under {_EXAMPLES_DIR}",
        )
    for ex in examples:
        try:
            proc = subprocess.run(
                [sys.executable, str(ex)],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            failures.append(
                {
                    "id": f"examples::{ex.name}",
                    "kind": "timeout",
                    "msg": "exceeded 180s",
                }
            )
            continue
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).splitlines()[-10:]
            failures.append(
                {
                    "id": f"examples::{ex.name}",
                    "kind": "exit",
                    "msg": f"exit={proc.returncode}; tail: " + " | ".join(tail),
                }
            )
        else:
            passed += 1
    return GateResult(
        name="examples",
        duration_s=time.monotonic() - start,
        passed=passed,
        failed=len(failures),
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Tier composition
# ---------------------------------------------------------------------------


_UNIT_TEST_FILES = (
    "tests/test_fusion_ir.py",
    "tests/test_graph_analyzer.py",
    "tests/test_compiler.py",
    "tests/test_epilogue_codegen.py",
    "tests/test_tile_config.py",
    "tests/test_smoke.py",
)


def _run_tier(tier: str, pytest_filter: str | None = None) -> TierResult:
    """Execute every gate in the tier and aggregate."""
    start = time.monotonic()
    gates: list[GateResult] = []

    # `-k` filter forwards to every pytest gate.
    extra = ["-k", pytest_filter] if pytest_filter else []

    # smoke: unit tests.
    if tier in ("smoke", "quick", "full"):
        gates.append(_run_pytest("unit_tests", _UNIT_TEST_FILES, extra))

    # quick: + matmul sweep + fusion sweep + examples.
    if tier in ("quick", "full"):
        gates.append(
            _run_pytest(
                "matmul_sweep",
                ["tests/test_matmul_sweep.py"],
                extra,
            )
        )
        gates.append(
            _run_pytest(
                "fusion_sweep",
                ["tests/test_fusion_sweep.py"],
                extra,
            )
        )
        gates.append(_run_examples())

    # full: + matmul sweep on the entire 384-config catalog.
    if tier == "full":
        gates.append(
            _run_pytest(
                "matmul_sweep_full",
                ["tests/test_matmul_sweep.py"],
                extra,
                env_overrides={"CUDNN_GEMM_TEST_FULL": "1"},
            )
        )

    return TierResult(
        tier=tier,
        duration_s=time.monotonic() - start,
        gates=gates,
    )


# ---------------------------------------------------------------------------
# Coverage manifest (`--list`)
# ---------------------------------------------------------------------------


def _load_test_module(path: Path) -> object:
    """Load a test file by absolute path (tests/ isn't a package).

    Register in ``sys.modules`` before ``exec_module`` — the test files use
    ``@dataclass(frozen=True)``, which looks up ``sys.modules[cls.__module__]``
    during class build.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _coverage_manifest() -> dict:
    """Snapshot of what each tier exercises.

    Loads both sweep modules by path (they import torch/cuDNN, so activated-env
    only) to keep the manifest in sync with the test files.
    """
    matmul_sweep = _load_test_module(_TESTS_DIR / "test_matmul_sweep.py")
    fusion_sweep = _load_test_module(_TESTS_DIR / "test_fusion_sweep.py")

    return {
        "tiers": {
            "smoke": {
                "duration_est_s": 30,
                "gates": ["unit_tests"],
                "scope": "codegen + IR + analyzer + tile_config + smoke; no GPU sweep",
            },
            "quick": {
                "duration_est_s": 300,
                "gates": [
                    "unit_tests",
                    "matmul_sweep",
                    "fusion_sweep",
                    "examples",
                ],
                "scope": "default regression gate: curated configs × all dtype-pairs × OOB shapes; full fusion API surface",
            },
            "full": {
                "duration_est_s": 1900,
                "gates": [
                    "unit_tests",
                    "matmul_sweep",
                    "fusion_sweep",
                    "examples",
                    "matmul_sweep_full",
                ],
                "scope": "quick + matmul sweep across all 384 catalog configs (CUDNN_GEMM_TEST_FULL=1)",
            },
        },
        "coverage": {
            "matmul_sweep": {
                "configs_quick": list(matmul_sweep._QUICK_CONFIGS),
                "n_configs_full": len(matmul_sweep.CATALOG),
                "dtype_pairs": list(matmul_sweep._CORE_DTYPE_PAIRS),
                "shapes": list(matmul_sweep._WEIRD_SHAPES),
            },
            "fusion_sweep": {
                "default_config": fusion_sweep.COVERAGE.default_config,
                "default_dtype_pair": [
                    fusion_sweep.COVERAGE.default_in_dt,
                    fusion_sweep.COVERAGE.default_out_dt,
                ],
                "default_shape": list(fusion_sweep.COVERAGE.default_shape),
                "default_axis_chains": list(fusion_sweep.COVERAGE.chains_default_axis),
                "cross_configs": list(fusion_sweep.COVERAGE.cross_configs),
                "cross_chains": list(fusion_sweep.COVERAGE.chains_cross_config),
                "cross_dtype_pairs": [list(p) for p in fusion_sweep.COVERAGE.cross_dtype_pairs],
            },
        },
    }


# ---------------------------------------------------------------------------
# Pretty printer (human-readable)
# ---------------------------------------------------------------------------


def _print_human(tier: TierResult, *, file=sys.stderr) -> None:
    """Pretty summary. stderr by default so stdout stays JSON-clean."""

    def w(s: str = "") -> None:
        print(s, file=file)

    w(f"\n=== cudnn.frost.gemm.verify --tier {tier.tier} ===")
    w(f"duration: {tier.duration_s:.1f}s   exit_code: {tier.exit_code}\n")
    w(f"{'gate':<22} {'pass':>6} {'fail':>6} {'skip':>6} {'xfail':>6} {'time':>8}")
    w("-" * 60)
    for g in tier.gates:
        status = " " if g.is_clean else "x"
        w(f"{status} {g.name:<20} {g.passed:>6} {g.failed:>6} " f"{g.skipped:>6} {g.xfailed:>6} {g.duration_s:>7.1f}s")
    w()
    if not tier.is_clean:
        w("FAILURES:")
        for g in tier.gates:
            if g.error:
                w(f"  [{g.name}] HARNESS ERROR: {g.error}")
            for fail in g.failures:
                w(f"  [{g.name}] {fail.get('id', '?')}")
                w(f"           {fail.get('msg', '')}")
        w()
    w("OK" if tier.is_clean else "FAILED")


def _print_list(manifest: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(manifest, indent=2))
        return
    print("\n=== cudnn.frost.gemm.verify --list ===\n")
    print("Tiers:")
    for tier_name, info in manifest["tiers"].items():
        print(f"  {tier_name:<7} ~{info['duration_est_s']}s — {info['scope']}")
        print(f"          gates: {', '.join(info['gates'])}")
    print()
    mm = manifest["coverage"]["matmul_sweep"]
    print("matmul_sweep coverage:")
    print(f"  quick configs ({len(mm['configs_quick'])}):")
    for c in mm["configs_quick"]:
        print(f"    {c}")
    print(f"  full catalog: {mm['n_configs_full']} configs (set CUDNN_GEMM_TEST_FULL=1)")
    print(f"  dtype pairs:  {mm['dtype_pairs']}")
    print(f"  shapes:       {mm['shapes']}")
    print()
    fs = manifest["coverage"]["fusion_sweep"]
    print("fusion_sweep coverage:")
    print(f"  default-axis: config={fs['default_config']} dtype={fs['default_dtype_pair']} shape={fs['default_shape']}")
    print(f"  chains ({len(fs['default_axis_chains'])}): {', '.join(fs['default_axis_chains'])}")
    print(f"  cross-config: {fs['cross_configs']}")
    print(f"  cross-dtype:  {fs['cross_dtype_pairs']}")
    print(f"  cross-chains: {fs['cross_chains']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cudnn.frost.gemm.verify",
        description="Functional verification driver for cudnn.frost.gemm.",
    )
    p.add_argument(
        "--tier",
        choices=("smoke", "quick", "full"),
        default="quick",
        help="which gate set to run (default: quick, ~5min)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON summary on stdout (human-readable " "summary still goes to stderr unless --quiet)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the human-readable summary on stderr",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="print the coverage manifest and exit (no tests run)",
    )
    p.add_argument(
        "-k",
        "--filter",
        dest="pytest_filter",
        default=None,
        help="forward `-k <expr>` to every pytest gate (substring case filter)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)

    if args.list:
        manifest = _coverage_manifest()
        _print_list(manifest, as_json=args.json)
        return 0

    tier = _run_tier(args.tier, pytest_filter=args.pytest_filter)
    if not args.quiet:
        _print_human(tier)
    if args.json:
        print(json.dumps(tier.to_dict(), indent=2))
    return tier.exit_code


if __name__ == "__main__":
    sys.exit(main())
