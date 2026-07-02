"""Benchmark every CATALOG config on a single matmul shape vs cuBLAS.

`--shape` is `B,M,N,K`: B independent same-shape GEMMs (B=1 → a plain matmul;
B>1 → batched, A=(B,M,K) @ B=(B,N,K).T). Default 1,4096,4096,4096 (BF16). The
cuBLAS reference is torch.matmul (natively batched). Pure matmul, no fusion. No
CPU verify (trust the existing test suite — this script is for perf).

Three timing modes:

  * **delayed** (default) — the trick: queue a long `torch.cuda._sleep`
    onto the stream BEFORE the timed loop, then queue `iters` kernels
    behind it. While the GPU is busy executing the sleep, the host has
    plenty of time to enqueue every kernel launch, so by the time the
    sleep finishes all launches are sitting in the stream and execute
    back-to-back with no host-side gaps. `start.record()` queues the
    timestamp right after the sleep, `end.record()` after the last
    kernel — the elapsed time covers kernel work only. Matches nsys
    median to <2% on 4096³.

  * **nsys** — re-exec under `nsys profile`, then parse median per-kernel
    duration from `nsys stats --report cuda_gpu_kern_sum`. Ground truth
    (reads GPU-side timestamps), and you can inspect the report file.
    A bit slower (~20s instead of ~15s for 18 configs) because of nsys
    profile overhead.

  * **events** — plain `torch.cuda.Event` wall-clock around a Python
    loop. Inflated by ~50us/call from Python + TVM-FFI dispatch — for
    sub-ms kernels this is significant (GEMM measured ~0.13ms but real
    kernel time is ~0.085ms). Kept as a fallback / sanity-check mode.

Why we can't use `torch.cuda.CUDAGraph`: GEMM kernels are launched via
`cuLaunchKernelEx` from libcute_dsl_runtime.so, outside torch's stream
API, so torch's graph capture produces an empty graph. `cute.testing.benchmark`
*could* capture, but it requires the cute jit fn to take `stream` as an
explicit parameter; GEMM's `_host` doesn't, so we'd have to touch every
kernel template. The delay trick is a less invasive workaround.

Usage (from the cudnn-frontend repo root):

    python benchmark/TBD/gemm/benchmark_matmul.py                              # delayed (default)
    python benchmark/TBD/gemm/benchmark_matmul.py --timing nsys                # ground-truth
    python benchmark/TBD/gemm/benchmark_matmul.py --timing events              # incl Python overhead
    python benchmark/TBD/gemm/benchmark_matmul.py --shape 1,8192,8192,8192    # B,M,N,K
    python benchmark/TBD/gemm/benchmark_matmul.py --shape 16,512,512,512      # batched (16 GEMMs)
    python benchmark/TBD/gemm/benchmark_matmul.py --configs CONFIG_a,CONFIG_b
    python benchmark/TBD/gemm/benchmark_matmul.py --shape 1,1024,1024,1024 --rotate-buffers 64

Buffer rotation (`--rotate-buffers N`, default 'auto'): allocate N independent
copies of every tensor and rotate the timed launches across them, so a kernel
never re-reads the previous launch's inputs from a hot L2. Without it, small
shapes (whose working set fits in B200's ~126 MB L2) report inflated TFLOPS
because every launch after the first reads its inputs at L2 — not DRAM — speed,
which a real back-to-back workload wouldn't. The default 'auto' sizes the pool
to exceed L2 for the given shape (large shapes → 2 copies since one set already
dwarfs L2; small shapes → scaled up until the pool is L2-cold), capped at 4 GB.
Warmup runs against a separate dedicated buffer (never rotated). Pass an integer
to override; `--rotate-buffers 1` disables rotation.

(active_tbd.sh uses $PWD to find .micromamba; source it from workspace root.)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable

import cudnn  # noqa: F401
import cudnn.TBD.gemm  # noqa: F401
import torch

from cudnn.TBD.gemm.compiler import jit_from_cudnn_graph
from cudnn.TBD.gemm.fusion_ir import FusionChain as _FC, MatmulSpec as _MS
from cudnn.TBD.gemm.kernel_registry import candidates as _candidates


def _build_spec_map():
    """Legacy label -> (geometry cfg, cta_group, scheduler) for every sweepable
    matmul strategy, via the registry funnel (excludes known-bad/unsupported).
    Labels reconstruct the old CONFIG_..._Nctamma[_static] form so --configs
    still accepts them."""
    chain = _FC(matmul=_MS(M=4096, N=4096, K=4096, a_major="k", b_major="k", a_dtype="bf16", b_dtype="bf16", accum_dtype="fp32"), output_dtype="bf16")
    m = {}
    for t, cfg in _candidates(chain):
        label = f"{cfg.name}_{t.cta_group}ctamma" + ("_static" if t.static_sched else "")
        m[label] = (cfg, t.cta_group, t.scheduler)
    return m


_SPEC_MAP = _build_spec_map()
from cudnn.TBD.gemm.tile_config import CATALOG, TileConfig


def _vp(handles, a, b, c):
    """Variant-pack dict {cuDNN tensor: buffer} keyed by the graph's tensors."""
    A, B, C = handles
    return {A: a, B: b, C: c}


def _build_plan(g, cfg, name):
    """JIT-compile the recorded graph with a forced tile config via jit_from_cudnn_graph.
    Returns the compiled kernel (callable with a variant-pack dict)."""
    return jit_from_cudnn_graph(g, config=cfg, cta_group=_SPEC_MAP[name][1], scheduler=_SPEC_MAP[name][2])


# ---------------------------------------------------------------------------
# Graph + data setup
# ---------------------------------------------------------------------------


def _graph_matmul(batch: int, M: int, N: int, K: int):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1])
    Bt = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=Bt, name="mm")
    C.set_output(True)
    return g, (A, Bt, C)


def _mkdata(batch: int, M: int, N: int, K: int):
    torch.manual_seed(0)
    a = torch.empty(batch, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(batch, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(batch, M, N, dtype=torch.bfloat16, device="cuda")
    return a, b, c


# ---------------------------------------------------------------------------
# Buffer rotation — defeat the hot-L2 artifact on small shapes
# ---------------------------------------------------------------------------
#
# Back-to-back launches against the SAME a/b/c keep those tensors resident in
# L2, so a small matmul reads its inputs at L2 latency/bandwidth instead of
# DRAM — inflating the measured TFLOPS vs a cold first launch. To measure
# realistic (DRAM-fed) performance, allocate a POOL of N independent copies of
# every tensor and rotate the launch across them: launch i uses pool[i % N].
# If the pool footprint exceeds L2 (~126 MB on B200), by the time the rotation
# wraps back to buffer 0 its data has been evicted, so every launch pays the
# DRAM cost — the same situation a real workload sees.


# B200 L2 is ~126 MB. A pool smaller than this still gets fully cached after one
# rotation, so its launches stay warm — warn the user to bump --rotate-buffers.
_L2_BYTES_B200 = 126 * 1024 * 1024


def _mkdata_pool(batch: int, M: int, N: int, K: int, nbuf: int):
    """Return a list of `nbuf` independent (a, b, c) triples at distinct GMEM
    addresses. nbuf<=1 returns the single base triple (legacy behavior)."""
    a, b, c = _mkdata(batch, M, N, K)
    pool = [(a, b, c)]
    # Distinct allocations (clone → fresh GMEM). Contents are irrelevant for
    # perf timing; what matters is that each launch hits a different address.
    for _ in range(max(0, nbuf - 1)):
        pool.append((a.clone(), b.clone(), c.clone()))
    return pool


def _per_set_bytes(batch: int, M: int, N: int, K: int) -> int:
    # BF16 = 2 bytes/elem; a:(batch,M,K) b:(batch,N,K) c:(batch,M,N).
    return 2 * batch * (M * K + N * K + M * N)


def _pool_footprint_bytes(batch: int, M: int, N: int, K: int, nbuf: int) -> int:
    return _per_set_bytes(batch, M, N, K) * nbuf


# Cap the auto-sized pool so a large shape (whose single tensor set already
# dwarfs L2) doesn't allocate dozens of needless multi-GB copies.
_AUTO_POOL_BUDGET_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
_AUTO_NBUF_CAP = 1024


def _auto_nbuf(batch: int, M: int, N: int, K: int) -> int:
    """Pick the smallest buffer count whose pool exceeds L2 (with 1.5× margin
    so the wrap-around is a guaranteed miss), clamped to a memory budget.

    Large shapes → 2 (one tensor set already far exceeds L2, so minimal
    rotation suffices). Small shapes → scaled up until the pool is L2-cold."""
    per_set = _per_set_bytes(batch, M, N, K)
    target = int(1.5 * _L2_BYTES_B200)
    nbuf = max(2, -(-target // per_set))  # ceil-div

    # Don't exceed a memory budget: min(4 GB, half of currently-free GMEM).
    budget = _AUTO_POOL_BUDGET_BYTES
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        budget = min(budget, free // 2)
    max_by_budget = max(1, budget // per_set)

    return max(1, min(nbuf, max_by_budget, _AUTO_NBUF_CAP))


def _resolve_nbuf(spec: str, batch: int, M: int, N: int, K: int) -> int:
    """Resolve the --rotate-buffers CLI value: 'auto' → shape-sized count,
    else the given integer (floored at 1 = rotation disabled)."""
    if spec.strip().lower() == "auto":
        return _auto_nbuf(batch, M, N, K)
    return max(1, int(spec))


def _rotating(fn_of_buf: Callable, pool: list) -> Callable:
    """Wrap a `(a, b, c) -> None` callable into an `i -> None` callable that
    selects pool[i % len(pool)] for launch index i."""
    n = len(pool)
    return lambda i: fn_of_buf(pool[i % n])


def _compatible(cfg: TileConfig, M: int, N: int, K: int) -> bool:
    tm, tn = cfg.cgrp_tile_mn
    tk = cfg.cta_tile_k(elem_bytes=2)
    return M % tm == 0 and N % tn == 0 and K % tk == 0


# ---------------------------------------------------------------------------
# Timing (events mode)
# ---------------------------------------------------------------------------


def _time_ms_events(
    timed_fn: Callable,
    warmup_fn: Callable,
    *,
    warmup: int,
    iters: int,
) -> float:
    """Wall-clock CUDA Event timing around a python loop. Inflated by
    Python+TVM-FFI dispatch overhead (~50us/call); use `--timing delayed`
    or `--timing nsys` for clean kernel-only timing.

    `timed_fn(i)` is the per-launch callable for the timed loop (i = launch
    index, used to rotate buffers). `warmup_fn()` runs the warmup launches
    against a separate dedicated buffer (not part of the rotation pool)."""
    for _ in range(warmup):
        warmup_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        timed_fn(i)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _time_ms_delayed(
    timed_fn: Callable,
    warmup_fn: Callable,
    *,
    warmup: int,
    iters: int,
) -> float:
    """Kernel-only timing by hiding host-launch overhead behind a delay kernel.

    `timed_fn(i)` is the per-launch callable for the timed loop (i = launch
    index, used to rotate buffers); `warmup_fn()` runs the warmups against a
    separate dedicated buffer (not part of the rotation pool).

    Pattern on the stream (everything async, GPU executes in order):

        torch.cuda._sleep(D)        # ~120ms — host enqueues all of the below
                                    # while GPU is busy here
        for _ in range(post_warmup):
            warmup_fn()             # post-sleep warmup so SM clocks are boosted
                                    # by the time we start timing
        start.record()
        for i in range(iters):
            timed_fn(i)
        end.record()

    Why the post-sleep warmup matters: the first few kernels after the sleep
    (especially small/fast ones) run at lower SM clock — `_sleep` keeps the
    GPU busy but doesn't fully load the FMA / TC pipes, so DVFS hasn't ramped
    to boost. Without this warmup, the first delayed measurement of a fast
    config (e.g. 64×256) inflates ~2× while the second is fine. With it the
    first run already matches the nsys median.

    The delay needs to outlast `iters × per_launch_host_overhead` (~50us/call
    for GEMM kernels). Floor 1e8 cycles ≈ 60ms; for big `iters` we scale up.
    """
    for _ in range(warmup):
        warmup_fn()
    torch.cuda.synchronize()

    # Auto-scale the delay: 50us/launch * iters + 20ms slack, in B200 cycles.
    # B200 SM clock ~1.7 GHz → 1ms = 1.7e6 cycles. Floor at 1e8 cycles.
    delay_cycles = max(
        int(1e8),
        int((iters * 0.05 + 20.0) * 1.7e6),
    )
    torch.cuda._sleep(delay_cycles)

    # Post-sleep warmup (queued behind the sleep, same stream) — ramps GPU
    # clocks back up before we start timing.
    post_warmup = max(5, warmup)
    for _ in range(post_warmup):
        warmup_fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        timed_fn(i)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# nsys mode
# ---------------------------------------------------------------------------


def _nsys_run_and_parse(
    shape: str,
    configs: list[str],
    warmup: int,
    iters: int,
    nbuf: int,
) -> dict[str, float]:
    """Re-exec self under nsys, run every (config) sequentially, parse the
    `cuda_gpu_kern_sum` report to get median kernel time (ms) per config.

    Returns {config_name_or_'cuBLAS': median_ms}.
    """
    # Prefer the system-wide nsys (/usr/local/bin/nsys). The cuda-13.x
    # bundled nsys in this env's PATH has an unresolved libbpf.so.1
    # dependency on B200 — falling back to it would just error out.
    nsys = "/usr/local/bin/nsys" if os.path.exists("/usr/local/bin/nsys") else shutil.which("nsys")
    if nsys is None:
        sys.exit("nsys not found — install nsight-systems or use the default events mode.")

    workdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"benchmark_matmul_nsys_{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    report_prefix = os.path.join(workdir, "report")

    # nsys writes to /tmp/nvidia/nsight_systems by default; on this host that
    # path is owned by root and the user can't write to it. Redirect via env.
    nsys_env = os.environ.copy()
    nsys_env.setdefault("TMPDIR", os.environ.get("TMPDIR", tempfile.gettempdir()))

    # Build the inner command — same script with --_nsys-worker.
    inner = [
        sys.executable,
        "-u",
        os.path.abspath(__file__),
        "--_nsys-worker",
        "--shape",
        shape,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--rotate-buffers",
        str(nbuf),
    ]
    if configs:
        inner += ["--configs", ",".join(configs)]

    # Step 1: profile (record only; --stats output to stdout is unreliable
    # across nsys versions when capture_output=True).
    profile_cmd = [
        nsys,
        "profile",
        "-o",
        report_prefix,
        "--force-overwrite=true",
        "--cuda-um-cpu-page-faults=false",
        "--cuda-um-gpu-page-faults=false",
        "--trace=cuda",
    ] + inner

    print(f"  + {' '.join(profile_cmd)}\n")
    proc = subprocess.run(profile_cmd, capture_output=True, text=True, env=nsys_env)
    if proc.returncode != 0:
        print("nsys stdout:\n" + proc.stdout)
        print("nsys stderr:\n" + proc.stderr, file=sys.stderr)
        sys.exit(f"nsys profile exited {proc.returncode}")

    # Step 2: extract the kernel-summary table.
    stats_cmd = [
        nsys,
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        "--force-export=true",
        report_prefix + ".nsys-rep",
    ]
    proc = subprocess.run(stats_cmd, capture_output=True, text=True, env=nsys_env)
    if proc.returncode != 0:
        print("nsys stats stdout:\n" + proc.stdout)
        print("nsys stats stderr:\n" + proc.stderr, file=sys.stderr)
        sys.exit(f"nsys stats exited {proc.returncode}")

    return _parse_nsys_stats(proc.stdout)


def _parse_nsys_stats(text: str) -> dict[str, float]:
    """Parse `nsys stats --report cuda_gpu_kern_sum` output. Returns
    {kernel_name (possibly truncated with …): median_ms}.

    Format (nsys 2025+):

      Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name
      --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----
       22.4     25,508,650       35         728,818.6 728,898.0 727,970   729,699   496.6        kernel_cutlass__kernel_CONFIG_sm100_...…

    Numeric columns are space-separated but each number may contain commas
    as thousands separators (no spaces inside a number). The Name column
    starts at a fixed text offset in the header (used to extract the rest
    of the line — names contain spaces, parens, and may be truncated with `…`).
    """
    lines = text.splitlines()
    header_i = None
    for i, ln in enumerate(lines):
        if "Med (" in ln and "Name" in ln and ("ns)" in ln or "us)" in ln or "ms)" in ln):
            header_i = i
            break
    if header_i is None:
        sys.exit("could not find kernel-summary header in nsys stats output:\n  " + "\n  ".join(lines[:60]))

    m_unit = re.search(r"Med \((\w+)\)", lines[header_i])
    unit = m_unit.group(1) if m_unit else "ns"
    unit_div = {"ns": 1e6, "us": 1e3, "ms": 1.0, "s": 1e-3}.get(unit, 1e6)

    # Numeric columns (0-indexed): 0 Time%, 1 Total, 2 Instances, 3 Avg, 4 Med,
    # 5 Min, 6 Max, 7 StdDev, then the kernel name (everything that follows).
    # Numbers may contain commas as thousands separators but no internal spaces,
    # so whitespace tokenization is reliable.
    NUM_NUMERIC_COLS = 8
    MED_COL = 4

    result: dict[str, float] = {}
    in_data = False
    for j in range(header_i + 1, len(lines)):
        row = lines[j]
        stripped = row.strip()
        if not stripped:
            if in_data:
                break
            continue
        if set(stripped) <= set("- "):
            in_data = True
            continue
        if not in_data:
            continue
        if stripped.startswith("**") or stripped.startswith("##"):
            break

        toks = stripped.split()
        if len(toks) <= NUM_NUMERIC_COLS:
            continue
        try:
            med = float(toks[MED_COL].replace(",", ""))
        except ValueError:
            continue
        name = " ".join(toks[NUM_NUMERIC_COLS:]).rstrip()
        if not name:
            continue
        result[name] = med / unit_div
    return result


def _match_kernel_name(kern_name: str, config_name: str) -> bool:
    """nsys reports demangled symbols like
    'cutlass::_kernel_CONFIG_sm100_...<...>(...)'. Match by substring."""
    return config_name in kern_name


def _find_cublas_time(kern_times: dict[str, float]) -> tuple[str, float] | None:
    """Largest 'nvjet_*' kernel (cuBLAS) — assume single-kernel matmul, but
    if multiple match, pick the one with the largest invocation total (Med
    is what we have, so longest median)."""
    cands = [(k, v) for k, v in kern_times.items() if k.startswith("nvjet_")]
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Worker mode: just run the kernels under nsys profile, no Python timing.
# ---------------------------------------------------------------------------


def _nsys_worker(
    shape: str,
    configs: list[str],
    warmup: int,
    iters: int,
    nbuf: int,
) -> None:
    """Inner mode: re-exec'd under nsys. Run each config (and cuBLAS) for
    warmup+iters launches each, no timing. nsys captures everything.

    Warmup runs against a dedicated buffer; the timed iters rotate across a
    pool of `nbuf` independent buffers to avoid the hot-L2 artifact."""
    B, M, N, K = (int(x) for x in shape.split(","))
    wa, wb, wc = _mkdata(B, M, N, K)  # dedicated warmup buffer
    pool = _mkdata_pool(B, M, N, K, nbuf)  # rotation pool for timed iters

    print(f"[worker] shape={B}x{M}x{N}x{K}, configs={len(configs)}, " f"warmup={warmup}, iters={iters}, rotate_buffers={nbuf}")

    # 1. cuBLAS — torch.matmul.
    for _ in range(warmup):
        torch.matmul(wa, wb.transpose(-1, -2), out=wc)
    for i in range(iters):
        a, b, c = pool[i % nbuf]
        torch.matmul(a, b.transpose(-1, -2), out=c)
    torch.cuda.synchronize()

    # 2. each GEMM config.
    name_to_cfg = {lbl: sp[0] for lbl, sp in _SPEC_MAP.items()}
    config_names = configs or list(_SPEC_MAP)
    for name in config_names:
        cfg = name_to_cfg.get(name)
        if cfg is None:
            continue
        if not _compatible(cfg, M, N, K):
            continue
        try:
            g, h = _graph_matmul(B, M, N, K)
            plan = _build_plan(g, cfg, name)
            for _ in range(warmup):
                plan(_vp(h, wa, wb, wc))
            for i in range(iters):
                a, b, c = pool[i % nbuf]
                plan(_vp(h, a, b, c))
            torch.cuda.synchronize()
            print(f"[worker] OK   {name}")
        except Exception as e:
            print(f"[worker] FAIL {name}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        default="1,4096,4096,4096",
        help="B,M,N,K (default 1,4096,4096,4096; B = batch / number of " "independent same-shape GEMMs)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    # Perf-testing rule (CLAUDE.md): keep iters <= 20 — more doesn't sharpen the
    # measurement here, it just lengthens the run / holds the GPU.
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--configs",
        default=None,
        help="comma-separated config names to test (default: every CATALOG entry)",
    )
    parser.add_argument(
        "--timing",
        choices=("delayed", "events", "nsys"),
        default="delayed",
        help="delayed (default): events-around-loop, with a long delay kernel "
        "queued first to hide host-launch overhead — matches nsys to "
        "<2%%. events: plain events around a loop (has ~50us/call "
        "Python overhead — inflates sub-ms kernels). nsys: ground "
        "truth via nsys profile (read GPU-side kernel timestamps).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="print each config's result line as soon as it finishes (and a "
        "'running …' line before measurement starts). Useful when a "
        "config hangs — the last 'running' line points at the culprit. "
        "events/delayed modes only; no effect under --timing nsys.",
    )
    parser.add_argument(
        "--rotate-buffers",
        default="auto",
        metavar="N",
        help="allocate N independent copies of every tensor and rotate the "
        "timed launches across them (launch i uses copy i%%N) so a kernel "
        "doesn't re-read the previous launch's data from a hot L2 — the "
        "main source of inflated TFLOPS on small shapes. Warmup uses a "
        "separate dedicated buffer (never rotated). Default 'auto': size "
        "the pool to exceed the ~126 MB B200 L2 for this shape (large "
        "shapes → 2 copies, small shapes → scaled up), capped at 4 GB. "
        "Pass an integer to override; 1 disables rotation.",
    )
    parser.add_argument(
        "--_nsys-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K (four values; use B=1 for a plain matmul)")
    B, M, N, K = parts
    nbuf = _resolve_nbuf(args.rotate_buffers, B, M, N, K)

    if getattr(args, "_nsys_worker"):
        configs = [c.strip() for c in args.configs.split(",")] if args.configs else []
        _nsys_worker(args.shape, configs, args.warmup, args.iters, nbuf)
        return 0

    flops = 2 * B * M * N * K
    config_names = [c.strip() for c in args.configs.split(",")] if args.configs else list(_SPEC_MAP)
    name_to_cfg = {lbl: sp[0] for lbl, sp in _SPEC_MAP.items()}

    print(f"\n=== matmul B={B} {M}x{N}x{K}  (~{flops / 1e9:.1f} GFLOP) — BF16 ===")

    if nbuf > 1:
        footprint = _pool_footprint_bytes(B, M, N, K, nbuf)
        print(f"  [rotate-buffers: {nbuf} copies/tensor, " f"{footprint / 1024 / 1024:.0f} MB pool — defeats hot-L2 on small shapes]")
        if footprint < _L2_BYTES_B200:
            print(
                f"  [WARNING: pool ({footprint / 1024 / 1024:.0f} MB) < B200 L2 "
                f"(~{_L2_BYTES_B200 / 1024 / 1024:.0f} MB) — it fits in cache, so "
                f"launches stay warm after one rotation. Bump --rotate-buffers.]"
            )
    else:
        print("  [rotate-buffers: disabled (1) — small-shape TFLOPS may be hot-L2-inflated]")

    rows: list[tuple[str, float, float, str]] = []  # (name, tflops, ms, note)
    t0 = time.time()

    def _fmt_row(name: str, tflops: float, ms: float, note: str, ref_tflops: float) -> str:
        if note:
            return f"  {name:50s} {'':8s}   {'':7s}   {note}"
        ratio = tflops / ref_tflops if ref_tflops > 0 else 0.0
        return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {ratio:>9.2f}×"

    if args.timing == "nsys":
        # nsys mode: one driver invocation, parse kernel times.
        print(f"  [timing: nsys median kernel duration]\n")
        kern_times = _nsys_run_and_parse(args.shape, config_names, args.warmup, args.iters, nbuf)

        cublas_hit = _find_cublas_time(kern_times)
        if cublas_hit:
            cublas_name, cublas_ms = cublas_hit
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            print(f"  cuBLAS kernel: {cublas_name}")
        else:
            cublas_tflops, cublas_ms = float("nan"), float("nan")
            print("  cuBLAS kernel: not detected in nsys output")

        for name in config_names:
            cfg = name_to_cfg.get(name)
            if cfg is None:
                rows.append((name, 0.0, float("inf"), "UNKNOWN_CONFIG"))
                continue
            if not _compatible(cfg, M, N, K):
                rows.append((name, 0.0, float("inf"), "incompatible"))
                continue
            matches = [(k, v) for k, v in kern_times.items() if _match_kernel_name(k, name)]
            if not matches:
                rows.append((name, 0.0, float("inf"), "NO_KERNEL_IN_NSYS"))
                continue
            # If multiple specializations share the same config name, pick the
            # heaviest (most representative).
            _, ms = max(matches, key=lambda x: x[1])
            rows.append((name, flops / (ms * 1e-3) / 1e12, ms, ""))
    else:
        # In-process events timing (events or delayed).
        timer = _time_ms_delayed if args.timing == "delayed" else _time_ms_events
        if args.timing == "delayed":
            print(f"  [timing: events bracketed around delayed back-to-back " f"launches — host overhead hidden behind a CUDA _sleep]\n")
        else:
            print(
                f"  [timing: torch.cuda.Event wall-clock around python loop — "
                f"includes ~50us/call Python+TVM-FFI dispatch overhead; use "
                f"--timing delayed or --timing nsys for kernel-only timing]\n"
            )
        wa, wb, wc = _mkdata(B, M, N, K)  # dedicated warmup buffer
        pool = _mkdata_pool(B, M, N, K, nbuf)  # rotation pool for timed iters
        if args.stream:
            print(f"  ▶ running cuBLAS reference ...", flush=True)
        cublas_ms = timer(
            _rotating(lambda t: torch.matmul(t[0], t[1].transpose(-1, -2), out=t[2]), pool),
            lambda: torch.matmul(wa, wb.transpose(-1, -2), out=wc),
            warmup=args.warmup,
            iters=args.iters,
        )
        cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
        if args.stream:
            print(
                _fmt_row("cuBLAS (reference)", cublas_tflops, cublas_ms, "", cublas_tflops),
                flush=True,
            )

        # Once a kernel emits an async device-side fault (e.g. illegal address),
        # the CUDA context stays sticky-poisoned for the rest of the process —
        # every subsequent cuLaunchKernel returns CUDA_ERROR_LAUNCH_FAILED even
        # for known-good configs. JIT-compiling + launching every remaining
        # config in that state still costs seconds each, which looks like a
        # hang. After the first such error, short-circuit the remaining
        # configs and mark them CTX_DEAD instead of trying to launch them.
        ctx_dead = False
        for name in config_names:
            cfg = name_to_cfg.get(name)
            if cfg is None:
                row = (name, 0.0, float("inf"), "UNKNOWN_CONFIG")
            elif not _compatible(cfg, M, N, K):
                row = (name, 0.0, float("inf"), "incompatible")
            elif ctx_dead:
                row = (name, 0.0, float("inf"), "skipped (CUDA context dead)")
            else:
                if args.stream:
                    print(f"  ▶ running {name} ...", flush=True)
                try:
                    g, h = _graph_matmul(B, M, N, K)
                    plan = _build_plan(g, cfg, name)
                    ms = timer(
                        _rotating(lambda t, _plan=plan, _h=h: _plan(_vp(_h, t[0], t[1], t[2])), pool),
                        lambda _plan=plan, _h=h: _plan(_vp(_h, wa, wb, wc)),
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    row = (name, flops / (ms * 1e-3) / 1e12, ms, "")
                except Exception as e:
                    msg = str(e).splitlines()[0][:50] if str(e) else type(e).__name__
                    row = (name, 0.0, float("inf"), f"ERR {msg}")
                    # CUDA context-poisoning errors are unrecoverable in this
                    # process. Stop trying — everything after will fail too.
                    if any(
                        s in str(e)
                        for s in (
                            "illegal memory access",
                            "unspecified launch failure",
                            "CUDA_ERROR_LAUNCH_FAILED",
                        )
                    ):
                        ctx_dead = True
            rows.append(row)
            if args.stream:
                print(_fmt_row(*row, cublas_tflops), flush=True)

    # Sort best → worst, print.
    rows.sort(key=lambda r: -r[1])
    print("=" * 88)
    print(f"  {'config':50s} {'TFLOPS':>8s}   {'ms':>7s}   {'vs cuBLAS':>10s}")
    print("=" * 88)
    for name, tflops, ms, note in rows:
        print(_fmt_row(name, tflops, ms, note, cublas_tflops))
    print("=" * 88)
    if cublas_tflops > 0:
        print(f"  {'cuBLAS (reference)':50s} {cublas_tflops:8.2f}   {cublas_ms:7.3f}   {'1.00×':>10s}")
    else:
        print("  cuBLAS reference: n/a")

    ok = [r for r in rows if not r[3]]
    if ok and cublas_tflops > 0:
        best_name, best_tflops, _best_ms, _ = ok[0]
        print(f"\nbest GEMM: {best_name}" f" — {best_tflops:.2f} TFLOPS" f" ({best_tflops / cublas_tflops:.2f}× cuBLAS)")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
