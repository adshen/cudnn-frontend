"""Benchmark every block-scale-compatible CATALOG config on a single shape.

Default: nvfp4 inputs (FP4 E2M1 + per-block E4M3 scale, block16) → BF16 output.
`--shape` is `B,M,N,K`: B independent same-shape block-scale GEMMs (B=1 → single
GEMM; B>1 → batched). Default 1,4096,4096,4096. Pure block-scaled matmul, no
epilogue fusion. No CPU verify (trust the test suite — this script is for perf).

Reference (`--ref`, default `auto`):
  * **scaled_mm** — cuBLAS's OWN block-scaled FP4/FP8 GEMM of the matching combo,
    via `torch.nn.functional.scaled_mm` (cuBLASLt). This is the true
    apples-to-apples comparison. Some driver / cuBLASLt builds have no heuristic
    algorithm for the scaled path and every call raises
    `CUBLAS_STATUS_NOT_INITIALIZED` — then this option errors out.
  * **bf16** — dense BF16 cuBLAS of the same M×N×K. A throughput yardstick only:
    fp4/fp8 peak is ~2× BF16, so ratios land above 1× and are NOT a like-for-like
    win.
  * **auto** (default) — use the real block-scaled kernel; fall back to BF16
    (clearly labeled) if this env's cuBLASLt can't run it.

Timing modes are identical to benchmark_matmul.py:

  * **delayed** (default) — events bracketed around back-to-back launches with a
    long `torch.cuda._sleep` queued first to hide host-launch overhead (~50us/call).
    Matches the nsys median to <2%.
  * **nsys** — re-exec under `nsys profile`, parse median per-kernel duration.
  * **events** — plain CUDA-Event wall-clock around a python loop (inflated by the
    per-call dispatch overhead; kept as a sanity check).

Usage (from workspace root, after `source active_tbd.sh`):

    python cudnn.TBD.gemm/benchmarks/benchmark_block_scale_matmul.py                    # nvfp4, auto ref, delayed
    python cudnn.TBD.gemm/benchmarks/benchmark_block_scale_matmul.py --ref scaled_mm    # force cuBLAS nvfp4 ref
    python cudnn.TBD.gemm/benchmarks/benchmark_block_scale_matmul.py --timing nsys      # ground-truth
    python cudnn.TBD.gemm/benchmarks/benchmark_block_scale_matmul.py --shape 1,8192,8192,8192   # B,M,N,K
    python cudnn.TBD.gemm/benchmarks/benchmark_block_scale_matmul.py --shape 256,256,3072,2048  # batched (256 GEMMs)
    python cudnn.TBD.gemm/benchmarks/benchmark_block_scale_matmul.py --combo mxfp8      # fp8 + e8m0, block32
    python cudnn.TBD.gemm/benchmarks/benchmark_block_scale_matmul.py --configs CONFIG_a,CONFIG_b

Buffer rotation (`--rotate-buffers N`, default 'auto') is identical to
benchmark_matmul.py: allocate N independent copies of every tensor (here each set is
a/b/c + the two scale-factor tensors) and rotate the timed launches across them
so a kernel never re-reads the previous launch's inputs from a hot L2 — the main
source of inflated TFLOPS on small shapes. The default 'auto' sizes the pool to
exceed the ~126 MB B200 L2 (large shapes → 2 copies; small shapes → scaled up),
capped at 4 GB. Both the GEMM configs AND the reference rotate. Warmup uses a
separate dedicated buffer (never rotated). Pass an integer to override; 1
disables rotation.

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
from cudnn.TBD.gemm.tile_config import CATALOG as _CATALOG


def _build_spec_map():
    """Legacy label -> (geometry cfg, cta_group, scheduler) for block-scale
    strategies (geometry must satisfy the SF 128x4 swizzle). Labels reconstruct
    the old CONFIG_..._Nctamma[_static] form so --configs still accepts them."""
    m = {}
    for cfg in _CATALOG:
        if cfg.cta_tile_m % 128 or cfg.cta_tile_n % 128 or cfg.cta_tile_k_bytes != 128:
            continue
        for cg in (1, 2):
            if cg == 2 and (cfg.cgrp_size_m % 2 or cfg.cta_tile_m == 64):
                continue
            for sched, tok in (("clc", ""), ("static", "_static")):
                m[f"{cfg.name}_{cg}ctamma{tok}"] = (cfg, cg, sched)
    return m


_SPEC_MAP = _build_spec_map()
from cudnn.TBD.gemm.tile_config import CATALOG, TileConfig, validate_block_scale_config


def _vp_bs(compiled, a, b, c, sfa, sfb):
    """Block-scale single-GEMM variant-pack dict from the compiled binding."""
    bd = compiled.binding
    return {bd.a_operands[0]: a, bd.b_operands[0]: b, bd.sfa_operands[0]: sfa, bd.sfb_operands[0]: sfb, bd.outputs[0]: c}


# ---------------------------------------------------------------------------
# Combo table (input dtype family + scale dtype + block size)
# ---------------------------------------------------------------------------

_COMBOS = {
    # combo : (is_fp4, block_size, a_dtype, sf_dtype)
    "nvfp4": (True, 16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (True, 32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (False, 32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


# ---------------------------------------------------------------------------
# Graph + data setup
# ---------------------------------------------------------------------------


def _graph_block_scale(batch: int, M: int, N: int, K: int, combo: str):
    is_fp4, block_size, a_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1], data_type=a_dt)
    Bt = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K], data_type=a_dt)
    # Scale factors are in the CUDNN_TENSOR_REORDERING_F8_128x4 swizzle — the
    # support table keys on this, so it must be declared (else the block-scale
    # gate rejects the config). Matches examples/09 + tests/test_block_scale.
    SFA = g.tensor(name="SFA", dim=[batch, M, sf_k], stride=[M * sf_k, sf_k, 1], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
    SFB = g.tensor(name="SFB", dim=[batch, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block_size])
    Bd = g.block_scale_dequantize(input=Bt, descale=SFB, block_size=[block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)  # BF16 output
    return g


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_blocked(x):
    """Pack a (rows, cols) scale tensor into the 128x4 blocked layout the kernel
    expects (matches tests/test_block_scale.py)."""
    rows, cols = x.shape
    nrb, ncb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _rand_e8m0(shape, dev):
    return torch.randint(125, 129, shape, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)


def _mkdata(batch: int, M: int, N: int, K: int, combo: str):
    """Block-scale runtime tensors: (a, b, c, sfa_blocked, sfb_blocked).

    a/b are packed FP4 (viewed as float4_e2m1fn_x2) or FP8; c is BF16. Each is
    rank-3 with a leading batch dim (B independent same-shape block-scale GEMMs);
    SFA/SFB are F8_128x4-reordered per batch then concatenated."""
    dev = "cuda"
    torch.manual_seed(0)
    is_fp4, block_size, _, _ = _COMBOS[combo]
    sf_k = K // block_size

    if is_fp4:
        a_u8 = torch.randint(0, 256, (batch, M, K // 2), dtype=torch.uint8, device=dev)
        b_u8 = torch.randint(0, 256, (batch, N, K // 2), dtype=torch.uint8, device=dev)
        a = a_u8.view(torch.float4_e2m1fn_x2)
        b = b_u8.view(torch.float4_e2m1fn_x2)
    else:
        a = (torch.randn(batch, M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b = (torch.randn(batch, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (batch, M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (batch, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((batch, M, sf_k), dev)
        sfb_log = _rand_e8m0((batch, N, sf_k), dev)

    c = torch.empty(batch, M, N, dtype=torch.bfloat16, device=dev)
    sfa = torch.cat([_to_blocked(sfa_log[i]) for i in range(batch)]).view(batch, M, sf_k)
    sfb = torch.cat([_to_blocked(sfb_log[i]) for i in range(batch)]).view(batch, N, sf_k)
    return a, b, c, sfa, sfb


# ---------------------------------------------------------------------------
# Buffer rotation — defeat the hot-L2 artifact on small shapes (see
# benchmark_matmul.py for the full rationale). Rotate the *timed* launches across a
# pool of independent tensor sets; warmup uses a separate dedicated buffer.
# ---------------------------------------------------------------------------

# B200 L2 is ~126 MB. A pool smaller than this gets fully cached after one
# rotation, so its launches stay warm — warn the user to bump --rotate-buffers.
_L2_BYTES_B200 = 126 * 1024 * 1024
_AUTO_POOL_BUDGET_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
_AUTO_NBUF_CAP = 1024


def _set_bytes(tensors) -> int:
    """GMEM footprint of one tensor set (handles packed FP4: element_size of
    float4_e2m1fn_x2 is 1 byte per packed pair)."""
    return sum(t.numel() * t.element_size() for t in tensors)


def _mkdata_pool(batch: int, M: int, N: int, K: int, combo: str, nbuf: int):
    """Return a list of `nbuf` independent (a, b, c, sfa, sfb) sets at distinct
    GMEM addresses. nbuf<=1 returns the single base set (legacy behavior)."""
    base = _mkdata(batch, M, N, K, combo)
    pool = [base]
    for _ in range(max(0, nbuf - 1)):
        pool.append(tuple(t.clone() for t in base))
    return pool


def _auto_nbuf(per_set_bytes: int) -> int:
    """Smallest buffer count whose pool exceeds L2 (1.5× margin so the
    wrap-around is a guaranteed miss), clamped to a memory budget. Large shapes
    → 2 (one set already dwarfs L2); small shapes → scaled up until L2-cold."""
    target = int(1.5 * _L2_BYTES_B200)
    nbuf = max(2, -(-target // per_set_bytes))  # ceil-div
    budget = _AUTO_POOL_BUDGET_BYTES
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        budget = min(budget, free // 2)
    max_by_budget = max(1, budget // per_set_bytes)
    return max(1, min(nbuf, max_by_budget, _AUTO_NBUF_CAP))


def _resolve_nbuf(spec: str, per_set_bytes: int) -> int:
    """Resolve --rotate-buffers: 'auto' → shape-sized count, else the integer
    (floored at 1 = rotation disabled)."""
    if str(spec).strip().lower() == "auto":
        return _auto_nbuf(per_set_bytes)
    return max(1, int(spec))


def _rotating(fn_of_set: Callable, pool: list) -> Callable:
    """Wrap a `set -> None` callable into an `i -> None` callable selecting
    pool[i % len(pool)] for launch index i."""
    n = len(pool)
    return lambda i: fn_of_set(pool[i % n])


def _scaled_mm_ref(batch: int, M: int, N: int, K: int, combo: str):
    """(label, call) for cuBLAS's OWN block-scaled GEMM of this combo, via
    torch.nn.functional.scaled_mm (cuBLASLt). This is the apples-to-apples
    reference. Returns None if the env can't run it (some driver / cuBLASLt
    builds return no heuristic algorithm — every scaled_mm then raises
    CUBLAS_STATUS_NOT_INITIALIZED).

    Scale layout: cuBLASLt block scaling on sm100 wants the per-block scales in
    the padded SWIZZLE_32_4_4 buffer — numel round_up(M,128)·round_up(ceil(K/bs),4)
    (e4m3 for nvfp4, e8m0 for mx) — plus, for nvfp4, a per-tensor fp32 global
    scale. mat_a row-major, mat_b column-major. (Values are random — this is a
    perf yardstick, not a correctness check.)"""
    import torch.nn.functional as F
    from torch.nn.functional import ScalingType, SwizzleType

    dev = "cuda"
    is_fp4, bs, _, _ = _COMBOS[combo]
    ru = lambda x, m: ((x + m - 1) // m) * m
    cd = lambda a, b: (a + b - 1) // b
    sw = [SwizzleType.SWIZZLE_32_4_4]

    if is_fp4:
        a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
        b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2).t()
    else:
        a = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn).t()

    na = ru(M, 128) * ru(cd(K, bs), 4)
    nb = ru(N, 128) * ru(cd(K, bs), 4)
    if combo == "nvfp4":
        sa = torch.randn(na, device=dev).to(torch.float8_e4m3fn)
        sb = torch.randn(nb, device=dev).to(torch.float8_e4m3fn)
        ga = torch.ones(1, device=dev)
        gb = torch.ones(1, device=dev)
        recipe = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
        scA, scB = [sa, ga], [sb, gb]
    else:  # mxfp4 / mxfp8 — e8m0, block32, no global scale
        sa = torch.randn(na, device=dev).to(torch.float8_e8m0fnu)
        sb = torch.randn(nb, device=dev).to(torch.float8_e8m0fnu)
        recipe = [ScalingType.BlockWise1x32]
        scA, scB = [sa], [sb]

    # scaled_mm has no batched form, so a batched reference is B back-to-back
    # single-GEMM scaled_mm calls (same operands — a perf yardstick, not a
    # correctness check). FLOPS counts all B, so the throughput stays comparable.
    def call():
        out = None
        for _ in range(batch):
            out = F.scaled_mm(a, b, scA, recipe, scB, recipe, sw, sw, None, torch.bfloat16)
        return out

    try:
        call()
        torch.cuda.synchronize()
    except Exception:
        return None
    suffix = f" ×{batch}" if batch > 1 else ""
    return f"cuBLAS {combo} scaled_mm{suffix}", call


def _make_reference(batch: int, M: int, N: int, K: int, combo: str, ref_mode: str, verbose: bool = True):
    """(label, call) for the throughput reference row. Each call allocates its
    OWN tensors, so calling this repeatedly yields independent buffers (used to
    build the rotation pool). `verbose=False` suppresses the fallback note when
    building extra pool entries.

    ref_mode='scaled_mm' → cuBLAS's own block-scaled kernel (errors out if the
    env can't run it). 'bf16' → dense BF16 cuBLAS. 'auto' (default) → try the
    real block-scaled kernel, fall back to BF16 with a clear label if it can't
    run here."""
    dev = "cuda"
    if ref_mode in ("auto", "scaled_mm"):
        ref = _scaled_mm_ref(batch, M, N, K, combo)
        if ref is not None:
            return ref
        if ref_mode == "scaled_mm":
            sys.exit(
                f"--ref scaled_mm: cuBLAS block-scaled GEMM for '{combo}' is not "
                f"runnable here — torch.nn.functional.scaled_mm raised at the "
                f"cuBLASLt heuristic (no algorithm for this driver / cuBLASLt "
                f"build). Use --ref bf16, or --ref auto to fall back automatically."
            )
        if verbose:
            print("  [note: scaled_mm reference unavailable in this env — " "falling back to dense BF16 cuBLAS]")
    # BF16 fallback: a single BATCHED matmul covers all B (torch.matmul is
    # natively batched), so no per-batch loop needed here.
    a = torch.randn(batch, M, K, dtype=torch.bfloat16, device=dev)
    b = torch.randn(batch, N, K, dtype=torch.bfloat16, device=dev)
    c = torch.empty(batch, M, N, dtype=torch.bfloat16, device=dev)
    label = "BF16 cuBLAS (fallback)" if ref_mode == "auto" else "BF16 cuBLAS"
    return label, (lambda: torch.matmul(a, b.transpose(-1, -2), out=c))


def _make_reference_pool(batch: int, M: int, N: int, K: int, combo: str, ref_mode: str, nbuf: int):
    """(label, warmup_call, [timed_call × nbuf]). The warmup call uses its own
    dedicated buffer (not rotated); the timed calls each allocate independent
    tensors so the timed loop can rotate across distinct GMEM addresses."""
    label, warmup_call = _make_reference(batch, M, N, K, combo, ref_mode, verbose=True)
    timed_calls = [_make_reference(batch, M, N, K, combo, ref_mode, verbose=False)[1] for _ in range(nbuf)]
    return label, warmup_call, timed_calls


def _cta_k_elems(cfg: TileConfig, combo: str) -> int:
    is_fp4 = _COMBOS[combo][0]
    return cfg.cta_tile_k_bytes * 8 // (4 if is_fp4 else 8)


def _compatible(cfg: TileConfig, M: int, N: int, K: int, combo: str) -> bool:
    """A config runs this block-scale shape iff it passes the block-scale
    validator AND the shape divides the CGRP tile / K-tile."""
    block_size = _COMBOS[combo][1]
    cta_k = _cta_k_elems(cfg, combo)
    try:
        validate_block_scale_config(cfg, block_size, cta_k)
    except NotImplementedError:
        return False
    tm, tn = cfg.cgrp_tile_mn
    return M % tm == 0 and N % tn == 0 and K % cta_k == 0


# ---------------------------------------------------------------------------
# Timing (events / delayed) — identical to benchmark_matmul.py
# ---------------------------------------------------------------------------


def _time_ms_events(
    timed_fn: Callable,
    warmup_fn: Callable,
    *,
    warmup: int,
    iters: int,
) -> float:
    """`timed_fn(i)` = per-launch callable (i rotates buffers); `warmup_fn()`
    runs warmups against a separate dedicated buffer (not rotated)."""
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
    """Kernel-only timing by hiding host-launch overhead behind a delay kernel
    (see benchmark_matmul.py for the full rationale). `timed_fn(i)` rotates buffers;
    `warmup_fn()` uses a separate dedicated buffer."""
    for _ in range(warmup):
        warmup_fn()
    torch.cuda.synchronize()

    delay_cycles = max(int(1e8), int((iters * 0.05 + 20.0) * 1.7e6))
    torch.cuda._sleep(delay_cycles)

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
# nsys mode — mirrors benchmark_matmul.py
# ---------------------------------------------------------------------------


def _nsys_run_and_parse(shape, combo, configs, warmup, iters, ref_mode, nbuf) -> dict[str, float]:
    nsys = "/usr/local/bin/nsys" if os.path.exists("/usr/local/bin/nsys") else shutil.which("nsys")
    if nsys is None:
        sys.exit("nsys not found — install nsight-systems or use the default events mode.")

    workdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"bench_bs_nsys_{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    report_prefix = os.path.join(workdir, "report")

    nsys_env = os.environ.copy()
    nsys_env.setdefault("TMPDIR", os.environ.get("TMPDIR", tempfile.gettempdir()))

    inner = [
        sys.executable,
        "-u",
        os.path.abspath(__file__),
        "--_nsys-worker",
        "--shape",
        shape,
        "--combo",
        combo,
        "--ref",
        ref_mode,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--rotate-buffers",
        str(nbuf),
    ]
    if configs:
        inner += ["--configs", ",".join(configs)]

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
    return config_name in kern_name


def _find_cublas_time(kern_times: dict[str, float]) -> tuple[str, float] | None:
    """The reference kernel in the nsys report. cuBLAS GEMMs (BF16 and the
    block-scaled FP4/FP8 path) show up as `nvjet_*`; if none match, fall back to
    the heaviest kernel that is NOT one of our tagged GEMM kernels
    (`..._kernel_CONFIG_sm100_...`)."""
    cands = [(k, v) for k, v in kern_times.items() if k.startswith("nvjet_")]
    if not cands:
        cands = [(k, v) for k, v in kern_times.items() if "_kernel_CONFIG_sm100_" not in k]
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Worker mode: run the kernels under nsys profile, no Python timing.
# ---------------------------------------------------------------------------


def _nsys_worker(shape, combo, configs, warmup, iters, ref_mode, nbuf) -> None:
    B, M, N, K = (int(x) for x in shape.split(","))
    wset = _mkdata(B, M, N, K, combo)  # dedicated warmup buffer
    pool = _mkdata_pool(B, M, N, K, combo, nbuf)  # rotation pool for timed iters
    ref_label, ref_warmup, ref_timed = _make_reference_pool(B, M, N, K, combo, ref_mode, nbuf)

    print(f"[worker] shape={B}x{M}x{N}x{K}, combo={combo}, ref={ref_label}, " f"configs={len(configs)}, warmup={warmup}, iters={iters}, rotate_buffers={nbuf}")

    # 1. reference kernel (cuBLAS block-scaled, or BF16 fallback).
    for _ in range(warmup):
        ref_warmup()
    for i in range(iters):
        ref_timed[i % nbuf]()
    torch.cuda.synchronize()

    # 2. each GEMM block-scale config.
    name_to_cfg = {lbl: sp[0] for lbl, sp in _SPEC_MAP.items()}
    config_names = configs or list(_SPEC_MAP)
    for name in config_names:
        cfg = name_to_cfg.get(name)
        if cfg is None or not _compatible(cfg, M, N, K, combo):
            continue
        try:
            g = _graph_block_scale(B, M, N, K, combo)
            compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=_SPEC_MAP[name][1], scheduler=_SPEC_MAP[name][2])
            wa, wb, wc, wsfa, wsfb = wset
            for _ in range(warmup):
                compiled(_vp_bs(compiled, wa, wb, wc, wsfa, wsfb))
            for i in range(iters):
                s = pool[i % nbuf]
                compiled(_vp_bs(compiled, s[0], s[1], s[2], s[3], s[4]))
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
        "--shape", default="1,4096,4096,4096", help="B,M,N,K (default 1,4096,4096,4096; B = batch / number " "of independent same-shape block-scale GEMMs)"
    )
    parser.add_argument("--combo", choices=tuple(_COMBOS), default="nvfp4", help="block-scale dtype family (default nvfp4: FP4 + E4M3 SF, block16)")
    parser.add_argument("--warmup", type=int, default=10)
    # Perf-testing rule (CLAUDE.md): keep iters <= 20 — more doesn't sharpen the
    # measurement here, it just lengthens the run / holds the GPU.
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--configs", default=None, help="comma-separated config names (default: every compatible CATALOG entry)")
    parser.add_argument(
        "--timing", choices=("delayed", "events", "nsys"), default="delayed", help="delayed (default), events, or nsys — see benchmark_matmul.py"
    )
    parser.add_argument(
        "--ref",
        choices=("auto", "scaled_mm", "bf16"),
        default="auto",
        help="reference kernel: scaled_mm = cuBLAS's OWN block-scaled "
        "FP4/FP8 GEMM (apples-to-apples, via F.scaled_mm); bf16 = "
        "dense BF16 cuBLAS yardstick; auto (default) = scaled_mm, "
        "falling back to bf16 if this env's cuBLASLt can't run it.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="print each config's result as soon as it finishes " "(events/delayed only; the last 'running' line points at a hang).",
    )
    parser.add_argument(
        "--rotate-buffers",
        default="auto",
        metavar="N",
        help="allocate N independent copies of every tensor and rotate the timed "
        "launches across them so a kernel doesn't re-read the previous "
        "launch's data from a hot L2 (inflates small-shape TFLOPS). Warmup "
        "uses a separate dedicated buffer. Default 'auto': size the pool to "
        "exceed the ~126 MB B200 L2 (large shapes → 2, small → scaled up), "
        "capped at 4 GB. Pass an integer to override; 1 disables.",
    )
    parser.add_argument("--_nsys-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    combo = args.combo
    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K (four values; use B=1 for a single GEMM)")
    B, M, N, K = parts
    per_set = _set_bytes(_mkdata(B, M, N, K, combo))
    nbuf = _resolve_nbuf(args.rotate_buffers, per_set)

    if getattr(args, "_nsys_worker"):
        configs = [c.strip() for c in args.configs.split(",")] if args.configs else []
        _nsys_worker(args.shape, args.combo, configs, args.warmup, args.iters, args.ref, nbuf)
        return 0

    flops = 2 * B * M * N * K
    config_names = [c.strip() for c in args.configs.split(",")] if args.configs else list(_SPEC_MAP)
    name_to_cfg = {lbl: sp[0] for lbl, sp in _SPEC_MAP.items()}

    print(f"\n=== block-scale matmul B={B} {M}x{N}x{K}  (~{flops / 1e9:.1f} GFLOP) — " f"{combo} in / BF16 out ===")

    if nbuf > 1:
        footprint = per_set * nbuf
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

    ref_label = "reference"
    if args.timing == "nsys":
        print(f"  [timing: nsys median kernel duration; ref={args.ref}]\n")
        kern_times = _nsys_run_and_parse(args.shape, combo, config_names, args.warmup, args.iters, args.ref, nbuf)

        cublas_hit = _find_cublas_time(kern_times)
        if cublas_hit:
            cublas_name, cublas_ms = cublas_hit
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            ref_label = cublas_name
            print(f"  reference kernel: {cublas_name}")
        else:
            cublas_tflops, cublas_ms = float("nan"), float("nan")
            print("  reference kernel: not detected in nsys output")

        for name in config_names:
            cfg = name_to_cfg.get(name)
            if cfg is None:
                rows.append((name, 0.0, float("inf"), "UNKNOWN_CONFIG"))
                continue
            if not _compatible(cfg, M, N, K, combo):
                rows.append((name, 0.0, float("inf"), "incompatible"))
                continue
            matches = [(k, v) for k, v in kern_times.items() if _match_kernel_name(k, name)]
            if not matches:
                rows.append((name, 0.0, float("inf"), "NO_KERNEL_IN_NSYS"))
                continue
            _, ms = max(matches, key=lambda x: x[1])
            rows.append((name, flops / (ms * 1e-3) / 1e12, ms, ""))
    else:
        timer = _time_ms_delayed if args.timing == "delayed" else _time_ms_events
        if args.timing == "delayed":
            print(f"  [timing: events around delayed back-to-back launches — " f"host overhead hidden behind a CUDA _sleep]\n")
        else:
            print(f"  [timing: torch.cuda.Event wall-clock around python loop — " f"includes ~50us/call dispatch overhead]\n")

        wset = _mkdata(B, M, N, K, combo)  # dedicated warmup buffer
        pool = _mkdata_pool(B, M, N, K, combo, nbuf)  # rotation pool for timed iters
        ref_label, ref_warmup, ref_timed = _make_reference_pool(B, M, N, K, combo, args.ref, nbuf)
        if args.stream:
            print(f"  ▶ running {ref_label} reference ...", flush=True)
        cublas_ms = timer(
            _rotating(lambda call: call(), ref_timed),
            ref_warmup,
            warmup=args.warmup,
            iters=args.iters,
        )
        cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
        if args.stream:
            print(_fmt_row(f"{ref_label} (reference)", cublas_tflops, cublas_ms, "", cublas_tflops), flush=True)

        ctx_dead = False
        for name in config_names:
            cfg = name_to_cfg.get(name)
            if cfg is None:
                row = (name, 0.0, float("inf"), "UNKNOWN_CONFIG")
            elif not _compatible(cfg, M, N, K, combo):
                row = (name, 0.0, float("inf"), "incompatible")
            elif ctx_dead:
                row = (name, 0.0, float("inf"), "skipped (CUDA context dead)")
            else:
                if args.stream:
                    print(f"  ▶ running {name} ...", flush=True)
                try:
                    g = _graph_block_scale(B, M, N, K, combo)
                    compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=_SPEC_MAP[name][1], scheduler=_SPEC_MAP[name][2])
                    wa, wb, wc, wsfa, wsfb = wset
                    ms = timer(
                        _rotating(
                            lambda s, _c=compiled: _c(_vp_bs(_c, s[0], s[1], s[2], s[3], s[4])),
                            pool,
                        ),
                        lambda _c=compiled: _c(_vp_bs(_c, wa, wb, wc, wsfa, wsfb)),
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    row = (name, flops / (ms * 1e-3) / 1e12, ms, "")
                except Exception as e:
                    msg = str(e).splitlines()[0][:50] if str(e) else type(e).__name__
                    row = (name, 0.0, float("inf"), f"ERR {msg}")
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

    is_bf16_ref = "BF16" in ref_label
    if is_bf16_ref:
        print("  [reference is dense BF16 cuBLAS — fp4/fp8 peak is ~2× BF16, so >1× " "is expected and NOT a like-for-like win]")
    vs_col = "vs BF16" if is_bf16_ref else "vs cuBLAS"

    rows.sort(key=lambda r: -r[1])
    print("=" * 88)
    print(f"  {'config':50s} {'TFLOPS':>8s}   {'ms':>7s}   {vs_col:>10s}")
    print("=" * 88)
    for name, tflops, ms, note in rows:
        print(_fmt_row(name, tflops, ms, note, cublas_tflops))
    print("=" * 88)
    if cublas_tflops > 0:
        print(f"  {ref_label + ' (reference)':50s} {cublas_tflops:8.2f}   " f"{cublas_ms:7.3f}   {'1.00×':>10s}")
    else:
        print(f"  {ref_label} reference: n/a")

    ok = [r for r in rows if not r[3]]
    if ok and cublas_tflops > 0:
        best_name, best_tflops, best_ms, _ = ok[0]
        print(f"\nbest GEMM ({combo}): {best_name}" f" — {best_tflops:.2f} TFLOPS" f" ({best_tflops / cublas_tflops:.2f}× {ref_label})")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
