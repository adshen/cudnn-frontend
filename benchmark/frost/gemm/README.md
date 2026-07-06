# cudnn.frost.gemm benchmarks

Performance measurement scripts for the cudnn.frost.gemm GEMM kernels on sm100.

## benchmark_matmul.py

Sweeps every CATALOG TileConfig on a single matmul shape and reports
TFLOPS / ms / ratio-vs-cuBLAS, sorted best → worst. BF16, pure matmul,
no fusion, no CPU verify — this script is for **perf only**, trust the
test suite for correctness.

```bash
# from the cudnn-frontend repo root
python benchmark/frost/gemm/benchmark_matmul.py                              # default: 4096³, --timing delayed
python benchmark/frost/gemm/benchmark_matmul.py --shape 8192,8192,8192
python benchmark/frost/gemm/benchmark_matmul.py --shape 4096,11008,4096      # Llama-FFN up
python benchmark/frost/gemm/benchmark_matmul.py --configs CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma,CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma
python benchmark/frost/gemm/benchmark_matmul.py --timing nsys                # ground-truth, slower
python benchmark/frost/gemm/benchmark_matmul.py --timing events              # plain wall-clock (includes ~50us/call Python overhead)
```

### Timing modes

| mode | accuracy vs nsys | speed | notes |
|---|---|---|---|
| `delayed` (default) | <2% | ~15s for 18 configs | queues `torch.cuda._sleep(~120ms)` before the timed loop so host overhead is hidden behind it; kernels execute back-to-back |
| `nsys` | ground truth | ~22s (incl. nsys profile + stats) | re-execs self under `nsys profile`, parses `cuda_gpu_kern_sum` median (ns) |
| `events` | bad for sub-ms kernels | ~15s | plain CUDA Events around a python loop; inflates GEMM ~50us/call from Python+TVM-FFI dispatch (cuBLAS unaffected because its launch path is C++) |

For `--timing nsys` you may need `TMPDIR=$TMPDIR  # or any user-writable dir` (or
any user-writable path) — system `/tmp/nvidia` is root-owned on this host.

### Why no `torch.cuda.CUDAGraph` mode

GEMM kernels launch via `cuLaunchKernelEx` from `libcute_dsl_runtime.so`,
outside torch's stream API. Torch's `CUDAGraph` capture sees an empty
graph. `cute.testing.benchmark` *could* capture, but it requires the cute
jit fn to take `stream: CUstream` as an explicit parameter, which GEMM's
`_host` doesn't — wiring it through every kernel template would be more
invasive than the delay-kernel workaround in `--timing delayed`.

## Sample output (B200, 4096³ BF16, `--timing delayed`)

```text
=== matmul 4096x4096x4096  (~137.4 GFLOP) — BF16 ===
========================================================================================
  config                                               TFLOPS        ms    vs cuBLAS
========================================================================================
  CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma            1596.77     0.086        0.93×
  CONFIG_sm100_128x256x128_128x256x32_cluster4x1_2ctamma            1589.94     0.086        0.92×
  CONFIG_sm100_128x256x128_128x256x32_cluster2x2_2ctamma            1586.18     0.087        0.92×
  CONFIG_sm100_128x256x128_128x256x32_cluster1x2_1ctamma            1420.17     0.097        0.82×
  ...
  CONFIG_sm100_64x32x128_64x32x32_cluster1x1_1ctamma               188.01     0.731        0.11×
========================================================================================
  cuBLAS (reference)                                  1726.09     0.080        1.00×
```

## nsys profiling (manual)

Generated GPU kernels carry the TileConfig name in their symbol (see
`compiler._render_template`), so `nsys stats --report cuda_gpu_kern_sum`
shows one row per config — sort by `Time (%)` to see the winner against
cuBLAS's `nvjet_sm100_*` kernel.

```bash
TMPDIR=$TMPDIR  # or any user-writable dir \
nsys profile -o report --stats=true --force-overwrite=true \
  python benchmark/frost/gemm/benchmark_matmul.py --timing events --shape 4096,4096,4096 \
    --configs CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma
```

Or just use `--timing nsys` and let the script drive nsys + parse for you.
