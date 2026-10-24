# Ascend / CUDA follow-up evidence

2026-10-24. [Full report](../docs/ASCEND_QWEN36_AND_CUDA.md).
Implementation baseline: `93010f3e6208a58fdeb27c174b0ca6d2f876250e`.
The portable comparison harness is on the separate
[code branch](https://github.com/Cheongroach/dotLLM/tree/ascend/910b4-dense-baseline/tests/DotLLM.Tests.BackendComparison).

Both temporary workers completed actual hardware execution: one Ascend 910B4
64 GB with CANN 8.3.RC1; one NVIDIA H20 with explicit CUDA compatibility and
cuBLAS 12 libraries. Both had 2 CPU cores of quota and 16 GiB host memory.
These are different implementations and precision paths, not normalized
hardware benchmarks.

| Result | Evidence |
| --- | --- |
| Smol CLI decode: Ascend 10.16, CUDA 275.10 tok/s, three-run medians | `cli/`, `comparison-summary.json` |
| Qwen CLI: Ascend 8.32 tok/s; CUDA baseline numerically invalid | `cli/`, report |
| Each Ascend model and CUDA Smol: 33/33 teacher comparisons PASS | `ascend-*-followup/`, `cuda-smol/` |
| CUDA Qwen baseline: 0/33 teacher comparisons PASS | `cuda-qwen-baseline/` |
| CUDA Qwen diagnostic raw RoPE type 1: 33/33 PASS | `cuda-qwen-rope-diagnostic/` |
| Direct Ascend vs CUDA logits: 33/33 per model PASS; same top-1, not bit-exact | `cross-device-numerics.jsonl` |
| Actual CUDA profiling: 50.434 ms hot interval, 22.820 ms kernels | `nsys-steady-summary.json` |
| Actual NPU profiling: 1585.148 ms hot interval, 114.907 ms operator tasks | `msprof-steady-summary.json` |
| Both test jobs stopped, no active pods; previous HL request retained | `resource-cleanup.json` |

![Warm Forward measurements](warm-forward.png)

![Profile compute costs](profile-costs.png)

All comparison metrics remain available, including failures and the initial
GPU harness run. The new portable harness's warm measurements are the report's
primary warm table. The original NPU full-suite results remain unchanged in
`../docs/ascend-results/20261024/`.

`execution_complete: PASS` / exit zero only confirms execution. Inspect all
numerical records independently. Thresholds are finite values,
`RMS(error) <= 0.03 + 0.015*RMS(reference)`, `max_abs <= 0.25`,
`cosine >= 0.999`; direct device comparisons use Ascend logits as the reference.
Split checks and exact rollback are documented in the harness README.
Qwen diagnostic timing is not a corrected production CLI benchmark.

## Reproduce profiling

Prepare the package described by the code branch's Ascend runtime and full-suite
instructions, plus the built `DotLLM.Tests.BackendComparison` output. Use the
same model/golden hashes in `source-manifest.json`.

CUDA, with the tested library versions in `environment.json`:

```bash
nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
  --output=cuda-profile \
  /absolute/package/runtime/dotnet \
  /absolute/package/compare-followup/DotLLM.Tests.BackendComparison.dll \
  cuda /absolute/package/models/SmolLM-135M.Q8_0.gguf \
  /absolute/package/models/smol-text.json \
  /absolute/package/models/smol-numpy-f32.bin /absolute/results/cuda profile
nsys export --type=sqlite --output=cuda-profile.sqlite cuda-profile.nsys-rep
python analyze_nsys.py cuda-profile.sqlite app.stdout exported
```

Capture the application's stdout separately for its UTC phase markers.
The actual program is C# → native CUDA; NVTX was enabled but the application
does not emit custom NVTX ranges in this experiment.

Ascend, with `set_env.sh` sourced and `DOTLLM_ASCEND_LIBRARY` pointing to the
built bridge, uses **absolute paths** because msprof can change the working
directory:

```bash
msprof --output=/absolute/msprof-smol --ascendcl=on --runtime-api=on \
  --task-time=on --aicpu=on --ai-core=on --aic-metrics=PipeUtilization \
  --application="/absolute/package/runtime/dotnet /absolute/package/compare-followup/DotLLM.Tests.BackendComparison.dll ascend /absolute/package/models/SmolLM-135M.Q8_0.gguf /absolute/package/models/smol-text.json /absolute/package/models/smol-numpy-f32.bin /absolute/results/ascend profile" \
  > app.stdout 2> app.stderr
python analyze_msprof.py /absolute/msprof-smol/PROF_*/mindstudio_profiler_output \
  app.stdout exported
```

Both use load, one warm-up, one cached 32-token prefill and eight forced decode
steps. The exporter crops prefill_begin → last decode marker using tool session
clocks or absolute microsecond task timestamps. No profiling range is invented
from guessed kernel counts.

## Open and interpret the traces

Drag `cuda-steady-trace.json` or `ascend-steady-trace.json` into
[Perfetto](https://ui.perfetto.dev/). Exporters retain only allowlisted timing
fields; process environment, original machine identifiers, connection details,
original trace args and internal resources are not included.
`msprof-op-steady.csv` also preserves actual shapes, dtypes and PipeUtilization
fields from the device task export. Tiny-task counters should not be interpreted
as whole-device utilization or sustained HBM throughput.

The `msprof-*-statistic*.csv` tables are full-process tool exports; they include
warm-up/lifecycle and **are not** the steady table. Prefer the cropped summaries
when assessing inference. API calls are nested or overlap device execution.
Device task/kernel busy fractions are timeline measures; neither is SM/AICore
occupancy. NPU's standalone copy activity is not included in its operator-task
busy fraction. All profiled times include profiler overhead.

Raw Nsight report/SQLite and original msprof exports are retained locally under
the repository's `.git/followup-work/`; their hashes establish provenance.
They are not pushed because they contain extra process metadata. Public traces
and CSVs suffice to inspect the recorded inference timing.

Qwen3.6 architecture/memory estimates and public quality benchmarks are research,
not measurements of Qwen3.6 running in DotLLM. Source links and limitations are
in the full report.
