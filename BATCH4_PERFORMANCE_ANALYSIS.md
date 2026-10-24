# Continuous Batching Performance Analysis at Batch Size 4

This note publishes a technical summary of a performance-analysis discussion
for the continuous-batching prototype. It intentionally omits private chat
metadata, local absolute paths, and unverified performance numbers.

Reference implementation:

https://github.com/Cheongroach/dotLLM/tree/research/continuous-batching-20261018

## Discussion summary

**Question:** The active-sequence limit is currently four. How should
performance analysis continue?

**Answer:** First determine whether the scheduler actually sustains four
sequences during steady-state decode. If it does not, investigate arrivals,
admission, conservative KV reservation, and early request completion. If it
does, decompose each iteration into GPU kernels, transfers, synchronization,
and CPU-side scheduling/sampling before changing the implementation.

The existing trace benchmark already compares `max_sequences=1` with the
requested concurrency using the same scheduler and paged-attention path. It
reports aggregate output throughput, end-to-end TTFT and ITL percentiles,
forward count, observed batch occupancy, and KV-page reclamation.

Two implementation details require special attention on CUDA:

1. A supported quantized projection can use direct quantized GEMV for one row.
   Multiple rows take the GEMM path; when no persistent FP16 weight exists,
   that path first dequantizes the weight into scratch memory. Therefore,
   changing batch size from one to four may also change the projection
   algorithm.
2. Paged KV scatter and attention are launched once per sequence per layer,
   while the projections and FFN operate on packed rows. Small attention
   kernels and launch overhead may limit scaling even when projections benefit
   from batching.

These are profiling hypotheses, not measured conclusions.

## 1. Freeze the experiment

Record the following with every result:

- exact commit SHA;
- model file identity and quantization;
- backend, GPU/CPU model, driver, runtime, and .NET version;
- prompt tokens and actual generated tokens;
- request count, arrival pattern, output limit, and thread count;
- `MaxSequences`, `MaxBatchedTokens`, `PrefillChunkSize`, block size, and total
  KV capacity;
- warm-up policy, repetition count, and raw JSON output.

Use greedy decoding so output is deterministic. Do not compare runs that use
different prompts, model files, quantization, KV capacity, or arrival traces.

## 2. Establish the batch-1 and batch-4 baseline

The benchmark arguments are:

```text
--continuous MODEL BACKEND CONCURRENCY REQUESTS MAX_TOKENS ARRIVAL_MS CPU_THREADS
```

Run a long enough decode-heavy trace to reduce startup and tail effects:

```bash
dotnet run --project benchmarks/DotLLM.Benchmarks -c Release -- \
  --continuous /path/to/model.gguf gpu 4 64 128 0 4
```

The command emits one JSON object for `max_sequences=1` and another for
`max_sequences=4`. Repeat the complete command at least five times and preserve
every raw result. Report the median and dispersion rather than the best run.

Interpret the fields as follows:

| Field | Interpretation |
|---|---|
| `tokens_per_second` | Aggregate output throughput using actual generated tokens |
| `ttft_p50_ms`, `ttft_p95_ms` | Submission-to-first-token latency, including queueing |
| `itl_p50_ms`, `itl_p95_ms` | Per-request time between generated tokens |
| `mean_batch_sequences` | Mean sequence rows per forward across the whole trace |
| `max_batch_sequences` | Largest observed sequence batch |
| `forwards` | Total model forward calls |
| `kv_pages_reclaimed` | Whether all pages were returned after completion |

Do not treat `MaxSequences=4` as proof that every iteration contains four
sequences. The current mean combines prefill, steady-state decode, startup, and
drain phases. A useful benchmark extension is a decode-only occupancy histogram:

```text
batch_size -> iteration_count, total_forward_ms, p50_forward_ms, p95_forward_ms
```

Also record queued and active request counts per iteration. This distinguishes
insufficient offered load from admission or compute bottlenecks.

## 3. Sweep batch size

With all other variables fixed, evaluate:

```text
MaxSequences = 1, 2, 4, 8, 16
```

Keep total KV capacity fixed across the sweep. The current standalone benchmark
derives pool capacity from requested concurrency, so modify or parameterize it
before claiming a strict scaling comparison across separate invocations.

During steady-state decode, each active sequence contributes one input token
and normally produces one output token per iteration. A useful approximation is:

```text
aggregate throughput ~= active sequences / iteration time
per-request ITL      ~= iteration time
```

The selected operating point should maximize throughput or goodput while
respecting TTFT and ITL service-level objectives. Peak throughput alone is not
enough if P95 latency becomes unacceptable.

## 4. Separate workload classes

Use at least three traces:

| Trace | Purpose |
|---|---|
| Short prompt, long output | Isolate decode, weight reads, and kernel-launch overhead |
| Long prompt, short output | Measure packed prefill and chunking behavior |
| Mixed lengths with timed arrivals | Measure TTFT/ITL under continuous admission and drain |

The prototype benchmark currently rotates through four short prompts. It is
adequate for an initial decode comparison, but not for a full prefill or mixed
production-load claim.

## 5. Profile the CUDA timeline

Build once, then profile the built benchmark executable so compilation and
restore activity do not pollute the trace:

```bash
dotnet build benchmarks/DotLLM.Benchmarks -c Release

nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  --force-overwrite=true \
  --output=dotllm-batch4 \
  dotnet benchmarks/DotLLM.Benchmarks/bin/Release/net10.0/DotLLM.Benchmarks.dll \
  --continuous /path/to/model.gguf gpu 4 64 128 0 4

nsys stats dotllm-batch4.nsys-rep
```

Capture a warm steady-state region when possible. Inspect:

- GPU utilization and gaps between launches;
- counts and total time for dequantization, GEMM/GEMV, paged attention, KV
  scatter, normalization, activation, and logits conversion;
- H2D/D2H transfer time and size;
- time spent waiting at stream synchronization;
- CPU scheduler/sampling work between GPU iterations.

Synchronization time includes outstanding GPU work. Do not add it to kernel
time as if it were independent CPU overhead.

Use Nsight Compute only after Nsight Systems identifies the expensive kernels:

```bash
ncu \
  --set full \
  --target-processes all \
  --kernel-name-base demangled \
  --launch-skip 100 \
  --launch-count 20 \
  --export dotllm-batch4-kernels \
  dotnet benchmarks/DotLLM.Benchmarks/bin/Release/net10.0/DotLLM.Benchmarks.dll \
  --continuous /path/to/model.gguf gpu 4 64 128 0 4
```

Adjust skip/count after inspecting the Systems timeline. For the selected
kernels, inspect memory throughput, achieved occupancy, eligible warps,
Tensor Core use, and launch dimensions. Collecting every metric for the whole
application is unnecessarily expensive.

## 6. Test the leading hypotheses

Run one controlled experiment per hypothesis:

| Hypothesis | Evidence | Possible follow-up |
|---|---|---|
| On-the-fly weight dequantization dominates batched projections | Large `dequant_*` share at batch 4 but not batch 1 | Quantized small-batch GEMM or persistent FP16 weights, with VRAM cost reported |
| Per-sequence attention launches dominate | Many short attention/scatter kernels and launch gaps | Batch sequence metadata and launch one kernel per layer |
| Logits transfer or CPU sampling stalls iterations | Large D2H/sync or CPU gap after logits | Transfer fewer logits, GPU sampling, or optimize host sampling |
| Scheduler does not sustain four rows | Decode occupancy histogram remains below four despite queued work | Inspect KV reservation, admission, EOS, and arrival policy |
| Batch four is already beyond the efficient point | Step time grows faster than useful throughput | Select a smaller batch or improve the identified kernel path |

After each change, rerun the same correctness tests and complete baseline
matrix. Do not combine multiple optimizations before establishing attribution.

## 7. Reporting template

```text
Commit:
Model / quantization:
Hardware / driver / runtime:
Workload:
Scheduler and KV configuration:
Warm-up and repetitions:

Batch | Mean decode batch | Output tok/s | TTFT P50/P95 | ITL P50/P95
----- | ----------------- | ------------ | ------------- | -----------
1     |                   |              |               |
2     |                   |              |               |
4     |                   |              |               |
8     |                   |              |               |

Nsight Systems top operations:
Nsight Compute findings:
Correctness checks:
Raw result and report locations:
Conclusion and remaining uncertainty:
```

Until this matrix and profile evidence exist, the defensible statement is that
the prototype supports an active-sequence limit of four and provides the
measurement hooks above. It is not yet evidence of fourfold scaling or of a
specific CUDA bottleneck.
