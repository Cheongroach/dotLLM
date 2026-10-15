# Scheduling & Continuous Batching — dotLLM

`ContinuousBatchScheduler` owns model compute and request KV caches. Multiple requests
share each `IBatchedModel.ForwardBatch` call: projections and FFNs operate on packed
rows, while attention reads only the corresponding request's pages. No padding or
cross-request attention is introduced. CPU and full CUDA offload are supported.

## Enable serving

```bash
dotllm serve model.gguf --continuous-batching --max-sequences 4 --max-batched-tokens 128 --kv-cache-tokens 4096
dotllm serve model.gguf --device gpu:0 --continuous-batching --max-sequences 4 --max-batched-tokens 128 --kv-cache-tokens 4096
```

The mode is opt-in. Both completions endpoints use it, for regular JSON responses
and SSE streaming. Model swaps block new requests and drain existing HTTP leases
before stopping the scheduler and disposing its pool/model. Shutdown cancels work
and waits for leases to unwind. GPU paged kernels must be compiled and deployed
alongside the existing PTX files (`native/kernels/paged_attention.cu`).

The model supports one compute owner; do not run `TextGenerator` or another forward
concurrently with the scheduler. Dispose in order: scheduler, page pool, model.

## Iteration policy

1. Drain arrivals and cancel queued/active requests, returning their pages.
2. Admit by priority, FIFO within a priority, up to `MaxSequences` and the reserved
   page budget. A capacity waiter is not bypassed by a smaller request of equal priority.
3. Give every decoding request one token slot. `MaxBatchedTokens > MaxSequences`
   guarantees space for them.
4. Fill remaining slots with prompt chunks up to `PrefillChunkSize`, rotating the
   first prompt considered each iteration. Prefill and decode may share one forward.
5. Sample only completed prompt chunks and decode rows. Each request owns its RNG,
   constraint clone, token history, output stream and KV cache.
6. Finish EOS, stop, context-limit and max-token requests, reclaim their pages, then
   admit more requests on the next iteration.

A request reserves `ceil(capacity / blockSize)` logical pages at admission, where
`capacity = min(contextLimit, promptTokens + max(0, maxTokens - 1))`. This covers
its worst-case cached length; the final generated token does not need a KV slot.
Physical page IDs are assigned as the cache grows. Reservation is conservative:
unused output budgets can reduce concurrency. The pool's underlying storage is a
fixed allocation; releasing page IDs makes that storage reusable, not smaller.

`MaxTokens = 0` returns without a forward. A full context can still produce one
next-token prediction. Requests too large for the pool and invalid inputs fail at
submission. A bounded waiting queue rejects overload instead of growing indefinitely.

## Streaming and failures

`GenerateStreamingAsync` uses a bounded per-request channel. A consumer that falls
behind its `StreamBufferCapacity` fails its own request; it cannot block shared
compute. Disposing enumeration cancels that request and waits for KV release.
Stop-string prefixes are withheld until resolved, including stops spanning tokens
or ending inside a token. The terminal event reports `GeneratedTokenCount` so EOS
and zero-token responses are counted correctly.

Sampling/constraint errors affect one request. A forward failure stops the loop
and fails every outstanding request; GPU work is synchronized before page reuse.
Custom sampler steps, logit processors and stop conditions must be immutable or
owned by a single request. Constraints are cloned by the scheduler.

## Scope

- CPU uses Float32 pages and direct SIMD page reads; CUDA uses Float16 pages,
  device page tables, a scatter kernel for new KV and tiled online softmax.
- Weight quantization remains supported. Quantized **KV**, hybrid CPU/GPU models,
  speculative draft models and per-request adapters are unsupported in this mode.
- Prefix reuse is disabled while batching. CPU block refcounts/CoW remain available
  separately, but automatic cross-request prefix sharing is not implemented.
- Priority controls waiting-queue admission only. There is no active preemption,
  swap/recompute, priority aging, distributed prefill/decode, or CUDA graph capture.
- Attention launches are per sequence/layer; the shared projection work is batched.
  Further GPU kernel fusion and batching-aware quantized GEMM are performance work.
- Timing fields in individual responses sum the shared iterations in which that
  request participated. They are not exclusive per-request compute costs. Measure
  end-to-end TTFT/ITL separately, including queueing.

## Tests and benchmarks

```bash
dotnet test tests/DotLLM.Tests.Unit -c Release
dotnet test tests/DotLLM.Tests.Integration -c Release --filter 'FullyQualifiedName~ContinuousBatchingIntegrationTests|FullyQualifiedName~PagedKvCacheIntegrationTests'
DOTLLM_GPU_TEST_MODEL=/path/model.gguf dotnet test tests/DotLLM.Tests.Unit -c Release --filter 'FullyQualifiedName~CudaPagedAttentionTests|FullyQualifiedName~CudaContinuousBatchingTests'

# MODEL BACKEND CONCURRENCY REQUESTS MAX_TOKENS ARRIVAL_MS CPU_THREADS
# Runs the same trace with max_sequences=1, then CONCURRENCY; emits JSON lines.
dotnet run --project benchmarks/DotLLM.Benchmarks -c Release -- --continuous /path/model.gguf cpu 4 16 32 0 4
dotnet run --project benchmarks/DotLLM.Benchmarks -c Release -- --continuous /path/model.gguf gpu 4 16 32 10 4

# Kernel-only comparison: historical gather+attention vs direct page reads.
dotnet run --project benchmarks/DotLLM.Benchmarks -c Release -- --filter '*PagedAttentionBenchmarks*'
```

The trace benchmark records warm-up-excluded throughput, TTFT/ITL percentiles,
actual sequence rows per forward and page reclamation. Its single-sequence baseline
uses the same scheduler/paged kernel with admission limited to one request, isolating
the batching effect. More requests can improve throughput/queueing while increasing
individual decode latency; evaluate both. The kernel benchmark includes the legacy
attention math as well as gather cost, so its ratio does not isolate copying alone.
