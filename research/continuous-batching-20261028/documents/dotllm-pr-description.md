# Add continuous batching and direct paged attention on CPU and CUDA

Concurrent HTTP requests currently serialize whole generations, and the paged CPU
cache gathers historical KV into contiguous staging before attention. This change
adds opt-in iteration-level scheduling with packed model projections and direct
physical-page attention on CPU and full CUDA offload. Requests arriving during a
running generation can join subsequent forwards; completed/cancelled requests return
their pages for reuse.

- Introduce `IBatchedModel`/sequence segments, a bounded scheduler, decode-first token
  budgets, round-robin chunked prefill and conservative page reservations.
- Keep RNG, constraint clones, output streams and KV ownership separate per request;
  handle EOS, context/max-token limits, stop strings across/inside tokens, cancellation,
  slow consumers and forward/cleanup failures.
- Read CPU Float32 pages directly with tiled online softmax. Add shared CUDA Float16
  page pools, device block tables, scatter writes and direct paged attention.
- Wire both OpenAI completion endpoints and CLI options, drain leases on model swaps,
  and release scheduler/pool resources before model teardown.
- Add real-model parity tests, scalar kernel references and throughput/latency benchmarks.
  Fix the compute thread-pool first-dispatch race exposed during this work.

Validation: Release builds; 1196 unit tests passed locally (41 CUDA/environment-dependent
cases skipped); four SmolLM-135M Q8_0 CPU integration tests passed. On one NVIDIA H20,
four fragmented-page kernel cases matched a scalar reference and concurrent greedy
generation matched serial generation for three distinct prompts. Page reclamation
checks passed. In three H20 runs of 32 requests × 64 tokens, increasing active
sequences from 1 to 4 raised median throughput from 347.70 to 600.75 tok/s (1.73×),
with median per-run TTFT p50 falling from 2763.91 to 1284.46 ms and ITL p50 rising
from 2.87 to 6.70 ms. The baseline uses the same scheduler/kernel with one active
sequence. This is a combined concurrency/packed-execution comparison, not a legacy
upstream baseline or an isolated Direct Pages speedup. Measurements use scheduler
submission and channel consumption, excluding HTTP/SSE transport; runs lasted about
3.4–5.9 seconds. See the accompanying validation notes for workload details and CPU regressions.

An A0–A4 ablation plan now separates request scheduling, packed forwards, paged
storage with gather, and direct page reads. It requires a common attention kernel
body, CUDA Event timing, contiguous/fanout/gather experiment paths, representative
3B/7B workloads, and real HTTP/SSE load tests. These ablations are not implemented
or measured yet; no per-component speedup is claimed.

Limitations: no active preemption, automatic prefix sharing, quantized KV, hybrid
offload, speculative drafts, request adapters, multi-GPU or CUDA graphs in batching
mode. GPU attention launches remain per sequence/layer. CPU short-context attention
can regress; batching may increase individual inter-token latency. CUDA quantized
batch projections use dequantization plus GEMM and can require additional scratch.

This is a local PR description draft. No PR has been published and no issue is marked closed.
