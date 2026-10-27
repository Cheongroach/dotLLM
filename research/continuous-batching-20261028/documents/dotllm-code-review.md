# dotLLM Continuous Batching + PagedAttention Review

## Intent

Turn request-level serialized inference into iteration-level continuous batching:
pack independent token segments for shared projections, preserve per-request attention
and sampling state, and replace KV gather with direct physical-page reads on CPU/CUDA.

## Findings

| Severity | Finding | Impact | Recommended action |
|---|---|---|---|
| P1 | CUDA paged KV is enabled for every default server invocation, not only continuous batching | The fixed 65,536-token pool reserves about 7 GiB for Llama-3.2-3B and 8 GiB for a typical 7B GQA model before serving the first request. This can turn an otherwise valid single-request CUDA deployment into startup OOM. | Keep the existing per-request `CudaKvCache` outside continuous batching, or size/allocate GPU pages lazily from an explicit VRAM budget. |
| P1 | Non-batched CUDA stores `CudaPagedKvCache` objects in a prefix cache that cannot reuse them | `TextGenerator.ResolveKvCache` supports only `SimpleKvCache` and CPU `PagedKvCache`; CUDA entries always miss but remain alive. Repeated requests retain pages until LRU store-time eviction, and a finite pool can exhaust before the next request reaches that eviction point. | Disable prefix caching for this cache type, or add a backend-neutral truncation/reuse contract and evict before allocation. |
| P1 | Quantized CUDA batched decode dequantizes every full projection on every token step | Any packed batch with more than one row selects the dequantize-to-FP16 plus HGEMM path. For the GGUF-first workload, low dynamic batch sizes can move more bytes than independent quantized GEMVs and need a full-size FP16 weight scratch. The 135M result does not establish the crossover for 3B/7B models. | Add a batched quantized kernel, or dispatch to per-row quantized GEMV below a measured batch threshold. Benchmark representative 3B/7B quantizations. |
| P2 | CUDA paged and contiguous attention disagree on sliding-window width | The paged kernel attends exactly `W` positions, matching CPU, while the existing CUDA kernels use `pos_q - tkv > W` and include `W+1`. Toggling paged mode therefore changes logits for sliding-window models. | Unify all kernels on one documented boundary and add paged/non-paged CUDA parity tests for window sizes 1 and 17. |

## Verification

- `1196` non-GPU unit tests passed.
- `4` focused real-model integration tests passed.
- Existing H20 evidence covers fragmented pages, MHA/GQA/MQA, scalar parity,
  serial/batched greedy parity and page reclamation.
- `git diff --check` passed.

The successful tests establish the main algorithm and lifecycle path. They do not
cover default-pool sizing, repeated non-batched CUDA prefix-cache requests, a
representative 3B/7B quantized crossover, or paged/non-paged CUDA sliding-window parity.

## Merge Scope

The branch substantially exceeds issue #256, whose acceptance criteria request a
foundation/MVP with no modifications to existing files and explicitly leave server
wiring, priority admission, kernel batching and streaming to follow-up work. Submit as
stacked PRs or open a new design issue before requesting review.
