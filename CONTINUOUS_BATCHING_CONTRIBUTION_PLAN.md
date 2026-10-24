# Continuous Batching Contribution Plan

Author: Cheongroach  
Prototype base: `d88040451d7db56e5dfef9d5754ad0955b0f7fe5`  
Prototype tip: `1ceba8c98f841bfa03f704c3aab32949aae7983d`

Reference implementation:

https://github.com/Cheongroach/dotLLM/tree/research/continuous-batching-20261018

Published performance-analysis discussion and batch-size-4 methodology:

[BATCH4_PERFORMANCE_ANALYSIS.md](BATCH4_PERFORMANCE_ANALYSIS.md)

This branch documents the intended upstream contribution boundaries. It is not
an upstream pull request and does not claim ownership of existing issues.

## Why the prototype is not one PR

The prototype validates an end-to-end CPU/CUDA path across 43 files and six
commits, including:

- direct paged KV attention;
- packed multi-sequence model forward;
- continuous scheduling and chunked prefill;
- per-request cancellation and streaming;
- ASP.NET and CLI integration;
- correctness tests and benchmark tooling.

Submitting the whole branch would overlap existing scheduler and ForwardBatch
work and would not provide reviewable or independently revertible boundaries.

## Existing upstream work

The contribution must coordinate with:

| Work | Existing tracking |
|---|---|
| Scheduler foundation and MVP | #256 |
| Async scheduler service | #258 |
| Server scheduler lifecycle and non-streaming routing | #268 |
| `IModel.ForwardBatch` and `SequenceForwardRequest` | #222 |
| CPU ForwardBatch model refactoring | #276 |
| Batched decode | #348 |
| Batched prefill | #349 |
| Priority admission implementation | Draft PR #287 |

The prototype's `IBatchedModel` / `SequenceBatch` API is therefore not proposed
as a competing public contract. Later work should adapt to
`IModel.ForwardBatch` / `SequenceForwardRequest` unless maintainers decide
otherwise.

## Proposed first contribution

### CPU direct paged attention

This scope is useful independently of the scheduler and has the least overlap:

- expose borrowed physical KV block views and logical-to-physical mappings;
- read paged K/V directly from the CPU attention kernel;
- integrate the path into single-sequence transformer forward;
- retain the contiguous compatibility path;
- test MHA, GQA, MQA, fragmented pages, partial blocks, causal masks, decode,
  and multi-token queries;
- verify existing page reuse, rollback, and copy-on-write semantics;
- benchmark the direct path against staging gather with exact commit and
  environment metadata.

Not included:

- CUDA;
- scheduler changes;
- public ForwardBatch API changes;
- server or SSE integration;
- prefix-cache scheduler integration.

This work receives its own issue, branch, and PR after the scope is checked in
#256.

## Follow-up boundaries

Suggested order:

1. CPU direct paged attention.
2. CUDA direct paged attention.
3. CPU packed forward adapted to #222/#276.
4. Scheduler batched decode and prefill through #348/#349.
5. Per-request scheduler streaming, cancellation, backpressure, stop handling.
6. Server SSE integration and lifecycle management.

Each item must have independent acceptance criteria, tests, documentation, and
rollback boundaries. A follow-up is omitted when existing upstream work already
covers it.

## Branch policy

- Every upstream change starts with an issue.
- Every issue uses a branch from the latest upstream `main`:
  `issue/{number}-{short-description}`.
- Commit subjects end with `(#issue)`.
- PR descriptions use `Closes #issue`.
- Dependent work may be developed locally as a stack, but the upstream PR is
  rebased onto `main` after its prerequisite merges.
- The reference prototype is never used directly as an upstream PR base.

## Validation policy

Each focused PR reports only evidence rerun for that exact commit:

- scalar/reference numerical comparison and tolerance;
- targeted regression tests plus relevant non-GPU test coverage;
- real-model output comparison where applicable;
- BenchmarkDotNet before/after results for performance changes;
- CPU/GPU, OS, .NET, model, quantization, context length, batch size, thread
  count, warmup, iteration count, and allocation measurements;
- negative or mixed performance results are disclosed.

Prototype-level results are supporting evidence, not a substitute for rerunning
the focused branch.

## Coordination and inactivity policy

1. Publish one coordination comment on #256 with the reference implementation
   and proposed CPU scope.
2. Wait seven days while preparing the focused branch and validation.
3. If there is no response, add one concise follow-up.
4. After another seven days without response, open the focused issue and Draft
   PR, explicitly state the overlap analysis, and keep the change independent
   of scheduler APIs.
5. Do not post repeated reminders or empty commits.

If the scope is rejected, preserve the tested implementation on the fork and
either adapt it to the requested contract or redirect effort to a non-overlapping
component. Public commits, tests, benchmarks, and review history remain the
contribution record even when upstream maintenance is paused.
