# dotLLM 推理系统串讲：八种架构切面

阅读入口：[关键代码逐段讲解](#11-关键代码逐段讲解) · [分层消融实验与归因边界](#12-分层消融实验与归因边界) · [完整消融设计](dotllm-ablation-study.md)。

## 0. 先给项目定性

dotLLM 是一个原生 .NET/C# LLM 推理引擎，而不是 Python 对 llama.cpp/vLLM
的包装。核心编排、模型加载、tokenizer、采样、约束解码、调度和 CPU kernel
使用 C#；NVIDIA GPU kernel 使用 CUDA C 编译为 PTX，由 C# 通过 CUDA Driver
API 和 cuBLAS 直接调用。

当前真实部署形态是：

```text
一个进程
  ASP.NET / CLI
  + Tokenizer / Chat Template
  + TextGenerator 或 ContinuousBatchScheduler
  + 一个 Model Runtime
  + 一个 CPU/CUDA KV Pool
  + Sampling / Constraints
```

它不是当前就具备 Frontend、Master、每卡 Worker、独立 KV Manager 的分布式
系统。多进程 Router、NCCL TP/PP、PD 分离、Tokenizer Worker 和 OpenTelemetry
exporter 都属于演进方向。

状态标记：

- **已实现**：源码中存在并被实际调用。
- **部分实现**：接口或单机版本存在，但工业化形态不完整。
- **设计预留**：有接口/文档，没有可用执行路径。
- **未实现**：当前代码没有该能力。

---

## 1. 按功能域切分

### 1.1 模型执行层 Model Runtime

**职责**

- 从 GGUF mmap 读取模型元数据和权重。
- 将元数据转换为 `ModelConfig`。
- 执行 embedding、N 层 Transformer、final norm、LM head。
- 根据 token 数选择 prefill GEMM 或 decode GEMV。
- 返回 CPU Float32 logits，交给采样层。

**当前实现**

- `IModel`：普通 forward 抽象。
- `IBatchedModel`：无 padding packed sequence forward。
- `TransformerModel`：CPU Float32 路径。
- `CudaTransformerModel`：CUDA FP16 activation 路径。
- `HybridTransformerModel`：前 N 层 GPU、后续层 CPU。
- `TransformerWeights` / `CudaWeights`：权重加载与布局。
- `TransformerForwardState` / `CudaForwardState`：复用 scratch。

**CPU 特点**

- `System.Numerics.Tensors.TensorPrimitives`。
- AVX2/AVX-512 intrinsics。
- Q8_0/Q4_K/Q5_K/Q6_K 等 GGUF 量化 GEMV/GEMM。
- R4 row-interleaved 权重重排。
- 专用 `ComputeThreadPool`、NUMA/CPU affinity。

**CUDA 特点**

- C# 直接 P/Invoke `libcuda` 和 `libcublas`。
- `.cu` 编译为 PTX，运行时由 driver JIT 到 SASS。
- Prefill：反量化到 FP16 scratch，再调用 cuBLAS HGEMM。
- 单 token decode：支持的量化格式直接执行 quantized GEMV。
- 一个 context、stream、cuBLAS handle，forward 末尾统一同步。

**边界**

- `ModelConfig` 能参数化 Llama/Mistral/Phi/Qwen 类标准结构。
- MLA、MoE、Gemma 特殊结构仍是 roadmap。
- CUDA FP32 residual stream、FlashAttention、融合量化 GEMM仍未完成。

### 1.2 注意力与 KV Cache

**为什么需要 KV**

自回归 decode 第 t 步只需要计算新 token 的 Q/K/V。历史 token 的 K/V 已缓存，
否则每一步都要重算整个前缀。

```text
KV bytes =
  2(K+V) × layers × kv_heads × head_dim × tokens × bytes_per_element
```

**实现层次**

1. `SimpleKvCache`
   - 每个请求预分配连续最大长度。
   - 地址简单，但短请求浪费大量空间。

2. `PagedKvCache`
   - 逻辑 token block 映射到共享物理 block。
   - 每个 sequence 有独立 `KvBlockTable`。
   - CPU pool 有 refcount、fork、copy-on-write。
   - 请求结束后 block 回到 free list。

3. `QuantizedKvCache`
   - 老 token 保存为 Q8_0/Q4_0。
   - 最近窗口保留 FP32/FP16。
   - 降低上下文内存，但与 continuous batching paged path 尚未组合。

4. `PrefixCache`
   - 保存一条请求的 live KV cache。
   - 多轮对话匹配最长公共 token prefix，只 prefill 新后缀。
   - 当前不是跨请求共享 prefix trie。

**Direct PagedAttention**

逻辑位置 `t`：

```text
block_id = block_table[t / block_size]
offset   = t % block_size
row      = block_id * block_size + offset
```

CPU：

- 从 Float32 物理页直接读 K/V。
- 按 attention head 并行。
- 使用固定 256-token score tile。
- online softmax，不构造完整 score matrix。

CUDA：

- 共享 Float16 物理页池。
- sequence page table 上传到 device。
- scatter kernel 把新 K/V 写入物理页。
- paged attention kernel 直接解引用 page table。
- Q 放入 shared memory，FP32 累积。

**当前边界**

- CPU/CUDA direct pages 已实现。
- CUDA paged cache 没有 prefix fork/CoW。
- Continuous batching 暂不支持 quantized KV。
- 固定页池会提前占用内存；默认大小必须谨慎。

### 1.3 批调度与请求编排 Scheduler

`ContinuousBatchScheduler` 是当前改造的核心。

**并发模型**

- ASP.NET 可以并发接收请求。
- 只有一个 scheduler loop 可以进入 model forward。
- 模型 scratch、CUDA stream、页表变更不需要大范围加锁。
- 每个请求持有独立 cache、sampler、constraint clone、token history 和 channel。

**每轮迭代**

1. 收集新请求。
2. 清理取消请求并释放 KV 页。
3. 按 priority 和最坏情况页容量准入。
4. 每个 decoding sequence 先放入一个 token。
5. 剩余 token budget 分给轮转式 chunked prefill。
6. 一次 `ForwardBatch`。
7. 每个 sequence 独立 sample、stop check、stream。
8. 完成请求释放页；下一轮准入新请求。

**Packed batch**

```text
tokens   = [A_decode, B_decode, C_prefill..., D_prefill...]
segments = [
  (offset=0, count=1, cache=A),
  (offset=1, count=1, cache=B),
  (offset=2, count=n, cache=C),
  ...
]
```

QKV、FFN、LM head 对 packed rows 共享计算；attention 根据 segment 使用自己的
cache 与 causal range，所以不会跨请求注意。

**资源策略**

- `MaxSequences`：同时驻留的 sequence 数。
- `MaxBatchedTokens`：单 forward token budget。
- `PrefillChunkSize`：一个请求每轮最多 prefill 数。
- `MaxQueuedRequests`：排队上限。
- `StreamBufferCapacity`：单请求流式缓冲。
- 准入按 `ceil(max_context_for_request / block_size)` 保守预留页。

**已实现**

- 动态加入、decode-first、chunked prefill。
- 排队 priority。
- bounded queue/channel。
- cancellation、EOS、stop string、context limit。
- 慢消费者只失败自身。
- 请求完成/取消/异常后的 KV 回收。

**未实现**

- active preemption。
- swap/recompute。
- fairness aging。
- distributed scheduler。
- PD disaggregation。
- CUDA graphs。

### 1.4 并行与分布式 Parallelism

必须区分“单进程内并行”和“跨 GPU/跨进程并行”。

**已实现**

- CPU data parallel：一个算子按 row/head 分给固定线程池。
- CPU NUMA 与 affinity。
- GPU kernel thread/block 并行。
- CPU/GPU hybrid layer offload。
- 多请求 packed batch。

**设计预留**

- `ITensor.DeviceId`。
- `IBackend.AllReduce/Send/Receive`。
- 多设备文档中的 `ParallelismConfig`。

**未实现**

- NCCL。
- Tensor Parallel。
- Pipeline Parallel。
- Expert Parallel。
- 每 GPU 一个独立 Executor 进程。
- 多机通信。

目标 TP 模式下，每层一般需要：

```text
QKV/Attention local
→ O projection partial
→ AllReduce
→ Gate/Up local
→ Down projection partial
→ AllReduce
```

当前 `CudaBackend.AllReduce/Send/Receive` 会直接抛
`NotSupportedException`，因此不能把 MULTI_GPU.md 当成现状。

### 1.5 解码策略与加速 Decoding

**普通采样流水线**

```text
raw logits
→ constraint mask
→ repetition processor
→ temperature
→ top-k
→ top-p
→ min-p
→ greedy / categorical sample
→ stop check
```

组件：

- `SamplerPipeline`
- `ISamplerStep`
- `ILogitProcessor`
- `IStopCondition`
- `IDecodingConstraint`
- JSON/Schema/Regex/Grammar constraints
- incremental detokenizer
- logprobs capture

**Speculative decoding**

- 小 draft model 自回归生成 K 个候选。
- target model 一次验证候选。
- 当前只在 effectively-greedy 模式启用。
- 拒绝后 rollback target/draft KV 和 constraint state。
- 与 continuous batching 暂不组合。

**尚未实现或不完整**

- 非 greedy 的 pipeline-aware rejection sampling。
- beam search 的可用执行路径。
- guided decoding 中途注入。
- fused quantized batched GEMM。

### 1.6 服务与接入层 Serving

**当前实现**

- ASP.NET Minimal API。
- `/v1/chat/completions`。
- `/v1/completions`。
- `/v1/models`、`/v1/tokenize`、`/v1/detokenize`。
- `/health`、`/ready`。
- SSE streaming。
- chat template、tool-call parsing。
- Web UI。
- 运行时模型热切换。
- legacy 单请求模式或 opt-in continuous batching。

**请求边界**

- `ServerState.ExecuteAsync` 持有完整请求 lease。
- legacy 模式通过 gate 串行化。
- batching 模式允许 HTTP 请求重叠，但 model 仍由 scheduler 单独拥有。
- model swap 阻止新请求、等待旧 lease、再按 scheduler → pool → model 顺序释放。

**安全定位**

当前 server 是本地开发工具：

- 无内建认证。
- 无 TLS。
- permissive CORS。
- `/v1/models/load` 和 `/v1/config` 属于高权限 endpoint。
- 默认 localhost；暴露网络时必须放在反向代理后。

### 1.7 可观测性与运维 Observability

**已实现**

- `InferenceTimings`：prefill/decode/sampling 时间与 token 数。
- 响应 usage、finish reason、logprobs。
- scheduler active/queue depth 的最小快照。
- `/health`、`/ready`。
- BenchmarkDotNet 与自定义 workload benchmark。

**只有接口**

- `IInferenceMetrics`。
- `IRequestTracer`。

**未实现**

- `DotLLM.Telemetry` 当前还是 placeholder。
- Prometheus `/metrics`。
- OpenTelemetry Meter/Activity 实际埋点。
- Grafana dashboard。
- 每请求分布式 trace。
- GPU allocator、KV fragmentation、batch occupancy 的完整指标。

生产环境至少应补：

- TTFT、ITL、E2E latency。
- queue wait、prefill、decode 分段。
- batch sequence/token 数。
- KV block used/free、allocation failure。
- cancellation、timeout、OOM、CUDA error。
- model/version/device/quant tags。

### 1.8 安全与合规

**当前已有**

- 基本 request validation。
- model inspect 路径限制。
- .NET 类型安全边界。
- unmanaged/CUDA 资源确定性 Dispose。
- GPL-3.0-only 许可证。
- Hugging Face token 只用于下载请求。

**当前缺失**

- API authentication/authorization。
- TLS。
- secret management。
- endpoint RBAC。
- model artifact 签名和来源校验。
- quota/rate limit。
- audit log。
- SBOM、依赖漏洞扫描、供应链策略。
- 多租户 KV 数据擦除/隔离保证。
- prompt/output 脱敏和数据保留策略。

生产部署应把 `/models/load`、`/config`、`/cache/clear` 放在管理面，不应与推理
数据面共享匿名入口。

---

## 2. 按运行时角色切分

### 当前角色映射

| 工业角色 | dotLLM 当前对应 | 状态 |
|---|---|---|
| Frontend / Router | `DotLLM.Server` ASP.NET endpoints | 同进程、无独立 Router |
| Scheduler / Master | `ContinuousBatchScheduler` + `ServerState` | 单机有状态 owner |
| Worker / Executor | `TransformerModel` / `CudaTransformerModel` | 同进程，一个 model instance |
| KV Manager | `PagedKvCacheFactory` / `CudaPagedKvCacheFactory` | 同进程内存对象 |
| Tokenizer Worker | `DotLLM.Tokenizers` | 同进程同步执行 |

### 目标多进程形态

```text
Frontend/Router
  无状态、认证、协议转换、session affinity
      |
Scheduler/Master
  全局队列、placement、KV 元数据、fairness
      |
GPU Executor 0..N
  每卡进程、model shard、CUDA stream、local KV pages
      |
KV Manager / Transfer
  prefix registry、RDMA/NCCL transfer、PD handoff

Tokenizer Pool
  独立 CPU 扩缩容，encode/decode/chat template
```

要演进到该形态，需要把当前 C# 对象引用协议化：

- `SequenceBatch.Cache` 不能再是进程内对象，应变成 `sequence_id + block_table handle`。
- `ITensor.DataPointer` 不能跨 RPC，应变成 device-local handle。
- scheduler/model 调用应变成 batch command。
- KV 生命周期需要租约、版本、ack 和故障恢复。
- tokenizer 输出需要 token IDs、offsets、prompt hash 等稳定协议。

---

## 3. 按抽象层次切分

用户给出的 Python/C++/CUDA/IR 是典型 vLLM/容器化引擎分层；dotLLM 的实际语言
分层不同：

| 抽象层 | 典型工业实现 | dotLLM 当前实现 |
|---|---|---|
| 业务编排 | Python/Rust | C#：Server、Engine、Scheduler |
| 性能敏感 runtime | C++ | C# unsafe + Span + NativeMemory + intrinsics |
| GPU kernel | CUDA/Triton | CUDA C → PTX |
| Vendor library | cuBLAS/NCCL | cuBLAS 已用；NCCL 未实现 |
| IR/图编译 | TensorRT/MLIR/Inductor | 未实现；依赖 .NET JIT 与 driver PTX JIT |

dotLLM 的独特点是把传统 C++ runtime 的大量职责放进 C#：

- unmanaged tensor metadata。
- mmap GGUF。
- pointer arithmetic。
- block allocator。
- scheduler state machine。
- P/Invoke CUDA Driver API。

因此它不是“Python control plane + C++ engine”，而是：

```text
C# control plane + C# native-style runtime + CUDA PTX kernels
```

---

## 4. 按请求生命周期切分

### 阶段 A：接入与规范化

1. ASP.NET 反序列化 OpenAI-compatible DTO。
2. 校验 message、prompt、max_tokens 和 context length。
3. 应用 chat template 和 tools。
4. tokenizer 编码为 `int[]`。
5. 构造 `InferenceOptions` / `InferenceRequest`。

### 阶段 B：排队与准入

1. legacy：等待 `SemaphoreSlim` 独占模型。
2. batching：进入 bounded incoming queue。
3. scheduler 清理 cancellation。
4. 按 priority、sequence 上限、逻辑页 reservation 准入。
5. 创建 request-owned KV cache。

### 阶段 C：Prefill

1. prompt 按 chunk 放进 packed token buffer。
2. embedding。
3. 每层 norm → QKV → RoPE。
4. K/V 写入请求自己的 cache。
5. attention 只读取该请求历史页。
6. O projection、FFN、residual。
7. 只取 segment 最后一行 logits。

### 阶段 D：Decode

1. 上一轮 sampled token 作为下一轮输入。
2. 所有 decoding sequence 各贡献一个 row。
3. 共享 projection 计算。
4. 更新每个 cache 的下一个 position。
5. 返回每个 sequence 的 logits row。

### 阶段 E：Sampling 与流输出

1. constraint mask。
2. penalties 和 sampling transforms。
3. sample token。
4. 更新 constraint、token history 和 incremental detokenizer。
5. 检查 EOS/max/context/stop。
6. 写入 bounded `Channel<GenerationToken>`。
7. ASP.NET 转为 SSE。

### 阶段 F：完成与回收

1. 完成 response/task/channel。
2. dispose request cache，物理页回 free list。
3. 释放 cancellation registration。
4. 下一轮利用空出的 sequence/page capacity 准入请求。

### 故障定位

| 症状 | 优先检查阶段 |
|---|---|
| TTFT 高 | tokenize、queue wait、prefill、warm-up |
| ITL 高 | decode GEMV、batch size、attention、sampling |
| OOM | weight、dequant scratch、KV pool、logits scratch |
| 输出串请求 | segment offset、page table、cache ownership |
| SSE 卡死 | bounded channel、客户端取消、request lease |
| 模型切换 hang | inflight drain、scheduler stop、dispose 顺序 |
| token 不一致 | RoPE position、cache length、sliding window、precision |

---

## 5. 横切关注点

### Config

当前：

- `ModelConfig`：模型结构。
- `InferenceOptions`：采样与停止。
- `ThreadingConfig`：CPU 并行。
- `KvCacheConfig`：KV 量化。
- `ContinuousBatchingOptions`：调度上限。
- `ServerOptions`：部署入口。

不是 Pydantic/YAML，而是强类型 C# record + CLI parsing。未来需要分层配置、环境变量、
配置校验、热更新边界和配置快照。

### Logging & Metrics

当前主要是 console、响应 timings、health/readiness。Metrics/Trace 接口有了，实际
OpenTelemetry 还没有落地。

### Error & Retry

当前：

- cancellation token。
- request-level sampling error 隔离。
- forward failure 终止 scheduler 并失败所有请求。
- CUDA API 返回码转 exception。
- model swap drain。

缺少统一 error code、retry classification、deadline budget、熔断和 worker restart。

### Testing

- Unit：kernel、scheduler、server lifecycle。
- Integration：真实 GGUF 与 TextGenerator。
- Correctness：scalar reference、serial/batched parity。
- GPU：可选 GPU trait/H20 harness。
- Performance：BenchmarkDotNet + arrival-trace benchmark。

### Benchmark

`benchmarks/DotLLM.Benchmarks` 与业务代码分离。必须同时看：

- throughput。
- TTFT。
- ITL。
- batch occupancy。
- forward 次数。
- KV 是否完全回收。

---

## 6. 按稳定性治理

### Stable Core 候选

- `IModel`
- `IBatchedModel`
- `IKvCache`
- `IScheduler`
- `ITensor`
- `ModelConfig`

这些接口一旦被 Server、CLI、CPU、CUDA 同时依赖，修改成本很高。应保持最小语义，
不要把某个 backend 的 page pointer 或 CUDA stream 塞入 Core。

### Pluggable Backend

- CPU/CUDA model。
- attention implementation。
- quantized kernels。
- tokenizer/chat template。
- constraint。
- KV implementation。

目前“可插拔”主要通过接口和具体类型分支实现，还没有完整 backend registry。

### Experimental

- Continuous batching 当前应 feature flag。
- Speculative decoding 仅 greedy。
- Native AOT。
- 未来 prefix sharing、preemption、TP/PP。

### Vendor-specific

当前 NVIDIA 代码集中在 `DotLLM.Cuda` 和 `native/kernels`。ROCm/TPU/NPU 都未实现。
未来应隔离：

```text
platforms/nvidia
platforms/amd
platforms/<npu-vendor>
```

Core 只保留 device、tensor、collective 等稳定语义。

---

## 7. 按硬件/后端切分

### Device Backend

| 后端 | 状态 |
|---|---|
| CPU | 已实现 |
| CUDA | 已实现，单 GPU |
| CPU+CUDA hybrid | 已实现，layer offload |
| ROCm | 未实现 |
| TPU/NPU | 未实现 |

### Attention Backend

| 后端 | 状态 |
|---|---|
| CPU contiguous tiled attention | 已实现 |
| CPU direct PagedAttention | 已实现 |
| CUDA contiguous tiled attention | 已实现 |
| CUDA direct PagedAttention | 已实现 |
| FlashAttention/FlashInfer/Triton | 未实现 |

### Quant Backend

dotLLM 面向 GGUF，而不是 AWQ/GPTQ Python 生态：

- weight：F32/F16/Q8_0/Q4_0/Q5_0/Q4_K/Q5_K/Q6_K 等。
- CPU：专用 quantized GEMV/GEMM。
- CUDA decode：部分格式专用 quantized GEMV。
- CUDA prefill/packed rows：临时反量化 + cuBLAS。
- KV：Q8_0/Q4_0 dual-region。

### Communication Backend

- 单进程内：方法调用、Span、Channel。
- CPU thread：专用 ComputeThreadPool。
- GPU：单 CUDA stream。
- NCCL/Gloo/RCCL/custom allreduce：未实现。

---

## 8. 按部署形态切分

### 8.1 单卡单进程

**当前主路径。**

- ASP.NET、scheduler、tokenizer、model 同进程。
- 一个 CUDA context/stream。
- 适合本地、开发、单节点服务。

### 8.2 单机 CPU

**已实现。**

- 多线程 SIMD。
- NUMA/affinity。
- continuous batching + CPU paged attention。

### 8.3 单机 CPU/GPU Hybrid

**已实现，但不支持 continuous batching。**

- 前 N 层 GPU，后续层 CPU。
- 一次 D2H hidden-state 边界传输。
- GPU/CPU KV 分层存放。

### 8.4 单机多卡 TP

**未实现。**

需要：

- 权重切分。
- device-local KV。
- 每层两次 all-reduce。
- NCCL communicator/stream。
- rank-aware model runtime。

### 8.5 多机 TP+PP+EP

**未实现。**

需要全局 topology、rank placement、collective groups、pipeline microbatch、故障恢复，
以及 MoE token dispatch。

### 8.6 PD 分离

**未实现。**

Prefill 和 Decode 的计算特征不同：

- Prefill：大 GEMM、compute-bound。
- Decode：小 batch、读权重/KV、memory-bandwidth-bound。

PD 分离后必须增加：

- KV transfer protocol。
- block table ownership transfer。
- target decode worker placement。
- prefix/session affinity。
- transfer completion ack。
- timeout 后 KV 清理。

### 8.7 Serverless / Spot

**未实现。**

挑战不只是拉起 worker：

- 模型权重冷启动。
- PTX JIT 和 warm-up。
- KV state 不能随 worker 消失。
- in-flight request 重试是否会重复输出 token。
- graceful drain 和 checkpoint。

---

## 9. 用一句话串起所有切面

从请求生命周期看，dotLLM 是：

```text
ASP.NET 接入
→ C# tokenize/template
→ 单 owner scheduler 动态组 batch
→ CPU/CUDA model runtime 执行 packed forward
→ request-local page table 访问 shared KV pages
→ sampling/constraint/detokenize
→ SSE/JSON
→ cache/page 确定性回收
```

从架构演进看，它正从：

```text
单进程本地推理引擎
```

走向：

```text
可并发的单节点 serving runtime
```

但还没有到：

```text
多进程、多 GPU、PD 分离、可观测、可鉴权的生产推理平台
```

这三层必须明确区分，才是准确的项目介绍。

## 10. 推荐阅读代码顺序

1. `DotLLM.Core/Models/IModel.cs`
2. `DotLLM.Models/Architectures/TransformerModel.cs`
3. `DotLLM.Cuda/CudaTransformerModel.cs`
4. `DotLLM.Engine/TextGenerator.cs`
5. `DotLLM.Engine/ContinuousBatchScheduler.cs`
6. `DotLLM.Core/Attention/IKvCache.cs`
7. `DotLLM.Engine/KvCache/PagedKvCache.cs`
8. `DotLLM.Cpu/Kernels/PagedAttention.cs`
9. `native/kernels/paged_attention.cu`
10. `DotLLM.Engine/ScheduledSequence.cs`
11. `DotLLM.Server/ServerState.cs`
12. `DotLLM.Server/Endpoints/ChatCompletionEndpoint.cs`

---

## 11. 关键代码逐段讲解

### 11.1 第一组：系统契约

#### `IModel`：最小模型执行接口

文件：`src/DotLLM.Core/Models/IModel.cs`

关键点：

```csharp
ITensor Forward(
    ReadOnlySpan<int> tokenIds,
    ReadOnlySpan<int> positions,
    int deviceId,
    IKvCache? kvCache);
```

这个接口把 Engine 和具体 CPU/CUDA 实现解耦。Engine 只需要提供：

- 本轮新增 token。
- token 的绝对 position。
- 该 sequence 的 KV cache。

返回值是 CPU Float32 logits。即使 forward 在 GPU 上执行，最终 sampling 仍在 CPU。

读代码时关注两个约定：

1. 普通 CPU forward 可能返回多行 logits。
2. GPU/cache forward 通常只返回最后 token 的一行 logits。

这个历史差异也是后面引入 `IBatchedModel` 的原因之一。

#### `IBatchedModel`：连续批处理的模型边界

文件：`src/DotLLM.Core/Models/IBatchedModel.cs:10-58`

```csharp
ITensor ForwardBatch(
    ReadOnlySpan<int> tokenIds,
    ReadOnlySpan<int> positions,
    ReadOnlySpan<SequenceBatch> sequences,
    int deviceId);
```

它没有使用传统 `[batch, max_seq_len]` padded tensor，而是把所有有效 token
直接拼成一维 packed rows。

`SequenceBatch` 描述其中一段：

```csharp
(TokenOffset, TokenCount, Cache)
```

最重要的不是 record 本身，而是 `SequenceBatch.Validate()`：

- segment 必须无缝覆盖整个 packed buffer。
- token 和 position 合法。
- position 必须从 `cache.CurrentLength` 连续追加。
- 同一个 cache 不能同时属于两个 segment。

这是防止跨请求污染的第一道边界。验证在任何 cache/model mutation 之前执行。

#### `IKvCache` / `IPagedKvCache`

文件：

- `src/DotLLM.Core/Attention/IKvCache.cs`
- `src/DotLLM.Core/Attention/IPagedKvCache.cs`

`IKvCache` 提供：

- `CurrentLength` / `MaxLength`
- `Update`
- 连续 K/V view
- `Rollback`

`IPagedKvCache` 额外暴露：

- `BlockSize`
- `KvStride`
- `BlockIds`
- 每层所有物理 key/value block 的 base pointer

这里的设计取舍是：Core 不知道 CPU allocator 的具体实现，但 direct paged
attention 必须得到物理页和页表，因此通过最小接口暴露 borrowed views。

### 11.2 第二组：Scheduler 状态机

#### `ContinuousBatchScheduler.Submit`

文件：`src/DotLLM.Engine/ContinuousBatchScheduler.cs:91-126`

职责：

1. 校验 prompt/token/max_tokens。
2. 计算该请求最坏情况 cache capacity。
3. 换算需要预留的 block 数。
4. clone token 数组和 stop sequence，隔离调用方后续修改。
5. 流式请求创建 bounded channel。
6. 放入 `_incoming`，通过 `_wake` 唤醒 scheduler。

容量公式：

```text
capacity =
  min(model_context,
      prompt_tokens + max(0, max_tokens - 1))
```

减 1 的原因：prefill logits 已经产生第一个输出 token；最后一个 sampled token
不需要再写入 KV 才能返回。

#### `RunLoopAsync`

文件：`ContinuousBatchScheduler.cs:131-175`

主循环：

```text
DrainIncoming
→ RemoveCanceled
→ Admit
→ RunIteration
```

这里采用 single-owner：

- 外部线程只写 `_incoming` 和 cancellation。
- `_waiting`、`_active`、页 reservation、sequence 状态只由 loop 修改。
- model forward 也只有这个 loop 调用。

这样避免给 model scratch、CUDA stream 和 sequence cache 加细粒度锁。

#### `Admit`

文件：`ContinuousBatchScheduler.cs:201-223`

关注三点：

1. priority 只影响 waiting admission，不抢占 active sequence。
2. `_reserved + ReservedBlocks <= _totalBlocks` 才允许准入。
3. cache 在准入成功后才创建。

保守 reservation 的优点是 active 请求后续增长时不会 OOM；缺点是用户声明很大
`max_tokens` 却提前停止时，会降低实际并发度。

#### `RunIteration`

文件：`ContinuousBatchScheduler.cs:225-279`

这是最核心的函数。

第一阶段放 decode：

```csharp
foreach (active sequence)
    if (prompt already prefilled)
        append one generated token row;
```

第二阶段填 prefill：

```csharp
remaining = prompt.Length - Prefilled;
take = min(remaining, PrefillChunkSize, remainingBatchBudget);
```

`_prefillCursor` 每轮旋转，避免长 prompt 永远先拿走剩余 budget。

随后构造：

```csharp
ForwardBatch(tokens[..used], positions[..used], segments[..count])
```

返回 logits 是 `[segmentCount, vocab]`，不是 `[tokenCount, vocab]`。每个 segment
只需要最后 token 的 logits。

最后逐 sequence：

- prefill chunk 未完成：只推进 `Prefilled`。
- prompt 刚完成：用该段最后一行 logits 采第一个 token。
- decode：统计 decode step 并采下一个 token。
- 完成后 `Finish()` 回收 reservation 与 cache。

#### `ScheduledSequence`

文件：`src/DotLLM.Engine/ScheduledSequence.cs`

它是请求级状态容器：

```text
Request / Cancellation / Cache / Stream
Generated / Prefilled / DecodeSteps
Sampler / Constraint / Detokenizer
PendingText / Output / Logprobs
Finished
```

`Initialize()` 创建 request-owned sampler，并 clone decoding constraint。

`Sample()` 的关键顺序：

```text
constraint mask
→ log-softmax snapshot（可选）
→ sampler
→ constraint.Advance
→ EOS/custom stop
→ incremental detokenize
→ stop-string withholding
→ max tokens/context
→ channel.TryWrite
```

`Complete()` 负责：

- 先构造最终 response。
- dispose cache。
- dispose cancellation registration。
- complete stream 或 response TCS。
- 最后完成 `Released`，让流枚举 dispose 能等待页真正回收。

### 11.3 第三组：CPU 模型执行

#### `TransformerModel.ForwardBatch`

文件：`src/DotLLM.Models/Architectures/TransformerModel.cs:142-147`

它只做两件事：

1. `SequenceBatch.Validate`
2. 进入共享 `ForwardCore`

真正的差异通过 `sequences.IsEmpty` 判断普通 forward 还是 packed forward。

#### `TransformerModel.ForwardCore`

文件：`TransformerModel.cs:149-434`

每层主链：

```text
Embedding
→ copy residual
→ RMSNorm
→ Q/K/V projection
→ bias / QK norm
→ RoPE
→ request-isolated attention
→ O projection
→ residual
→ FFN norm
→ Gate/Up
→ SwiGLU
→ Down
→ residual
```

重要性能分支：

- `seqLen == 1`：decode，走 fused RMSNorm+quantize、fused QKV/GateUp dispatch。
- `seqLen > 1`：prefill/packed batch，走 GEMM。

注意这里 `seqLen` 指本轮 packed token rows，而不是单一请求长度。因此多个
decode 请求组成 batch 后，也会进入多 row GEMM 路径。

#### `Attend`

文件：`TransformerModel.cs:436-462`

这是 attention backend dispatch：

```text
cache.Update(new K/V)

if IPagedKvCache
    PagedAttention.Execute
else if IQuantizedKvCache
    quantized tiled attention
else
    contiguous attention
```

它把“Transformer block 怎么组织”与“KV 具体如何存储/读取”分开。

#### Last-row LM head

文件：`TransformerModel.cs:409-431`

普通 prefill/speculative verification 可能需要所有 token 的 logits。

Continuous batching 中，每个 segment 只需要最后 row。代码先把每个 segment 的
最后 hidden row compact 到 `normOut`，再做一次共享 LM head GEMM：

```text
[packed token hidden]
→ select last row per segment
→ [sequence_count, hidden]
→ LM head
→ [sequence_count, vocab]
```

否则会生成 `[packed_tokens, vocab]`，在大词表下浪费大量内存和计算。

### 11.4 第四组：CPU 页表和 PagedAttention

#### `KvBlockPool`

文件：`src/DotLLM.Engine/KvCache/KvBlockPool.cs`

它是物理页框分配器：

- 每层有连续 key buffer 和 value buffer。
- `_freeStack` 管理 free block ID。
- `_refCounts` 支持 prefix sharing/COW。

`Allocate()`：

- 从 free stack pop。
- refcount 设为 1。

`Release()`：

- refcount 减 1。
- 降到 0 时 push 回 free stack。

Pool 分配的是 block ID，不为每个请求重新申请大块 unmanaged memory。

#### `KvBlockTable`

文件：`src/DotLLM.Engine/KvCache/KvBlockTable.cs`

这是 sequence 的逻辑页表。

`EnsureCapacity()` 为逻辑长度补足 block ID。

`Resolve(position)`：

```csharp
logicalBlock = position / blockSize;
offset = position % blockSize;
return (blockIds[logicalBlock], offset);
```

`Fork()` 增加 refcount，共享 prefix。

`EnsureWritable()` 检查 refcount，大于 1 时 `CopyBlock`，实现 copy-on-write。

#### `PagedKvCache.Update`

文件：`src/DotLLM.Engine/KvCache/PagedKvCache.cs:101-163`

流程：

1. 校验 K/V shape 和 dtype。
2. 根据最大 position 预分配 block。
3. 对每个 logical position 调 `EnsureWritable`。
4. `Resolve` 到物理 block + offset。
5. 同一 block 内连续 token 合并为一次 `Buffer.MemoryCopy`。
6. 推进 visible `CurrentLength`。

`GetKeysRef`/`GetValuesRef` 仍保留 legacy gather，但 direct paged path 使用
`GetKeyBlocksRef`/`BlockIds`，不会申请 staging。

#### `PagedAttention.Execute`

文件：`src/DotLLM.Cpu/Kernels/PagedAttention.cs`

入口先校验：

- heads 与 KV stride 匹配。
- query position 已写入 cache。
- block table 完整。
- block ID 没有越界。

`Worker()` 按 query head 分给线程。GQA 映射：

```csharp
kvHead = queryHead / groupSize;
```

每个 query 的 online softmax：

```text
scores(tile)
tile_max
new_max = max(old_max, tile_max)
old_accumulator *= exp(old_max - new_max)
tile_weights = exp(scores - new_max)
sum += tile_weights
output += tile_weights × V
```

最终除以 `sum`。临时空间只需要固定 256 个 score。

### 11.5 第五组：CUDA 执行与分页 kernel

#### `CudaTransformerModel.ForwardBatch`

文件：`src/DotLLM.Cuda/CudaTransformerModel.cs:147-172`

入口校验所有 cache：

- 必须是 `CudaPagedKvCache`。
- 必须属于同一个 CUDA context。
- layers、KV stride 必须与 model 匹配。

然后为每个 segment 调 `cache.Prepare(newLength)`：

- 分配所需物理页。
- 更新 host block table。
- 将 block IDs 上传到 device page table。

异常时必须 `_stream.Synchronize()`，避免 scheduler 提前回收仍被 GPU kernel 使用的页。

#### `CudaTransformerModel.ForwardCore`

文件：`CudaTransformerModel.cs:174-374`

主要路径：

1. H2D token IDs / positions。
2. embedding kernel。
3. Q/K/V projection。
4. bias、QK norm、RoPE。
5. 按 segment scatter K/V。
6. 按 segment launch PagedAttention。
7. O/FFN projection。
8. compact 每个 segment 最后一行 hidden。
9. LM head。
10. stream sync。
11. logits D2H。

注意：projection/FFN 是 packed batch；当前 CUDA attention 仍按
`sequence × layer` 分别 launch，并没有把多个 sequence 的 attention 融成一个 kernel。

#### `Project`

文件：`CudaTransformerModel.cs:381-413`

dispatch 规则：

```text
rows > 1
  → quantized weight dequant to FP16 scratch
  → cuBLAS HGEMM

rows == 1 and specialized quantized GEMV exists
  → direct quantized GEMV

otherwise
  → dequant + cuBLAS GEMV
```

这是当前量化 batch 性能的关键拐点：batch 大于 1 会切换到完整权重反量化路径。

#### `CudaPagedKvCacheFactory`

文件：`src/DotLLM.Cuda/CudaPagedKvCacheFactory.cs`

它一次性为每层分配：

```text
[totalBlocks, blockSize, kvStride] key
[totalBlocks, blockSize, kvStride] value
```

并使用单线程 `Stack<int>` 管理 free IDs。这里依赖 scheduler single owner，不是
通用线程安全 allocator。

#### `CudaPagedKvCache.Prepare`

文件：`src/DotLLM.Cuda/CudaPagedKvCache.cs:38-55`

核心步骤：

- 计算 `requiredBlocks`。
- 从 factory rent block IDs。
- block count 变化时把 host table H2D。
- 更新 `CurrentLength`。

`Rollback()` 会释放尾部整页；最后一个部分页保留。

#### `LaunchScatterPagedKv` / `LaunchPagedAttention`

文件：`src/DotLLM.Cuda/CudaKernels.cs:510-535`

C# 侧负责：

- 准备 CUDA kernel 参数指针数组。
- 计算 grid/block。
- 计算 dynamic shared memory：

```text
Q[headDim]
+ scores[256]
+ output accumulator[headDim]
+ warp scratch[32]
```

#### `paged_attention_f16`

文件：`native/kernels/paged_attention.cu:20-203`

一个 block 负责一个 `(query token, query head)`。

关键过程：

1. query head 映射到 KV head。
2. Q 从 FP16 加载到 shared FP32。
3. 根据 causal/sliding window 确定 KV 范围。
4. 通过 device page table 解引用 K。
5. warp shuffle 求 tile max/sum。
6. online softmax 重标定旧 accumulator。
7. 通过同一页表读取 V 并累积。
8. normalize，转回 FP16。

物理地址：

```cuda
page = page_table[token / page_size];
row = page * page_size + token % page_size;
```

`scatter_paged_kv_f16` 位于同文件 206-219 行，只写本轮新增 K/V，不 gather 历史。

### 11.6 第六组：解码与文本语义

#### `SamplerPipeline.Sample`

文件：`src/DotLLM.Engine/Samplers/SamplerPipeline.cs:141-157`

顺序：

1. 所有 `ILogitProcessor`。
2. greedy 直接 argmax。
3. temperature/top-k/top-p/min-p 等 sampler steps。
4. categorical sample。

每个 `ScheduledSequence` 创建自己的 pipeline，因此 seeded `Random` 不会在请求间共享。

#### `IncrementalDetokenizer`

文件：`src/DotLLM.Engine/IncrementalDetokenizer.cs`

为什么不能每步 `Decode(allGeneratedTokens)`：

- 第 1 步解 1 个 token。
- 第 N 步解 N 个 token。
- 总复杂度 O(N²)。

它维护一个小 token window，只在安全时把稳定前缀 commit 到 StringBuilder。
`TakeDelta()` 只返回上次以后新增的完整文本，并处理不完整 UTF-8/高代理项。

#### `TextGenerator`

文件：`src/DotLLM.Engine/TextGenerator.cs`

它是 legacy 单请求编排器，负责：

- prompt tokenize。
- prefix cache lookup。
- prefill。
- decode loop。
- speculative decoding。
- response/timing。

读它的重点不是逐行，而是比较它与 Scheduler：

- `TextGenerator`：一个请求独占完整 generate loop。
- `ContinuousBatchScheduler`：把多个请求的 loop 拆成每轮一步并交错执行。

#### `SpeculativeDecoder.DraftAndVerify`

文件：`src/DotLLM.Engine/SpeculativeDecoder.cs:67+`

流程：

1. draft model 生成 K 个候选。
2. target model 一次 forward 验证。
3. greedy 模式逐位置比较。
4. 接受匹配前缀。
5. target/draft cache rollback。
6. constraint clone/restore。

当前仅 effectively-greedy，且 continuous batching 明确拒绝 speculative model。

### 11.7 第七组：服务生命周期

#### `ServerStartup.LoadModel`

文件：`src/DotLLM.Server/ServerStartup.cs:66-263`

它是 composition root：

1. 解析 GGUF、ModelConfig、Tokenizer。
2. 根据 `gpuLayers` 选择 CPU/CUDA/Hybrid model。
3. 创建 chat template/tool parser。
4. 创建 KV factory。
5. 创建 prefix cache、draft model。
6. 创建 `TextGenerator`。
7. warm-up。
8. 可选创建 ContinuousBatchScheduler 并启动 loop。
9. 返回完整 `ServerState`。

最值得关注的是 catch 中的逆序 cleanup；模型加载任何阶段失败都不能泄漏 mmap、
thread pool、CUDA context 或 KV pool。

#### `ServerState.ExecuteAsync`

文件：`src/DotLLM.Server/ServerState.cs:93-110`

legacy 模式：

- 持有 `_requestGate` 到请求完成，序列化 model。

batching 模式：

- 增加 `_inflight` lease 后立即释放 gate。
- 多个 HTTP 请求可以同时等待 scheduler/读取 SSE。
- model compute 仍由 scheduler 串行拥有。

#### `SwapModelAsync`

文件：`ServerState.cs:139-169`

模型热切换顺序：

```text
阻止新请求
→ DrainAsync 等待所有 request lease
→ stop scheduler
→ dispose prefix/paged pool
→ dispose draft/main model
→ dispose GGUF
→ load new state
```

这里的顺序是 correctness 逻辑，不只是资源清理习惯。先销毁 model 或页池会导致
仍在流式输出的请求访问已释放内存。

#### Chat/Completion endpoints

文件：

- `Endpoints/ChatCompletionEndpoint.cs`
- `Endpoints/CompletionEndpoint.cs`

endpoint 只处理协议：

- validate。
- DTO → prompt/options。
- 调 `ServerState.Generate*`。
- response/SSE 序列化。

它们不应直接进入 model，也不应持有具体 KV 类型。

### 11.8 第八组：单机并行

#### `ComputeThreadPool.Dispatch`

文件：`src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:154-185`

使用 N-1 个固定 worker + caller thread：

- context 是 stack pointer。
- work 是 `delegate*` 函数指针。
- 无 Task/closure 分配。
- generation counter 唤醒 spin worker。
- event 作为 prefill 或 spin 超时后的 fallback。

`SetDispatchMode`：

- decode：SpinWait，并限制线程到内存通道估算。
- prefill：EventBased，使用所有线程。

`WorkerLoop` 的 `lastGeneration = 0` 修复了 worker 启动慢于首次 dispatch 时漏任务的竞态。

#### CUDA stream

`CudaTransformerModel` 使用一个 stream 和一个 cuBLAS handle。所有 kernel/GEMM
按 stream 顺序提交，forward 末尾一次 sync。

这也是 single-owner model 的原因：如果允许多个线程同时 forward，就需要独立
stream/state/handle 或复杂的资源池。

#### Hybrid runtime

文件：`src/DotLLM.Cuda/HybridTransformerModel.cs`

- embedding 和前 N 层在 GPU。
- hidden state D2H。
- 后续层、final norm、LM head 在 CPU。
- GPU/CPU KV cache 按 layer 分拆。

它解决模型放不下单卡，不等同于 multi-GPU parallelism。

### 11.9 第九组：测试与性能证据

#### Scheduler 单测

`tests/DotLLM.Tests.Unit/Engine/ContinuousBatchSchedulerTests.cs`

重点用例：

- packed prefill + decode 混批。
- late arrival 动态加入。
- oversubscription reservation。
- queued/active cancellation 回收页。
- forward failure 全局失败。
- stop string 跨 token。
- slow consumer 只失败自身。
- priority、seed、constraint 隔离。
- stream dispose 等待 in-flight forward。

#### CPU PagedAttention correctness

`tests/DotLLM.Tests.Unit/Cpu/Kernels/PagedAttentionTests.cs`

- 随机 K/V。
- 故意用 blocker 打散物理页。
- 与 scalar contiguous attention 对比。
- 覆盖 MHA/GQA/MQA、跨 256 tile、sliding window 和多线程。

#### CUDA correctness

`tests/DotLLM.Tests.Unit/Cuda/CudaPagedAttentionTests.cs`

- CUDA 输出与 double scalar reference 比较。
- 非连续页。
- rollback/recycle。
- FP16 误差阈值。

#### 真实模型 parity

- `ContinuousBatchingIntegrationTests`
- `CudaContinuousBatchingTests`

用真实 SmolLM GGUF 比较 serial greedy 与 concurrent greedy token IDs，而不只是比较
单个 kernel。

#### Benchmark

- `ContinuousBatchingBenchmark`：max_sequences=1 与 N 的到达轨迹对比。
- `PagedAttentionBenchmarks`：legacy gather+attention 与 direct pages。

两者都不能单独给出模块归因：前者固定分页路径、同时改变活跃请求数及执行形态，而且不经过 HTTP；后者包含两套 attention 数学实现的差异。1.73× 是已测试配置的组合收益，不能称为 Direct Pages 的独立贡献。严格消融设计见第 12 章。

需要同时报告：

- token/s。
- TTFT p50/p95。
- ITL p50/p95。
- forward 次数。
- mean batch sequences。
- KV pages reclaimed。

只报告吞吐会掩盖 batch 增大导致的单请求 ITL 上升。

### 11.10 一条最短代码调用链

```text
ChatCompletionEndpoint.HandleRequestAsync
→ ServerState.ExecuteAsync
→ ServerState.GenerateStreamingAsync
→ ContinuousBatchScheduler.GenerateStreamingAsync
→ Submit
→ RunLoopAsync
→ Admit
→ RunIteration
→ IBatchedModel.ForwardBatch
→ TransformerModel/CudaTransformerModel.ForwardCore
→ cache.Update / scatter_paged_kv
→ CPU/CUDA PagedAttention
→ logits
→ ScheduledSequence.Sample
→ IncrementalDetokenizer.TakeDelta
→ Channel<GenerationToken>
→ SSE
→ ScheduledSequence.Complete
→ cache.Dispose
→ physical pages returned
```

掌握这条链后，再分别深入 CPU quant kernel、CUDA PTX、speculative decoding 和
Server 热切换，会比按目录逐文件阅读高效得多。

---

## 12. 分层消融实验与归因边界

**本章状态：新增实验设计，尚未执行 A0–A4 或 G0–G2；没有新增性能数据。**

完整方法、测试矩阵、统计口径与代码缺口见：[dotLLM 分层消融实验设计](dotllm-ablation-study.md)。

### 12.1 已有 1.73× 的含义

此前 NVIDIA H20、SmolLM-135M Q8_0、32 请求 × 64 token 的三轮记录，对比的是同一 scheduler 与 direct paged 路径的 `max_sequences=1` 和 `4`。吞吐中位数从 347.70 到 600.75 token/s，ITL p50 从 2.87 到 6.70 ms。

它能说明允许多个活跃 sequence 后的组合收益，不能拆分 Scheduler、Packed Forward、分页布局和 Direct Pages 的贡献，也不是 upstream 原始系统与完整实现的对照。计时发生在 scheduler/Channel 层，不能称为真实 HTTP/SSE 端到端压测；每轮约 3.4–5.9 秒，后续正式实验应延长并增加独立重复。[原始记录](../evidence/dotllm-gpu-benchmark.jsonl)

### 12.2 阶梯归因

| 版本 | 调度 | 模型计算 | KV 存储与读取 | 相邻比较 |
|---|---|---|---|---|
| A0 Serial | 整请求串行 | 独立 forward | 连续 KV + 同源 contiguous attention | 公平基线 |
| A1 Scheduler | 连续调度 | 每 sequence fanout | 与 A0 相同 | 调度策略组 |
| A2 Packed | 与 A1 相同 | packed QKV/FFN/LM head | 与 A1 相同 | packed execution |
| A3 Paged+Gather | 与 A2 相同 | 与 A2 相同 | 共享页池 + gather + 同源 contiguous attention | 分页管理及 gather 开销 |
| A4 Paged+Direct | 与 A3 相同 | 与 A3 相同 | 同一页池 + 直接页表读取 | 去掉 gather 的条件收益 |

“调度”包含动态入场、decode-first 和 chunked prefill，若要单独评价每个策略，应固定模型和 KV 路径，再一次只改变一个策略。分页也必须分成存储布局与地址访问两个因素，A3/A2 不等于纯 allocator 开销。

保留独立 A0-original 记录指定 upstream commit 的性能。主阶梯 A0 使用统一 attention 数学，二者差异另表，不把修复窗口语义后的版本称为原始版本。

### 12.3 同源数学与测量分层

A3/A4 使用同一 kernel body，只有地址解析不同：连续路径 `row=t`，分页路径 `row=page_table[t/B]*B+t%B`。固定 TILE_KV=256、FP16 输入、FP32 accumulator、expf、causal/window、block/grid、shared memory、编译 flags，并记录 PTX register/spill 差异。

目前旧 CUDA contiguous attention 与新 paged attention 存在 W/W+1 的窗口边界差，CPU 两套路径也存在 exp 精度差异；应先统一再测，不能用既有两条不同实现解释纯 gather 收益。

| 实验层 | 排除项与测量工具 | 主要指标 |
|---|---|---|
| Kernel | 不含模型与 HTTP；同 stream CUDA Events | gather、attention、组合/direct µs、launch、逻辑复制流量 |
| Model | 不含 tokenize/队列；固定 cache snapshot；GPU Events + host wall | forward、dequant/GEMM/GEMV/attention、sync/D2H、峰值显存 |
| Serving | 真实 HTTP/SSE open-loop arrival replay | tok/s、req/s、TTFT/ITL/E2E p50/p95/p99、queue、取消与回收 |

Gather+attention 组合要直接计时，不能仅相加独立测量。CUDA Event 时间与 host enqueue、同步 D2H、SSE 可见 ITL 分开。一个请求的 query rows 不等于多个请求各 decode 一行。

### 12.4 Workload、容量与控制条件

正式模型覆盖 SmolLM-135M Q8 correctness/smoke、3B Q4_K 和 7B/8B Q4_K。按模型支持情况扫 context=512/2K/4K/8K/16K/32K、page=8/16/32/64、MHA/GQA/MQA、head_dim=64/128、连续/随机/交错页及 sliding window；先筛选，再扩展性能拐点。

服务负载包含 burst、5/10/20/50 ms 固定或 Poisson 间隔、50% 128/32 + 30% 1K/128 + 20% 8K/256 的混合长度，以及基线容量 30%/60%/80%/95% 的同绝对到达率对照。超过 context 或不能稳态运行的配置如实单列。

性能比较固定逻辑 token 准入容量，报告实际 VRAM。容量比较固定总显存预算，报告可驻留 sequence、live KV tokens、尾页浪费、未用 reservation、metadata 与 gather scratch。固定物理页池的 free ID 回收不表示底层 GPU allocation 缩小。

所有版本共用模型哈希、GPU、runtime、precision、greedy、token corpus、arrival seed、sequence/token budget 和 warm-up。正式模型/服务每轮至少 30 秒、5～10 次重复，随机化版本顺序；报告 median、min/max 或基于独立 run 的 95% CI，不只报告最好一轮。

### 12.5 正确性门禁与结果解释

先验证 serial/batched token IDs、A3/A4 logits、head mapping、跨页、window、chunked prefill、position、stop/EOS、取消及异常回收，再运行 Compute Sanitizer memcheck/racecheck。所有请求释放后检查 `free_pages==total_pages`，连续 KV 检查 live allocation 为零。

A0–A4 主表全部标“待测”，已有 1.73× 不填入任何新增版本。使用 `throughput_ratio=X.tok_s/Y.tok_s` 与同一分位数的延迟变化；相邻条件收益不能相加成模块独立贡献。queue/prefill/decode 分项按请求时间戳计算，不能相加各项 p95。

预注册 H1–H6：短 context 单序列连续 KV 优势、context 增长时 direct/gather 拐点、随机页 locality、packed projection 的主要贡献、量化小 batch 全量反量化回退，以及吞吐与 ITL 的权衡。均为待验证假设。

### 12.6 最小实施方案

优先做 G0 contiguous、G1 paged+gather、G2 paged+direct，同源 math，context=512/4K/16K、active=1/4/8、page=16、连续/随机页，覆盖 3B/7B Q4_K；随后补 A1/A2 与真实服务负载。

需要新增 FanoutBatchModel、contiguous packed 入口、GPU gather、共享 attention body、CUDA Event 计时、组件 profiling 和 HTTP arrival replay。以上是实验接口计划，本次文档补充没有实现或运行它们。
