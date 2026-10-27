# dotLLM 对话记录：当前可见消息摘录

导出日期：2026-10-28。

**范围与来源：** 以下内容摘录自本次归档时助手可见的对话上下文，按可见顺序排列。不是完整历史导出。开头若干用户消息对应的助手回复未出现在可见上下文中，因此未补写。没有包含系统/开发者指令、思考过程、工具调用、工具输出或云端记忆摘要。消息时间戳不可恢复，不根据消息内容推测。

保留助手当时的回答，包括可能已经过时的项目状态和未经独立验证的评价。阅读事实与实验结论时，以归档 README 指向的实施记录和消融设计为准。链接目标在归档整理时转换成 GitHub/归档内路径；机器路径在正文中的引用保持原样。

## 001 · 用户

调研一下C#原生训推引擎

## 002 · 用户

适合交pr充实简历的方向？

## 003 · 用户

是的。可以看一下和主流推理框架的对齐程度，看看哪里发力比较合适。

## 004 · 用户

好，把 dotLLM 的实现一下：连续批处理、PangedAttention。

## 005 · 用户

怎么提Pr？

## 006 · 用户

让我review一下你的改动。
然后以面试者的角度介绍一下。
面试介绍要求：
1. 看起来高大上，不要有种“哦，AI也能做”的感觉，有独特价值。
2. 提出多个可选的比选方案，体现思考决策过程。
3. 体现技术深度和自己的价值。
3. 能以图形、PPT来介绍。

## 007 · 用户

你做的这个工作，符合一般硕士的什么样的水准如何变成我面试的有力材料？

## 008 · 用户

我想你把该推理系统串讲一遍。
从不同的切面介绍。
按功能区分：
- 模型执行层（Model Runtime）
- 注意力与显存优化（KV Cache 管理）
- 批调度与请求编排（Scheduler）
- 并行与分布式（Parallelism）
- 解码策略与加速（Decoding）
- 服务与接入层（Serving Layer）
- 可观测性与运维（Observability）
- 安全与合规

按运行时角色（Process Role）：
Frontend / Router：无状态，纯 Python/Rust，负责协议和路由。
Scheduler / Master：有状态单点，维护全局队列和 KV 元数据。
Worker / Executor：每张 GPU 一个进程，拿到 batch 后执行 forward。
KV Manager：PD 分离架构中独立出来的 KV 池 / 传输服务。
Tokenizer Worker：把 CPU 密集的 tokenize/detokenize 从 GPU 进程剥离。

三、按抽象层次（Abstraction Level）
从 "离硬件远近" 切分，常对应不同语言。
Python 层：业务编排、调度策略、Python 侧的 Module 定义。
C++ 层：性能敏感的调度核心、KV 块管理、ZMQ/RPC 通信。
CUDA / Triton 层：手写 kernel，PagedAttention、FlashAttention、量化 GEMM。
IR / 编译层：TensorRT-LLM / MLIR / Torch Inductor 的图编译与 fusion。

四、按请求生命周期（Pipeline Stage）
把代码按 "一个请求从进到出" 的阶段切成流水线，便于定位问题和做 PD 分离。

五、按横切关注点（Cross-cutting Concerns）
贯穿所有模块的 "切面"，通常独立成包。
Config：Pydantic/YAML 的配置系统，模型 / 并行 / 调度 / 量化配置分文件。
Logging & Metrics：Prometheus 指标、OpenTelemetry Trace。
Error & Retry：统一异常体系、超时、取消 (cancellation)。
Testing：unit /integration/correctness /performance 四类独立目录。
Benchmark：独立的 benchmarks/，与业务代码解耦。
六、按变化频率与稳定性（Stability Tiers）
这是架构治理视角，决定 "什么该抽象、什么该硬编码"。
Stable Core：调度器接口、Executor 接口、KV 接口 —— 很少改。
Pluggable Backend：模型实现、量化方法、attention backend—— 可插拔。
Experimental：speculative decoding、新并行策略 —— 放 experimental/ 或 feature flag。
Vendor-specific：NVIDIA / AMD / 国产卡的平台适配，隔离在 platforms/<vendor>/。
七、按硬件 / 后端（Backend Dimension）
同一份高层代码，适配多后端。
Device Backend：cuda/、rocm/、cpu/、tpu/、npu/。
Attention Backend：flash_attn/、xformers/、triton/、flashinfer/。
Quant Backend：awq/、gptq/、fp8/、int4/。
Comm Backend：nccl/、gloo/、rccl/、custom_allreduce/。
八、按部署形态（Deployment Topology）
同一代码库，不同部署模式入口不同。
单卡单进程：本地调试用。
单机多卡：TP within node。
多机多卡：TP + PP + EP 组合。
PD 分离集群：Prefill 集群 + Decode 集群 + KV 传输层各自独立部署。
Serverless / Spot：弹性扩缩容，worker 可随时上下线。

## 009 · 助手最终回复

**客观水准**
按工程型硕士项目评估，这项工作大致处于：

| 维度 | 水准 |
|---|---|
| 系统跨度 | 强硕士上沿：HTTP → Scheduler → Model → KV → CUDA |
| 底层深度 | 明显高于一般硕士：unmanaged memory、SIMD、PTX、online softmax |
| 实验严谨性 | 较强：标量参考、真实模型 parity、H20 三轮测试 |
| 研究创新性 | 中等：核心思想已有，价值在原生 .NET 落地与跨层设计 |
| 生产成熟度 | 中等偏低：仍有 3 个 P1、1 个 P2，未长期压测 |
| 外部认可 | 当前偏弱：尚未形成上游合并 PR |

综合判断：**远超普通课程项目，属于强硕士系统项目上沿；如果修完问题并获得上游 PR review，可以成为推理基础设施岗位非常有力的核心项目。**

不能包装成“研究创新达到论文水平”，更适合定位为：

> 在非主流但高约束的 Native .NET 推理栈中，完成从调度、执行、内存到 CUDA kernel 的系统级改造。

**决定材料强度的公式**
```text
面试说服力 = 技术难度 × 个人所有权 × 可复现证据 × 外部验证
```

现在：

- 技术难度：高
- 可复现证据：较高
- 外部验证：低
- 个人所有权：取决于你能否脱离代码独立解释、修改和复现

由于实现有 AI 辅助，真正拉开差距的不是掩盖 AI，而是做到：

1. 能从空白画出架构和状态机。
2. 能解释关键公式与内存布局。
3. 面试官任意修改条件时，能现场推导影响。
4. 能亲自修复 review 中的问题。
5. 能提交 PR 并处理真人 reviewer 的反对意见。

如果做不到这些，4,000 行代码反而会成为风险。

**转化路线**
1. **先获得技术所有权**
   - 独立重写 scheduler 的伪代码。
   - 手推 online softmax。
   - 计算指定模型的 KV 显存。
   - 解释 page table 如何映射物理地址。
   - 不看材料讲清 cancellation、EOS、stop string 和页回收。

2. **补齐工程闭环**
   - 修复默认 CUDA 页池显存问题。
   - 修复 CUDA PrefixCache 不兼容。
   - 测出量化 GEMV/HGEMM 的 batch-size 拐点。
   - 统一 sliding-window 语义。
   - 做 3B/7B 模型和 30 分钟并发压测。

3. **获得外部信用**
   - 拆成可审查的小 PR。
   - 保留 reviewer 讨论和修改记录。
   - 至少争取一个 PR 合并。
   - 发布可复现 benchmark 和设计文档。

4. **形成展示资产**
   - 一页架构图。
   - 一页方案比选。
   - 一页状态机与内存映射。
   - 一页正负性能结果。
   - 一个 3 分钟现场 demo。

**简历版本**
> **Native .NET LLM Continuous Batching & PagedAttention**
> - 设计 iteration-level continuous batching，将独立请求组织为无 padding packed segments，共享 QKV/FFN/LM-head 计算，同时隔离 attention、KV、采样状态与 SSE 生命周期。
> - 在 CPU SIMD 与 CUDA PTX 后端实现 direct PagedAttention，基于逻辑页表直接读取物理 KV 页，并使用 tiled online softmax 消除历史 KV gather。
> - 在 NVIDIA H20、32 请求 × 64 token 负载上，将吞吐中位数从 347.70 提升至 600.75 token/s（1.73×）；通过 scalar reference、真实模型 serial/batched parity 和异常页回收测试验证正确性。
> - 分析吞吐与 ITL、短上下文间接寻址、量化 GEMV/HGEMM 以及固定页池显存之间的性能权衡。

**面试定位**
针对不同岗位，强调点不同：

- **AI Infra / Serving**：调度策略、请求状态、背压、取消、资源准入。
- **CUDA / HPC**：页表寻址、shared memory、online softmax、GQA/MQA、数值验证。
- **系统工程**：single-owner 并发模型、生命周期不变量、无托管内存、异常恢复。
- **后端工程**：SSE、热切换、bounded queue、慢消费者隔离、服务降级。

**60 秒介绍**
> 原系统虽然能接收并发请求，但模型仍按完整请求串行生成。我把它改造成迭代级连续批处理：不同请求以无 padding segment 方式共享 Transformer 投影计算，但各自维护独立的 attention 范围、KV 页表、采样器和流式状态。
>
> 为避免分页 KV 每步 gather，我又在 CPU 和 CUDA 两端实现 direct PagedAttention，CUDA 使用 device block table、自定义 scatter kernel 和 tiled online softmax。H20 三轮测试中吞吐提升 1.73 倍，同时我也测出了 ITL 上升和短上下文回退，并进一步识别了默认页池显存和量化 batch 路径的生产风险。这个项目最核心的价值，是我建立并验证了调度、模型、内存和 CUDA 生命周期之间的一致性。

**面试官最可能深挖**
必须准备以下问题：

- 为什么不用静态 padded batch？
- 为什么采用 single compute owner？
- 为什么 decode 优先？
- worst-case KV reservation 有什么缺点？
- online softmax 如何保持数值稳定？
- GQA 的 query head 如何映射 KV head？
- 如何证明没有跨请求 attention 污染？
- 为什么 CPU 512-token 场景反而变慢？
- 量化模型为什么 batch 后可能更慢？
- 请求取消时，什么时候才能安全回收 GPU 页？
- 1.73× 的基线、模型、硬件和 workload 是什么？
- 如果扩展到多 GPU、PD 分离，需要改哪些接口？

已有材料可直接使用：

- [面试 PPT](../manuscripts/dotllm-continuous-batching-interview.pptx)
- [完整讲稿](../manuscripts/dotllm-interview-script.md)
- [Review 报告](../documents/dotllm-code-review.md)

最关键的一步是：**亲自修复那 4 个 review 问题并推动至少一个 PR 获得真人 review。**完成后，它就不只是“一个看起来复杂的本地项目”，而是可以证明你具备真实推理基础设施能力的材料。

## 010 · 用户

我想你把该项目。
介绍时，从不同的切面介绍。
按功能区分：
- 模型执行层（Model Runtime）
- 注意力与显存优化（KV Cache 管理）
- 批调度与请求编排（Scheduler）
- 并行与分布式（Parallelism）
- 解码策略与加速（Decoding）
- 服务与接入层（Serving Layer）
- 可观测性与运维（Observability）
- 安全与合规

按运行时角色（Process Role）：
Frontend / Router：无状态，纯 Python/Rust，负责协议和路由。
Scheduler / Master：有状态单点，维护全局队列和 KV 元数据。
Worker / Executor：每张 GPU 一个进程，拿到 batch 后执行 forward。
KV Manager：PD 分离架构中独立出来的 KV 池 / 传输服务。
Tokenizer Worker：把 CPU 密集的 tokenize/detokenize 从 GPU 进程剥离。

三、按抽象层次（Abstraction Level）
从 "离硬件远近" 切分，常对应不同语言。
Python 层：业务编排、调度策略、Python 侧的 Module 定义。
C++ 层：性能敏感的调度核心、KV 块管理、ZMQ/RPC 通信。
CUDA / Triton 层：手写 kernel，PagedAttention、FlashAttention、量化 GEMM。
IR / 编译层：TensorRT-LLM / MLIR / Torch Inductor 的图编译与 fusion。

四、按请求生命周期（Pipeline Stage）
把代码按 "一个请求从进到出" 的阶段切成流水线，便于定位问题和做 PD 分离。

五、按横切关注点（Cross-cutting Concerns）
贯穿所有模块的 "切面"，通常独立成包。
Config：Pydantic/YAML 的配置系统，模型 / 并行 / 调度 / 量化配置分文件。
Logging & Metrics：Prometheus 指标、OpenTelemetry Trace。
Error & Retry：统一异常体系、超时、取消 (cancellation)。
Testing：unit /integration/correctness /performance 四类独立目录。
Benchmark：独立的 benchmarks/，与业务代码解耦。
六、按变化频率与稳定性（Stability Tiers）
这是架构治理视角，决定 "什么该抽象、什么该硬编码"。
Stable Core：调度器接口、Executor 接口、KV 接口 —— 很少改。
Pluggable Backend：模型实现、量化方法、attention backend—— 可插拔。
Experimental：speculative decoding、新并行策略 —— 放 experimental/ 或 feature flag。
Vendor-specific：NVIDIA / AMD / 国产卡的平台适配，隔离在 platforms/<vendor>/。
七、按硬件 / 后端（Backend Dimension）
同一份高层代码，适配多后端。
Device Backend：cuda/、rocm/、cpu/、tpu/、npu/。
Attention Backend：flash_attn/、xformers/、triton/、flashinfer/。
Quant Backend：awq/、gptq/、fp8/、int4/。
Comm Backend：nccl/、gloo/、rccl/、custom_allreduce/。
八、按部署形态（Deployment Topology）
同一代码库，不同部署模式入口不同。
单卡单进程：本地调试用。
单机多卡：TP within node。
多机多卡：TP + PP + EP 组合。
PD 分离集群：Prefill 集群 + Decode 集群 + KV 传输层各自独立部署。
Serverless / Spot：弹性扩缩容，worker 可随时上下线。

## 011 · 助手最终回复

我已经按八种切面整理成完整文档：

[dotLLM 推理系统串讲](../documents/dotllm-system-walkthrough.md)

注意：Continuous Batching 与 CUDA Direct PagedAttention 指当前工作区分支实现，尚未合并上游。

**项目定位**
dotLLM 当前是：

> 单进程、单模型、单计算所有者的 Native .NET 推理引擎，通过 C# 完成服务、调度、模型 Runtime 和内存管理，通过 CUDA PTX/cuBLAS 执行 GPU 计算。

它目前不是 vLLM 式多进程分布式系统。

**功能域**
| 功能域 | 当前状态 |
|---|---|
| Model Runtime | CPU、CUDA、CPU/GPU Hybrid 已实现 |
| KV Cache | Simple、Paged、Quantized、简单 PrefixCache 已实现 |
| Scheduler | 动态 batching、decode-first、chunked prefill 已实现 |
| Parallelism | CPU 多线程、单 GPU、Hybrid 已实现；TP/PP/NCCL 未实现 |
| Decoding | Sampling、约束解码、greedy speculative 已实现 |
| Serving | ASP.NET、OpenAI API、SSE、Web UI、模型热切换已实现 |
| Observability | Timings、health、benchmark 已实现；OTel/Prometheus 仅有接口 |
| Security | 本地开发工具级；无认证、TLS、RBAC、审计 |

**运行时角色**
工业系统通常拆为五种进程，但 dotLLM 当前全部合并在一个进程中：

| 工业角色 | dotLLM 当前对象 |
|---|---|
| Frontend / Router | `DotLLM.Server` |
| Scheduler / Master | `ContinuousBatchScheduler`、`ServerState` |
| Worker / Executor | `TransformerModel`、`CudaTransformerModel` |
| KV Manager | `PagedKvCacheFactory`、`CudaPagedKvCacheFactory` |
| Tokenizer Worker | `DotLLM.Tokenizers` 同步执行 |

要拆成多进程，必须把当前 C# 对象引用改成稳定协议，例如把 `SequenceBatch.Cache` 变成 `sequence_id + block_table_handle`。

**抽象层次**
dotLLM 与常见 Python/C++ 引擎不同：

```text
典型 vLLM:
Python 编排 → C++ Runtime → CUDA/Triton → Torch/Compiler IR

dotLLM:
C# 编排 → C# unsafe Runtime → CUDA PTX/cuBLAS → Driver JIT
```

C# 层承担了传统 C++ Runtime 的职责：

- unmanaged tensor
- GGUF mmap
- block allocator
- scheduler state machine
- SIMD intrinsics
- CUDA Driver API P/Invoke

没有 TensorRT、MLIR、Torch Inductor 图编译层。

**请求生命周期**
```text
HTTP DTO
→ Chat template
→ Tokenize
→ Queue / KV 容量准入
→ Chunked prefill
→ Packed model forward
→ Request-local PagedAttention
→ Sampling / Constraint
→ Incremental detokenize
→ SSE / JSON
→ KV 页回收
```

性能定位：

- TTFT 高：tokenize、queue wait、prefill、warm-up。
- ITL 高：decode GEMV、batch size、attention、sampling。
- OOM：权重、反量化 scratch、activation、KV pool。
- 输出串请求：segment offset、page table、cache ownership。
- 模型切换卡住：request drain、scheduler stop、dispose 顺序。

**稳定性分层**
- Stable Core：`IModel`、`IBatchedModel`、`IKvCache`、`IScheduler`、`ITensor`。
- Pluggable：CPU/CUDA model、attention、quantization、tokenizer、constraint。
- Experimental：Continuous Batching、Speculative Decoding、Native AOT。
- Vendor-specific：`DotLLM.Cuda` 与 `native/kernels/`。
- Future：ROCm、NPU、TP/PP、PD 分离。

**后端维度**
- Device：CPU、CUDA、Hybrid 已实现。
- Attention：CPU/CUDA contiguous 与 direct paged 已实现。
- Quant：GGUF Q8/Q4/Q5/Q6 系列，不是 AWQ/GPTQ 体系。
- Communication：进程内调用、Channel、CPU thread pool、单 CUDA stream。
- NCCL/Gloo/RCCL：未实现。

**部署形态**
| 部署方式 | 状态 |
|---|---|
| 单进程 CPU | 已实现 |
| 单卡单进程 | 已实现 |
| CPU/GPU Hybrid | 已实现 |
| 单机多卡 TP | 未实现 |
| 多机 TP+PP+EP | 未实现 |
| Prefill/Decode 分离 | 未实现 |
| Serverless/Spot | 未实现 |

核心代码建议按此顺序阅读：

1. [IModel.cs](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Core/Models/IModel.cs)
2. [TransformerModel.cs](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Models/Architectures/TransformerModel.cs)
3. [CudaTransformerModel.cs](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Cuda/CudaTransformerModel.cs)
4. [ContinuousBatchScheduler.cs](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Engine/ContinuousBatchScheduler.cs)
5. [PagedKvCache.cs](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Engine/KvCache/PagedKvCache.cs)
6. [paged_attention.cu](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/native/kernels/paged_attention.cu)
7. [ScheduledSequence.cs](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Engine/ScheduledSequence.cs)
8. [ServerState.cs](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Server/ServerState.cs)

一句话总结：

> dotLLM 已从本地单请求推理程序演进到具备 CPU/CUDA、分页 KV 和连续批处理的单节点 Serving Runtime，但距离多进程、多 GPU、PD 分离、完整可观测和安全治理的生产推理平台仍有清晰边界。

## 012 · 用户

介绍各处关键代码。

## 013 · 助手最终回复

已将完整“关键代码逐段讲解”追加到：

[dotLLM 推理系统串讲：第 11 章](../documents/dotllm-system-walkthrough.md)

**最重要的三组代码**
1. **请求隔离契约**
   - [IBatchedModel.cs:L10-L58](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Core/Models/IBatchedModel.cs#L10-L58)
   - `SequenceBatch.Validate()` 保证 segment 连续覆盖、position 追加、cache 不共享。
   - 这是防止跨请求 attention 污染的第一道边界。

2. **单 Owner 调度状态机**
   - [ContinuousBatchScheduler.cs:L131-L285](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Engine/ContinuousBatchScheduler.cs#L131-L285)
   - `RunLoopAsync → Admit → RunIteration → Finish`
   - Decode 请求先各占一个 token slot，剩余 budget 做轮转式 chunked prefill。

3. **逻辑位置到物理页**
   - [KvBlockTable.cs:L38-L103](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Engine/KvCache/KvBlockTable.cs#L38-L103)
   - `Resolve(position)` 完成逻辑页到物理 block 的映射。
   - `Fork + EnsureWritable` 提供 refcount 和 copy-on-write 基础。

**模型执行**
- [TransformerModel.ForwardCore](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Models/Architectures/TransformerModel.cs#L149-L434)
  - CPU Transformer 完整执行链。
  - `seqLen == 1` 走 fused decode；多 row 走 GEMM。
- [TransformerModel.Attend](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Models/Architectures/TransformerModel.cs#L436-L462)
  - 在 contiguous、quantized、paged attention 之间 dispatch。
- [Last-row LM Head](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Models/Architectures/TransformerModel.cs#L409-L431)
  - 每个 sequence 只取最后 hidden row，再共享一次 LM-head GEMM。

**CPU PagedAttention**
- [PagedKvCache.Update](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Engine/KvCache/PagedKvCache.cs#L101-L163)
  - 分页、COW、同页连续 token 批量 memcpy。
- [PagedAttention.Execute](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Cpu/Kernels/PagedAttention.cs#L22-L136)
  - GQA head 映射、页表寻址、256-token tile、online softmax。

**CUDA 路径**
- [CudaTransformerModel.ForwardBatch](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Cuda/CudaTransformerModel.cs#L147-L172)
  - 校验 context/cache geometry，准备 device page table。
- [CudaTransformerModel.ForwardCore](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Cuda/CudaTransformerModel.cs#L174-L374)
  - packed projection、按请求 attention、last-row logits、stream sync。
- [Project dispatch](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Cuda/CudaTransformerModel.cs#L381-L413)
  - 单 row 量化 GEMV；多 row 反量化后 cuBLAS HGEMM。
- [paged_attention.cu](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/native/kernels/paged_attention.cu#L20-L219)
  - 一个 block 负责一个 query-token/head。
  - Q 放 shared memory，K/V 经 device page table 读取，FP32 online softmax。

**请求状态与输出**
- [ScheduledSequence.Sample](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Engine/ScheduledSequence.cs#L58-L160)
  - constraint → sampler → stop → detokenize → channel。
- [ScheduledSequence.Complete](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Engine/ScheduledSequence.cs#L172-L198)
  - 构造 response、释放 cache、完成 stream/TCS。
- [IncrementalDetokenizer](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Engine/IncrementalDetokenizer.cs)
  - 避免每步重新 decode 全历史造成 O(N²)。

**服务生命周期**
- [ServerStartup.LoadModel](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Server/ServerStartup.cs#L66-L263)
  - 整个系统的 composition root。
- [ServerState.ExecuteAsync](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Server/ServerState.cs#L93-L110)
  - legacy 模式独占模型；batching 模式只保留 request lease。
- [ServerState.SwapModelAsync](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Server/ServerState.cs#L139-L169)
  - drain 请求，再按 scheduler → pool → model → GGUF 顺序释放。

**验证代码**
- [Scheduler 单测](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/tests/DotLLM.Tests.Unit/Engine/ContinuousBatchSchedulerTests.cs)
- [CPU PagedAttention 标量对照](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/tests/DotLLM.Tests.Unit/Cpu/Kernels/PagedAttentionTests.cs)
- [CUDA 标量对照](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/tests/DotLLM.Tests.Unit/Cuda/CudaPagedAttentionTests.cs)
- [真实模型并发一致性](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/tests/DotLLM.Tests.Integration/Engine/ContinuousBatchingIntegrationTests.cs)
- [连续批处理 Benchmark](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/benchmarks/DotLLM.Benchmarks/ContinuousBatchingBenchmark.cs)

推荐实际阅读顺序：

```text
IBatchedModel
→ ContinuousBatchScheduler.RunIteration
→ TransformerModel.ForwardCore
→ KvBlockTable / PagedKvCache
→ CPU PagedAttention
→ CudaTransformerModel
→ paged_attention.cu
→ ScheduledSequence
→ ServerState
→ correctness tests
```

这次只补充了讲解文档，没有修改推理实现，也没有重新运行测试。

## 014 · 用户

补充一下消融实验到报告里。

---

每层只改变一个变量，否则 1.73× 无法归因。
可以把消融设计成一个**分层归因实验**，目标不是再证明“系统变快了”，而是回答：

```text
1. Scheduler 贡献多少？
2. Packed forward 贡献多少？
3. Paged KV 的容量收益和性能代价是多少？
4. Direct Pages 相比 Gather 到底贡献多少？
5. 收益在哪些模型、上下文和并发下成立？
```

## 一、消融变量

把系统拆成三个独立变量：

| 变量 | 关闭 | 开启 |
|---|---|---|
| `S` 调度 | 整请求串行 | iteration-level continuous batching |
| `M` 模型执行 | 每请求单独 forward | packed projection/FFN/LM head |
| `K` KV 访问 | contiguous 或 paged+gather | direct paged attention |

理想情况下做全因子实验，但部分组合需要额外代码，因此建议采用阶梯式消融。

## 二、五组实验版本

### A0：Legacy 串行基线

```text
请求级串行
+ 每请求独立 Forward
+ CudaKvCache 连续 KV
+ contiguous attention
```

回答：

> 原始系统性能是多少？

### A1：只开启 Continuous Scheduler

```text
iteration-level scheduler
+ 每个 sequence 仍单独调用 Forward
+ 连续 KV
```

实现一个 `FanoutBatchModel`：

```csharp
ForwardBatch(batch):
    foreach segment:
        model.Forward(segmentTokens, segmentPositions, segment.Cache)
    concatenate last-row logits
```

A1 对 A0：

> 动态入场、decode-first、chunked prefill 本身改善了多少排队延迟？

它不会共享模型计算，因此吞吐未必明显提升，但 TTFT、公平性可能改善。

### A2：增加 Packed Forward

```text
continuous scheduler
+ packed QKV/FFN/LM-head
+ 每请求连续 KV
+ 相同 attention 数学实现
```

A2 对 A1：

> 共享 projection 计算贡献多少？

重点观察：

- forward 次数
- 平均 batch rows
- GEMV → GEMM 后的吞吐
- kernel launch 数
- TTFT 与 ITL

GPU 当前 `ForwardBatch` 只接受 paged cache，需要增加一个实验性
`ForwardBatchContiguous`，按 segment 使用独立 `CudaKvCache`，但 projection 保持 packed。

### A3：Paged KV + Gather

```text
continuous scheduler
+ packed forward
+ 共享物理页池
+ page table
+ gather K/V 到连续 scratch
+ contiguous online-softmax attention
```

A3 对 A2：

> 分页内存管理本身带来了多少容量收益和 gather 成本？

这一版本非常重要，因为它把“Paged KV 管理”和“Direct PagedAttention”分开。

需要新增 CUDA gather kernel：

```cuda
gather_paged_kv(
    page_pool,
    page_table,
    contiguous_scratch,
    sequence_length);
```

### A4：Paged KV + Direct Pages

```text
continuous scheduler
+ packed forward
+ 共享物理页池
+ paged_attention_f16 直接读取页表
```

这就是当前完整实现。

A4 对 A3：

> 消除 gather、直接读物理页的纯收益是多少？

这是回答“GPU Direct Pages 到底快多少”的关键对比。

## 三、最重要的公平性约束

A3 和 A4 必须使用**完全相同的 Attention 数学实现**。

不能拿：

```text
旧 contiguous attention
vs
新 paged online-softmax
```

否则差异同时包含：

- Gather copy
- 页表寻址
- 不同 softmax
- 不同 exp 精度
- 不同 tile
- 不同 shared-memory 布局

正确方法是从同一个 kernel 模板派生：

```text
attention_online_contiguous:
    K/V 地址 = base + token * stride

attention_online_paged:
    K/V 地址 = resolve(page_table, token)
```

除了地址解析表达式，其他代码保持一致：

- `TILE_KV = 256`
- FP16 输入
- FP32 accumulator
- 相同 `expf`
- 相同 causal/sliding-window
- 相同 block/grid
- 相同 shared-memory 大小

这样 A4/A3 才能归因于 Direct Pages。

## 四、实验分成三层

### 1. Kernel 微基准

测量：

```text
Gather kernel
Contiguous online attention
Paged online attention
```

使用 CUDA Event，不使用 host `Stopwatch`：

```csharp
event.Record(stream);
for (int i = 0; i < iterations; i++)
    LaunchKernel();
end.Record(stream);
end.Synchronize();
```

报告：

- gather 时间
- attention 时间
- gather+attention 总时间
- direct attention 时间
- speedup
- kernel launch 数
- gather 搬运字节数

测试矩阵：

| 维度 | 取值 |
|---|---|
| Context | 512、2K、4K、8K、16K、32K |
| Query rows | 1、4、8、16 |
| Heads/KV heads | MHA、GQA、MQA |
| Head dim | 64、128 |
| Page size | 8、16、32、64 |
| Fragmentation | 连续页、随机页、交错页 |
| Sliding window | 关闭、128、4096 |

### 2. Model Forward 微基准

不含 HTTP 和 tokenize，只测模型。

比较：

```text
A1 fanout forward
A2 packed forward
A3 packed + gather
A4 packed + direct pages
```

报告：

- forward latency
- prefill token/s
- decode token/s
- GEMM/GEMV 时间
- attention 时间
- D2H logits 时间
- CUDA kernel launch 数
- 峰值显存
- 平均 batch rows

### 3. 服务端到端压测

从请求提交开始，到 SSE 收到 token。

报告：

- 系统总吞吐 `token/s`
- request/s
- TTFT p50/p95/p99
- ITL p50/p95/p99
- E2E latency p50/p95/p99
- queue wait
- prefill/decode 时间
- GPU utilization
- KV pool utilization
- cancellation latency
- 页是否完全回收

## 五、Workload 设计

不能只测 SmolLM-135M，它太小，无法代表真实推理。

至少选：

| 模型 | 用途 |
|---|---|
| SmolLM-135M Q8 | correctness 和 smoke test |
| Llama/Qwen 3B Q4_K | 中等模型，观察小 batch 拐点 |
| Mistral/Llama 7B/8B Q4_K | 真实 bandwidth-bound decode |

请求类型：

### Burst

```text
32 个请求同时到达
```

验证最大 batch 合并能力。

### Poisson/固定间隔

```text
平均每 5/10/20/50 ms 到达一个请求
```

模拟在线服务，而不是人工完美 batch。

### 混合长度

```text
50%: prompt 128, output 32
30%: prompt 1K, output 128
20%: prompt 8K, output 256
```

验证 chunked prefill 是否避免长请求阻塞短请求。

### 稳态负载

分别压到单卡能力的：

```text
30%、60%、80%、95%
```

低负载时 batching 可能增加 ITL；高负载时才体现 throughput/TTFT 收益。

## 六、实验控制

所有版本固定：

- 同一模型文件及 SHA256
- 同一 GPU
- 同一 driver/CUDA/.NET runtime
- 同一 precision
- 同一随机种子
- greedy sampling
- 同一 prompt 集合
- 同一 `MaxBatchedTokens`
- 同一 `MaxSequences`
- 同一 KV pool 容量
- 相同 warm-up
- 禁止同时运行其他 GPU 任务

每个配置：

```text
warm-up 10 次
正式运行 5～10 次
每次至少 30 秒
报告 median + min/max 或 95% CI
```

不要只报告最好的一轮。

## 七、正确性门禁

性能实验前，每个版本都必须通过：

1. serial 与 batched token IDs 完全一致。
2. A3 与 A4 logits 最大误差在 FP16 阈值内。
3. MHA/GQA/MQA。
4. 跨页和非连续页。
5. sliding window。
6. 请求取消后页全部回收。
7. 运行结束：

```text
free_pages == total_pages
```

8. CUDA Compute Sanitizer：

```bash
compute-sanitizer --tool memcheck ...
compute-sanitizer --tool racecheck ...
```

## 八、最终结果表

### 端到端主表

| Variant | tok/s | TTFT p50/p95 | ITL p50/p95 | Forward 次数 | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| A0 Serial | | | | | |
| A1 Scheduler | | | | | |
| A2 Packed | | | | | |
| A3 Paged+Gather | | | | | |
| A4 Paged+Direct | | | | | |

### 贡献拆解

```text
Scheduler contribution     = A1 - A0
Packed compute contribution = A2 - A1
Paging overhead/capacity    = A3 - A2
Direct Pages contribution   = A4 - A3
Full-system gain            = A4 - A0
```

延迟指标不能直接做普通减法，建议同时报告百分比：

```text
throughput_gain = A4.tok/s / A0.tok/s
ttft_change = (A4.TTFT - A0.TTFT) / A0.TTFT
itl_change = (A4.ITL - A0.ITL) / A0.ITL
```

## 九、建议优先完成的最小版本

如果时间有限，只做下面三组：

```text
G0: contiguous KV + contiguous online attention
G1: paged KV + gather + contiguous online attention
G2: paged KV + direct paged online attention
```

测试：

```text
Context: 512 / 4K / 16K
Batch rows: 1 / 4 / 8
Page size: 16
Layout: 连续页 / 随机页
Model: 3B Q4_K + 7B Q4_K
```

这能严谨回答三个问题：

1. Paging 本身付出了多少代价？
2. Gather 占用了多少时间和 HBM 流量？
3. Direct Pages 在什么 context/batch 下开始获益？

**预期假设**
实验开始前先写清假设，避免看结果后编故事：

- `H1`：低 context、batch=1 时 contiguous 最快。
- `H2`：context 增长后 gather 成本线性增加，direct pages 开始占优。
- `H3`：物理页随机化会削弱 direct pages 的 locality，但仍避免 gather 写流量。
- `H4`：continuous batching 的主要收益来自 packed projection，而不是 attention。
- `H5`：量化 3B/7B 在小 batch 下可能因全量反量化而出现性能回退。
- `H6`：吞吐提升会伴随单请求 ITL 上升。

这样的消融即使最终发现某一模块没有加速，也是高质量结果，因为它给出了**明确的适用区间和性能拐点**。

## 015 · 助手最终回复

已补充到[系统报告第 12 章](../documents/dotllm-system-walkthrough.md#12-分层消融实验与归因边界)，并新增[完整消融实验设计](../documents/dotllm-ablation-study.md)。

包含 A0–A4 阶梯归因、同源 Attention、三层测量、3B/7B workload、容量实验、统计规范、正确性门禁及 H1–H6 假设。

也同步修正了实施记录和 PR 描述：**现有 1.73× 是组合收益，不能归因于某个模块。新增消融均标为待实施、待测，没有填入虚构结果。**

## 016 · 用户文件批注

文件：`/mlx_devbox/users/zhangyang.exp/vllm-omni/vllm-omni/external-refs/dotllm-implementation-notes.md`。

批注文字为空，选中内容：

```text
16384
22147.66
18892.66
1.17×
分层消融实验计划（待执行）
新增完整设计：dotllm-ablation-study.
```

## 017 · 用户

`/mlx_devbox/users/zhangyang.exp/vllm-omni/vllm-omni/external-refs/dotllm-implementation-notes.md`这里的结论补充一下。

## 018 · 助手最终回复

已补充[实施记录中的结论](../documents/dotllm-implementation-notes.md)：

- **16K CPU 注意力耗时减少 14.70%，约节省 3.26 ms**；1.17× 是速度比。
- **H20 吞吐提高 72.78%**，同时明确 TTFT 改善与 ITL 增大的代价。
- 区分已验证的合批与页回收，以及尚未证明的模块归因、显存容量收益和大模型适用范围。

没有新增实验数据，消融仍标为待执行。

## 019 · 用户

将文档、手稿还有和你的对话记录。上传到github一个单独的分支。
