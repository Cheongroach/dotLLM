# dotLLM 连续批处理与 PagedAttention 实施记录

- 仓库：`dotLLM/`，分支：`issue/256-continuous-batching-paged-attention`。
- 基线提交：`d88040451d7db56e5dfef9d5754ad0955b0f7fe5`。变更尚未提交或发布 PR。
- 范围：43 个文件（24 修改、19 新增），新增 4229 行、删除 394 行；新增行中包含 1125 行生成 PTX。
- 运行说明：[SCHEDULING.md](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/docs/SCHEDULING.md)。本地 PR 描述：[dotllm-pr-description.md](dotllm-pr-description.md)。
- 实验归因：[分层消融设计](dotllm-ablation-study.md)，A0–A4 与 G0–G2 均待实施；现有 1.73× 不代表模块独立贡献。

## 已实现

1. `IBatchedModel.ForwardBatch` 将请求拼成无 padding 的 token 批次，共享投影/FFN，分别执行注意力，只计算每段最后 token 的 logits。
2. `ContinuousBatchScheduler` 负责动态准入、优先级等待队列、decode 优先、分块 prefill、容量预留和回收；每个请求拥有独立采样器、约束与状态。
3. CPU `IPagedKvCache` + `PagedAttention` 直接查页表；旧 gather 接口按需分配，用于兼容旧调用。
4. CUDA 页池、GPU 页表、新 KV scatter、分页 online-softmax、batched LM head；GPU 完成后才释放页。LM head 大缓冲区按需扩容。
5. 两种 completion API 的同步/SSE 路径接入，模型切换等待请求 lease 结束，关闭时取消并清理请求。
6. 流式停止字符串跨 token / token 内部截断、慢客户端隔离、取消、异常传播、种子和约束隔离；线程池启动时首个 dispatch 竞态修复。

## 正确性验证

- Release 构建成功，无新增警告；`git diff --check` 通过。
- 本地单元测试：1196 passed，41 skipped（本机无 CUDA）；总数 1237。最后补充的 SSE logprobs 保留场景与模型切换等待请求场景，定向服务端测试 2 passed。
- SmolLM-135M Q8_0 集成测试：4 passed，其中 3 个不同 prompt、分块预填充、12 token 生成与串行完全一致。
- H20：MHA/GQA/MQA、跨页/非连续页、277 token 跨 softmax tile、窗口长度 1 和 17，与 double 标量参考比对误差不超过 0.001；回收检查通过。
- H20 真实模型：3 个不同 prompt、每个 12 token，并发 greedy token IDs 与串行完全一致。
- GPU 验证通过独立 .NET harness 执行相同的公开测试方法及断言；没有运行仓库其余 GPU 测试。

模型文件：`SmolLM-135M.Q8_0.gguf`，144810528 字节；SHA-256：`7d4afa1d8a5587f4beb8b75dab755f90d13d3ae2b8b9f148545b414ecfb9123a`。

## 最终 H20 基准：三轮复验

在按需分配 LM head scratch 的最终 CUDA 实现上，32 个请求同时到达、每个 64 token、共 2048 token，分别限制 1 / 4 个活跃序列，重复三轮。以下为三轮各项指标的中位数，不是合并全部请求重新计算的百分位。

| 活跃序列 | tok/s 中位数（范围） | TTFT p50 / p95(ms) | ITL p50 / p95(ms) | forward 次数 |
|---:|---:|---:|---:|---:|
| 1 | 347.70 (346.46–348.31) | 2763.91 / 5519.92 | 2.87 / 3.33 | 2048 |
| 4 | 600.75 (599.84–602.16) | 1284.46 / 2988.37 | 6.70 / 8.45 | 513 |

吞吐提升约 **1.73×**。所有请求生成完整 64 token，页回收检查均通过。分页数值测试和三 prompt 的 serial/batched greedy 一致性也重新通过。结果仅适用于该模型和负载；更高吞吐伴随单请求逐 token 延迟增加。

归因边界：两组均使用同一 continuous scheduler 和 direct paged 路径，只限制活跃 sequence 数为 1 / 4，因此该比值是并发与 packed 执行的组合收益，不是 A4/A0，也不能分别归因于 Scheduler、Paged KV 或 Direct Pages。计时在请求进入 scheduler 到 Channel 读取之间，不包含 HTTP/SSE 网络；每轮约 3.4–5.9 秒，尚不满足至少 30 秒、5～10 次重复的正式实验要求。

原始记录：`dotllm-gpu-validation-3.log`；结构化结果：`dotllm-gpu-benchmark.jsonl`。

## 首轮短时基准

相同调度器与分页内核，对比最大活跃序列数 1 / 4；8 个请求同时到达，每个 32 token，输出共 256 token。加载与 warm-up 不计入计时。CPU 使用 4 线程。这里的串行基线是 `max_sequences=1`，不是 upstream 未修改版本。

| 环境 | 最大活跃序列 | tok/s | TTFT p50 / p95(ms) | ITL p50 / p95(ms) | forward 次数 |
|---|---:|---:|---:|---:|---:|
| CPU | 1 | 91.07 | 1119.68 / 2528.00 | 9.02 / 10.79 | 256 |
| CPU | 4 | 94.17 | 247.39 / 1591.66 | 35.33 / 49.48 | 65 |
| H20 | 1 | 358.67 | 262.29 / 622.21 | 2.68 / 2.94 | 256 |
| H20 | 4 | 701.68 | 7.47 / 188.35 | 5.71 / 6.55 | 65 |

H20 为 1 卡，driver 535.161.08，约 95.6 GiB；.NET Runtime 10.0.12。以上 GPU 单轮仅持续 0.36–0.71 秒，只作为 smoke 数据，不代表所有模型与负载。

CPU kernel 微基准：16 个 query heads、4 个 KV heads、head_dim=64、单 token decode；5 轮 × 30 次，报告中位数。包含 legacy gather+attention 数学实现的差异，不能只归因于省去 copy。

| 上下文 token | gather+attention(µs) | direct pages(µs) | 比值 |
|---:|---:|---:|---:|
| 512 | 604.16 | 977.14 | 0.62× |
| 4096 | 3628.46 | 2958.73 | 1.23× |
| 16384 | 22147.66 | 18892.66 | 1.17× |

**CPU 微基准结论。** 在已测配置下，Direct Pages 并非普遍加速：512 token 时耗时增加约 **61.74%**；4K 时耗时降低约 **18.46%**；16K 时从 **22.15 ms 降至 18.89 ms**，每次单层注意力调用节省约 **3.26 ms（14.70%）**。表中的 **1.17× 是速度比，不代表耗时减少 17%**。这些是一个 query token、16 query heads、4 KV heads、head_dim=64 的注意力微基准，不能直接乘以层数推算模型加速，也不能外推为 GPU 或服务端收益。

4K、16K 的观测支持“较长上下文下完整 direct 路径可能更快”，但只有三个上下文点，尚不能确定准确的性能拐点；相对收益也没有随长度单调增加。短上下文中页表寻址、局部性和算子开销如何影响性能，需组件计时验证。由于两条路径的 attention/exp 数学实现不同，当前节省的时间不能全部归因于去掉 gather。[原始微基准记录](../evidence/dotllm-paged-kernel-benchmark.txt)

## 分层消融实验计划（待执行）

新增完整设计：[dotllm-ablation-study.md](dotllm-ablation-study.md)；主报告摘要：[系统报告第 12 章](dotllm-system-walkthrough.md#12-分层消融实验与归因边界)。

| 配对 | 改变的因素 | 输出 |
|---|---|---|
| A1/A0 | 整请求串行 → 迭代调度；独立 forward、连续 KV 保持一致 | 排队、TTFT、公平性、吞吐 |
| A2/A1 | fanout → packed projection/FFN/LM head | GEMV/GEMM、dequant、launch、sync、吞吐 |
| A3/A2 | 连续 KV → 同一页池、页表、scatter、gather | 显存容量与管理/gather 开销 |
| A4/A3 | gather 后读取 → direct page address | 同源 math 下的 Direct Pages 条件收益 |

当前 CUDA `ForwardBatch` 只接受 paged cache；Fanout wrapper、contiguous packed、gather、统一 math、CUDA Events 和 HTTP 负载生成器均需补充代码。实验未运行，不能用历史记录代替任何缺失的版本。

优先级：同源 G0/G1/G2 kernel → 3B/7B model sweep → A1/A2 → HTTP/SSE A0–A4 → 固定显存容量与长稳验证。正式运行前经过 token/logits parity、window/head/page/取消回收以及 sanitizer 门禁。结果报告配对 run 的统计区间、失败配置和反例。

## 当前证据支持的结论

**功能与正确性：** 当前分支已经实现多个请求共享 packed forward、按请求隔离 KV/采样状态及完成后的页回收。现有单元测试、真实 SmolLM greedy parity 和 H20 标量对照支持已测试路径的正确性；它们不等于所有模型的数值一致性，也不等于长期生产稳定性。

**H20 并发收益与延迟代价：** 在 SmolLM-135M Q8_0、32 请求同时到达、每个生成 64 token 的条件下，1→4 个活跃 sequence 的三轮吞吐中位数提高 **72.78%（1.73×）**；TTFT p50 降低 **53.53%**，p95 降低 **45.86%**。与此同时，ITL p50 增加 **133.39%**，p95 增加 **153.66%**。因此，这组结果支持总吞吐和首 token 等待的改善，同时明确显示单请求 token 间隔的代价；这些百分比均由逐轮分位数的中位数计算，不是合并请求后重算的分位数。

底层 forward 次数从 **2048 降至 513（减少 74.95%）**，多活跃序列组平均每次 forward 包含约 **3.99 个 sequence**，证明计算调用实际合批，而不是仅将请求排队。但调用减少不等于耗时同比减少，也不能据此给 Scheduler、Packed Forward 或 Direct Pages 分配贡献比例。三轮短时结果的离散度较小，仍缺少更长运行、更多独立重复和置信区间；不能据此宣称统计显著或跨负载稳定。[H20 原始记录](../evidence/dotllm-gpu-benchmark.jsonl)

**后端与规模边界：** CPU 的 8 请求 × 32 token 短时实验中，吞吐仅提高约 **3.41%**，与 H20 的负载不同，不能横向推导 CPU/GPU 加速差。当前真实模型证据仅覆盖 SmolLM-135M，尚不能证明 3B/7B 量化模型、混合长度、随机在线到达或高负载下仍有相同收益。[CPU 原始记录](../evidence/dotllm-cpu-benchmark.jsonl)

**内存收益：** 已验证测试结束后页号可回收，支持请求间复用；尚未测定同等显存预算下能多容纳多少 sequence、节省多少峰值显存。固定物理页池不会因页号归还而缩小，保守 reservation 也可能限制准入，不能把“页全部归还”表述为“显存占用下降”。

**现阶段可采用的结论表述：** 在指定 H20/SmolLM burst 负载下，连续批处理与 packed execution 的组合获得约 1.73× 吞吐，同时付出更高 ITL；CPU direct 路径在已测 4K/16K 注意力微基准中更快，512 token 则回退。模块独立贡献、准确性能拐点、分页容量收益和代表性大模型适用范围，均需后续分层消融确认。A0–A4 与 G0–G2 仍为待执行计划，本次结论补充没有新增实验数据。

## 明确限制

尚无 active preemption、前缀共享、量化 KV、hybrid、speculative、请求级 LoRA、多 GPU 或 CUDA graphs。显存/内存页池为固定底层分配；释放页号供复用，并不减少整个池占用。worst-case token 预留保证不因活跃请求增长而抢页失败，但保守预留会降低并发上限。单请求 decode 延迟可能随批大小增大；短上下文 CPU 内核可能变慢。

## 日志与复现

- `dotllm-unit-test.log`、`dotllm-real-model-test.log`、`dotllm-server-test.log`。
- `dotllm-cpu-benchmark.jsonl`、`dotllm-paged-kernel-benchmark.log`。
- `dotllm-gpu-validation-2.log`：H20 第一轮成功验证；`dotllm-gpu-validation-3.log`：最终代码复验与三轮稳定性测量。
- 可移植补丁：`dotllm-continuous-batching.patch`，包含全部 tracked/untracked 变更，已用 `git apply --reverse --check` 核对。
- `dotllm-launch-gpu.py` / `dotllm-gpu-remote.py` 为此环境临时验证脚本；正式可复现的公开命令见仓库调度文档。
- IDE `apply_patch` 调用超时，文件编辑使用 Python 后备路径；AppHost 在沙箱内创建 memfd 被拒绝，构建使用 `-p:UseAppHost=false -p:MinVerSkip=true`。这不影响正常环境的常规 `dotnet` 命令。
