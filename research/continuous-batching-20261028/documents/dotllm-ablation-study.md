# dotLLM 分层消融实验设计与结果边界

## 1. 目标与当前证据

**状态：实验设计已完成；本章新增的 A0–A4、G0–G2 消融尚未执行，没有新增性能结果。**

实验回答五个问题：调度贡献多少、Packed Forward 贡献多少、分页的容量收益与性能代价是多少、Direct Pages 相比 Gather 贡献多少，以及这些收益在哪些模型、上下文与并发下成立。

已有 H20 记录来自 SmolLM-135M Q8_0、32 个同时到达请求、每请求最多生成 64 token，比较同一调度器/分页路径下 `max_sequences=1` 与 `4`。三轮吞吐中位数分别为 347.70 和 600.75 token/s，比例为 1.7278；该记录既不是 upstream 原始系统与最终系统的 A4/A0，也不是 Direct Pages 的独立加速。[原始六次测量](../evidence/dotllm-gpu-benchmark.jsonl)

| 证据 | 已证明的范围 | 尚未证明 |
|---|---|---|
| H20 1.73× | 同一分页实现下，允许更多活跃请求的组合收益 | Scheduler、Packed Forward、Direct Pages 的单独贡献 |
| CPU gather/direct 微基准 | 两条完整执行路径的时间差 | 纯 gather 省略收益；两条路径的 exp/math 不完全相同 |
| H20 scalar 与生成一致性 | 已测试的页布局、head 配置与 SmolLM prompts 正确性 | A0–A3 新路径、3B/7B、长期服务负载的正确性 |

现有 benchmark 从 scheduler 提交到 Channel 读取计时，没有经过 HTTP/SSE 网络，因此不能称为服务端端到端压测。现有三轮 GPU 测量每轮约 3.4–5.9 秒，不满足下述至少 30 秒、5～10 次独立重复的正式实验要求。[Benchmark 源码](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/benchmarks/DotLLM.Benchmarks/ContinuousBatchingBenchmark.cs#L39-L98)

## 2. 控制变量与归因范围

基本变量按以下方式定义：

| 变量 | 关闭 | 开启 |
|---|---|---|
| S：请求调度 | 整请求串行生成 | iteration-level continuous scheduling |
| M：模型执行 | 每 sequence 独立 forward | packed QKV、FFN、LM head |
| K：KV 路径 | 连续 KV 或分页后 gather | direct paged attention |

K 实际包含存储与访问两个因素，不能作为一个二元开关解释。实验记录必须进一步拆成：

| 细化因素 | 水平 |
|---|---|
| P：KV 存储布局 | contiguous / shared physical pages |
| D：注意力地址访问 | contiguous address / page-table address |

因此实验是沿 S、M、P、D 的阶梯式消融，而非完整的三个二元变量全因子实验。完整全因子需要更多有效组合及实现，并非所有组合都具有工程意义。

**每一级只变一个因素组，组内还有子因素时另设消融。** A1/A0 估计的是整个调度策略组的增量，包括动态入场、decode-first、chunked prefill，不能据此单独宣称某个调度子策略贡献多少。A2/A1 包含 packed execution 导致的 GEMV/GEMM dispatch 变化；这是当前 Packed Forward 的实现效应，而非数学批处理的硬件无关收益。

阶梯结果都是“给定前面已启用因素后的条件效应”。各项比值可以沿阶梯连乘得到完整比值，但不能把相邻加速百分比相加，也不能据此消除模块之间的交互作用。

## 3. A0–A4 实验版本

| 版本 | S | M | P | D | 回答的问题 |
|---|---|---|---|---|---|
| A0：Serial | 整请求串行 | 独立 forward | 连续 KV | 连续地址 | 公平串行基线 |
| A1：Scheduler | 连续调度 | fanout forward | 连续 KV | 连续地址 | 调度策略组的效果 |
| A2：Packed | 连续调度 | packed forward | 连续 KV | 连续地址 | 共享 projection/FFN/LM head 的效果 |
| A3：Paged+Gather | 连续调度 | packed forward | 分页 KV | gather 后连续地址 | 分页管理与 gather 的合并开销、容量变化 |
| A4：Paged+Direct | 连续调度 | packed forward | 分页 KV | 直接页表寻址 | 用页表读取替换 gather 的增量效果 |

### 3.1 A0：串行基线与历史基线

A0 使用 `CudaKvCache`，对请求逐个完成生成，采用与 A1–A4 同源的 contiguous online attention。独立 forward 也只返回最后一行 logits，与 GPU 既有契约一致。PrefixCache、draft model、LoRA、额外日志等在所有版本统一关闭。

另外保留 **A0-original**：指定 upstream commit 的原始串行执行实现，用于回答“原系统原本多快”。历史版本与公平 A0 必须分开报告。若统一 attention 修复了原 CUDA 的 W/W+1 窗口差异，不能悄悄把新数学实现称为原始基线。

主阶梯以统一数学的 A0 开始；A0-original → A0 的改动和时间差另列。旧报告的 `max_sequences=1` 属于 A4 路径上的单活跃序列控制组，不能填入 A0。

### 3.2 A1：只开启调度

增加实验用 `FanoutBatchModel : IBatchedModel`：

```text
ForwardBatch(tokens, positions, segments):
    validate the whole batch before mutation
    allocate [segments.Count, vocab] Float32 output
    for each segment:
        logits = model.Forward(segment.tokens, segment.positions, segment.cache)
        copy logits.lastRow to this segment's output row
        dispose logits
    return output
```

A1 仍使用每请求连续 KV，没有共享模型 projection。容器必须区分 logical batch 调用和实际 executor forward：一次 `ForwardBatch` 的 fanout 可能包含多次底层 forward、stream synchronization 和 logits D2H。拼接成本也属于 A1，需单独测量。

A1/A0 的重点是 queue wait、TTFT、短请求延迟、公平性及吞吐。吞吐不要求必然提高；切片带来的额外调度和同步可能有代价。

为细分调度策略，再固定 M/P/D，增加：

- A1-FIFO：迭代调度、完整 prefill、固定 FIFO。
- A1-chunk：只改变 prefill chunk size。
- A1-decode-first：在同一 chunk size 下只改变 decode 优先级。
- A1-rotation：在其余策略固定时只改变 prefill 轮转。

给出同优先级短/长请求的 queue wait、TTFT 分组分布及最长等待，避免“总体 TTFT 下降”掩盖长请求饥饿。

### 3.3 A2：增加 Packed Forward

增加实验性 contiguous packed 路径（可称 `ForwardBatchContiguous`）。QKV/FFN/LM head 对所有有效 token row 做 packed projection；每个 segment 分别使用 `CudaKvCache.UpdateDevice` 和同源 contiguous attention。

连续 attention 的输入指针按 segment offset 选择，position 仍使用该请求的绝对位置。返回值统一为 `[sequences, vocab]`，所有版本只输出最后 row。

A2/A1 测 shared projection 的条件收益，并报告 GEMM/GEMV 选择、反量化时间与 scratch、kernel launch 数和同步次数。A2 必须验证与 A1 的生成一致性；不能只因接口名称里有 batch 就认定底层计算已合批。

### 3.4 A3：分页管理与 Gather

保持 A2 的 packed projection 与调度，只把连续 KV 改成与 A4 相同的物理页池及页表。每层先 scatter 新 K/V，再执行 GPU gather：

```text
gather_paged_kv(key_pool, value_pool, page_table, scratch_K, scratch_V, length):
    for each logical token and KV element:
        physical = resolve(page_table, logical_token)
        scratch_K[logical_token, element] = key_pool[physical, element]
        scratch_V[logical_token, element] = value_pool[physical, element]
```

Gather scratch 预热时分配，测量区间内禁止重新申请或扩容。A3 的 scratch 单列显存开销，不能因 A4 没有 scratch 就同时增加 A4 的 KV pool 容量。

A3/A2 是 **分页 allocator/page metadata/scatter/gather 的组合效果**，不是纯 allocator 的贡献。使用组件计时与独立容量实验进一步区分。

### 3.5 A4：Direct Pages

保留 A3 的 pool、页表、scatter、scheduler、packed projection，仅让 attention 直接通过页表读取物理页，删除 gather 调用和其使用。

A4/A3 是最重要的 Direct Pages 配对比较。allocator、fragmentation、page size、KV precision、可见 KV 长度均保持不变。attention launch 数不变，仅减少 gather launch，不能同时引入跨 sequence attention fusion。

## 4. 同源 Attention 的公平性约束

当前 CUDA paged kernel 与既有 contiguous kernel 的 sliding-window 边界并不完全一致，因此直接使用旧两套文件不能完成严格归因。[Paged 边界](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/native/kernels/paged_attention.cu#L81-L84) · [旧 contiguous 边界](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/native/kernels/attention.cu#L95-L104)

应从一个共享 kernel body 派生两个入口：

```text
attention_online_contiguous:
    row(token) = token

attention_online_paged:
    row(token) = page_table[token / page_size] * page_size + token % page_size
```

除地址解析和页表参数外，固定：

- `TILE_KV=256`、FP16 输入/输出、FP32 累积、同样的 `expf`。
- 相同 causal 和 sliding-window：`[max(0, pos + 1 - W), pos]`。
- 相同 Q/KV head 映射、dot-product 计算与 softmax reduction 顺序。
- 相同 block/grid、shared-memory 大小、tile 起止及遍历顺序。
- 相同 `nvcc`、architecture、编译优化和 precise/fast-math flags。

记录两个入口的 PTX 资源用量（register、spill、shared memory）。页表索引可能改变生成的指令或寄存器数，这是 Direct Pages 实际成本的一部分；不能声称编译后只相差一条地址表达式。

CUDA 使用编译期模板/地址解析函数，避免运行期分支混入 contiguous 路径。CPU 消融也需要同源精确或同源近似 exp，不能用旧 approximate-exp 对新标准 exp 解释 gather 开销。

## 5. 三层测量

### 5.1 Kernel 微基准

测量 gather、contiguous online attention、paged online attention，以及实际组合的 gather+attention。使用同一 stream 的 CUDA Events：

```text
warm up kernels and buffers
record start event on stream
repeat iterations:
    launch the measured kernel(s)
record end event on stream
synchronize end event
elapsed_ms = event_elapsed_time(start, end) / iterations
```

组合时间要直接测，不能只把两个独立微基准相加；组合可能受缓存预热、launch 间隙和数据局部性影响。禁止每次 launch 后 host synchronize。另记录 host enqueue 成本，避免把 CUDA Event 时间当作总服务时延。

输出：gather µs、contiguous attention µs、gather+attention µs、direct attention µs、配对比值、launch 数、复制字节数。

| 维度 | 候选取值 |
|---|---|
| Context tokens | 512、2K、4K、8K、16K、32K |
| 同一 sequence 的 query rows | 1、4、8、16 |
| Attention | MHA、GQA、MQA，记录确切 Q/KV heads |
| Head dim | 64、128 |
| Page size | 8、16、32、64 |
| 物理布局 | 连续、随机、交错；保存页表与 seed |
| Sliding window | 关闭、128、4096 |

**query rows 与并发 sequence 数分开记录。** 当前 kernel 的 `seq_q` 是一个 segment 内的 token 数；4 个 sequence 各 decode 1 row，不等于一个 sequence 有 4 个 query rows。模型/服务实验另设 active sequences=1/4/8/16。

先做筛选矩阵，再只扩展边界区域，避免机械跑完数千组合。每次比较只改变一个维度，页表和输入内容在配对运行中固定。

Gather 字节统计以 KV heads 为准，不用 query heads 重复计算：

```text
K+V payload = 2 * context_tokens * kv_heads * head_dim * sizeof(FP16)
logical gather traffic = payload read + payload write = 2 * payload
```

示例：单层、16K context、8 KV heads、head_dim=128，payload=64 MiB，逻辑读写合计=128 MiB。该值是算法逻辑流量，不等于实测 HBM 流量；后者需要硬件 profiler，cache 命中会改变结果。乘以实际层数/sequence/迭代次数才得到模型层面的总量。

### 5.2 Model Forward 微基准

不含 HTTP、tokenize、等待队列；固定已有前缀、输入 token、position 和 segment 列表，比较 A1 fanout、A2 packed-contiguous、A3 packed-gather、A4 packed-direct。

decode 用相同输入/相同 cache snapshot，通过预先准备独立 cache 或在同步后 rollback 保持上下文长度不随迭代增长。恢复、重新 prefill、allocator 及初始化不计入固定 forward 测量，但在单独测试中报告其成本。持续增长 decode 另设场景。

记录：

- forward host wall latency；prefill token/s、decode token/s。
- CUDA Event 的反量化、projection/GEMM/GEMV、attention、gather GPU 时间。
- logits D2H、同步与 host orchestration 成本。
- CUDA launch 数、底层 forward 数、stream sync 数、平均 sequence/token rows。
- 峰值显存及 weights、scratch、KV pool、page table 的分项。

CUDA Events 适用于同一 stream 的 GPU 时间；现有同步 `cuMemcpyDtoH` 的 host wall 时间须单独采集，不能强行用 event 给它分项。若实验引入 pinned async D2H，所有比较版本一起采用，并保留同步版对照。

使用 Nsight Systems/NVTX 做短程诊断、统计 kernel/反量化路径；profiling 与正式吞吐测量分开，避免 profiler 改变运行开销。

### 5.3 服务端端到端

真实 HTTP 请求发往两种 completion endpoint，客户端从计划到达时刻到 SSE token 接收记录：

- 成功生成 token/s、成功 request/s；错误/取消/rejection 分开统计。
- TTFT、ITL、E2E p50/p95/p99，含按 prompt 长度分组的分布。
- queue wait、prefill/decode、客户端分词/输出开销。
- GPU utilization、GPU memory、KV pool utilization。
- 取消响应时延和资源回收时延，以及最终页回收状态。

用 open-loop 到达生成器保持给定 arrival trace，不用“上一个请求结束后再发”的客户端替代目标负载。记录计划到达、实际提交、收到首 token、末 token、终止、回收 ack 等时刻；实际提交延误必须报告，避免 coordinated omission。

ITL 是客户端可见的 token/event 间隔；SSE 可能合并、暂扣文本或网络聚合，需同时记录服务内部 token-ready 时间。Channel 读取间隔、GPU decode 时间和 SSE 可见 ITL 不能混成同一指标。取消 endpoint 不等于 cache 已回收，以 scheduler 的资源释放事件为终点。

## 6. 容量实验与性能实验分离

**固定 token 容量不等于固定物理显存。** 当前 GPU 页池预分配所有物理页，页号归还只是供复用，不会缩小底层 GPU allocation。[Pool 源码](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Cuda/CudaPagedKvCacheFactory.cs#L42-L56)

性能实验固定可准入序列/逻辑 KV token 上限，分别报告各版本实际显存；A3/A4 还需固定完全相同的 pool tokens。不能强迫 A2 的每请求连续 cache 与分页池拥有相同内部对象，也不能把 A3 gather scratch 挤占容量的影响隐藏掉。

容量实验反过来固定总显存预算：weights 和其他 scratch 计入预算，然后扫 active sequences、请求长度和 max_tokens，测：

- 最大可成功准入 sequence 数与峰值 live KV tokens。
- 实际缓存 token / 分配 token 的比例、尾页浪费和未用 reservation。
- page pool used/free、分配失败、rejection。
- continuous memory 的未用长度；paged metadata 与 gather scratch 开销。

分两个连续基线：① 每请求分配完整模型 context；② 每请求按同样的 prompt+max_tokens 保守需求分配。否则“分页容量收益”可能主要来自连续基线过度预留。

当前 conservative reservation 不能回收尚未使用的输出预算供其他活跃请求准入，需把物理页碎片减少与准入策略限制分开解释。

## 7. 模型与 Workload

| 模型层级 | 用途 | 前置条件 |
|---|---|---|
| SmolLM-135M Q8_0 | correctness、smoke，与历史记录对照 | 已有本地文件及哈希 |
| Llama/Qwen 3B Q4_K 系列 | 中等模型、动态小 batch 拐点 | 确认对应 GGUF 和 tokenizer/结构支持 |
| Mistral/Llama 7B/8B Q4_K 系列 | 代表性量化 decode | 确认模型能放入 weights+KV+scratch 预算 |

“Q4_K”文件可能含 Q6_K 等混合 tensor，记录每个 tensor 的实际 dtype/布局。不能把两个不同模型的吞吐差当成一个消融变量。Qwen CUDA 精度与 sliding-window 差异要先过正确性验证，再纳入性能结论。

| Workload | 设计 | 主要指标 |
|---|---|---|
| Burst | 同时到达 32 请求，另扫并发 1/4/8/16/32 | batch occupancy、吞吐、TTFT |
| 固定间隔 | 平均间隔 5/10/20/50 ms | 动态 batch size 与队列稳定性 |
| Poisson | 使用相同 seed 的指数间隔，均值同上 | 在线随机到达行为 |
| 混合长度 | 50%：128/32；30%：1K/128；20%：8K/256（prompt/output） | 长短请求相互干扰、公平性 |
| 稳态 | baseline 能力的 30%/60%/80%/95% 负载 | SLA、排队、取消、资源回收 |

超出 context 的配置标为不支持，不静默截断后仍宣称测了 8K。token 长度按目标 tokenizer 生成并记录最终 token IDs；不同模型有独立固定 corpus，同一模型的所有版本重放完全相同的输入。

稳态主比较使用 **同一绝对 arrival rate**，基于公平串行 A0 对该 workload 的容量估计设置；按各版本自身容量归一化的结果另表，不与主比较混用。5 ms 等固定间隔可能超过设备能力，过载场景应单列，报告 admitted rate、队列增长和 rejection，而非宣称稳态。

代表性 3B/7B、active=1/2/4/8 的 projection sweep 单独验证量化拐点：对照多次 direct quantized GEMV、临时反量化+HGEMM；可增加常驻 FP16 权重副本，但它改变显存预算，必须作为额外版本报告。

## 8. 实验控制、重复与统计

所有版本固定：模型文件/SHA256、GPU/driver/CUDA/.NET、precision、prompt IDs、greedy、seed、可准入序列数、token budget、KV 逻辑容量、warm-up、采样与输出配置。A0 是串行语义，其 active sequence 数为 1；其余版本共用相同并发上限。

**语义固定不等于 kernel path 相同。** A1 的 quantized GEMV 与 A2 的反量化 HGEMM是被测 Packed 机制的一部分；两者的 dispatch、反量化量及精度必须如实记录。

每配置预热至少 10 次、正式重复 5～10 次；模型/服务运行每轮至少 30 秒，kernel 用足够批量迭代达到稳定 event 时间。微基准可适配总时长，不能用一次几微秒 event 代表正式结果。

随机化或平衡 A0–A4 的运行顺序，配对用同一输入与 arrival seed，记录功耗/温度/时钟与是否 throttle。避免永远串行先跑、batch 后跑造成 JIT/cache/热状态偏差。GPU 不运行其他任务；host CPU 核数、affinity、客户端负载生成器资源也固定。

每轮 raw result 保存，报告 run-level median、min/max 和配对 ratio；如提供 95% CI，以独立 run 或独立 arrival trace 为重采样单位。不能把同一运行内数万个相关 token 当成数万个独立样本。

延迟 percentile 每轮先计算，再报告 run-level 汇总；若合并请求重算 percentile，需明确标注。p99 样本不足时报告样本数与不确定性，不用三个请求的 p99 支撑 SLA。

## 9. 正确性与回收门禁

性能前所有新增版本通过：

1. 同一模型/请求，serial 与 batch greedy token IDs 一致。
2. A3/A4 同源 attention logits 在预先定义的精度门限内，固定 max absolute/relative error 与 near-zero 处理。
3. MHA/GQA/MQA、partial page、跨 tile、随机/交错页。
4. 相同 causal/window；window=1、17、128、4096。
5. chunked 与 unchunked prefill、position/cache length、last-row logits 一致。
6. EOS、max_tokens、UTF-8、stop-string 和请求采样/constraint 状态隔离。
7. cancel/error/slow consumer 后等待 GPU 完成并释放 cache。
8. run 结束清空请求：`free_pages == total_pages`；连续 KV 检查 live cache allocation=0。不要只检查 free IDs，不检查 event/device buffer 泄漏。
9. Compute Sanitizer 针对新增路径通过：

```bash
compute-sanitizer --tool memcheck dotnet /path/to/ablation-harness.dll
compute-sanitizer --tool racecheck dotnet /path/to/ablation-harness.dll
```

以上是未来 harness 的命令模板，当前没有该程序集。racecheck 聚焦设备 shared-memory hazard，不能替代 C# 队列与页表 ownership 的并发验证。Sanitizer 不与性能测量同跑。

不同 GEMV/GEMM 数值路径发生 argmax 翻转时，不能放宽“token IDs 一致”后继续算作通过。记录失败模型和输入，诊断精度/position/数学实现；失败配置不给出有效加速结论。

## 10. 结果表与解释规则

正式结果当前全部待测。保留已有 1.73× 记录，独立于以下表，不将其移入任一新消融行。

| Variant | tok/s | TTFT p50/p95 | ITL p50/p95 | 底层 Forward 次数 | Peak VRAM |
|---|---|---|---|---|---|
| A0 Serial | 待测 | 待测 | 待测 | 待测 | 待测 |
| A1 Scheduler | 待测 | 待测 | 待测 | 待测 | 待测 |
| A2 Packed | 待测 | 待测 | 待测 | 待测 | 待测 |
| A3 Paged+Gather | 待测 | 待测 | 待测 | 待测 | 待测 |
| A4 Paged+Direct | 待测 | 待测 | 待测 | 待测 | 待测 |

额外表分别存 E2E p50/p95/p99、queue、成功/错误/取消数、launch、sync、显存分项及容量扫描，避免主表无限扩宽。

| 配对 | 吞吐归因 | 解释边界 |
|---|---|---|
| A1/A0 | 调度策略组的条件效应 | 子策略用 A1 子消融分离 |
| A2/A1 | packed execution 的条件效应 | 包含 dispatch、反量化、sync/D2H 变化 |
| A3/A2 | 分页+gather 相对连续 KV 的效果 | 容量与延迟分别报告 |
| A4/A3 | direct pages 取代 gather 的效果 | 同源 math、pool/layout 必须相同 |
| A4/A0 | 完整实现相对公平串行的收益 | 不能用旧 1.73× 代替 |

```text
throughput_ratio(X,Y) = X.success_tokens_per_second / Y.success_tokens_per_second
ttft_change(X,Y) = (X.TTFT - Y.TTFT) / Y.TTFT
itl_change(X,Y) = (X.ITL - Y.ITL) / Y.ITL
```

同一分位数的延迟差可作为条件变化报告（ms 和百分比），但不同 percentile 的相邻差不能拆成逐请求可相加的组成时延。queue/prefill/decode 分项使用每个请求的时间戳，而不是各自 p95 相加。

## 11. 最小可执行优先级

先完成统一数学、CUDA Event、gather 和 contiguous packed 测试接口，再按以下顺序：

1. **Kernel 最小 G0/G1/G2**：contiguous / paged+gather / paged+direct，context=512/4K/16K，page=16，连续/随机页；分开测 single-sequence query rows 和多 sequence decode。
2. **Model G0/G1/G2**：3B Q4_K 与 7B Q4_K，active sequences=1/4/8、每 sequence decode row=1，保持 projection 相同。G1/G2 若当前实现不具备，不以既有 benchmark 代替。
3. **A1 fanout 与 A2 packed**：拆出计算共享贡献，先 burst 再固定间隔。
4. **HTTP/SSE A0–A4**：再加入混合长度、open-loop 稳态和 cancellation。
5. **容量扫描与长稳实验**：固定 VRAM budget，检查 fragmentation、reservation 及回收。

该最小方案先回答分页存储代价、gather 时间/流量和 direct pages 的 context/batch 拐点，不覆盖 Scheduler 的全部贡献。

当前代码缺口：Fanout wrapper、CUDA contiguous packed 入口、CUDA gather、同源 attention body、CUDA Event interop/计时器、组件 profiling、HTTP arrival replay 和统一 raw-result 导出。CUDA `ForwardBatch` 当前要求 GPU paged cache，不能直接拿连续 KV 跑 A2。[入口限制](https://github.com/Cheongroach/dotLLM/blob/1ceba8c98f841bfa03f704c3aab32949aae7983d/src/DotLLM.Cuda/CudaTransformerModel.cs#L147-L171)

本章是实现计划，未添加上述代码或运行 GPU 任务。

## 12. 预注册假设与交付证据

| 假设 | 待验证预测 | 支持或推翻所需证据 |
|---|---|---|
| H1 | 低 context、单活跃序列时 contiguous 最快 | G0/G1/G2 同源 event 时间 |
| H2 | context 增大后 direct pages 可抵消 gather 开销 | context sweep、配对 crossover |
| H3 | 随机页降低 locality，但保留避免 gather 的收益 | 同输入不同页表、GPU 时间与实测流量 |
| H4 | 连续批收益主要来自 packed projection | A1/A0、A2/A1 与组件时间 |
| H5 | 3B/7B 小 batch 下全量反量化可能回退 | GEMV/HGEMM 与 dequant sweep |
| H6 | 吞吐增加可能伴随单请求 ITL 上升 | 同绝对 arrival rate 的 HTTP/SSE 分布 |

H2/H3 只是预测：gather 时间可能受 cache、window、布局影响而不呈严格线性，也不保证 direct 在随机页上获胜。没有收益同样是有效结果。

正式交付至少包括：版本 commit 与开关、模型 manifest/SHA256、固定 token corpus、arrival trace、页表 seed、硬件/runtime/flags、原始 per-run 结果、正确性与 sanitizer 记录、组件时间、显存分项和失败配置。

面试或 PR 的结论句应采用：“在指定模型、context、到达率和 batch 范围内，A4/A3 测得多少；在其他区域不成立”。消融完成前，只能介绍实验设计和历史组合收益，不能声称已获得模块贡献比例。
