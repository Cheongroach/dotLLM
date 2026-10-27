# dotLLM 连续批处理与 PagedAttention：面试讲稿

## 一句话定位

我在一个原生 C#/.NET LLM 推理引擎中，把“请求级串行生成”升级为真正的
iteration-level continuous batching，并在 CPU 与 CUDA 两端实现直接读取物理 KV
页的 PagedAttention。核心价值不是增加一个队列，而是打通调度、模型前向、内存
管理、采样状态、流式 API 和 CUDA kernel 的完整执行链路。

## 60 秒版本

原系统虽然能同时接收 HTTP 请求，但模型执行仍以完整请求为单位串行化；decode
阶段每次只计算一个 token，GPU 的矩阵计算退化成低算术强度 GEMV，后来的请求还会
把前面请求的完整生成时间算进 TTFT。

我做了三层重构。第一层是调度器：采用单计算所有者，每轮优先为所有 decoding
请求分配一个 token，再用剩余预算做轮转式 chunked prefill，新请求可以在旧请求
完成前加入。第二层是模型接口：把不同请求打包成无 padding 的 token segments，
共享 QKV、FFN 和 LM head 投影，但 attention 按 segment 隔离。第三层是 KV：
每个请求只持有逻辑页表，CPU 和 CUDA attention 直接访问共享物理页，并用 tiled
online softmax 避免 gather 和完整 score matrix。

我没有只跑通 happy path。测试覆盖取消、慢消费者、停止串跨 token、独立随机种子
与 constraint、异常回收、跨页 GQA/MQA/MHA；H20 上用 double 标量参考验证 CUDA
内核，并用真实 SmolLM 验证 serial/batched token 完全一致。32 个请求、每个 64
token 的三轮实验中，4 个活跃序列把吞吐中位数从 347.70 提高到 600.75 token/s，
约 1.73 倍；同时我明确记录了 ITL 从 2.87 ms 增至 6.70 ms 的代价。

## 三分钟版本

### 1. 先定义问题

连续批处理解决的是迭代级资源利用率，不是 HTTP 并发。传统请求锁会让一个请求
占用模型直到生成结束；静态 batch 又要求同时到达且需要 padding，都不适合在线
生成。

### 2. 说明方案比选

我比较了四类方案：

1. 请求级串行：实现简单，但吞吐和 TTFT 最差。
2. 静态 padded batch：能共享 GEMM，但不能动态加入，长度差造成浪费。
3. 连续批处理 + gather KV：调度成立，但每层每步仍搬运全部历史 KV。
4. 连续批处理 + direct pages：调度和内存布局共同设计，最终选择。

调度侧也有三个关键选择：

- single owner，而不是让多个线程并发进入同一个模型，避免 scratch buffer、CUDA
  stream 和页表的锁竞争与数据竞争；
- packed segments，而不是 padding，保持计算量与真实 token 数成正比；
- conservative reservation，而不是运行中抢页，牺牲部分并发上限，换取活跃请求
  不因 KV 增长而失败。

### 3. 讲清执行机制

每轮先清理取消请求，再按优先级和 worst-case KV 容量准入。所有 decode 请求先各取
一个 token slot；剩余 `MaxBatchedTokens` 由 prefill 请求轮转分配，每个请求最多
取一个 `PrefillChunkSize`。

模型收到的是：

```text
tokens    = [A_decode, B_decode, C_prompt_chunk..., D_prompt_chunk...]
segments  = [(A, 1, cacheA), (B, 1, cacheB), (C, n, cacheC), (D, m, cacheD)]
```

embedding、QKV、O、FFN 和 LM head 对 packed rows 统一计算；attention 根据 segment
选择自己的页表和 causal range，因此不会发生跨请求注意力。模型只返回每个 segment
最后一行 logits，避免产生 `tokens × vocab` 的巨大中间结果。

### 4. 讲 PagedAttention

逻辑位置 `t` 的物理地址是：

```text
physical_row = block_table[t / block_size] * block_size + t % block_size
```

CPU 路径直接从 Float32 物理页读取，按 attention head 分配线程，并用
`TensorPrimitives` 做 dot 和 multiply-add。

CUDA 路径预分配 Float16 物理页池，每个请求持有 device block table。新 K/V 通过
scatter kernel 写入物理页；attention kernel 把 Q 放入 shared memory，按 256-token
tile 扫描 KV，用 FP32 online softmax 维护 running max、normalizer 和 value
accumulator。共享内存复杂度与上下文长度解耦。

### 5. 证明工程价值

我的验证不是“能生成文字”：

- packed 与独立请求的 logits/token parity；
- CPU/CUDA 对 double scalar reference；
- 非连续页、跨 256-token tile、MHA/GQA/MQA、sliding window；
- 取消、异常、慢 SSE 消费者后的页回收；
- 相同 seed 和 constraint clone 的请求隔离；
- H20 三轮负载测试，同时报告吞吐、TTFT 和 ITL。

## 面试中主动讲的取舍

### 为什么不用每请求并行线程

模型持有复用的 unmanaged scratch、CUDA stream 和 cuBLAS handle。多线程进入模型
意味着大范围加锁或复制整套状态。单 owner 把并发移到调度层，使状态转移线性化，
也让页回收点可证明。

### 为什么 Decode 优先

在线推理中 decode 影响 ITL，prefill 影响 TTFT。先保证每个 decoding sequence
获得一个 slot，再让 prefill 消耗剩余预算，可避免长 prompt 阻塞已有对话。

### 为什么不用完整 attention score

完整 score matrix 的临时空间随上下文增长。online softmax 按 tile 更新：

```text
new_max = max(old_max, tile_max)
old_acc *= exp(old_max - new_max)
tile_weight = exp(score - new_max)
```

因此只需固定大小 score tile 和 head-sized accumulator。

### 为什么不是所有场景都宣称提速

CPU 512-token 上下文中，页表间接寻址比 gather 后连续访问慢，测得约 0.62 倍；
4K/16K 上下文才分别约 1.23/1.17 倍。H20 连续批处理提高总吞吐和 TTFT，但增加
单请求 ITL。这说明优化目标是多租户服务吞吐，不是单请求最低延迟。

## 自我 Review：面试时不要回避

1. CUDA server 默认开启 paged，65,536 token 的固定页池会为典型 7B 配置预占约
   8 GiB KV 显存。应仅在 continuous batching 时默认启用，或按显存预算自适应。
2. 非 batching CUDA 路径仍启用 prefix cache，但 `TextGenerator` 不支持复用
   `CudaPagedKvCache`，会保留无效缓存并可能耗尽有限页池。
3. 量化 CUDA 在 batch rows 大于 1 时会逐投影全量反量化权重再做 HGEMM。需要
   batched quantized GEMM，或根据 batch size 在“多次量化 GEMV”和“反量化 GEMM”
   之间设置性能拐点。
4. 新 paged CUDA kernel 的 sliding window 是严格 W 个 token，而旧 contiguous
   CUDA kernel 实际包含 W+1；切换 `--no-paged` 会改变结果，应统一语义并补 parity。
5. 当前分支远超 issue #256 的 MVP 范围，应拆成 CPU direct pages、CUDA pages、
   packed model API、scheduler/server 四个可独立审查的 PR。

主动指出这些问题不是减分项。它体现的是：你知道实现、验证、产品默认值和开源
交付是四个不同层次。

## 高频追问

### PagedAttention 是不是照抄 vLLM？

设计原则参考了 vLLM 的逻辑页表和物理块池，但实现适配了 dotLLM 的约束：
CPU Float32 unmanaged tensor、`TensorPrimitives`、CUDA Driver API + PTX、单 stream、
GGUF 量化权重，以及现有 `IKvCache` 生命周期。真正的工作是把这些约束组合成
可验证的端到端系统，而不是复述算法名。

### 为什么不做 preemption？

当前选择 conservative reservation，让任何已准入请求都能增长到声明上限，避免
swap/recompute 引入的状态恢复复杂度。后续可以加入基于优先级的 recompute
preemption，但需要同时定义公平性、重算成本与流式 SLA。

### 最大的技术风险是什么？

不是调度循环，而是跨层语义一致性：position、cache length、最后一行 logits、
stop token 是否计数、CUDA stream 完成前能否复用页。我的测试围绕这些不变量设计。

### 你个人的独特贡献是什么？

我把一个算法概念落到原生 .NET 推理栈，跨越 API、调度器、模型前向、无托管内存、
SIMD 和 CUDA PTX；同时建立了真实模型 parity、标量参考和 H20 性能闭环，并能明确
指出当前版本的资源模型与性能拐点。
