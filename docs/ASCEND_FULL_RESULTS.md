# DotLLM Ascend 单卡完整验收结果

2026-10-24，工作分支 `prepare/ascend-910b-bringup`，基线 `d88040451d7db56e5dfef9d5754ad0955b0f7fe5`，代码尚未提交。**910B4 上完整硬件套件 16/16 阶段 PASS，测试进程退出 0**。已交付 dense Llama/Qwen2 的 GGUF F32/F16/Q8_0 加载、prefill/decode、连续 KV、CLI run/chat/serve 和 OpenAI SSE。安装与合约见 [ASCEND_RUNTIME](ASCEND_RUNTIME.md)，原始记录和机读统计见 [证据目录](ascend-results/20261024/) 与 [summary.json](ascend-results/20261024/summary.json)。

当前默认精度为 **FP32 展开权重与激活、FP16 连续 KV、FP16 RoPE 表**。Q8_0 是源文件存储格式，加载后执行 FP32 Matmul。该版本是实验性正确性基线，未达到推理加速目标。

## 环境与可追溯性

| 项目 | 实测/记录 |
| --- | --- |
| 硬件 | 1 × Ascend910B4-64G，设备名 910B4-1，logical ordinal 0，physical ID 6 |
| Worker | Debian 12 x86_64；g++ 12.2.0；CANN toolkit/ops 8.3.RC1；driver/npu-smi 25.5.0 |
| .NET | 执行 .NET/ASP.NET 10.0.0；managed build SDK 10.0.100 |
| 容器限制 | CPU quota 200000/100000 µs，即 2 核；内存 17179869184 B（16 GiB） |
| 固件 | `npu-smi ... -t board` 实际返回 Firmware Version=NA；独立固件版本未取得 |
| 自有 native ABI | 3，native 编译与 required-symbol/managed resolver 验证通过 |
| 平台执行 | cn，group [internal resource]，cluster [internal resource]；run [internal resource ID] / trial 438207790 / pod 5389268523 |
| 程序退出 | `exit.json` = 0，`stages.jsonl` 16 PASS；Pod 是外层保活任务，套件退出码独立记录 |

[environment.json](ascend-results/20261024/environment.json) 保存环境摘要；归档已移除主机标识、芯片唯一标识和运行环境原始输出。归档不含平台凭据、SSH 私钥、模型或二进制大文件。

模型来源与 SHA256：

- `QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf`：`7d4afa1d8a5587f4beb8b75dab755f90d13d3ae2b8b9f148545b414ecfb9123a`。
- `Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf`：`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`。
- 自有 `libdotllm_ascend.so`：`9d00adc5d58b994e29b92d931672c15941ab3140228e5b89b2feb30048ce5bca`。

[source-manifest.json](ascend-results/20261024/source-manifest.json) 记录受测未提交源码的逐文件哈希、ABI/precision 与官方 ASP.NET runtime URL/SHA512；[SHA256SUMS](ascend-results/20261024/SHA256SUMS) 记录受测包所有文件（包括模型、fixture、golden、managed DLL）的 SHA256。native build/symbol 记录及 native binary hash 单独归档；SHA256SUMS 的相对路径对应原验证包，不代表证据目录包含大文件。

## 验收范围与计数

16 个阶段是执行器阶段数，不是所有断言数。归档包含 **318 条 enforced logits 数值比较记录**，最终再次核对其各自 max/RMS/cosine 门槛均通过；其他 shape、ownership、失败路径和 API 断言另计：

| 阶段 | 已执行覆盖 | 结果 |
| --- | --- | --- |
| kernels | FP16 Linear 7 shape × 3 输入；RMSNorm、bias/residual、SwiGLU、两种 RoPE/partial rotary、MHA/GQA/MQA/causal offset；FP32 Linear/Norm/残差/cast 和 dtype 拒绝；copy/ownership、跨线程和多 session 生命周期 | PASS |
| capability | 在 native 初始化前拒绝 6 类非法/未支持配置 | PASS |
| smol、qwen | 各 1 次 8-token prefill + 32 步相同 token teacher forcing；split/rollback/reset；cache/device/model 所属、token/position 边界 | 2 PASS |
| tiny-{llama,qwen2}-{F32,F16,Q8_0} | 6 个独立 fixture，每个 24 步 teacher forcing；row-major 映射、Llama tied head、Qwen2 各 projection bias | 6 PASS |
| smol-text、qwen-text | 各 13 case：英文/中文/代码 × 1/32/128/512 token + EOS；独立 logits、split、bit-exact rollback、容量拒绝；每 case 热测 3 次 | 2 PASS |
| memory、nonfinite、fault | 3 次真实 unload 内存回归；非有限权重加载失败清理；真实 ACL 分配配合合成 bridge failure 验证 faulted-session 拒绝与释放 | 3 PASS |
| service | 18 项检查：每模型 3 次 CLI run、CPU 回归、4 类非法 flag、PTY 两轮 chat、health/tokenizer/warmup、completion/repeat/multiturn、两种 SSE、断连后恢复、双客户端串行隔离、非法请求、reload 恢复 | PASS |

两种 SSE 均检查 terminal finish_reason 和 `data: [DONE]`；服务复用既有串行 gate，双客户端测试证明独立状态和可完成性，不代表 Continuous Batching。故障注入是合成算子错误，不是物理设备重启或 driver crash 恢复。硬件阶段未因缺库/缺卡而 skip。

相关 managed 项目构建成功，0 warning / 0 error；[final-build.log](ascend-results/20261024/final-build.log) 保存构建器结果，后续 V8 precision/dtype 补充测试构建和包哈希也保留。SDK 工具在此宿主上偶发 crashpad/memfd/metadata 异常，不能把目录简写 CLI 的退出 22 当成编译通过；最终 DLL 由明确 `.csproj` 构建，且完整 CLI/API 硬件阶段实际执行成功。

本地针对 GGUF、KV/rollback/prefix、CLI 参数与服务验证的回归：**168 PASS、2 SKIP、0 FAIL，总 170**，见 [unit-final.log](ascend-results/20261024/unit-final.log)。2 个 CUDA 硬件 case 因无 CUDA GPU 跳过；未宣称 CUDA 实机或全仓库所有单测通过。

## 数值门槛与结果

独立参考使用同一 GGUF 解码后的 FP32 权重与相同 token IDs，在 NumPy 中执行 FP32 dense math；不使用原始 HF 权重代替量化 GGUF，也不以共享 CANN 的 torch_npu 输出作为唯一金标准。

固定门槛：`RMS(actual-reference) <= 0.03 + 0.015 × RMS(reference)`，`max_abs <= 0.25`，`cosine >= 0.999`。KV split 相对完整重算使用零 atol RMS、max_abs ≤ 0.05；文本 rollback 与此前 cached 结果必须 bit-exact。误差和阈值全部保留，`cpu_vs_numpy_prefill` 标记 `enforced=false`，不计入通过的精度比较。

| 模型 | 对照集合 | logits 比较数 | 最坏 max_abs | 最大 RMS(error) | 最低 cosine |
| --- | --- | --- | --- | --- | --- |
| SmolLM-135M | 8-token prefill + 32-step teacher forcing | 33 | 0.012556076 | 0.007312979 | 0.9999991813 |
| SmolLM-135M | 13 English/Chinese/code/EOS cases | 13 | 0.019275665 | 0.003245958 | 0.9999997393 |
| Qwen2.5-0.5B | 8-token prefill + 32-step teacher forcing | 33 | 0.133827686 | 0.025929799 | 0.9999659763 |
| Qwen2.5-0.5B | 13 English/Chinese/code/EOS cases | 13 | 0.003406882 | 0.000673554 | 0.9999999675 |

13 case 的 split 最大误差：SmolLM `0.0110497475`、Qwen `0.0144920349`；各 13 次 rollback 的 max/RMS 误差均为 0。Top-1/Top-5 和 centered error 是辅助诊断；例如 teacher 集合最小 Top-5 overlap 为 0.8，未把接近并列排名当作唯一判据。

首次 all-FP16 权重/激活方案在冻结门槛下失败：SmolLM 中文 1-token max_abs≈1.276、RMS≈0.254；Qwen 英文 1-token max_abs≈0.370。独立数学及中间张量分析指向舍入/跨层累计误差，因此改为 FP32 correctness path，门槛未放宽。原 FP16 kernel smoke 继续独立保留；不能由 kernel PASS 推导 FP16 全模型 PASS。

## 性能与内存

同一 worker、同一 GGUF、英文 prompt `Explain what a key value cache does in a transformer.`，`max-tokens=32, temp=0, seed=42`，batch=1。每模型/后端运行 3 个独立 CLI 进程；CPU 显式 `--threads 2`，Ascend 使用同一 2 核 quota。SmolLM 输入 11 token，Qwen 输入 12 token，均生成 32 token且各次 CPU/NPU 文本一致。

以下为中位数 `[最小, 最大]`。CLI `total_ms` 是 generation interval，不含模型加载、进程启动或 HTTP 网络耗时；decode_tok_s 按 CLI 已有口径计算，首 token 来自 prefill，其余 31 token 计入 decode。prefill 为进程冷调用，包含规划/JIT等成本，不能当作纯 NPU kernel 吞吐。

| 模型 | 后端 | load ms | cold prefill ms | decode tok/s | generation total ms |
| --- | --- | --- | --- | --- | --- |
| SmolLM-135M | Ascend | 1361.80 [1342.00, 1371.60] | 662.70 [650.30, 676.70] | 10.69 [10.58, 10.72] | 3596.40 [3593.80, 3648.50] |
| Qwen2.5-0.5B | Ascend | 2246.40 [2232.10, 2291.10] | 705.40 [702.90, 715.70] | 8.96 [8.95, 9.23] | 4207.10 [4107.20, 4222.80] |
| SmolLM-135M | CPU | 217.20 [211.20, 223.40] | 701.70 [688.10, 710.00] | 60.98 [60.12, 61.70] | 1243.10 [1243.00, 1257.50] |
| Qwen2.5-0.5B | CPU | 633.50 [626.40, 633.90] | 1006.00 [988.50, 1026.60] | 15.68 [15.51, 15.96] | 3005.40 [2988.90, 3064.60] |

CPU 使用压缩 Q8 权重及现有 repacking/量化 activation 路径，NPU 展开为 FP32，二者是当前实现的实际对照，不是相同精度的硬件能力比较。本版 NPU decode 明显慢于 CPU；不宣称 speedup，也未运行 llama.cpp CANN、vLLM-Ascend 等引擎 benchmark。

热 prefill 使用已加载模型，在独立完整 Forward 后再同步测量 3 次；包括 host descriptor/规划、device 操作、同步和最后 logits D2H，排除采样/HTTP。英文 case：

| 模型 | prompt token | synchronous Forward ms，中位数 [min,max] | token/s |
| --- | --- | --- | --- |
| SmolLM-135M | 1 | 74.86 [73.94, 76.18] | 13.36 |
| SmolLM-135M | 32 | 390.71 [382.50, 404.58] | 81.90 |
| SmolLM-135M | 128 | 1228.73 [1215.69, 1291.57] | 104.17 |
| SmolLM-135M | 512 | 4971.86 [4953.56, 5000.86] | 102.98 |
| Qwen2.5-0.5B | 1 | 96.34 [88.78, 111.79] | 10.38 |
| Qwen2.5-0.5B | 32 | 456.28 [448.81, 477.37] | 70.13 |
| Qwen2.5-0.5B | 128 | 1404.90 [1398.46, 1407.89] | 91.11 |
| Qwen2.5-0.5B | 512 | 5259.38 [5226.14, 5273.26] | 97.35 |

设备权重常驻：SmolLM `538060032 B`（513.13 MiB），Qwen `2520669696 B`（2403.90 MiB）；tied output 不重复计量。512-token cache 分别 `11796480 B` / `6291456 B`；该长度 tracked compute 分别 `24646144 B` / `57321472 B`。FP16 KV 公式：`2 × layers × maxLength × kvHeads × headDim × 2 B`。这些是程序跟踪 buffer 的统计，**不是 driver/runtime workspace 峰值**。

unload baseline free=`64946835456 B`，3 次卸载后的最坏 retained=`544768 B`，低于固定 64 MiB budget；非有限权重失败前后 free 完全一致。短时回归不能证明长期零泄漏。

TTFT、HTTP 端到端 wall time、driver workspace peak、kernel vs host submission 分解、SM/AICore profile 尚未测量。每步 decode 平均 forward 成本可从 `decode_ms/31` 推得，但不替代逐 token TPOT/P99 分布。

## 复现与下一阶段

1. 按 [runtime 文档](ASCEND_RUNTIME.md) 构建 managed/native；无 CANN 的 host 可先运行 `--reference-self-test`。
2. 准备上述固定 GGUF，用 `gguf_reference.py --tiny` 生成六个 fixture，并将各源文件 `--dtype F32` 展开成 `.reference`。NumPy golden 由 `numpy_reference.py --model <F32 reference> --output <golden>` 生成；真实模型默认生成 8-token prefill + 32-step teacher sequence。
3. 测试程序集 `--emit-text <model> <cases.json>` 生成固定 text tokens；使用 `numpy_reference.py --cases <cases.json>` 生成其 golden。源模型与 reference/golden 哈希均与归档核对。
4. [run-full.sh](https://github.com/Cheongroach/dotLLM/blob/ascend/910b4-dense-baseline/tests/DotLLM.Tests.Ascend/run-full.sh) 约定包目录 `cli/ tests/ native/ascend/ fixtures/ models/ runtime.tar.gz`；包含 `fault_bridge.c`、`service_smoke.py` 和 `SHA256SUMS`。非有限 fixture 为 tiny F32 模型的 norm weight 注入 NaN，用于失败路径。脚本逐项执行并输出 `stages.jsonl` 与 `exit.json`。
5. CPU 对照命令与 [cpu-benchmark.py](ascend-results/20261024/cpu-benchmark.py) 一致；脚本中的 `/tmp/full-v6` 改为实际包目录。NPU 对照使用 [service_smoke.py](https://github.com/Cheongroach/dotLLM/blob/ascend/910b4-dense-baseline/tests/DotLLM.Tests.Ascend/service_smoke.py) 中 CLI run 的固定输入。

当前完成 M0–M4 的单卡 correctness/CLI/API 基线；firmware 独立版本记录仍缺失，融合/低精度高性能目标未完成。下一阶段先测 host/同步与算子成本，再验证 fused attention、批量 native 提交或 graph replay、权重 packing/量化 GEMV。FP16 fast path 必须单独通过同一冻结预算后才能启用；分页、prefix sharing、Continuous Batching、多卡/HCCL 和新增架构使用独立后续范围。

资源清理状态见 [resource-cleanup.json](ascend-results/20261024/resource-cleanup.json)：run 已为 `done`，Pod status=4 且已记录 stopAt，无活跃 Pod。本次验证任务已释放，此前 HL 开发机申请按用户约定保留。未创建公开 issue、commit 或 PR。
