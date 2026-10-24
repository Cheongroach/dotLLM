# Ascend M0 实施与验证记录

日期：2026-10-22～2026-10-24。分支：`prepare/ascend-910b-bringup`。基线：
`d88040451d7db56e5dfef9d5754ad0955b0f7fe5`。

实验性 runtime 和 FP16 Linear 已通过 910B4 实机 smoke：C++ 编译、链接成功，
C# 经自有桥接调用 CANN，7 组 shape × 3 次输入全部通过，进程及 Pod 退出码均为 0。
全模型 forward、GGUF 权重加载、KV-cache 和 CLI/Server 接入尚未实现。

## 2026-10-24 回验：NPU 实机结果

[任务 [internal resource ID]]([internal URL omitted])
已在 2026-10-23 03:44（UTC+8）完成，10 月 24 日取回产物并核对。
控制面 `cn`，资源组 `2079/[internal resource]`，集群 `64/[internal resource]`。
成功 trial 为 `437423234`，Pod SID 为 `5385230849`，Pod 名为
`trial-437423234-trialrun-437423234-worker-0`。平台 run/trial 状态为
`done/finished`，Pod `exit_status=0`，产物 `exit.json` 为 `{"exitCode":0}`。

| 环境/执行项 | 实际证据 |
| --- | --- |
| 硬件 | Ascend910b4-64G；npu-smi 显示 `910B4-1`、Health OK、65536 MB HBM |
| 设备编号 | npu-smi 物理编号 7；C# 使用容器内逻辑 `deviceId=0` |
| 系统 | Debian 12，x86_64 |
| npu-smi 版本 | 25.5.0；未单独采集 firmware 版本 |
| CANN runtime/compiler/ops | `8.3.0.1.200:8.3.RC1` |
| C++ 编译器 | g++ 12.2.0，NPU worker 上 CMake 编译和链接成功 |
| .NET | 10.0.0 linux-x64 runtime；managed 程序预先在开发机编译 |
| 动态链接 | ascendcl、nnopbase、opapi、driver/libascend_hal.so 均解析成功 |
| 生命周期 | H2D/D2H 位一致、跨 session 拒绝、线程切换、销毁、重新初始化通过 |

每组误差汇总三个不同输入。参考使用实际上传的 FP16 输入，以 scalar double 累加，
再写入 FP32 参考输出。所有元素有限，逐元素满足
`abs(error) <= 0.002 + 0.002 * abs(reference)`；未根据本轮结果放宽阈值。

| M | N | K | 最大绝对误差 | Normalized RMSE | 结果 |
| --- | --- | --- | --- | --- | --- |
| 3 | 5 | 7 | 0.000103354454 | 0.000222761212 | PASS |
| 1 | 19 | 37 | 0.000319480896 | 0.000183843207 | PASS |
| 5 | 33 | 65 | 0.000486254692 | 0.000208996464 | PASS |
| 16 | 32 | 64 | 0.000961542130 | 0.000202423126 | PASS |
| 1 | 576 | 576 | 0.001923561096 | 0.000208728258 | PASS |
| 7 | 129 | 1025 | 0.003138542175 | 0.000210609787 | PASS |
| 2 | 11 | 13 | 0.000199973583 | 0.000205579725 | PASS |

ND 权重转置 stride `[1,K]` 与 executor 的 repeatable→单次执行→显式销毁路径
已在这套 CANN/硬件组合通过上述测试。结果不代表故障注入、长期无泄漏、
其他 CANN 版本、全模型质量或性能已经验证。

日志最终记录：

```json
{"status":"PASS","scope":"Ascend FP16 Linear and runtime smoke","cases":7,"iterationsPerCase":3}
```

取回的 `SHA256SUMS` 与提交包完全一致，包内 15 个文件重新校验通过，
native 源码与 runner 也和当前工作区一致。实际运行的是新包
`input/package-20261023.tar.gz`，不是下文历史记录中的旧包。

产物目录：

```text
[internal storage path]
```

其中包含 `smoke.log`、`exit.json`、`SHA256SUMS` 和 `libdotllm_ascend.so`。
原始产物已取回 `/tmp/dotllm-ascend-research/npu-hj-results-20261024/`。

| 产物 | SHA256 |
| --- | --- |
| 提交归档 | `23b5cdb7a9f409e427ba0fb7ba4ab8eb0025a22e578494b755c69cc0db3d0dce` |
| smoke.log | `f3f1c648c0770b9b516a0bf9caba1f5ff0bc2fd01ffa72ee063e884bd6d47ff0` |
| native 库 | `22348053dd4a6d6bca2cb2d3657ce3140be29fb4b0920133acc1d10778d13941` |
| SHA256SUMS | `ea79af8dd2f35bb48f9456585d220c88246fd7973874155f0e3c78d1c66b6ee5` |

本轮状态查询 request ID：
run `2026102401345471118CFDECBFD21340A2`，
pod `202610240135061C20424FB7A9D8DB42B9`。
备用任务已经结束；HL 开发机 `[internal resource ID]` 在本轮查询时仍为 `PENDING`，
按此前约定保留申请。实机 smoke 已由 HJ worker 完成，不依赖 HL 开发机调度。

## 已落盘的实现

- `src/DotLLM.Ascend/`：AscendCL LibraryImport、共享初始化、显式 context/stream、
  FP16 tensor、同步复制、Linear、session 所有权与异常清理。
- `native/ascend/`：独立 CMake 工程、ABI 1、ND FP16 Matmul，
  `X[M,K] × W[N,K]^T`，权重转置 view 的元素 stride 为 `[1,K]`。
- `tests/DotLLM.Tests.Ascend/`：独立可执行 smoke，7 组 shape、每组 3 次不同输入，
  数值断言以及复制、双 session、线程切换、销毁与重新初始化验证。
- `dotLLM.slnx`：注册实验工程。普通 managed build 不依赖 CANN。

运行说明见 [ASCEND_RUNTIME.md](ASCEND_RUNTIME.md)。

## 本地验证

| 检查 | 实际结果 |
| --- | --- |
| Managed build | MSBuild 报告成功，0 warning、0 error |
| 外层工具退出 | 返回 1：SDK metadata 沙箱访问受限；不能写成命令零退出 |
| `--reference-self-test` | JSON PASS，命令退出 0；仅验证独立数学参考 |
| Shell syntax | `bash -n` 退出 0 |
| 初次本机 CMake | 缺少 `acl/acl.h`，配置失败 |
| 补齐头文件后本机 C++ 编译 | 使用真实 CANN 8.3.RC1 headers，`g++ -c` 退出 0；本机尚未链接 CANN 库 |
| 本机硬件 smoke | 缺少 `libdotllm_ascend.so`，FAIL、退出 1；没有静默 skip |

可复现本地构建参数：

```bash
DOTNET_EnableWriteXorExecute=0 \
DOTNET_CLI_HOME=/tmp/dotllm-ascend-research/dotnet-home \
DOTNET_CLI_TELEMETRY_OPTOUT=1 \
dotnet build tests/DotLLM.Tests.Ascend/DotLLM.Tests.Ascend.csproj \
  -c Release -p:UseAppHost=false -p:MinVerSkip=true \
  --artifacts-path /tmp/dotllm-ascend-research/m0-artifacts --no-restore
```

`MinVerSkip` 和关闭 WriteXorExecute 仅是本地沙箱构建参数，未改仓库全局设置。
早先的 106 个 CPU 数学测试也已通过，但不能作为本次 NPU 正确性的替代证据。

## 2026-10-22 历史：首次单卡任务

控制面 `cn`；资源组 `2079/[internal resource]`；镜像 VID
`d6nakm3c77u7jl17ea90`。两次均请求 1 张 `ASCEND910B4_64G`、12 CPU、
65536 MiB host memory、1 worker，不自动重启。

| 任务 | 队列 | Trial | 最终状态 |
| --- | --- | --- | --- |
| [[internal resource ID]]([internal URL omitted]) | [internal resource] Ascend 保障队列 | 436941952 | stopped / killed |
| [[internal resource ID]]([internal URL omitted]) | [internal resource] 同机房 Spot | 436944849 | stopped / killed |

两次 `check-items check` 均返回 `status=success`，没有 warn/hint。创建前保留式合并
`request_context.precheck_sid`，均在 30 分钟内提交。若预检 error 则禁止提交。

保障队列最新查询为 16/16 GPU、240/240 CPU、2048000/2048000 MiB，聚合配额已用满。
任务一直 queued、pod list 为空，遂停止并回验，再尝试 Spot。Spot 也保持 queued，
trial `ErrMsg` 为空、pod list 为空。Kelemetry CN 对其 pod 名在最近一小时查询返回
HTTP 200、0 traces；这不证明无异常，只说明没有事件记录可用于判断具体调度原因。
停止后已回验两个 run 均为 stopped/killed，没有留存排队任务。

## 2026-10-22 历史：旧复现包与追溯

测试包使用官方 .NET 10.0.0 linux-x64 runtime，约 36.7 MB。包内包含 managed
二进制、native 源码、run-smoke.sh 和逐文件 SHA256SUMS。归档 SHA256：

```text
95b0930baaab32b40e02bde141d6e08a50761ced34b1f8882ec7ed3caa67f030
```

HDFS 留存路径：

```text
[internal storage path]
```

输出目录位于同级 `output/`；两次 NPU 任务没有启动用户程序，未产生 smoke 日志。
后续普通 CPU 编译产物保存在 `output/native-regular-cpu/`。
本地原始响应、构建产物和包保存在 `/tmp/dotllm-ascend-research/`，属于临时文件。

| 操作 | 请求 ID |
| --- | --- |
| 保障预检 | 202610221029138FDBDD74F40DF530F995 |
| 保障创建 | 202610221029542A48A91DBD36CAE9B9E1 |
| 最新保障配额查询 | [internal resource ID]ED9C8B3A3E18CDEB03 |
| Spot 预检 | [internal resource ID]AF3381F612075EEBAB |
| Spot 创建 | 20261022103640852C5C48B2287F4D2326 |

## 后续实现门槛

本页顶部记录的实机 smoke 已确认 CANN 编译/链接、正常 executor ownership 路径
与 7 组 Linear 数值断言。当前里程碑是 runtime/Linear 核心链路通过。

随后实现 Norm/SwiGLU/RoPE、连续 KV 与全模型；采用相同 GGUF、相同 token IDs、
双方最后 logits 行，验证分块 prefill、多步 decode 和 rollback/recompute 等价。
目前不得宣称 Ascend 已能生成文本或具备吞吐优势。

代码仍是未提交的本地实验。上游没有创建 Ascend issue，当前准备分支尚未满足仓库
issue 驱动贡献流程；正式上游提交前需建立 issue 并切换为 issue 分支。

## 开发环境与 CPU 队列补查

2026-10-22 11:06（UTC+8）补查确认：当前 `vayne/[internal resource]` 是旧 CPU 开发机，
集群 `[internal resource]`，x86_64 Linux，8 CPU、32 GiB，无 worker。cgroup 限额与平台
规格一致；`nproc/free` 的宿主机视图不能作为容器可用配额。
本机有 .NET SDK 10.0.100、CMake、g++，但 `/usr/local/Ascend/ascend-toolkit`
不存在，`driver/` 为空，也没有 `/dev/davinci*`。

额外 CPU 算力并非必需。需要补的是 CANN 编译依赖；NPU 数值验证还需要实际
设备及匹配驱动。当前账号可见 Devbox 2.0 列表为空，不代表全组没有 NPU 开发机。
若提供一台已启动的 910B4 开发机及可访问的实例链接，可直接复用其 toolkit 和设备。

之前的 CPU 镜像构建任务 `[internal resource ID]` 已进入用户脚本，trial
`436953692`、pod `5382716503`，退出 1。stdout 明确报错：

```text
Could not find compiler set in environment variable CXX:
/usr/local/gcc-9.3.0/bin/g++.
```

这是镜像编译器路径问题；不能解释成 CPU 队列不可运行 CANN 镜像。已修正
run-smoke.sh，在 source toolkit 后选择已安装 g++，并显式传给 CMake。
原始日志在 `/tmp/dotllm-ascend-research/cpu-build-final-{stdout,stderr}.log`。

CPU-only 资源查询扩大到当前账号全部已批准成员组，返回 9 行、6 条不同队列，
所有行的 CPU 类型和零 GPU 均已核对。以下是普通 CPU 候选：

| 组 / 集群 | 队列 | 当前余量 |
| --- | --- | --- |
| rh2_cpu / [internal resource] | compute-940-hl-cloudnative-default-rh2.cpu-guarantee | 10 CPU、7168 MiB |
| rh2_cpu / cloudnative-lq | compute-688-lq-cloudnative-default-rh2.cpu-guarantee | 0 CPU、109568 MiB |

目标 `[internal resource]` 组的 CPU 行只有 HL/LF 的 `am-ray.cpu` 队列，
分别返回 2804/1929 CPU；实际创建检查要求 `head` 角色。余量只是当前聚合状态，
不保证即时调度。查询请求 ID：
`20261022110632E8B4BAA77EA6A6A6A349`（组内）、
`[internal resource ID]B1DAC78375BDE67AE7`（全部已批准组）。

普通 CPU 队列已预检成功并提交短构建任务
[[internal resource ID]]([internal URL omitted])，
trial `436959125`，申请 4 CPU、4096 MiB，无 NPU，使用同一 CANN 镜像。
该任务已执行成功，run 为 `done/finished`，pod `5382724915` 退出码 0。
日志确认 Debian 12 x86_64、g++ 12.2.0、CANN
`8.3.0.1.200:8.3.RC1`，native CMake 编译和链接通过。

`nm -D` 确认三个 ABI 导出，`ldd` 找到 ascendcl、nnopbase、opapi；
CPU 容器缺少设备驱动 `libascend_hal.so`，因此不能进行设备执行。
该结果证明目标 toolkit 能编译桥接库，不证明 NPU Matmul 数值正确。
日志和库已取回 `/tmp/dotllm-ascend-research/regular-cpu-results/`。

库 SHA256：
`22348053dd4a6d6bca2cb2d3657ce3140be29fb4b0920133acc1d10778d13941`。
初次导出的头文件归档只有 `include -> x86_64-linux/include` 软链接，
随后使用解引用打包修正。重跑 trial `436962156`、pod `5382729391` 退出 0，
run 再次为 `done/finished`。实际头文件归档 1238192 bytes，SHA256：
`9b9da2bf5dc8f41e0d8cb147c987e82075c7ae141a0a35fb0fc876f43844c04e`。

公共头文件已取回并展开在
`/tmp/dotllm-ascend-research/cann-8.3.RC1-headers/include`。
当前开发机执行以下 C++ 编译命令退出 0，生成 7936-byte object：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -fvisibility=hidden \
  -I/tmp/dotllm-ascend-research/cann-8.3.RC1-headers/include \
  -Inative/ascend/include -c native/ascend/src/ops.cpp \
  -o /tmp/dotllm-ascend-research/ops-cann-8.3.RC1.o
```

后续 C++ 编译可留在当前开发机；链接仍需要 CANN 开发库，设备执行需要 driver、
ops package 与真实 NPU。取回的头文件和动态库都没有加入公开仓库。

所需实机开发环境：优先 x86_64 Linux、至少一张空闲 Ascend910b4-64G，
匹配的 driver/firmware/CANN（首选已编译验证的 8.3.RC1）、可用设备节点和
`npu-smi info`、可访问的 SSH/开发机入口。8 CPU、32 GiB host memory 足够首轮开发；
.NET 10 runtime 已有测试包可提供，torch_npu 不属于 DotLLM 运行时依赖。

## 2026-10-23：NPU 开发机与备用实机验证

用户已授权直接申请 NPU 资源并在机器上编译。已在指定组与集群成功创建
Devbox 2.0：[[internal resource ID]]([internal URL omitted])，
名称 `dotllm-ascend-910b4-dev-20261023`，创建时间 03:00（UTC+8）。

- 控制面 `cn`，组 `2079/[internal resource]`，集群 `20/[internal resource]`。
- 1 张 `ASCEND910B4_64G`、8 CPU、32 GiB host memory；实例返回持久盘 500 GiB。
- 复用 CANN 8.3.RC1 镜像，安装 g++/cmake，设置实际可用的 CC/CXX。
- HDFS 输入以只读方式挂载到 `/mnt/hdfs/dotllm_input`。
- 实例返回 SSH 地址：
  `[internal endpoint]`，端口 22222。

03:22 查询实例仍为 `PENDING`，关联 trial `437399112` 仍 queued，pod list 为空。
没有进行非运行态 SSH 探测。03:12 组配额查询显示目标 Ascend 队列 total/used 均为
16 卡、240 CPU、2048000 MiB。配额是聚合当前状态，不能给出排队 ETA 或占用任务。
开发机保留等待调度，没有因短暂排队撤销。

同组 `[internal resource]` 的 910B4 资源查询命中后，提交备用短验证任务
[[internal resource ID]]([internal URL omitted])，
trial `437417705`，创建时间 03:24（UTC+8）。申请 1 张 910B4、12 CPU、
65536 MiB；队列 `compute-598-hj-gpuspot1-default-mlsys.public.inc-guarantee`。
预检全部 success，没有 warn/hint；预检 SID 保留式合并到
`request_context.precheck_sid` 后在 30 分钟内提交。运行脚本限时 600 秒，
不自动重启，输出单独写入 `output/npu-hj-20261023/`。

最新 managed 代码已重新 build，MSBuild 0 warning/0 error；外层工具仍因 SDK
metadata 沙箱访问返回 1。重新构建的 `--reference-self-test` 退出 0。
新包包含当前 native 源码、编译器修正后的脚本与最新 managed 二进制。
本地逐文件校验和 HDFS 回读归档 SHA256 均通过：

```text
23b5cdb7a9f409e427ba0fb7ba4ab8eb0025a22e578494b755c69cc0db3d0dce
```

新归档 36669836 bytes，位于既有 `input/` 下的 `package-20261023.tar.gz`；
本地包与请求记录位于仓库忽略目录 `artifacts/ascend-smoke-20261023/`。

03:26 备用任务仍 queued、ErrMsg 为空、无 pod/runtime unit。
Kelemetry CN 最近一小时按该 worker 名查询 HTTP 200、0 traces；此结果只表示
该时间窗未找到事件，不能证明没有调度问题。尚未获得 NPU 执行结果。

| 操作 | 请求 ID |
| --- | --- |
| Devbox 创建 | 202610230300153493769BD69BA3A3A1DC |
| HL 组配额查询 | [internal resource ID]DC465C2A8E403C3E6C |
| HJ 可用队列查询 | 20261023031846DBE7B8BC3162210326A1 |
| HJ 预检 | 20261023032314EBF8E40E10AC8D9322FA |
| HJ 创建 | 20261023032411B48C1CF2A0B9A244C278 |

验收仍以 DotLLM bridge 的真实 NPU 编译与全部数值/生命周期测试通过为准。

03:30 对备用任务尝试显式 `advanced.is_preemptible=true`，重试前预检全 success。
新 trial `437422049` 在提交 task group 时失败，没有 Pod，后端原文：

```text
uce spot cluster cannot set preemtable
```

该实际限制优先于知识库通用可抢占操作说明；不能据此把原始排队归因为缺少该开关。
已再次预检并恢复 `is_preemptible=false`，原 run ID 复用，没有新建额外任务。
03:32 回验恢复后的 trial `437423234` 为 queued，两个当前请求的 Pod 列表仍为空。
原 trial `437417705` 已 stopped/killed，失败 trial `437422049` 无运行设备。

| 操作 | 请求 ID |
| --- | --- |
| 可抢占尝试预检 | 202610230329563EFBD1CEF89D8F727B6B |
| 可抢占重试 | 20261023033023B7BA45A9268A4F67E939 |
| 恢复原配置预检 | 20261023033136A413092C130F2A154849 |
| 恢复原配置提交 | 202610230331381ABD1A770DF606E3CD44 |
| 当前 Pod 回验 | 202610230332032493E156F43C753EF982 |

截至上述 03:32 查询，保留 HL 开发机申请与 HJ 自动 smoke 任务等待调度，
当时实机验收未完成；随后 HJ 成功结果见本页顶部。Devbox 启动后可使用已准备的
`artifacts/ascend-smoke-20261023/devbox-execute-when-running.json`
经 `job-v2 runs execute-script --from-file` 执行；执行前必须确认实例 Running 且 Pod 活跃。
备用任务已包含自动编译和测试入口，调度成功后会自行执行并将日志写入 HDFS。
