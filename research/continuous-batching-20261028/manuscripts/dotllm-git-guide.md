# dotLLM 变更导航

## 仓库边界

- 外层仓库：`vllm-omni`，当前工作区干净。
- 功能仓库：`external-refs/dotLLM`。
- 验证日志、PPT、模型和临时脚本位于 `external-refs/`，不属于 dotLLM Git 历史。
- 原始未提交状态已备份为 `external-refs/dotllm-continuous-batching.patch`。

## 提交序列

| 顺序 | Commit | 内容 | 建议重点 |
|---|---|---|---|
| 1 | `2bbd9b2` | 修复 ComputeThreadPool 首次 dispatch 竞态 | 可以独立抽成 bug-fix PR |
| 2 | `d98e604` | CPU/CUDA direct PagedAttention、页池、页表、PTX | 1124 行是生成的 PTX；先看 `.cu` 和 C# |
| 3 | `a3095a6` | Packed multi-sequence forward | 共享投影、隔离 attention、last-row logits |
| 4 | `bb8e1cf` | ContinuousBatchScheduler 与请求状态 | 准入、decode-first、chunked prefill、取消与回收 |
| 5 | `b40ef25` | CLI、HTTP/SSE、模型热切换与集成测试 | 生命周期和资源所有权 |
| 6 | `1ceba8c` | Benchmark、README 与设计文档 | 性能证据和已知限制 |

## 常用查看命令

```bash
cd external-refs/dotLLM

# 先看六步故事
git log --oneline --reverse origin/main..HEAD

# 看某一步改了哪些文件
git show --stat d98e604

# 看某一步的完整代码
git show --color-moved=dimmed-zebra a3095a6

# 只看调度器
git show bb8e1cf -- src/DotLLM.Engine

# 只看 CUDA 手写 kernel，跳过生成 PTX
git show d98e604 -- native/kernels/paged_attention.cu src/DotLLM.Cuda

# 看整个分支相对上游的文件级变化
git diff --stat origin/main..HEAD

# 检查工作区是否干净
git status --short
```

## 规模解释

- 总计：43 个文件，新增 4228 行，删除 394 行。
- 其中 1124 行是 `nvcc` 生成的 `paged_attention.ptx`。
- 剩余约 3100 行包含实现、测试、benchmark 和文档，不应作为一个整体审查。
- 最大的人工模块是 scheduler 及测试，其次是 server 生命周期和 CPU/CUDA backend。

## 当前验证

- 非 GPU 单元测试：1196 passed。
- 真实模型定向集成测试：4 passed。
- `git diff --check`：通过。
- 本轮沙箱需要 `DOTNET_EnableWriteXorExecute=0`，否则 .NET JIT 会访问被限制的
  `/memfd:doublemapper`。

## 提 PR 前

当前六个提交都暂时引用 `#256`，但该 issue 只要求 scheduler MVP，范围不包含
Server、CUDA PagedAttention 和 streaming。推送前应新建设计 issue，并将提交拆成
stacked PR 或重写对应 issue 编号。

代码审查中仍有四项待处理：

1. CUDA 默认 65,536-token 页池的显存预分配过大。
2. 非 batching CUDA paged cache 与 PrefixCache 不兼容。
3. 量化 CUDA 的 batch>1 路径会每步全量反量化权重。
4. paged/contiguous CUDA sliding-window 边界相差 1。
