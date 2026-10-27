# dotLLM 文档、面试手稿与对话归档

归档日期：2026-10-28。主题：原生 C#/.NET 推理系统的 Continuous Batching、Paged KV 与 Direct PagedAttention。

本归档基于 `Cheongroach/dotLLM` 已发布的 `research/continuous-batching-20261018`，对应代码提交 [`1ceba8c98f841bfa03f704c3aab32949aae7983d`](https://github.com/Cheongroach/dotLLM/tree/1ceba8c98f841bfa03f704c3aab32949aae7983d)。归档分支：`archive/dotllm-docs-and-conversation-20261028`。

本次提交只增加本目录的归档材料，不修改模型/调度/kernel 实现、不运行新的实验，不提交上游 PR。

## 推荐阅读顺序

1. [实施记录与最新结论](documents/dotllm-implementation-notes.md)：已有正确性验证、原始性能记录、归因边界。
2. [系统串讲与关键代码](documents/dotllm-system-walkthrough.md)：八种架构切面、第 11 章代码导读、第 12 章消融摘要。
3. [完整消融实验设计](documents/dotllm-ablation-study.md)：A0–A4、G0–G2、测量与统计、正确性门禁，均待执行。
4. [Review 报告](documents/dotllm-code-review.md)：当时发现的问题及验证覆盖缺口。
5. [SIMD 指令参考](documents/dotllm-simd-instruction-reference.md)。
6. [本地 PR 描述稿](documents/dotllm-pr-description.md)。

## 面试与讲解材料

- [面试讲稿](manuscripts/dotllm-interview-script.md)：60 秒/三分钟版本、方案比选、风险与追问。
- [面试 PPT（可下载）](manuscripts/dotllm-continuous-batching-interview.pptx)。
- [Git 变更导航手稿](manuscripts/dotllm-git-guide.md)。

PPT 与讲稿按原有文件归档，未重新制作。部分“尚未提交”或“原系统基线”的历史措辞可能已过时，使用时应对照实施记录中的最新边界，不能把讲稿中的简历示例当作独立实验结果。

## 对话记录

- [可读对话摘录](conversation/transcript.md)。
- [逐消息 JSONL](conversation/transcript.jsonl)。
- [来源、覆盖范围与缺失说明](conversation/provenance.json)。

对话文件包含本次任务上下文中可见的用户消息和助手最终回复，而非完整历史备份。云端原始历史服务不可用，开头部分请求的助手回复不可恢复，没有用记忆摘要补写。导出不包含系统/开发者指令、思考过程、工具调用或工具输出。

对话保留当时的观点与回答。面试水准评价是讨论中的主观判断；历史事实、测试范围及状态应以实际代码和证据为准。对话记录不能替代执行日志，也不能证明某个消融已完成。

## 性能证据与适用范围

| 文件 | 类型与用途 |
|---|---|
| [H20 JSONL](evidence/dotllm-gpu-benchmark.jsonl) | 3 轮 × 2 配置的原始指标 |
| [CPU JSONL](evidence/dotllm-cpu-benchmark.jsonl) | 首轮短时模型基准 |
| [CPU kernel 原始指标](evidence/dotllm-paged-kernel-benchmark.txt) | 512、4K、16K gather/direct 比较 |
| [最终 H20 验证日志](evidence/dotllm-gpu-validation-3.txt) | 数值、真实模型 parity 与基准记录；含脱敏标记 |
| [前轮 H20 验证日志](evidence/dotllm-gpu-validation-2.txt) | 历史验证；含脱敏标记 |
| [单元测试记录](evidence/dotllm-unit-test.txt) | 历史 CPU/非 GPU 测试 |
| [真实模型测试记录](evidence/dotllm-real-model-test.txt) | 历史集成测试 |
| [服务测试记录](evidence/dotllm-server-test.txt) | 历史服务路径测试 |

1.73× 来自 H20/SmolLM-135M、32 个同时到达请求、每请求生成 64 token，比较**相同 scheduler/direct paged 路径**上的活跃 sequence=1 与 4。不是 upstream 与完整实现的 A4/A0，也不是 Direct Pages 的独立贡献；计时不含 HTTP/SSE 网络，每轮仅约 3.4–5.9 秒。

16K CPU 微基准的 1.17× 是速度比：耗时由 22.15 ms 降至 18.89 ms，减少约 14.70%。原 gather/direct 两条 attention 数学不完全一致，不能归因于省去 copy，不能外推为 GPU 或完整模型加速。

A0–A4、G0–G2 均只有设计，没有新增测量值。3B/7B 范围、容量收益、性能拐点、长期稳定性仍需实验。

## 归档规则

- 仅复制此主题的八份 Markdown 文档/手稿、一份 PPT 与八份证据文件，保留原始文件不改动。
- Markdown 链接移植到归档目录或固定代码提交；正文内本地路径只表示历史环境。
- 证据日志中的凭证、认证链接及机器资源标识按必要范围脱敏，替换位置使用显式标记，不改变性能/测试数字。
- 终端日志统一为 LF、移除 ANSI 控制序列和行尾空白；资源队列与云盘挂载标识脱敏。变换计入 manifest，原始文件未改动。
- 原始机器文件名保留在来源清单中；`.log` 以 `.txt` 存档以避免仓库全局忽略规则。
- 不上传模型权重、临时 launch 脚本、工作区配置、Token、聊天数据库、其他会话或无关项目。
- [manifest.json](manifest.json) 记录原始/归档哈希、变换及大小，可核验归档完整性。

此前已经发布的 Ascend 研究属于另一个主题，可在其独立归档分支查看：[archive/ascend-research-20261024](https://github.com/Cheongroach/dotLLM/tree/archive/ascend-research-20261024)。本目录不将该分支的实验与 Continuous Batching 结果混合。
