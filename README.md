# From Minimind to More — 精读结构化笔记

> 对 [Tongyun1/from-minimind-to-more](https://github.com/Tongyun1/from-minimind-to-more) 全 17 篇正文(约 62 万字 / 1 万行)的逐篇精读合成笔记。
>
> **特别致谢**:本笔记基于 [Tongyun1](https://github.com/Tongyun1) 的原项目整理,原项目又基于 [minimind](https://github.com/jingyaogong/minimind)(作者 jingyaogong)与 [MiniMind-in-Depth](https://github.com/hans0809/MiniMind-in-Depth)。感谢原作者无私开源。本仓库**仅含精读笔记,不含原正文**,所有版权归原项目作者所有。

## 这是什么

一份对 `from-minimind-to-more` 仓库的**逐篇结构化精读笔记**,覆盖:

- **基石篇**(3):Tokenizer、Minimind 设计目录、Embedding 与位置编码
- **架构篇**(5):归一化技术、KV Cache→Flash Attention、MoE、超级拼装、推理训练优化
- **算法篇**(7):RL 概览、Pretrain、SFT、DPO、PPO、GRPO 及变体、SPO
- **求职篇**(1):大模型八股 100 问

## 笔记结构

每篇笔记含固定八段:

1. 一句话精炼
2. 核心概念(分点列出)
3. 关键公式(LaTeX)
4. 关键算法/流程(伪代码或步骤)
5. 源码要点(对照 minimind 代码逐行)
6. 作者独到见解/类比(摘录原文)
7. 面试考点
8. 批判性批注(事实性/覆盖面/过时风险)
9. 篇内思维导图(mermaid 或缩进树)

## 文件清单

| 文件 | 内容 |
|------|------|
| `00_OVERVIEW.md` | **总图 + 总索引**(全篇知识 mindmap、训练链路、GRPO 家族谱系、工程速查表、全局批判、阅读路径) |
| `01–16_*.md` | 16 篇正文逐篇精读笔记 |

**建议先读 `00_OVERVIEW.md` 建立全貌**,再按其中推荐的阅读路径精读单篇。

## 阅读路径

- **新手**:`02 → 01 → 03 → 04 → 07 → 10 → 11 → 09 → 16`
- **求职冲刺**:`16 → 04/05/06 → 09 → 12/13/14 → 速查表`
- **算法岗**:`09 → 12 → 14 → 15 → 13 → 10/11`
- **工程/推理岗**:`05 → 08 → 06 → 07 → 速查表`

## 免责声明

- 本笔记是**个人学习产物**,不代表原项目作者观点。
- minimind 是极简教学实现,与工业标准有差距(如 PPO 用 One-step MDP、DPO 用 mean 而非 sum 等),笔记在"批判性批注"段已逐项标注。
- 部分内容(Chinchilla scaling law、经典 RLHF 叙事、MMLU 等)在 2026 视角下已过时,请带"时间戳意识"交叉验证。
- 遇关键论断建议回原论文核对。

## License

笔记内容遵循原项目精神开源。原项目 `from-minimind-to-more` 未声明 License,版权归原作者所有。
