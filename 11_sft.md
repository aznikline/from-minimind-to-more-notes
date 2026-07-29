# 算法:Minimind 的 SFT

## 一句话精炼

SFT 与 Pretrain 训练框架几乎同构（同为监督学习），其灵魂全在 Dataset 层：用 chat template 把多轮对话"压扁"成一条长序列，再通过 Loss Masking（只对 assistant 回复 token 置 label、其余置 -100）强迫模型只在"回答"部分产生梯度，从而把续写机改造成听指令的助手。

## 核心概念（SFT/指令微调、chat template、loss mask 只对回答算 loss、full vs LoRA 等）

- **SFT（Supervised Fine-Tuning / 指令微调）**：用人工标注的对话数据，教模型理解"问题→回答"的对应关系；本质仍是监督学习，因此训练循环与 Pretrain 大同小异。
- **SFT 与 Pretrain 的关键差异**：Pretrain 对所有 token 算 loss（学语言分布）；SFT 只对 assistant 回复算 loss（学指令-回答映射）。这正是 Loss Masking 的作用。
- **Chat Template**：用 `apply_chat_template` 把 `system/user/assistant` 多轮列表渲染成一条纯文本，用 `<|im_start|>role\n...<|im_end|>\n` 这类特殊标记分段；`tokenize=False` 先拼字符串、`add_generation_prompt=False`（训练已有对话，不像推理时引导生成）。
- **Loss Mask / Label Mask**：把 user 段、system 段、padding 的 label 全置 -100（PyTorch CrossEntropyLoss 默认忽略 -100），只把 assistant 回复段恢复为真实 token id，从而只在回答上算梯度。
- **Full SFT vs LoRA**：本文件是 `train_full_sft.py`，即全参数微调（更新全部权重）。文件未涉及 LoRA（低秩适配只训少量 adapter 参数），但从工程取舍看：full SFT 效果上限高、显存大；LoRA 省显存、易切换，是小卡/多任务场景的常见替代。二者 Dataset/mask 逻辑一致，差异在"训哪些参数"。

## 关键公式（LaTeX：SFT loss（带 mask）、交叉熵）

**单步交叉熵（next token prediction）**：给定序列 $x=(x_1,\dots,x_T)$，模型在位置 $t$ 预测下一词的概率 $p_\theta(x_t\mid x_{<t})$：

$$
\mathrm{CE}(t)=-\log p_\theta(x_t\mid x_{<t};\theta)
$$

**带 mask 的 SFT loss**（令 $\mathcal{A}=\{t:\text{labels}_t\neq -100\}$ 为 assistant 回复 token 下标集合）：

$$
\mathcal{L}_{\mathrm{SFT}}=-\frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}}\log p_\theta(x_t\mid x_{<t};\theta)
$$

对比 **Pretrain loss**（对全部 token 求平均）：

$$
\mathcal{L}_{\mathrm{Pretrain}}=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t};\theta)
$$

**梯度累加缩放**（代码中 `loss = loss / accumulation_steps`）：

$$
\text{loss}_{\text{step}}=\frac{\mathcal{L}_{\mathrm{SFT}}+\mathcal{L}_{\mathrm{aux}}}{\text{accumulation\_steps}},\quad \theta\leftarrow\theta-\eta\,\frac{1}{K}\sum_{k=1}^{K}\nabla\text{loss}_{\text{step}}^{(k)}
$$

## 关键算法/流程（SFT 数据构造：system/user/assistant 拼接与 mask；训练循环）

**数据构造流程（SFTDataset）**：

1. `load_dataset('json', ...)` 懒加载 jsonl（内存映射，几 GB 也不一次性读入）。
2. 预算两个特征码 token 序列：
   - `bos_id = tokenizer(f'{bos_token}assistant\n', add_special_tokens=False).input_ids`
   - `eos_id = tokenizer(f'{eos_token}\n', add_special_tokens=False).input_ids`
3. `create_chat_prompt(cs)`：`apply_chat_template` 把多轮对话"压扁"成一条长字符串。
4. `tokenizer(prompt).input_ids[:max_length]`：分词 + 截断。
5. 不足 `max_length` 用 `pad_token_id` 补齐（保证 batch 内可堆叠成 Tensor）。
6. `generate_labels(input_ids)`（核心）：
   - 初始化 `labels = [-100] * len(input_ids)`（默认全不学）。
   - 线性扫描 `input_ids`，命中 `bos_id`（assistant 段起点）→ `start = i + len(bos_id)`。
   - 从 `start` 向后扫，命中 `eos_id` → `end`。
   - 把 `labels[start .. end+len(eos_id)]` 从 -100 恢复为真实 `input_ids`（解除 mask）。
   - 指针跳到回复末尾，继续找下一轮（支持多轮对话）。
7. 返回 `(input_ids_tensor, labels_tensor)`。

**训练循环（train\_epoch）**：

1. 取 `(input_ids, labels)` → 搬到 GPU。
2. 动态学习率：`get_lr(epoch*iters+step, total, base_lr)`（warmup + cosine decay）。
3. 前向（autocast）：`res = model(input_ids, labels=labels)`，`loss = res.loss + res.aux_loss`，`loss /= accumulation_steps`。
4. `scaler.scale(loss).backward()`（FP16 防下溢）。
5. 到累加步数：`scaler.unscale_` → `clip_grad_norm_(1.0)` → `scaler.step` → `scaler.update` → `zero_grad(set_to_none=True)`。
6. 日志：还原真实 loss（`* accumulation_steps`），分出 `logits_loss` 与 `aux_loss`。
7. 保存：仅主进程，`model.eval()`，DDP 取 `.module`、`torch.compile` 取 `_orig_mod`，权重存半精度。

## 源码要点（Minimind SFT 代码要点）

- **三文件分工**：`train_full_sft.py`（训练框架）+ `lm_dataset.py`（mask 灵魂）+ `trainer_utils.py`（基础设施，与 Pretrain 共用）。
- **SFT 超参默认值**：`epochs=2`（SFT 轮数少）、`learning_rate=1e-6`（比 Pretrain 小）、`grad_clip=1.0`、`max_seq_len=340`、`dtype=bfloat16`、`accumulation_steps=1`。
- **特征码硬编码风险**：`'assistant\n'` 与 `'\n'` 依赖 ChatML 模板的具体渲染；若 template 渲染无换行，匹配失效 → loss 全 0（作者自标风险）。
- **断点续训**：`lm_checkpoint` 存 model/optimizer/scaler/epoch/step；`SkipBatchSampler` 直接跳过已训 step，避免 DataLoader 空转。
- **DDP 细节**：`_ddp_params_and_buffers_to_ignore = {"freqs_cos","freqs_sin"}`（RoPE 表无需同步梯度）；`set_epoch` 保证每 epoch shuffle 不同。
- **半精度落盘**：`torch.save({k: v.half().cpu() ...})` 节省空间。
- **调试代码**（注释掉的）：肉眼核对每个 token 的 X/Y/label 是否对齐，作者强烈建议训练前跑一次。

## 作者独到见解/类比

- **"复读机 → 助手"跃迁**：Pretrain 让模型在"文本海洋中冲浪"，学会统计分布与世界知识，但只是续写的复读机；SFT 用 Loss Masking 强迫模型只对回答产生梯度，把语言概率模型改造成能沟通的助手。
- **SFT=模仿，RL=尝试反馈**：SFT 通过"模仿"标准答案学习；RL（PPO/DPO/GRPO）解决"标准答案无法覆盖所有场景"，提升对齐性、推理上限、鲁棒性。
- **RL 认知边界（引用 NeurIPS 2025 Best Paper Runner-Up）**：《Does RL Really Incentivize Reasoning Capacity... Beyond the Base Model?》实验表明 RL 只是提高了采样效率，而非真正让模型学会不会做的事；作者据此认为人类对 RL 的认识仍极为有限，"还有很多未探索的宝藏"。
- **幼儿学习类比**：人类幼年经历 RL 习得技能（不教说话→语言障碍；阻止摔东西→延长空间感知习得），迁移到 LLM 即"RL 可能是 AGI 关键拼图"——但被上述论文打了折扣。

## 面试考点（为何只对 assistant 部分算 loss、SFT 与 Pretrain loss 差异、过拟合风险）

- **为何只对 assistant 算 loss**：
  1. 目标是"给定问题，如何回答"，user 输入是**条件**而非**目标**；学用户提问方式会让模型退化成"模仿提问"。
  2. 若 user 段也算 loss，梯度会把模型拉向"生成用户的话"，稀释对回答能力的训练信号。
  3. Loss Masking 让监督信号精准落在"回答生成"上，效率与效果都更好。
- **SFT 与 Pretrain loss 差异**：Pretrain 对全序列算 loss，学语言统计分布；SFT 只对 assistant 段算 loss，学指令-回答映射；数据形态不同（纯文本 vs 多轮对话压扁）、mask 不同（全算 vs -100）、超参不同（lr 更小、epoch 更少）。
- **过拟合风险**：SFT 数据量远小于 pretrain，epochs 多/数据少→模型僵化、输出模式固化、丢失多样性。代码用 `epochs=2`、`lr=1e-6`、小 `max_seq_len` 压低过拟合；调试代码建议人工核对 mask 是否正确（mask 错→loss 异常）。

## 批判性批注

- **特征码硬编码脆弱**：`'assistant\n'`/`'\n'` 强依赖 ChatML 具体渲染，且无运行时校验；换 tokenizer/template 即静默失效（loss=0）。可加 assert：训练前统计非 -100 的 label 占比，过低即报错。
- **截断风险**：`input_ids[:max_length]` 直接截断，可能把多轮对话切在 user 段中间，导致该轮 assistant 段缺失或 mask 错位；未做"按对话边界截断"的保护。
- **单轮线性扫描 O(n)**：小模型够用，但超长序列可优化为 KMP/预记录区间。
- **未讨论 LoRA**：标题 full\_sft 暗示全参数，但未对比 LoRA，读者需自行补足。
- **超参缺乏解释**：`lr=1e-6`、`epochs=2`、`grad_clip=1.0` 是经验值，未给来源/消融。
- **作者 RL 思考略单薄**：仅引一篇论文即下"RL 只提升采样效率"结论，样本偏少；但作为启发式提醒（"对 RL 认识有限"）有价值。
- **亮点**：调试代码、风险提示、DDP/compile 取模细节、半精度落盘——工程严谨度高于多数教程。

## 篇内小思维导图（缩进树）

```
SFT
├── 训练框架 train_full_sft.py（与 Pretrain 同构）
│   ├── 梯度累加：loss / accumulation_steps
│   ├── 梯度裁剪：clip_grad_norm_(1.0)
│   ├── 混合精度：bfloat16 + GradScaler
│   ├── 动态 lr：get_lr(warmup+cosine)
│   └── 落盘：half().cpu()，DDP 取 .module / compile 取 _orig_mod
├── 数据灵魂 lm_dataset.py
│   ├── 懒加载 load_dataset（内存映射）
│   ├── create_chat_prompt：apply_chat_template 压扁多轮
│   ├── 截断 + pad_token_id 补齐
│   └── generate_labels（核心 mask）
│       ├── 全置 -100
│       ├── 命中 bos_id（assistant\n）→ 找回复区间
│       ├── 命中 eos_id（\n）→ 解除 mask（label=input_ids）
│       └── 多轮循环
├── 三大对比
│   ├── Pretrain：全 token 算 loss
│   ├── SFT：只 assistant 算 loss
│   └── full SFT vs LoRA：全参 vs 低秩 adapter（本文件仅 full）
└── 作者延伸
    ├── SFT=模仿标准答案
    ├── RL=尝试与反馈（PPO/DPO/GRPO）
    └── RL 边界论文：只提升采样效率，非真正"学会"
```
