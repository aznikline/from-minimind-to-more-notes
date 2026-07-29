# 算法:Minimind 的 Pretrain

## 一句话精炼

Minimind 的预训练以 Causal LM（自回归下一词预测）为目标，用「指令对数据」做无监督 Continued Pre-training，通过 cosine 学习率、梯度累积、梯度裁剪、混合精度、原子化断点续训等一整套工程化脚手架，在小显存上稳定训练 LLM。

## 核心概念

### 预训练目标：Causal Language Modeling
- 任务形式：给定前 $t$ 个 token，预测第 $t+1$ 个 token，即建模 $P(x_t \mid x_{<t})$。
- 自回归（causal）意味着使用下三角掩码，当前位置只能看到历史位置，不能看到未来。
- MiniMind 把 `pretrain_hq.jsonl`（本质是 QA 对话）直接放进 Pretrain 阶段且**不做 Loss Masking**，让模型在大规模无监督阶段就熟悉 `<|im_end|>`、熟悉「问→答」的文本概率分布。业界称为 **Instruction Pre-training / Continued Pre-training**。
- 区别于 SFT：Pretrain 的 label = input（除 pad 外全部参与 loss）；SFT 只对 assistant 段计算 loss，user 段被 mask 成 -100。

### Loss
- 主任务 Loss：交叉熵（Cross Entropy），PyTorch `CrossEntropyLoss(ignore_index=-100)` 自动跳过 label 为 -100 的位置。
- 辅助 Loss（aux_loss）：MoE 负载均衡 Loss，防止专家坍缩（所有 token 都涌向同一批专家）。
- 总 loss = `res.loss + res.aux_loss`，即使是 dense 模型也会显式加上（dense 时 aux_loss=0）。

### 学习率调度：Cosine Annealing（无显式 warmup）
- MiniMind 的 `get_lr` 是纯 cosine，**没有独立 warmup 段**，最低比例固定为 0.1：
  - `lr = base_lr * (0.1 + 0.45 * (1 + cos(pi * step / total)))`
  - step=0 时 → `lr = base_lr * 1.0`（最大）
  - step=total 时 → `lr = base_lr * 0.1`（最小）
- 每个 step 手动写回 `optimizer.param_groups`，不走 PyTorch `LambdaLR`。
- 注意：这与教科书「warmup + cosine」不完全一致，MiniMind 用「cosine 从最大起步」的简化版，依赖 AdamW 自适应来稳住前几步。

### 权重衰减
- 文中未单独提及 weight_decay 超参，仅使用默认 `optim.AdamW(model.parameters(), lr=...)`。注意点：AdamW 的解耦权重衰减是默认行为，但 MiniMind 代码没有显式调参，属于「用默认值」。

### 梯度裁剪
- `torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)`，按全局 L2 范数裁剪，防梯度爆炸。
- 关键顺序：先 `scaler.unscale_(optimizer)` 反缩放回 FP32，再裁剪，再 `scaler.step(optimizer)`。顺序错了裁剪就失效。

### 梯度累积
- `loss = loss / accumulation_steps`，每 N 步才 `optimizer.step()`。
- 目的：小显存模拟大 batch。累积期内只 `backward`，到达累积步才 step + zero_grad。
- 细节：`zero_grad(set_to_none=True)` 比 `zero_grad()` 更省显存（直接置 None 而非分配 0 张量）。

### 混合精度（AMP）
- 优先 `bfloat16`（范围广、训练稳），否则 `float16`。
- `torch.cuda.amp.autocast` + `GradScaler`：FP16 下放梯度防下溢；bf16 理论上不需要 scaler，但代码为兼容统一保留。

### 断点续训
- `lm_checkpoint` 保存两类文件：
  - `*.pth`：仅模型权重（half + cpu），用于推理/分发。
  - `*_resume.pth`：模型 + optimizer + scaler + epoch + step + world_size + wandb_id，用于恢复。
- **原子化保存**：先写 `.tmp`，再 `os.replace` 覆盖，防中途断电损坏文件。
- **GPU 数量变化自适应**：`saved_ws != current_ws` 时，`step = step * saved_ws // current_ws`，从 8 卡迁到 2 卡也能对齐数据消耗量。
- `SkipBatchSampler`：恢复时跳过前 `skip_batches` 个 batch，直接从断点处喂数据，避免重复训练。

## 关键公式

### 交叉熵 Loss（Causal LM，逐 token）
$$
\mathcal{L}_{\text{CE}} = -\frac{1}{|\{t: y_t \neq -100\}|} \sum_{t=1}^{T} \mathbb{1}[y_t \neq -100] \cdot \log P(x_{t} \mid x_{<t}; \theta)
$$
- $y_t = -100$ 的位置（pad、被 mask 的 user 段）被忽略。
- Pretrain 阶段仅 pad 被置 -100；SFT 阶段 user 段也置 -100。

### 总 Loss（含 MoE 辅助 Loss）
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{aux}}
$$
- $\mathcal{L}_{\text{aux}}$ 为 MoE 负载均衡 loss，dense 模型时为 0。

### 梯度累积下的等效 loss
$$
\mathcal{L}_{\text{accum}} = \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_{\text{total}}^{(i)}, \quad \text{累积 } N \text{ 步后一次 step}
$$
- 代码层面：每步 `loss / accumulation_steps`，累积 N 步 backward 后再 optimizer.step。

### Cosine 学习率（MiniMind 版）
$$
\text{lr}(s) = \text{lr}_{\text{base}} \cdot \left(0.1 + 0.45 \cdot \left(1 + \cos\left(\pi \cdot \frac{s}{S}\right)\right)\right)
$$
- $s$ = 当前全局 step，$S$ = 总 step（`epochs * iters`）。
- 注意：无独立 warmup，最大值即 $\text{lr}_{\text{base}}$（$s=0$），最小值 $0.1 \cdot \text{lr}_{\text{base}}$（$s=S$）。

### 梯度裁剪（全局 L2 范数）
$$
g \leftarrow \frac{g}{\max(1, \|g\|_2 / g_{\text{clip}})}
$$

### 权重衰减（AdamW 默认）
$$
\theta_{t+1} = \theta_t - \eta \cdot (\hat{m}_t + \lambda \theta_t)
$$
- AdamW 解耦形式，但 MiniMind 未显式调 `weight_decay`，使用默认值。

## 关键算法/流程

### 数据构造：固定长度 + Padding（非滑动窗口、非 sample packing）

MiniMind 的 `PretrainDataset.__getitem__` 采用最朴素的「一条样本 = 一个 input」方式：

1. 取一条 `text`。
2. tokenizer 编码，`max_length - 2` 截断（预留 BOS/EOS）。
3. 首尾包 `[BOS] ... [EOS]`。
4. 不足 `max_length` 的尾部补 `pad_token_id`。
5. `labels = input_ids.clone()`。
6. `labels[input_ids == pad_token_id] = -100`（pad 不算 loss）。

**对比说明（文中未出现但值得区分）**：
- **滑动窗口**：经典 nanoGPT 风格，把长文档按 stride 切出多个重叠样本，最大化利用长文。MiniMind **没用**，每条独立 padding。
- **Sample Packing / Packing**：把多条短样本拼进一个 max_length 序列以消除 pad 浪费。MiniMind **没用**，保留独立 padding（实现简单但有效率损失）。

### Loss Masking 的两副面孔
- **Pretrain**：只 mask pad（-100）。整段文本（包括 user 提问部分）都参与 loss，模型在学「整体语言分布」。
- **SFT**：mask pad + user 段（-100），只对 assistant 回答段计算 loss。`generate_labels` 扫描 input_ids，定位 `[bos]+assistant` 到 `eos` 之间，只把这段 label 设为真实 id，其余 -100。

### 训练循环（`train_epoch`）

```
for step, (input_ids, labels) in loader:
    1. 数据 → device
    2. lr = get_lr(epoch*iters + step, epochs*iters, base_lr)
       手动写回 optimizer.param_groups
    3. with autocast:
         res = model(input_ids, labels=labels)
         loss = res.loss + res.aux_loss
         loss = loss / accumulation_steps
    4. scaler.scale(loss).backward()           # 累积梯度
    5. if (step+1) % accumulation_steps == 0:
         scaler.unscale_(optimizer)            # 反缩放
         clip_grad_norm_(model, grad_clip)      # 裁剪
         scaler.step(optimizer)
         scaler.update()
         optimizer.zero_grad(set_to_none=True)
    6. 定期 log（loss/aux_loss/lr/eta）
    7. 定期 save：主进程 → eval → 剥 DDP/_orig_mod → half().cpu() → torch.save(.pth) → lm_checkpoint(resume)
    8. del input_ids, labels, res, loss         # 激进显存管理
```

### 断点续训流程
1. `from_resume=1` → `lm_checkpoint(...)` 走加载分支。
2. 读 `*_resume.pth` → 恢复 model/optimizer/scaler/epoch/step。
3. GPU 数变化时 `step = step * saved_ws // current_ws`。
4. DataLoader 用 `SkipBatchSampler(sampler, batch_size, skip=start_step)`，跳过已训练 batch。
5. `train_epoch` 的 `enumerate(loader, start=start_step+1)` 让进度条接续。

### 分布式要点
- `init_distributed_mode()` + `DistributedDataParallel` 包装。
- 每卡种子 `42 + rank`，防止各卡数据增强/Dropout 完全一致。
- `_ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}`：RoPE 预计算缓存是常量，不参与梯度同步。
- 仅 rank=0 初始化 wandb，仅 rank=0 保存 checkpoint，防多卡同时写文件冲突。

## 源码要点

### 三个核心文件
- `dataset/lm_dataset.py`：`PretrainDataset` / `SFTDataset` / `DPODataset`。
- `trainer/trainer_utils.py`：`lm_checkpoint` / `SkipBatchSampler` / `get_model_params` / `get_lr`。
- `trainer/train_pretrain.py`：环境初始化 → 模型/数据/优化器构建 → `train_epoch` → 清理。

### `PretrainDataset`（lm_dataset.py）
```python
tokens = [bos_id] + tokenizer(text, add_special_tokens=False,
                               max_length=max_len-2, truncation=True).input_ids + [eos_id]
input_ids = tokens + [pad_id] * (max_len - len(tokens))   # 定长 padding
labels = input_ids.clone()
labels[input_ids == pad_id] = -100                          # pad 不算 loss
```
- 返回 `(input_ids, labels)`，长度恒等于 `max_length`。

### `lm_checkpoint`（trainer_utils.py）
- 保存：剥 DDP（`model.module`）+ 剥 compile（`_orig_mod`）→ `half().cpu()` → 写 `.tmp` → `os.replace`。
- resume_data 含 `model/optimizer/scaler/epoch/step/world_size/wandb_id`，外加 kwargs 里有 `state_dict` 的对象（如 scheduler）。
- 加载：`torch.load(map_location='cpu')`，按 world_size 换算 step。

### `SkipBatchSampler`（trainer_utils.py）
- 包装基础 sampler，遍历时累计 batch，前 `skip_batches` 个直接丢弃不 yield。
- `__len__` = `max(0, total_batches - skip_batches)`。

### `get_model_params`（trainer_utils.py）
- 区分 dense vs MoE：
  - 通过参数名 `mlp.experts.0.` / `mlp.shared_experts.0.` 算单个专家大小。
  - base = total - 路由专家*总数 - 共享专家*总数。
  - active = base + 路由专家*激活数 + 共享专家*总数。
- MoE 输出 `1000.00M-A200.00M`（总-激活），dense 输出 `1000.00M`。

### `get_lr`（trainer_utils.py）
```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

### `train_pretrain.py` 主结构
- 阶段1：`init_distributed_mode()` + `setup_seed(42+rank)`。
- 阶段2：`MiniMindConfig` → `init_model` → 可选 `torch.compile` → `PretrainDataset` → `DistributedSampler` → `GradScaler` → `AdamW`。
- 阶段3：恢复 ckp → DDP 包装（忽略 `freqs_cos/freqs_sin`）→ `train_epoch`。
- 阶段4：`dist.destroy_process_group()`。

### `train_epoch` 关键点
- `lr` 每 step 手动更新（非 scheduler）。
- `loss = res.loss + res.aux_loss`，显式加 MoE aux_loss。
- `loss / accumulation_steps` 后 backward。
- unscale → clip → step → update → zero_grad(set_to_none=True)。
- 保存：主进程、eval 模式、剥 DDP/compile、half().cpu()、torch.save(.pth) + lm_checkpoint(resume)。
- `del input_ids, labels, res, loss` 显式释放。

## 作者独到见解/类比

- **「指令数据当 pretrain 用」的反直觉设计**：作者明确指出 `pretrain_hq.jsonl` 虽放在 Pretrain 阶段，内容却是 QA 对话，并点出业界叫法 Instruction Pre-training / Continued Pre-training。类比：传统 pretrain 像让模型读「百科全书」，MiniMind 这步像让模型先在「问答现场」泡一泡，提前熟悉 `<|im_end|>` 这种对话符号和「问→答」的文本概率分布，为后续 SFT 降低分布漂移。
- **「不要满足于跑通，逐行看懂」**：作者反复强调代码里藏着许多设计细节，把这当作学习 LLM 训练工程的最佳标本。
- **「原子化保存」类比**：像「先写草稿再交卷」——`os.replace` 是系统级原子改名，保证目标文件永远是完整的，不会出现「旧的坏了、新的也没存全」的中间态。
- **「GPU 数变化自适应」**：从 8 卡迁到 2 卡接着跑，自动按 world_size 换算 step，作者把这称为「非常硬核」的工程细节。
- **显存管理类比**：`del` 当前 step 变量、`zero_grad(set_to_none=True)`、`half().cpu()` 存权重，都是「显存紧张时的求生术」。

## 面试考点

### 1. 预训练和 SFT 的 loss 区别？
- **Pretrain**：labels = input_ids（除 pad 外全参与）。模型学整体语言分布，user 和 assistant 段都算 loss。
- **SFT**：只对 assistant 段算 loss，user 段 label 置 -100。通过 `generate_labels` 定位 assistant 片段。
- 一句话：Pretrain 学「下个词是什么」，SFT 学「作为助手该怎么接话」。

### 2. 为何要做 Loss Masking？
- SFT 阶段只希望模型学「如何回答」，不希望它学「如何提问」（否则会模仿用户的口吻甚至复述问题）。
- Mask user 段 = -100 → `CrossEntropyLoss(ignore_index=-100)` 跳过 → 梯度只来自 assistant 段。
- Pretrain 不做 user mask，因为目标是学整体文本分布；SFT 做 user mask，因为目标是学「应答行为」。

### 3. Warmup 的作用？MiniMind 用了吗？
- Warmup 的意义：训练初期权重随机，大 lr 会导致前几步梯度爆炸/损失剧烈震荡；warmup 让 lr 从 0 线性升到峰值，给优化器一个「找方向」的缓冲期。
- MiniMind 的 `get_lr` 是纯 cosine，**没有独立 warmup**，step=0 即最大 lr。依赖 AdamW 的自适应学习率（按梯度一阶/二阶矩自适应）来稳住初期。这是简化做法，大模型训练一般仍推荐 warmup。

### 4. 梯度累积为什么 loss 要除以 N？
- 累积 N 步 backward，梯度是和。若不除 N，等效 batch = N * batch_size 但 loss 没归一化，梯度会被放大 N 倍，等效学习率变大，训练不稳。
- `loss / N` 后累积 = 平均，与「真大 batch」语义一致。

### 5. 梯度裁剪的顺序为什么是 unscale → clip → step？
- AMP 下梯度被 scaler 放大过，直接 clip 会按放大后的范数判断，阈值失真。
- 必须 `scaler.unscale_` 还原成真实 FP32 梯度，再 `clip_grad_norm_`，最后 `scaler.step`（内部会再处理 inf/nan）。

### 6. `os.replace` 原子保存为何必要？
- `torch.save` 非原子，中途断电/进程被杀会留下半个坏文件，旧 checkpoint 也被覆盖没了。
- 先写 `.tmp` 完整后 `os.replace` 原子改名，目标文件永远是完整的。

### 7. 为什么 DDP 要 ignore `freqs_cos/freqs_sin`？
- RoPE 的预计算 sin/cos 表是常量缓冲区，不参与训练梯度，DDP 不需要对它们做 all-reduce 同步，显式 ignore 省通信开销。

### 8. MoE 的 aux_loss 作用？
- 负载均衡 loss，惩罚「所有 token 都涌向少数专家」，防止专家坍缩（expert collapse）。
- 总 loss = CE + aux_loss，让路由尽量均匀分布到所有专家。

### 9. 梯度累积 vs 真大 batch 的区别？
- 数学上 loss 期望一致，但 BN/通信/数值精度有差异；DDP + 累积等效 global batch = batch * world_size * accumulation_steps。
- 累积期内只 backward 不 step，对优化路径有微小影响（动量等滞后 N 步）。

## 批判性批注

- **无 warmup 的风险**：MiniMind 的 `get_lr` 从峰值起步，对小模型 + AdamW 通常能稳住，但放大到更大模型（如 7B+）初期容易震荡。生产场景建议加 warmup（如前 2% step 线性升峰），不应照搬此简化版。
- **独立 Padding 而非 Packing 的浪费**：每条样本单独 pad，短文本场景下大量位置是 pad（-100），计算/显存浪费明显。生产级预训练一般用 sample packing + attention mask 隔离样本，能显著提升吞吐。
- **无滑动窗口**：长文档被截断到 max_length，超出部分直接丢弃，长程依赖学不到。对短 QA 数据影响小，但对「百科+书籍」类 pretrain 数据损失大。
- **weight_decay 未显式调**：代码用 `AdamW` 默认值，对大模型预训练通常希望显式设较小的 weight_decay（如 0.1）以正则化，默认值偏弱。
- **`scaler` 在 bf16 下仍启用**：注释自承「bf16 通常不需要」，但代码为兼容统一保留 `GradScaler`，对 bf16 是冗余开销，可按 dtype 条件关闭。
- **每步手动写 lr**：灵活性高但失去了 PyTorch `LambdaLR`/`SequentialLR` 的可组合性，叠加 warmup/multistep 时维护成本高。
- **断点续训的 step 换算**：`step = step * saved_ws // current_ws` 是按「数据消耗量」对齐的粗略换算，假设了每卡 batch 一致；若 batch_size 也变了，换算会失真，应同时考虑 batch_size 因子。
- **「Instruction Pre-training」术语模糊**：作者把 QA 数据放进 pretrain 称为 Instruction Pre-training，但严格说 Instruction Pre-training 通常指用模型生成指令数据混入 base pretrain，这里更接近「Continued Pre-training on chat data」，术语可更精确。
- **保存频率**：`save_interval` 由 args 控制，若设太小在长训练中会产生大量 IO；且每次保存都 eval+half+cpu+torch.save+lm_checkpoint，开销不小，生产中可考虑异步保存。

## 篇内小思维导图

```mermaid
mindmap
  root((Minimind Pretrain))
    数据层 lm_dataset.py
      PretrainDataset
        BOS+text+EOS
        定长 pad
        labels=ids, pad→-100
      SFTDataset
        仅 assistant 段算 loss
        user 段 label=-100
      DPODataset
        chosen/rejected 成对
    基础设施 trainer_utils.py
      lm_checkpoint
        DDP 解包 .module
        compile 解包 _orig_mod
        "half().cpu()"
        ".tmp→os.replace 原子保存"
        resume: model+opt+scaler+epoch+step+ws
        GPU数变化 step 换算
      SkipBatchSampler
        跳过前 skip_batches
      get_model_params
        dense: total
        MoE: total-Aactive
      get_lr
        cosine, 0.1~1.0
        无 warmup
    训练引擎 train_pretrain.py
      初始化
        init_distributed_mode
        seed=42+rank
        bfloat16 优先
        torch.compile
      train_epoch
        每 step 更新 lr
        loss=CE+aux
        loss/=accum
        scaler.scale.backward
        unscale→clip→step→update
        "zero_grad(set_to_none)"
        "定期 save(rank0)"
        del 变量
      断点续训
        SkipBatchSampler
        world_size 换算
      DDP
        DistributedSampler
        ignore freqs_cos/sin
    关键设计取舍
      用指令数据做 pretrain
      无 warmup 简化
      独立 padding 非 packing
      显式 aux_loss 防专家坍缩
```
