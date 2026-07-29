# 算法:Minimind 的 DPO

## 一句话精炼

DPO 用同一个 SFT 模型充当"参考"与"策略"两份角色，直接在 chosen/rejected 偏序对上做带 log-ratio 的二元交叉熵，跳过显式奖励模型与 PPO，让策略在偏离参考模型的代价下学会偏好。

## 核心概念(DPO 动机:绕过 RM/RL 直接偏好优化;chosen/rejected;参考模型 frozen)

- **动机**:经典 RLHF 流程是 SFT -> 训练 Reward Model -> PPO 优化策略。RM 和 PPO 都不稳定、显存重、调参难。DPO 的洞见是:最优策略 $\pi^*$ 与参考策略 $\pi_{\text{ref}}$ 之间存在闭式关系 $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$,把这个关系代回偏好模型(Bradley-Terry),奖励就消失,只剩策略与参考的对数比,于是可以直接在偏好数据上做监督学习。
- **chosen / rejected**:每条样本是同一 prompt 下的"好回答"与"坏回答"偏序对。两者 user 输入必须完全一致,差异只在 assistant 回复。Minimind 的 `dpo.jsonl` 即 `{"chosen": [...], "rejected": [...]}` 多轮对话列表。
- **参考模型 frozen**:从 SFT 权重复制一份作为 $\pi_{\text{ref}}$,`eval()` + `requires_grad_(False)`,前向在 `torch.no_grad()` 下运行,只产出 `ref_log_probs` 作为基准锚点,不更新。
- **策略模型 trainable**:同一 SFT 权重初始化的 $\pi_\theta$,参与梯度更新,目标是在 chosen 上提概率、在 rejected 上降概率(相对参考而言)。
- **与 SFT 的相似性**:作者开篇点明"DPO 本质上还是交叉熵损失,代码与 SFT 比较相似",差异只在 loss 构造与双模型前向。

## 关键公式(LaTeX:DPO loss: Bradley-Terry 模型、π_ref、对数比、σ 形式;reward = β·log(π/π_ref))

**Bradley-Terry 偏好模型**(DPO 的前提,假设人类偏好服从奖励差值的 logistic):

$$p(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$$

**最优策略与奖励的闭式关系**(代入 BT 即消去显式奖励):

$$r(x,y) = \beta \log \frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$$

由于 $Z(x)$ 在 $y_w$ 与 $y_l$ 上相消,代入 BT 得到 **DPO 损失**:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l)}\Big[\log \sigma\Big(\beta\Big(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big)\Big]$$

**隐式奖励**(代码注释里给出的形式,可作为推理时的 reward 读数):

$$\hat{r}(x,y) = \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$$

**Minimind 代码中的等价实现**(注意它做了序列长度归一化):

- 单 token log 概率:`log_probs_per_token = gather(log_softmax(logits), labels)`
- 句子级平均 log 概率:$\bar{\ell}(y) = \frac{1}{|y|}\sum_{t \in y} \log \pi_\theta(y_t \mid y_{<t})$(除以 `seq_lengths` 做长短句公平)
- 核心 logits:$\text{logits} = \underbrace{(\bar{\ell}_{\text{policy}}^w - \bar{\ell}_{\text{policy}}^l)}_{\pi\text{-logratios}} - \underbrace{(\bar{\ell}_{\text{ref}}^w - \bar{\ell}_{\text{ref}}^l)}_{\text{ref-logratios}}$
- 损失:`loss = -F.logsigmoid(beta * logits)`,默认 `beta=0.1`

## 关键算法/流程(DPO 数据构造、训练循环、两个模型的前向)

### 数据构造(`DPODataset.__getitem__`)

1. 取 `chosen` 与 `rejected` 两个多轮对话列表。
2. `apply_chat_template(tokenize=False)` 拼成纯文本字符串。
3. `tokenizer(truncation=True, padding='max_length')` 得到 `input_ids`(定长)。
4. `generate_loss_mask`:扫描 `input_ids`,定位 `bos_id`(`<bos>assistant\n` 起始标志)到 `eos_id`(`<eos>\n` 结束标志)之间的区间,把这段 mask 置 1(含 EOS,让模型学会生成结束符);user 输入与 padding 全为 0。
5. **Shift Trick**:因果 LM 预测下一 token,故 `x = input_ids[:-1]`,`y = input_ids[1:]`,`mask = loss_mask[1:]`(mask 对齐 y 位置)。
6. 返回 6 个张量:`x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected`。

### 训练循环(`train_epoch`)

1. 拿到 batch 的 6 个张量,移到 GPU。
2. **拼接**:`x = cat([x_chosen, x_rejected])`,`y`、`mask` 同理,一次前向算完两份,batch 维变 `2*B`。
3. 动态调学习率(Cosine Annealing,`get_lr`)。
4. **双模型前向**(autocast 混合精度):
   - 参考模型:`with torch.no_grad(): ref_logits = ref_model(x).logits`,再 `logits_to_log_probs(ref_logits, y)`。
   - 策略模型:`logits = model(x).logits`,再 `logits_to_log_probs(logits, y)`。
5. **DPO 损失**:`dpo_loss(ref_log_probs, policy_log_probs, mask, beta)`。
6. `loss = dpo_loss_val + outputs.aux_loss`(aux_loss 是 MoE 负载均衡,非 MoE 为 0)。
7. `/ accumulation_steps` 后 `scaler.scale(loss).backward()`。
8. 梯度累积到点:`unscale_` -> `clip_grad_norm_(1.0)` -> `scaler.step` -> `zero_grad(set_to_none=True)`。
9. 日志打印 loss/dpo_loss/aux_loss/lr;按 `save_interval` 存半精度 `.pth` + 完整 checkpoint。

### `dpo_loss` 内部 7 步

1. `seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)` — 有效长度,防除零。
2. 句级平均 log 概率:`(log_probs * mask).sum(dim=1) / seq_lengths.squeeze()`。
3. 拆分 chosen / rejected:`[:batch//2]` 与 `[batch//2:]`(依赖拼接顺序)。
4. `pi_logratios = chosen_policy - reject_policy`。
5. `ref_logratios = chosen_ref - reject_ref`。
6. `logits = pi_logratios - ref_logratios`。
7. `loss = -F.logsigmoid(beta * logits)`,返回 `loss.mean()`。

## 源码要点(Minimind DPO 代码要点)

- **`logits_to_log_probs`**:`F.log_softmax(logits, dim=2)` 得到全词表 log 概率,再 `torch.gather(..., index=labels.unsqueeze(2)).squeeze(-1)` 只取 label 对应位置,输出 `(batch, seq)`。这是 DPO 与 SFT 最关键的工具函数差异。
- **长度归一化**:代码对句级 log 概率做了 `/ seq_lengths`。论文原式是对整句求和(等价于句子联合概率),这里改成"每 token 平均"是为了让长短句公平比较,避免短句天然 log 概率更高被误判为"更好"。这是 Minimind 相对原 DPO 的一个工程取舍。
- **batch 拼接顺序约束**:`dpo_loss` 用 `batch_size // 2` 切分 chosen/rejected,要求上层 `torch.cat([chosen, rejected])` 必须保持这个顺序,顺序错了 loss 含义就反了。
- **参考模型冻结**:`ref_model.eval()` + `ref_model.requires_grad_(False)` + 前向包 `torch.no_grad()`,且与策略模型用同一个 `from_weight`(`full_sft`)初始化,保证起点一致。
- **学习率极小**:`--learning_rate 4e-8`,注释强调"DPO 学习率通常 1e-7~5e-8,比 SFT 小很多",因为 DPO 是在已收敛的 SFT 权重上做微调,大步长会破坏已学知识。
- **`beta=0.1`**:`--beta` 默认 0.1,注释"值越大,越不允许偏离 Reference Model"。代码里 beta 直接乘在 logits 上进 logsigmoid。
- **loss mask 含 EOS**:`generate_loss_mask` 的 `range(start, min(end + len(self.eos_id), max_length))` 特意把 EOS token 也纳入 mask,让模型学会生成结束符。
- **MoE 兼容**:`loss = dpo_loss_val + outputs.aux_loss`,aux_loss 是 MoE 负载均衡损失,非 MoE 模型该字段为 0,框架统一处理。
- **DDP 忽略 RoPE**:`model._ddp_params_and_buffers_to_ignore = {"freqs_cos","freqs_sin"}`,避免 DDP 对非参数 buffer 报错。
- **checkpoint 保存**:半精度 `v.half().cpu()` 存权重省空间,另存完整 checkpoint(含 optimizer/scaler)支持断点续训。
- **断点续训**:`SkipBatchSampler` 跳过已训练 step,`from_resume=1` 时自动恢复 optimizer/scaler/epoch/step。

## 作者独到见解/类比

- 开篇定性:"DPO 本质上还是在做一个交叉熵损失,所以代码与 SFT 是比较相似的"——把 DPO 从神秘的 RL 拉回到"带 log-ratio 的监督学习",降低理解门槛。
- 把 `sum(dim=1)` 解释为"乘法变加法"的概率论操作:对数空间里整句话概率 = 各 token log 概率之和。用 "I love AI" 三词例子把数学落到代码。
- 主动点出 `/ seq_lengths` 的工程理由:长句 log sum 天然更负,不归一化会让 DPO 偏向短句。这是原论文没强调、但实战必须处理的细节。
- 强调"只需看懂数据构造与 `dpo_loss` 就能搞懂 DPO"——把 DPO 的复杂度收敛到两个核心点,其余训练框架与 SFT 同构。
- 用 mask 表格(位置/Token/mask/log_probs/含义)把抽象的 mask 过滤可视化,适合教学。

## 面试考点(DPO 与 PPO 的关系、为何能省去 RM、β 作用、chosen/rejected 梯度方向)

- **DPO 与 PPO 的关系**:两者都源于同一个 RLHF 目标 $\max_\pi \mathbb{E}[r] - \beta D_{\text{KL}}(\pi\|\pi_{\text{ref}})$。PPO 显式训 RM 再在线采样 + value baseline;DPO 利用 KL 约束的闭式解把奖励改写成策略对数比,把 RL 问题重写成监督学习。
- **为何能省去 RM**:因为 $\hat{r}(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 把奖励隐式地编码在策略与参考的对数比里,不需要单独的网络去拟合奖励。参考模型本身就是"奖励锚"。
- **β 作用**:β 越大,KL 惩罚越强,策略越不能偏离参考(更保守,保 SFT 知识);β 越小,越自由追随偏好数据(可能过拟合偏好、损害通用能力)。代码里 β 直接乘在 logits 上,默认 0.1。
- **chosen/rejected 梯度方向**:loss = $-\log\sigma(\beta(\Delta_\pi - \Delta_{\text{ref}}))$。要降 loss,就要增大 $\Delta_\pi = \log\pi(y_w) - \log\pi(y_l)$,即**策略对 chosen 提概率、对 rejected 降概率**;参考模型的 $\Delta_{\text{ref}}$ 是固定基准,把"绝对偏好"转成"相对参考的偏好增量"。
- **为何 ref 模型必须 frozen**:ref 是偏好基准,若它也更新,loss 的锚点会漂移,$\hat{r}$ 不再对应固定的奖励定义,训练会退化或发散。
- **学习率为何极小(4e-8)**:起点是已收敛的 SFT 模型,DPO 只做偏好微调,大步长会冲掉 SFT 知识。
- **长度归一化的影响**:不归一化时短句 log 概率天然更高,DPO 会偏向把 chosen 生成得越来越短;归一化后用每 token 平均 log 概率,长短句可比。
- **DPO 的失败模式**:chosen 与 rejected 质量差距小、或 rejected 本身也合理时,信号弱;偏好数据有标注噪声时 DPO 会放大偏差;β 过小会过拟合偏好、损害通用能力。

## 批判性批注

- **长度归一化 vs 原论文**:原 DPO 用整句 log 概率(联合概率),Minimind 改成每 token 平均。这改变了损失的概率语义(不再是严格的 BT 推导),更像一个工程化的 surrogate loss。理论上更接近 RPO/IPO 系列的思路,作者未点出这一偏离。面试时若被追问"DPO 原式怎么写",要记得原式是 sum 不是 mean。
- **隐式奖励的数值稳定性**:`log_softmax` + `gather` 是标准做法,但 bfloat16 下低概率 token 的 log 可能精度损失。代码用 `autocast` 包前向但 loss 计算未显式上转 float32,大规模训练需注意。
- **参考模型显存开销**:DPO 必须同时驻留两份模型权重,policy + ref。Minimind 模型小无压力,但大模型场景这是 DPO 的主要显存痛点(LoraDPO、ref-free DPO 等变体正是为此而生),作者未提。
- **无 KL 估计、无 reward 监控**:训练日志只打印 dpo_loss/aux_loss/lr,没有打印 chosen/rejected 各自的 log 概率、隐式 reward、KL 估计等。实战中这些是判断"是否在学偏好 vs 是否在崩"的关键,缺失会让调参凭感觉。
- **batch 拼接切分的脆弱性**:`batch_size // 2` 切分强依赖 `cat([chosen, rejected])` 顺序,没有断言保护。若有人改动 DataLoader 顺序会静默出错,建议加 `assert` 或用显式的 chunk 标记。
- **EOS 纳入 mask 的取舍**:让模型学会生成 EOS 是好事,但若 rejected 的 EOS 也被算进 loss,等于"教模型 rejected 的结束方式",信号可能互相矛盾。作者未讨论这一边界。
- **单 epoch 默认**:`--epochs` 默认 1,DPO 通常确实 1-2 epoch 即可,但缺少早停/验证集监控,容易过拟合偏好数据。

## 篇内小思维导图

```mermaid
flowchart TD
    A[DPO 动机:跳过 RM/PPO] --> B[核心关系 r=β·log π/π_ref]
    B --> C[代入 Bradley-Terry 消去奖励]
    C --> D["损失: -log σ(β·(Δπ - Δref))"]

    E[dpo.jsonl 偏序对] --> F[chosen 与 rejected 同 prompt]
    F --> G[apply_chat_template + tokenize]
    G --> H[generate_loss_mask: 只 assistant 段含 EOS]
    H --> I[Shift Trick: x=ids[:-1], y=ids[1:]]
    I --> J[6 张量: x/y/mask × chosen/rejected]

    J --> K[train_epoch]
    K --> L["cat 拼接 2B batch"]
    L --> M["ref_model: no_grad 前向 → ref_log_probs"]
    L --> N["policy model 前向 → policy_log_probs"]
    M --> O[dpo_loss]
    N --> O
    O --> P["sum/seq_len 长度归一化"]
    P --> Q["切分 chosen/rejected: batch//2"]
    Q --> R["pi_logratios - ref_logratios"]
    R --> S["-logsigmoid(beta * logits)"]
    S --> T[梯度更新 policy, ref frozen]

    U[超参] --> V["beta=0.1 控制 KL 偏离"]
    U --> W["lr=4e-8 比 SFT 小一档"]
    U --> X["epochs=1 偏好微调"]
```
