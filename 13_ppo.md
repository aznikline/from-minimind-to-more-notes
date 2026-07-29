# 算法:Minimind 的 PPO

## 一句话精炼

Minimind 把 PPO 四模型(actor / old_actor / ref / critic)压在同一台 GPU 上,用"整段回答当一个 step"的极简 one-step MDP 取代标准 GAE,通过 ratio clip + ref KL 惩罚 + value MSE 三项 loss 实现对 SFT 模型的 RLAIF 微调。

## 核心概念(PPO-CLIP、actor/critic、reward model、advantage A、GAE、reference policy、KL penalty、value loss、experience replay/rollout)

- **PPO-CLIP**:Proximal Policy Optimization 的截断变体;通过 `ratio = π_new/π_old` 的截断 `[1-ε, 1+ε]` 限制单次更新幅度,防止策略崩溃。Minimind 默认 `clip_epsilon=0.1`(比标准 0.2 更保守)。
- **Actor(策略模型,可训练)**:最终想要的模型;初始化自 SFT,负责生成 response 并被反向更新。
- **Old Actor(旧策略,冻结)**:重要性采样的分母 `π_old`;每隔 `update_old_actor_freq=4` step 从当前 Actor 拷贝一次权重。严格说它"阶段性冻结、周期同步",而非永远冻结。
- **Reference Model(参考模型,永久冻结)**:SFT 模型快照,用来算 KL 散度,充当"锚点"防止 reward hacking。
- **Critic(价值模型,可训练)**:在 MiniMindLM 基础上把 `lm_head` 换成 `Linear(hidden_size, 1)`,输出 token-level `V(s_t)`;但训练代码中只取最后一个有效 token 的 value 作为整段价值,把整个 response 视为 one-step MDP。
- **Reward Model(奖励模型,冻结)**:外置 `InternLM2-1.8B-Reward`,对完整对话打一个 scalar;被 clip 到 `[-3.0, 3.0]`。推理模型场景下,answer 部分单独打分并与整体得分加权(0.4 / 0.6)合并。
- **Advantage A**:实际 reward 减去 baseline;Minimind 极简版 `A = R - V.detach()`;标准版用 GAE 逆序折现累加 TD error。
- **GAE(Generalized Advantage Estimation)**:标准做法按 token 算 `δ_t = r_t + γV_{t+1} - V_t`,`A_t = Σ(γλ)^l δ_{t+l}`;Minimind 代码里**并未实现 GAE**,只是用了 `rewards - values.detach()`,作者在文中明确说明这是极简 one-step 近似。
- **KL penalty**:Actor 与 Reference 的对数概率差 `kl_ref = (actor_logp - ref_logp).mean()`,作为惩罚项进入总 loss;另有一个 `kl = (actor_logp - old_logp).mean()` 仅作监控,不进 loss。
- **Value loss**:Critic 预测的 `V` 与真实 `rewards` 的 MSE(`F.mse_loss(values, rewards)`);系数 `vf_coef=0.5`。
- **Experience replay / Rollout**:文中不使用经验回放池,而是 on-policy 在线 rollout——`torch.no_grad()` 下 `generate` 一次,用完即弃,下一个 batch 重新采样。这本质是 PPO 的多 epoch 重用 + 在线采样,而非 DDPG/DQN 式经验回放。

## 关键公式(LaTeX:PPO clipped objective、GAE 优势估计、value loss、总 loss 各项)

**重要性采样比率**(注意 Minimind 用整段 sum 后的 logp 而非 token-level ratio):
$$\text{ratio} = \exp\!\big(\log\pi_\theta(a_{1:T}|s) - \log\pi_{old}(a_{1:T}|s)\big)$$

**PPO-CLIP 目标**(标准 token 级):
$$L^{CLIP}_t = -\min\!\left( \text{ratio}_t \cdot A_t,\ \text{clip}(\text{ratio}_t, 1-\epsilon, 1+\epsilon)\cdot A_t \right)$$
Minimind 实际用句子级标量 `ratio` 和 `advantages`,等价于把整段视为一个动作。

**GAE(标准版,Minimind 未实现但文中给出)**:
$$\delta_t = r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t)$$
$$A_t = \sum_{l=0}^{T-t}(\gamma\lambda)^l\,\delta_{t+l}$$
(γ 通常 1.0,λ 通常 0.95)

**Step Reward 塑形(标准版)**:
$$r_t = \begin{cases} -\beta\,\text{KL}_t & t < T \\ R_{final} - \beta\,\text{KL}_T & t = T \end{cases}$$

**Value loss**:
$$L_{critic} = \frac{1}{2}\big(V_\omega(s) - R\big)^2$$
Minimind 代码:`F.mse_loss(values, rewards)`,默认 `reduction='mean'`。

**总 loss**(Minimind 实际形式):
$$L = \frac{1}{\text{accum}}\Big(L^{CLIP}_{policy} + c_v\,L_{critic} + c_k\,\text{KL}_{ref} + L_{aux}^{MoE}\Big)$$
其中 `vf_coef=c_v=0.5`、`kl_coef=c_k=0.02`,KL_ref 用 `(actor_logp - ref_logp).mean()` 近似(非严格 KL,是 log-prob 差的均值)。

## 关键算法/流程(rollout 采样 → 算 advantage → 算 ratio → clip → 更新;四模型/多模型协作)

1. **编码**:Prompt 字符串列表 → `tokenizer(..., padding_side="left")`,得到 `[B, P]`(左填充,保证最后一个有效 token 右对齐,便于自回归生成)。
2. **Rollout**:`actor_model.generate(do_sample=True, temperature=0.8)` 在 `torch.no_grad()` 下生成,得 `gen_out [B, P+R]`;解码回 `responses_text`。采样阶段不算梯度,显存只放推理图。
3. **奖励打分**:`calculate_rewards` 调外置 Reward Model 打 sentence-level scalar,clip 到 `[-3, 3]`;推理模式下叠加格式奖励(`</think>...<answer>...</answer>` 命中给 0.5)+标记奖励(四个标签各 0.25,最多 1.0)+ answer 内单独打分加权。
4. **Critic 前向**:`values_seq = critic_model(gen_out)` 得 `[B, P+R]`;用 `full_mask * arange` 找最后一个有效 token 索引,取出 `values [B]` 作为整段价值基线。
5. **Advantage**:`advantages = rewards - values.detach()`(one-step 极简;无 GAE、无 γ/λ)。
6. **四模型并发前向**(对同一 `gen_out`):
   - Actor(有梯度)→ `actor_logp [B]`
   - Old Actor(无梯度)→ `old_logp [B]`
   - Ref Model(无梯度)→ `ref_logp [B]`
   
   各自用 `log_softmax(logits[:, :-1]).gather(labels)` 取每 token logp,再乘 `final_mask` 后 `sum(dim=1)` 得句子级总 logp。
7. **Ratio + Clip**:`ratio = exp(actor_logp - old_logp)`,`surr1 = ratio * A`,`surr2 = clamp(ratio, 1-ε, 1+ε) * A`,`policy_loss = -min(surr1, surr2).mean()`。
8. **Value loss + KL ref**:`value_loss = mse(values, rewards)`,`kl_ref = (actor_logp - ref_logp).mean()`,`kl` 仅监控。
9. **总 loss**:`(policy_loss + vf_coef*value_loss + kl_coef*kl_ref + aux_loss)/accum_steps` → `backward`。
10. **更新**:每 `accumulation_steps` 步做 `clip_grad_norm_(1.0)` + `actor_optimizer.step()` + `critic_optimizer.step()` + 调度器 step + 清零。
11. **Old Actor 同步**:每 `update_old_actor_freq=4` step,把当前 Actor 的 state_dict 拷到 CPU 再加载到 old_actor(避免计算图纠缠)。
12. **保存**:每 `save_interval=10` step 存半精度 actor 权重 + 完整 checkpoint(含 critic/optimizer/scheduler)。
13. **显存清理**:每个 step 末 `del` 掉所有大张量,防止 PPO 一个 step 内"推理+前向+反向"叠加导致 OOM。

四模型协作本质:Actor 是选手、Critic 是教练预测分、Reward Model 是考官、Ref 是锚防止走火入魔、Old Actor 是"上一刻的我"作重要性采样分母。可训练的只有 Actor 和 Critic,其余三个冻结。

## 源码要点(Minimind PPO 代码主体:calculate_rewards、get_per_token_logps、训练 epoch、KL 估计方式)

- **CriticModel**:继承 `MiniMindForCausalLM`,把 `lm_head` 换成 `nn.Linear(hidden_size, 1)`;`forward` 取 `model.norm(outputs[0])` 再过 value_head,`squeeze(-1)` 输出 `[B, SeqLen]` token-level value。
- **calculate_rewards**(train_ppo.py 内嵌):
  - `reasoning_model_reward`:正则 `^思考和...` 命中给 0.5 格式分;`mark_num` 数 4 个标签各 0.25,最多 1.0。
  - Reward Model API 调用:`reward_model.get_score(reward_tokenizer, tmp_chat)`,得分 clip 到 `[-3, 3]`;推理模式把 `<answer>` 内容单独打分,`score = 0.4*整体 + 0.6*answer`。
  - 没有实现 per-token reward 分配,直接返回 `[B]` 句子级 reward。
- **logps 计算**(文中称 get_per_token_logps,实为内联代码):
  ```
  logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
  actor_logp = (logp_tokens * final_mask).sum(dim=1)
  ```
  注意是**句子级 sum**,而非 token-level 各自算 ratio——这是 Minimind 与标准 PPO 的核心差异。
- **final_mask 构造**(因果 LM 错位关键):
  ```
  labels = gen_out[:, 1:].clone()
  resp_mask = arange(seq_len) >= prompt_length - 1   # 错位 -1 是因为"用 B 预测 C"发生在 B 的位置
  final_mask = resp_mask & (~labels.eq(pad_token_id))
  ```
  通过矩阵运算构造 mask,避免 Python for 循环,充分利用 GPU 并行。
- **两次 padding 的使命**:
  - 第一阶段 Prompt 左填充:保证 Causal LM 最后一个有效 token 右对齐,能正确生成下一个词。
  - 第二阶段 generate 右填充:不同句子提前 `<eos>` 后,`generate` 自动在右侧补 `<pad>` 维持矩形张量。
  - `final_mask` 同时屏蔽左侧 prompt/pad 与右侧提前结束的 pad。
- **KL 估计方式**:**非严格 KL 散度**。严格 KL 为 `Σ p log(p/q)`,代码用 `(actor_logp - ref_logp).mean()` 近似——这是 log-prob 差的均值,在 PPO 文献中常被称为 "k3 estimator" 或 `KL_left`,是一种常用近似(等价于 `E[log(π/π_ref)]`),计算便宜但可能为负。`kl_coef=0.02` 很小。
- **采样阶段不算 logp 的原因**:
  1. 自回归 generate 若开梯度,要把 1000 步前向的激活值全堆显存,OOM。
  2. 生成完再一次性并行喂完整序列,Transformer 一次前向就能拿到所有 token logp + 计算图,效率远高于循环 1000 次。
- **超参默认值**:`lr=8e-8`(极小)、`clip_epsilon=0.1`、`vf_coef=0.5`、`kl_coef=0.02`、`update_old_actor_freq=4`、`max_seq_len=66`、`max_gen_len=1536`、`batch_size=2`、`accumulation_steps=1`、`grad_clip=1.0`、`dtype=bfloat16`。
- **Critic 热启动**:`CriticModel` 复用基座 SFT 权重 `load_state_dict(state_dict, strict=False)`,只新增 value_head 随机初始化。
- **DDP 注意**:RoPE 的 `freqs_cos/freqs_sin` buffer 要加入 `_ddp_params_and_buffers_to_ignore`,否则 DDP 报错;只有 Actor 和 Critic 包 DDP,old_actor/ref/reward 不包。

## 作者独到见解/类比

- **"先看代码后讲理论"**:作者明确反对"先理论后代码"的传统讲法,认为会让人晕。先把 Minimind 极简 PPO 跑通,再讲与工业 PPO 的差异,最后用理论解释。
- **"闭卷考试"类比**:Rollout 阶段是"闭卷考试",Actor 只管做题,不算分不算梯度;后续多模型前向是"各路老师对每个填空给看法";最后 Loss 阶段是"算总账"。
- **"步子迈太大扯着蛋"**:用 `torch.clamp(ratio, 1-ε, 1+ε)` 解释 PPO clip 的灵魂——"就算这次做得特别好,一次最多也只能进步 10%,防止策略崩溃"。
- **"考官/教练/选手/锚"四角色比喻**:Actor=选手、Critic=教练预测分、Reward Model=考官、Reference=锚防止走火入魔、Old Actor=上一刻的我。这是全书最生动的多模型协作类比之一。
- **"极简即教学"**:作者毫不避讳地承认 Minimind 跳过了 GAE、token-level ratio、reward 分配,明确说"为了极简把整个生成视为一步 One-step MDP",把教学清晰度置于算法完整性之上。
- **"两次 padding 使命不同"**:作者敏锐指出左填充(prompt 对齐)与右填充(生成矩形维持)是两个独立机制,容易让初学者混淆,专门用一节澄清。
- **"采样阶段不存 logp 是工程必然"**:作者把"为什么不在 generate 时算 logp"提为独立小节,从 OOM 与并行效率两个角度论证,这是工业实践中至关重要但常被忽略的工程要点。

## 面试考点(PPO 三种 loss、clip 范数、为何加 KL、reward 归一化、advantage 估计、为何 4 个模型显存爆)

- **PPO 三种 loss**:
  1. **Policy loss(Actor)**:`-min(ratio*A, clip(ratio,1-ε,1+ε)*A).mean()`,负号因为要最大化目标故取负做梯度下降。
  2. **Value loss(Critic)**:`MSE(V, R_target)`,Minimind 用 `mse(values, rewards)`(标准版用 `R_t = A_t + V_t`)。
  3. **KL penalty**:`kl_coef * (actor_logp - ref_logp).mean()`,防 reward hacking;Minimind 还有一个**监控用**的 `kl = (actor_logp - old_logp).mean()` 不进 loss。
- **clip 范数**:`clip_epsilon=0.1`(Minimind),标准 PPO 常用 0.2;ratio 被限制在 `[0.9, 1.1]`。梯度裁剪用 `clip_grad_norm_(1.0)`(注意这是梯度范数,与 ratio clip 是两回事)。
- **为何加 KL**:防 reward hacking——Actor 为讨好 Reward Model 可能输出乱码或钻空子;KL_ref 把 Actor 拉回 SFT 分布附近,保语言流畅性。`kl_coef=0.02` 小是因为它只是软约束。
- **reward 归一化**:Minimind **未做** advantage 归一化(标准做法是 `A = (A - mean)/std`);reward 只做了 clip `[-3, 3]`。这是 Minimind 与工业实现的差异之一,面试可指出"标准 PPO/TRL 会对 advantage 做标准化以稳定训练,Minimind 省略了"。
- **advantage 估计**:Minimind 用 one-step `A = R - V.detach()`;标准用 GAE `Σ(γλ)^l δ_{t+l}`。Minimind 的简化会丢失 token 级信用分配,适合小模型教学,不适合工业。
- **为何 4 个模型显存爆**:一个 step 内同时需要:Actor 推理(generate)→ Actor 前向(带梯度,留计算图)→ Critic 前向(带梯度)→ Old Actor 前向(无梯度但权重在显存)→ Ref 前向(无梯度但权重在显存)→ Reward Model 前向(无梯度但权重在显存)。即便只有 Actor 和 Critic 有梯度,五个模型的**参数本身**就占显存,且 Actor 的生成阶段 + 前向阶段激活值叠加。作者因此强调"手动 del 大张量"是防 OOM 的关键。这也是工业实现常用 LoRA / offload / 参数共享的原因。
- **ratio 为句子级而非 token 级**:Minimind 用 `exp(actor_logp - old_logp)` 其中 logp 是整段 sum,等价于把整段 response 视为一个联合动作,丢失了 token 级信用分配——这是与 TRL/PPO 标准实现最大的算法差异,面试可作为批判点。
- **KL 估计非严格**:`(actor_logp - ref_logp).mean()` 是 k3 左 KL 近似,计算便宜但可能为负、不保证非负;严格 KL 应 `Σ p(log p - log q)`。面试可指出这是工程近似。
- **Old Actor 同步频率**:`update_old_actor_freq=4`,即每 4 step 同步一次 old policy;在标准 PPO 中 old policy 在每个 rollout batch 开始时固定,然后在 K 个 epoch 内重复使用同一批数据。Minimind 的做法是滚动同步,没有"多 epoch 重用同一批数据"的显式逻辑。
- **学习率极小 `8e-8`**:RLHF 的 lr 通常比 SFT 小一到两个数量级,因为 RL 信号噪声大,大 lr 会直接把 SFT 能力打崩。

## 批判性批注

1. **One-step MDP 是教学妥协,代价是丢失信用分配**。把整段 response 当一个动作,所有 token 共享同一个 advantage,无法区分"哪个 token 贡献了高分"。对短回答尚可,对 1536 token 的长生成会严重退化——这正是标准 PPO 用 GAE 的理由。作者虽明确说明这是极简,但未量化其与 GAE 的效果差距,初学者易误以为"这就是 PPO"。
2. **句子级 ratio 数学上不严谨**。`ratio = exp(Σ logp_new - Σ logp_old)` 是联合概率比,当 response 长 1000 token 时,即使每个 token 概率只变 0.1%,乘积也会放大到 e^1 ≈ 2.7,极易撞到 clip 边界,使 clip 几乎总是触发,丧失 PPO 的"允许小步更新"的梯度信号。标准实现用 token-level ratio + per-token advantage 才是 PPO 论文的本意。
3. **KL 估计用 logp 差均值,可能为负**。严格 KL ≥ 0,而 `(actor_logp - ref_logp).mean()` 在 Actor 比 Ref 概率更集中时可能为负,作为惩罚项反而鼓励偏离——这是潜在 bug。工业实现多用 `0.5*(logp_old - logp_new)^2`(k3) 或真正的 KL `Σ p(log p - log q)`。
4. **advantage 未做标准化**。标准 PPO/TRL 会对一个 batch 内的 advantage 做 `(A - mean)/std`,Minimind 省略,reward 尺度依赖 Reward Model 输出范围(虽 clip 到 [-3,3]),训练稳定性会受影响。作者未提此点。
5. **Value loss 用 `mse(values, rewards)` 而非 `mse(values, R_t)`**。标准做法 target 是 `R_t = A_t + V_t`(即 GAE return),Minimind 直接用 raw reward 当 target,等价于假设 `V_target = R_final`,在 one-step MDP 下自洽,但若未来扩展到 GAE 会不一致。
6. **`update_old_actor_freq=4` 的滚动同步不是标准 PPO 的多 epoch 重用**。标准 PPO 在一个 rollout batch 上做 K(4~10)次 mini-batch epoch,old policy 在整个 batch 期间固定;Minimind 每 4 step 同步 old,等于把 old policy 漂移化,削弱了重要性采样的"同分布"假设。
7. **reward shaping 的格式奖励(0.5 + 1.0)与 RM 得分(范围 [-3,3])量级不匹配**。格式分最多 1.5 而 RM 最多 3.0,推理模式下两者叠加可能让格式奖励主导,偏离"语义质量"目标。作者未讨论此量级平衡。
8. **作者明确承认"未联网、纯教学"**,这些简化在教学语境下是合理的——Minimind 的目标是让人理解 PPO 四模型协作与 loss 结构,而非复现 SOTA。批判应落在"读者需知道这些简化在工业中不能直接照搬"。
9. **积极面**:final_mask 的错位 `-1` 解释、两次 padding 的区分、采样阶段不算 logp 的工程论证,这几段是全书质量最高的工程讲解之一,对初学者极有价值。

## 篇内小思维导图(缩进树)

```
Minimind PPO
├── 数据
│   ├── rlaif-mini.jsonl (1万条 SFT 摘录, assistant="空")
│   └── RLAIFDataset.__getitem__ 返回 {prompt, answer} 字符串(不 token 化)
├── 四模型
│   ├── Actor (可训练, SFT 初始化, 最终产出)
│   ├── Old Actor (阶段冻结, 每4step同步, 重要性采样分母)
│   ├── Reference (永久冻结, SFT 快照, KL 锚点)
│   └── Critic (可训练, lm_head→Linear(h,1), 热启动自 SFT)
├── 外置 Reward Model
│   └── InternLM2-1.8B-Reward (冻结, score clip [-3,3])
├── ppo_train_epoch 10 阶段
│   1. 编码 (left padding)
│   2. Rollout (no_grad generate → gen_out [B,P+R])
│   3. Reward (calculate_rewards: 格式+标记+RM, 推理模式 answer 加权)
│   4. Critic 前向 → values [B] (取最后有效 token)
│   5. Advantage = rewards - values.detach()  [one-step, 无 GAE]
│   6. Actor/Old/Ref 三路前向 (log_softmax+gather+final_mask+sum)
│   7. Loss = policy_clip + vf_coef*value_mse + kl_coef*kl_ref + aux
│   8. backward + clip_grad_norm + optimizer.step
│   9. 每4step 同步 old_actor (state_dict→CPU→load)
│   10. del 大张量防 OOM
├── 关键工程细节
│   ├── final_mask 错位 -1 (用 B 预测 C 发生在 B 位置)
│   ├── 两次 padding (左填充对齐生成 / 右填充维持矩形)
│   ├── 采样不算 logp (OOM + 并行效率)
│   └── KL 用 (logp_a - logp_r).mean() 近似, 可能非负不保证
├── 标准版差异 (文中给出但未实现)
│   ├── GAE: δ_t = r_t + γV_{t+1} - V_t; A_t = Σ(γλ)^l δ
│   ├── token-level ratio 而非句子级
│   ├── reward 分配到每 token (中间 -βKL, 末位 R_final-βKL)
│   ├── value target 用 R_t = A_t + V_t
│   └── advantage 标准化 (A-mean)/std
└── 超参默认
    ├── lr=8e-8, clip_epsilon=0.1, vf_coef=0.5, kl_coef=0.02
    ├── update_old_actor_freq=4, save_interval=10
    └── max_seq_len=66, max_gen_len=1536, batch_size=2
```
