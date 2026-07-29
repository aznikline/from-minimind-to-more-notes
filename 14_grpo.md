# 算法:Minimind 的 GRPO 及其变体

## 一句话精炼

GRPO 用"同 Prompt 组内采样 + Z-score 归一化优势"取代了 PPO 的 Critic,把显存砍掉一个 LLM 的同时,催生出 Dr.GRPO(修长度偏置)、DAPO(动态采样+解耦裁剪)、GSPO(序列级似然比稳住 MoE)、SAPO(软门控非对称温度)、GTPO(熵加权 token 级信用分配)一整条变体链,本质都是围绕"省掉 Critic 之后,如何让 Token 级优化与 Sequence 级奖励对齐、如何在长 CoT 中精分配 credit"这一主线打的补丁。

## 核心概念(GRPO:组采样去 critic;Dr.GRPO:长度偏置修正;DAPO:动态采样+解耦裁剪;GSPO:序列级/MoE 稳定;SAPO:soft advantage+非对称温度;GTPO:熵加权 token 级 credit assignment)

- **GRPO(Group Relative Policy Optimization)**:DeepSeek-R1 系的基石。给定 Prompt $q$,策略模型 $\pi_\theta$ 一次性生成 $G$ 个回答 $\{o_1,\dots,o_G\}$,奖励模型/规则器给分 $\{r_i\}$,在组内做 Z-score 归一化得到优势 $A_i=(r_i-\mu)/\sigma$。彻底移除 Critic,4 模型降为 3 模型(Actor / Ref / Reward),节省整整一个 LLM 的显存。
- **Dr.GRPO(GRPO Done Right)**:从策略梯度无偏估计出发修两处偏置——(1) Baseline 缩放因子应为 $1/(K-1)$ 而非 $1/K$;(2) Loss 除以响应长度 $|o|$ 会稀释负样本惩罚、压制正样本长链。剔除长度除法,堵掉"答错就拼命水字数"的漏洞。
- **DAPO(Decoupled clip & Dynamic sAmpling PO)**:四件套——Dynamic Sampling(丢掉 std=0 的无效组,持续采样直到攒够有效梯度)、Clip-Higher(非对称裁剪,上限放宽到 $1.28$、下限 $0.8$ 不变,给低概率优质 token 探索空间)、Token-level Loss(全局 token 拉平,长序列权重更大,鼓励 Long-CoT)、Overlong Filtering + Soft 软惩罚(截断样本 Mask 掉、过长则扣分)。
- **GSPO(Group Sequence Policy Optimization)**:Qwen 系。把 Token 级重要性采样抬到**序列级**,用长度归一化(几何平均)的似然比 $s_i(\theta)=(\pi_\theta(y_i|x)/\pi_{\theta_{old}}(y_i|x))^{1/|y_i|}$ 代替逐 token ratio,再做序列级裁剪。方差极低,MoE 路由不再漂移,可**抛弃昂贵的 Routing Replay**。
- **SAPO(Soft Advantage Policy Optimization)**:Qwen 系,已进 ms-swift / trl。用 Sigmoid 连续软门控代替硬 Clip——偏离越多权重越小但永不归零,榨干每次采样的梯度价值;并引入**非对称温度** $\tau_{pos}/\tau_{neg}$,对负优势(压低错误 token)设更陡的温度,防止"压低一个 token 把全词表垃圾 Logits 顶起来"。保留 Token 级操作,可作为 drop-in 替换 GRPO/PPO Loss 那几行。
- **GTPO(Group Token Policy Optimization)**:核心假设——正确推理序列中,模型高熵(纠结)的位置恰是决策点。用策略熵 $H_{i,t}$ 作为探针,把序列级优势 $A_i$ 按 $w_{i,t}\propto H_{i,t}/\sum_k H_{k,t}$ 动态再分配给每个 token,$A_{i,t}=A_i\cdot w_{i,t}$。**策略熵是 Forward 副产物,无需 PRM 即获伪过程奖励**;负向序列则回退平缓分配防误伤。

## 关键公式

### GRPO 目标函数
$$\mathcal{J}_{GRPO}(\theta)=\mathbb{E}_{q,\{o_i\}_{i=1}^G}\left[\frac{1}{G}\sum_{i=1}^G\left(\min\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i,\ \mathrm{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)},1-\epsilon,1+\epsilon\right)A_i\right)-\beta\,\mathbb{D}_{KL}(\pi_\theta\|\pi_{\mathrm{ref}})\right)\right]$$

### GRPO 组内 Z-score 优势
$$A_i=\frac{r_i-\mu}{\sigma},\qquad \mu=\frac{1}{G}\sum_{j=1}^G r_j,\quad \sigma=\mathrm{std}(\{r_j\}_{j=1}^G)$$
Minimind 实现里加了 `+1e-4` 防 $\sigma=0$,并 `clamp(-10,10)` 防爆炸。

### Minimind 逐 token KL(Schulman 无偏估计形式)
$$\mathrm{KL}_t=\exp(\log\pi_{\mathrm{ref}}(a_t|s_t)-\log\pi_\theta(a_t|s_t))-(\log\pi_{\mathrm{ref}}(a_t|s_t)-\log\pi_\theta(a_t|s_t))-1$$

### Dr.GRPO 关键修正
- 无偏缩放因子:$\frac{1}{K-1}$ 替代 $\frac{1}{K}$;
- 剔除长度除法:Loss 不再除以 $|o|$,正负 token 惩罚/奖励等权对齐,堵掉"靠字数稀释错误惩罚"漏洞。

### DAPO 非对称裁剪
$$\mathrm{clip}(r_t,\ 1-\epsilon_{\mathrm{low}},\ 1+\epsilon_{\mathrm{high}}),\qquad \epsilon_{\mathrm{high}}>\epsilon_{\mathrm{low}}\ (\text{如 }1.28/0.8)$$
Token-level Loss:所有 batch 内 token 拉平求策略梯度,长序列权重天然变大。

### GSPO 序列级似然比
$$s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}=\exp\!\left(\frac{1}{|y_i|}\sum_{t}\big(\log\pi_\theta(y_{i,t}|x,y_{i,<t})-\log\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})\big)\right)$$
目标对 $s_i(\theta)$ 做 $[1-\epsilon,1+\epsilon]$ 序列级裁剪,再乘序列级优势 $A_i$。

### SAPO 软门控与非对称温度
$$g(r_t)=\sigma\!\left(\frac{r_t-1}{\tau}\right)\ \text{(示意,平滑取代硬 clip)}$$
正负优势分用不同温度 $\tau_{pos},\tau_{neg}$,负优势($A<0$)给更陡温度,谨慎压低坏 token。

### GTPO 熵权重再分配
$$w_{i,t}\propto\frac{H_{i,t}}{\sum_{k}H_{k,t}},\qquad A_{i,t}=A_i\cdot w_{i,t}$$
高熵 token(决策点)分到更多奖励,低熵 token(标点/固定句式)分到更少;负向序列回退平缓分配。

## 算法对比表(6 个算法)

| 算法 | 核心改动 | 解决的偏置 | 适用场景 | 一句话 |
|---|---|---|---|---|
| **GRPO** | 组内 Z-score 优势替代 Critic | Critic 显存/Value 估计难 | 数学/代码等结果导向任务、Agentic RL | 砍掉私教,组内打架定胜负 |
| **Dr.GRPO** | 无偏缩放 $1/(K-1)$ + 剔除长度除法 | Baseline 偏差 + 长度稀释偏置 | 长思维链、抑制"水字数作弊" | 修掉 GRPO 数学小偏误,堵作弊漏洞 |
| **DAPO** | 动态采样 + 非对称裁剪 + Token-level Loss + Overlong 过滤 | 算力浪费 + 熵坍塌 + 长度偏置 + 截断噪声 | Long-CoT、需要强探索 | 长链该重、好 token 该放、无效组该丢 |
| **GSPO** | 序列级似然比 + 长度归一化 + 序列级裁剪 | Token 级高方差 + MoE 路由漂移 | 大规模 MoE(DeepSeek-V3/Qwen 系) | 把 ratio 拉到序列级,路由不再崩 |
| **SAPO** | Sigmoid 软门控 + 非对称温度 $\tau_{pos}/\tau_{neg}$ | 硬裁剪梯度归零 + 负优势带崩词表 | MoE + 长文本,需 drop-in 替换 | 梯度永不为零,惩罚更谨慎 |
| **GTPO** | 策略熵 $H_{i,t}$ 作权重动态再分配 $A_{i,t}=A_i w_{i,t}$ | 稀疏信用分配(大锅饭) | 超长 CoT、无 PRM 时的精细 credit | 白嫖 Forward 副产物,关键决策吃最多分 |

## 关键算法/流程(GRPO 组采样训练流程)

Minimind `grpo_train_epoch` 单次迭代流水线:

1. **出题(Prompt Tokenize)**:`batch['prompt'` 取出 B 条 prompt,`tokenizer(padding_side="left")` 左填充以保证右侧生成对齐;`max_seq_len` 时从左侧截断。
2. **答题(Rollout)**:`model.generate(num_return_sequences=num_generations, do_sample=True, temperature=0.8, max_new_tokens=max_gen_len)` 一次性为每个 prompt 生成 $G$ 条不同回答,产出 $B\times G$ 条序列 `[B*num_gen, P+R]`,切片出 `completion_ids = outputs[:, P:]`。
3. **打分(Reward)**:`calculate_rewards` 返回 `[B*num_gen]` 张量——推理模式下先做规则奖励(严格格式 0.5 分 + 标签完整每标签 0.25 共 1.0 分),再做 RM 打分 `max(min(score,3.0),-3.0)` 截断保护;推理模式额外对 `<answer>` 内容单独打分并加权 `score=score*0.4+answer_score*0.6`。
4. **组内排名(Advantage)**:`rewards.view(-1, num_gen)` 按问题分组,算 $\mu,\sigma$ 并 `repeat_interleave`;`advantages = clamp((r-μ)/(σ+1e-4), -10, 10)`。正数=同组拔尖,负数=拖后腿。
5. **算 logps**:`get_per_token_logps` 分别对策略模型(开梯度)和参考模型(`no_grad`)各算一次,错位切片 `logits[:, :-1, :]` + `torch.gather(log_softmax, ids_row)` 抽出实际生成 token 的对数概率。
6. **KL & Mask**:`is_eos` 算 `completion_mask`;`kl_div = ref_per_token_logps - per_token_logps`,Schulman 无偏形式 `per_token_kl = exp(kl_div) - kl_div - 1`。
7. **Loss & 反传**:`per_token_loss = -(exp(logps - logps.detach()) * advantages.unsqueeze(1) - β * per_token_kl)`,序列内按 mask 求平均再 batch 平均;`+ aux_loss`(MoE 辅助)/`accumulation_steps`;`backward()`。
8. **优化器步进**:累积到 `accumulation_steps` 后做 `grad_clip`、`optimizer.step()`、`scheduler.step()`、`zero_grad()`,记录 Wandb、按步数存 ckpt。

一句话总览:**出题 → 答题 → 打分 → 组内排名 → 分析得失(KL) → 反思并进步**。

## 源码要点(Minimind GRPO 代码)

### `calculate_rewards(prompts, responses, reward_model, reward_tokenizer)` → `[B*num_gen]`
- **两阶段**:规则奖励(仅 `args.reasoning==1` 触发) + 外部 RM 打分(全程执行)。
- **规则奖励内部**:`pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"` 严格正则,完全匹配给 0.5 分;`mark_num` 统计 `<think>/</think>/<answer>/</answer>` 四标签各 0.25 共 1.0,即使格式不完美也给"软保底"。
- **RM 打分**:ChatML 正则解析 prompt → `tmp_chat = messages + [{"role":"assistant","content":response}]` → `reward_model.get_score`;`scale=3.0`,`score = max(min(score, scale), -scale)` 截断防梯度爆炸。
- **推理模式加权融合**:正则提取 `<answer>...</answer>` 内容,RM 对最终答案单独打分,`score = score*0.4 + answer_score*0.6`,逼模型把注意力放在结论而非堆废话。

### `get_per_token_logps(mdl, input_ids, n_keep)` → `[B*num_gen, R]`
- **核心目的**:把"随机采样出的字"换成"该字的精确数学概率",才能反传求导。
- **错位切片**:`logits = mdl(input_ids, logits_to_keep=n_keep+1).logits[:, :-1, :]`——语言模型用第 $t$ 位置特征预测 $t+1$ 位置词,丢掉最后一位才能与目标 token 序列对齐(自回归错位,防越界)。
- **`torch.gather` 抽概率**:`log_softmax` 后用 `ids_row.unsqueeze(1)` 当索引,从 `[Seq, Vocab]` 巨阵里精准抠出实际生成 token 的那列概率——10 万词表只关心模型真正吐出的那一个字。
- **推理态 detach**:`is_inference()` 时对 `input_ids/ids_row` 做 `detach().clone()`,避免推理张量带进训练图。
- **外层双调用**:策略模型在 `autocast_ctx`(带梯度,反传源头);参考模型在 `torch.no_grad()`(纯锚点)。两者相减得 KL,防模型"走火入魔钻 RM 空子"。

### 主 Loss 行
```python
per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
                   - args.beta * per_token_kl)
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
loss = (policy_loss + aux_loss) / args.accumulation_steps
```
- `exp(logps - logps.detach())` = PPO 的重要性采样 ratio(等价但数值更稳);
- 序列内按 mask 求平均(注意:这里正是 Dr.GRPO 想砍掉的"长度除法");
- MoE 时叠加 `aux_loss`(负载均衡辅助 Loss)。

## 作者独到见解/类比

- **PPO vs DPO vs GRPO 三比喻**:PPO 像请昂贵私教(Critic)每步打分,贵且难题看走眼;DPO 像背题库,遇新题就懵;GRPO 像搞题海战术的小组学习——不要私教、同一题让脑子里的不同想法"打架"、谁过测试用例多谁就是老大。
- **"连坐"机制的本质**:单次看 GRPO 给 100 行代码全打负分像冤枉好人,但靠 Group Size=8 的**统计平均**——正确逻辑在多组里反复出现被推高、错误步骤只在负样本里被抑制,模型最终能区分"哪些 token 是成功的关键"。这正是 GRPO 用 Outcome reward 实现 Process-like 学习的统计学底座。
- **为什么 Agentic RL 爱 GRPO**:去肥增瘦 + 摆脱 Critic + Online 探索(对比 DPO 的 Offline 数据)。能命中"Aha Moment"——一旦偶然做对,奖励函数即强化该路径,涌现出数据集没覆盖的解题路径。
- **GTPO 的"白嫖"哲学**:策略熵是 Forward 副产物,模型本就算出来了,GTPO 直接拿来当 credit 探针,不引入 PRM 即获伪过程奖励——"高熵 = 决策点 = 关键 token"这一直觉假设极优雅。
- **SAPO 风险不对称**:压低一个错误 token 会把词表里成千上万垃圾 token 的 Logits 被动顶起来,所以惩罚要更谨慎——给负优势更陡温度是对"词表耦合"的深刻洞察。

## 面试考点

1. **为何 GRPO 省 Critic / 显存优势有多大**:GRPO 用同一 prompt 的 $G$ 个采样的均值/方差做 baseline,PPO 用 Critic 预测绝对 Value;模型数 4→3,Critic 通常与 Actor 同参数量,省掉整整一个 LLM 显存。Agentic 长程任务里 Critic 还难训练、估值不准致训练震荡,GRPO 用相对比较更稳。
2. **Dr.GRPO 修长度偏置的具体两处**:① Baseline 缩放 $1/K→1/(K-1)$ 的无偏校正;② Loss 不再除以响应长度 $|o|$——否则负样本靠水字数稀释惩罚、正样本长链被过度惩罚。
3. **信用分配(Credit Assignment)难点 & GRPO 的统计平均解法 + GTPO 的熵权重改进**:见上"连坐机制"与"白嫖熵"两节。核心是 Outcome-based → Process-like 的桥接靠采样统计或熵探针,而非昂贵的 PRM。
4. **GRPO 的优势是整句统一 vs 逐 token KL 是逐 token 的矛盾**:宏观 $A_{\mathrm{seq}}$ 整句级主导,但 KL 天然逐 token;某些实现拆成 $A_t=A_{\mathrm{seq}}-\beta\mathrm{KL}_t$ 让最终优势微变。
5. **MoE 训练 GRPO 为何崩 / GSPO 怎么救**:Token 级 ratio 方差大 → MoE 路由漂移 → 需要 Routing Replay 这种昂贵 hack;GSPO 抬到序列级 + 长度归一化似然比,方差极低,可丢弃 Routing Replay。
6. **DAPO 四件套分别修什么**:动态采样修 std=0 浪费、Clip-Higher 修熵坍塌、Token-level Loss 修长度偏置、Overlong Filtering+Soft 修截断噪声。
7. **SAPO 非对称温度为什么惩罚更谨慎**:压低一个 token 会把词表其他垃圾 token 的 Logits 顶起来,负优势需要更陡温度防带崩模型。
8. **PRM vs ORM & PRM 标注成本解法**:PRM 逐 step 打分、反馈密集;MCTS 自动生成分支或规则验证器(编译器/符号工具)自动化构建训练数据缓解标注贵。
9. **STaR "左脚踩右脚"冷启动**:生成→过滤→合理化补充(把答案当 hint 倒推)→SFT→循环;为 RL 注入初始 CoT 能力,避免模型"永远拿不到正奖励"的死循环。
10. **GRPO vs DPO 适用差异**:DPO 是 Offline preference 模仿,无探索、遇 OOD 即崩;GRPO 是 Online 试错,能涌现新路径,且天然契合 Rule-based Reward(编译过 0.2、测试过半 0.5、全对 1.0)。

## 批判性批注

- **作者对"无偏缩放 $1/(K-1)$"的论证偏口语化**:实际 Dr.GRPO 论文里更精确的表述是 std 归一化中分母应为 $\sqrt{\mathrm{Var}}$(无偏方差)而非样本 std 的有偏估计,且 baseline 还涉及 GAE 项的严格无偏。本文用"$1/K→1/(K-1)$"概括,工程上够用但学术面试若被追问"为什么是 K-1 而非其他无偏形式"时容易露怯,需回到 Dr.GRPO 原文(Liu et al. 2025)对照策略梯度定理推导。
- **"GSPO 把整条序列梯度压制=连坐惩罚"是作者借 SAPO 章节的转述,但 GSPO 论文未自承此缺陷**:这是作者用 SAPO 视角给 GSPO 补的批判,虽直觉合理(几千字序列里混进几个离谱 token 会让整条序列 ratio 失真),但缺定量实验支撑;面试时若被问"GSPO vs SAPO 谁更好"应回答"分场景:MoE 路由稳定选 GSPO,样本效率 + drop-in 选 SAPO,两者正交可叠加"。
- **DAPO 的 Clip-Higher 数值($1.28$/$0.8$)是经验值**:原文未给出该参数的敏感性消融,实际部署需按模型规模重调;且非对称裁剪破坏了 PPO 的信任域理论保证,长训可能 reintroduce off-policy 噪声——这一点作者未点明。
- **GTPO 的"高熵=决策点=关键 token"假设在反例上不稳健**:模型在"毫无头绪瞎猜"时也高熵,这种高熵 token 不一定是关键决策而是噪声;GTPO 论文靠负向序列回退平缓分配来兜底,但"瞎猜高熵"与"决策高熵"在分布上难分。面试若追问应主动提此局限。
- **Minimind 实现里 `score = max(min(score, scale), -scale)` 截断到 $[-3,3]$ 是工程 hack 而非理论需要**:会扭曲 RM 的相对排序,尤其当真实 reward 集中在边界附近时;Dr.GRPO/DAPO 的"长度偏置"修正并不能解决这种截断偏置。可作为"工程实践 vs 理论洁净"的批判点。
- **作者把"PRM 是护城河"说得过于绝对**:DeepSeek-Math 之外,R1-Zero 恰恰证明纯 ORM + GRPO 也能涌现强推理;PRM 在 R1 路线里反而被淡化。文中"不可逾越的护城河"措辞偏营销,宜降为"在超难定理证明/大型工程代码场景下 PRM 仍显著优于纯 ORM"。
- **STaR 章节把"生成-过滤-合理化"流程归为 RL 前置冷启动是合理但简化的视角**:实际 o1/R1 的冷启动更接近"PRM-guided MCTS + 拒绝采样 + SFT"复合管道,STaR 只是其中思想源头之一;作者用 STaR 代表整个冷启动略偏简化。

## 章内小思维导图

```
GRPO 变体谱系(按"省 Critic 之后修什么偏置"主线)
│
├─ 基石:GRPO ── 组内 Z-score 优势,去 Critic,4→3 模型
│   痛点遗留:① 长度偏置 ② 熵坍塌 ③ Token 方差 ④ 硬裁剪归零 ⑤ 稀疏 credit
│
├─ 修长度偏置 ─ Dr.GRPO
│   └─ 1/(K-1) 无偏缩放 + 剔除 |o| 除法
│
├─ 修熵坍塌 + 算力浪费 + 截断噪声 ─ DAPO(四件套)
│   ├─ Dynamic Sampling(丢 std=0 组)
│   ├─ Clip-Higher(非对称 1.28/0.8)
│   ├─ Token-level Loss(全局拉平,长链权重↑)
│   └─ Overlong Filter + Soft 软惩罚
│
├─ 修 Token 级方差 + MoE 路由漂移 ─ GSPO
│   └─ 序列级似然比 + 长度归一化 + 序列级 clip
│        → 可丢 Routing Replay
│
├─ 修硬裁剪归零 + 负优势带崩词表 ─ SAPO
│   ├─ Sigmoid 软门控(梯度永不归零)
│   └─ 非对称温度 τ_pos/τ_neg(惩罚更谨慎)
│        → drop-in 替换,保留 Token 级
│
└─ 修稀疏信用分配(大锅饭) ─ GTPO
    └─ 策略熵 H_{i,t} 作权重,A_{i,t}=A_i·w_{i,t}
         → 白嫖 Forward 副产物,无需 PRM 即伪过程奖励

旁支:
├─ PRM ── 过程奖励,逐 step 打分(MCTS/规则验证器自动标注)
└─ STaR ── RL 前置冷启动(生成-过滤-合理化-SFT 循环)
```
