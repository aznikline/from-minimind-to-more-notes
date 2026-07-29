# 架构:归一化技术(原理、演进与前沿)

## 一句话精炼
归一化从 BN→LN→RMSNorm、Post→Pre、再到 DeepNorm/QK-Norm/Sandwich-Norm,本质是一场围绕"梯度稳定性 vs 表达上限"权衡的工程演进,LLM 默认选 Pre-RMSNorm 是稳定性优先的现实妥协。

## 核心概念(BN/LN/GN、RMSNorm、Pre-Norm vs Post-Norm、DeepNorm 等)

- **内部协变量偏移(Internal Covariate Shift)**:深层网络前层参数更新导致后层输入分布漂移,随层数指数放大,触发激活饱和、收敛变慢。归一化把激活拉回零均值单位方差,降低 Loss 曲面 Lipschitz 常数,允许更大学习率。
- **BN(Batch Normalization)**:沿 batch 维统计,适合 CV 图像(通道结构规整)。
  - NLP 摒弃 BN 的三个本质原因:
    1. **变长序列 + Padding** 扭曲 μ/σ 统计;
    2. **小 micro-batch(可至 1)** 估计噪声爆炸;
    3. **Token 级跨样本无对应关系**,BN 在 batch 维统计破坏 Token 独立性。
- **LN(Layer Normalization)**:沿特征维对单样本统计,独立于 batch/seq。配套可学习 γ(缩放)、β(偏置),初始化 γ=1、β=0 近似恒等。两类不变性:
  - 重中心化不变:LN(x+δ)=LN(x)
  - 重缩放不变:LN(λx)=LN(x)→ 隐式 LR 衰减,防权重膨胀
- **GN(Group Normalization)**:把通道分组,介于 LN/BN 之间(本文未深入)。
- **RMSNorm**:去中心化 LN,只保留重缩放。LLaMA/Gemma/Mistral 标配,去 β,更轻量。
- **Post-Norm**:LN 在残差之后 x_{l+1}=LN(x_l+Sublayer(x_l))。BERT/原版 Transformer。需 Warm-up,梯度易爆/消,但性能上限略高。
- **Pre-Norm**:LN 在子层输入端 x_{l+1}=x_l+Sublayer(LN(x_l))。GPT-2/3、LLaMA 主流。主干恒等路径(高速公路效应),梯度稳定,可去 Warm-up;代价是主干方差随层累积→深层贡献被隐式缩小(1/L),即"深度诅咒(Curse of Depth)",深层退化为恒等映射。
- **DeepNorm**:改进 Post-Norm,引入 α>1 放大主干 + β 缩放子层权重,使"有界更新"成立,扩展到 1000 层(DeepNet)。β 只作用于 FFN 全权重 + Attention 的 W_V/W_O,W_Q/W_K 不缩放(保注意力分数尺度)。
- **QK-Norm**:对 Q、K 各做 LN/L2,用柯西-施瓦茨约束点积上界,治 Attention 熵坍塌(Logits 可达 1e4,Softmax 趋 one-hot,梯度消失)。ViT-22B 稳定训练关键 trick。
- **变体**:QKV-Norm(NormFormer,连 V 也归一化)、Softmax Capping(截断 logits 到 [-30,30],Mistral 采用)。
- **Sandwich-Norm**:CogView 提出,在 Pre-Norm 残差分支末端再加一个 LN,x_{l+1}=x_l+LN(Sublayer(LN(x_l))),二次归一化,治图像 token 数值溢出,FP16 稳定训练。
- **NormFormer**:Pre-LN + Post-Attention LN + Head-Scaling 三重过归一化,1.3B 实验达目标 PPL 快 24%。

## 关键公式(LaTeX)

**LayerNorm**:
$$\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\quad \sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2$$
$$\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y_i=\gamma_i\hat{x}_i+\beta_i$$
(γ=1, β=0 初始化 → 恒等;ε≈1e-5 防除零)

**RMSNorm**:
$$\text{RMS}(\mathbf{x})=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}$$
$$\bar{x}_i=\frac{x_i}{\text{RMS}(\mathbf{x})}\cdot\gamma_i$$
(LLaMA/Mistral 去 β;ε 加在平方和均值后、开根前;Gemma 修正:`RMSNorm(x)*(1+γ)`,γ≈0 时近似恒等)

**Post-Norm**:
$$\mathbf{x}_{l+1}=LN(\mathbf{x}_l+\text{Sublayer}(\mathbf{x}_l))$$

**Pre-Norm**:
$$\mathbf{x}_{l+1}=\mathbf{x}_l+\text{Sublayer}(LN(\mathbf{x}_l))$$

**DeepNorm(有界更新)**:
$$\mathbf{x}_{l+1}=LN(\alpha\cdot\mathbf{x}_l+G_l(\mathbf{x}_l,\theta_l)),\quad \alpha>1$$
残差缩放 α 放大主干、β 缩放子层权重,使每层更新期望有界。

**DeepNorm 初始化规则**:
- Encoder-only (N 层):$\alpha=(2N)^{1/4},\ \beta=(8N)^{-1/4}$
- Decoder-only (M 层):$\alpha=(2M)^{1/4},\ \beta=(8M)^{-1/4}$
- Enc-Dec (N Enc / M Dec):
  - Enc: $\alpha=0.81(N^4M)^{1/16},\ \beta=0.87(N^4M)^{-1/16}$
  - Dec: $\alpha=(3M)^{1/4},\ \beta=(12M)^{-1/4}$
- β 作用于 FFN 所有权重 + Attention 的 $W_V, W_O$;$W_Q, W_K$ 不缩放。

**QK-Norm**:
$$\mathbf{q}'=LN(\mathbf{q}),\quad \mathbf{k}'=LN(\mathbf{k})$$
$$\text{Attn}=\text{Softmax}\!\left(\frac{\mathbf{q}'(\mathbf{k}')^\top}{\sqrt{d}}\right)\mathbf{v}$$

**Sandwich-Norm**:
$$\mathbf{x}_{l+1}=\mathbf{x}_l+LN(\text{Sublayer}(LN(\mathbf{x}_l)))$$

## 演进脉络(归一化技术发展时间线/对比表)

| 阶段 | 技术/方案 | 架构位置 | 解决的核心痛点 | 代表模型 | 局限 |
|------|---------|---------|----------------|----------|------|
| 2015 | BatchNorm | — | CV 内部协变量偏移 | ResNet 等 CV | NLP 变长/小 batch 失效 |
| 2016 | LayerNorm | — | NLP 变长/小 batch,特征维统计 | Transformer/BERT 基石 | 双扫描,内存带宽受限 |
| 2017 原 Transformer / BERT | Post-Norm | 残差后 | 每层输入标准化,性能上限高 | Original Transformer, BERT | 需 Warm-up,梯度爆/消 |
| 2019 GPT-2 | Pre-Norm | 子层前 | 高速公路恒等路径,梯度稳定 | GPT-2/3, LLaMA, PaLM, Mistral | 深度诅咒,深层退化 |
| 2019 | RMSNorm | — | 去 centering,单次扫描省带宽 | LLaMA, Gemma, Mistral | 表达力略弱于 LN(可忽略) |
| 2022 | DeepNorm | Post 变体(α 放大主干) | 千层超深,有界更新 | DeepNet(1000 层) | 需 α/β 按层规模调参 |
| 2023 | QK-Norm | Attention 内 | 熵坍塌,logits→1e4 | ViT-22B, 多模态 | 额外算子 |
| — | Sandwich-Norm | Pre + 末端 LN | 图像 token 数值溢出 | CogView | 双 LN 开销 |
| — | NormFormer | Pre + Post-Attn LN + Head scaling | 加速收敛 | 研究模型 | 参数量增加 |
| — | Softmax Capping | Softmax 前 | 双保险防 logits 爆 | Mistral | 截断阈值需调 |

## 源码要点(Minimind 中用哪种,实现细节)

Minimind(`model/model_minimind.py`)采用 **RMSNorm**(Pre-Norm 架构,与 LLaMA 同款)。

实现细节要点(对应文中代码):
- `self.weight = nn.Parameter(torch.ones(dim))`:仅保留缩放 γ,无 β,与 LLaMA/Mistral 一致。
- `eps=1e-5` 默认(注:LLaMA 官方常用 1e-6,文中 minimind 代码取 1e-5)。
- `_norm`:`x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`
  - 用 `torch.rsqrt` 直接算 `1/sqrt`,比先 sqrt 再除更快;
  - `.mean(-1, keepdim=True)` 在最后一维(特征维)规约并保维。
- **精度策略(关键)**:`forward` 中 `self._norm(x.float()).type_as(x)`——先 upcast 到 float32 计算归一化(数值稳定),再 cast 回原精度(fp16/bf16),避免 fp16 下 x² 求和溢出。这是 LLM 推理稳定性的工程必备。
- 仿射变换在后:`self.weight * self._norm(...)`,即先归一再缩放,符合 RMSNorm 公式。
- 注释明确点出与 LN 的差异:"RMSNorm 不减去均值,只进行缩放,计算更快"。

## 作者独到见解/类比

- **"高速公路(Highway)效应"**:Pre-Norm 的残差主干是一条恒等高速公路,梯度可无损直通底层,非线性 LN 不挡道——这是现代大模型能训深的关键隐喻。
- **"深度诅咒(Curse of Depth)"**:Pre-Norm 主干方差随层线性累积,而残差分支经 LN 后方差被重置为 1,深层对主干贡献被隐式缩成 1/L,深网退化为恒等映射——与 ResNet 的"恒等越好"直觉相反,这里"恒等太好"反而是病。
- **RMSNorm 的反思**:LN 的成功主要靠"重缩放不变性"而非"重中心化",中心化在深网激活均值≈0 时收益微乎其微,却多一次全向量扫描——"去中心化"是极简主义的胜利。
- **内存带宽视角**:LayerNorm 是 memory-bandwidth bound,瓶颈不在 FLOPs 而在两次显存扫描+同步屏障;RMSNorm 单次扫描+易算子融合,在 FlashAttention/vLLM 等推理框架中被深度优化——这是"工程收益 >> 理论 FLOPs 减少"的典型。
- **DeepNorm 的"有界更新"**:让每层更新期望有界,本质是用 α 放大主干、β 缩放子层,把 Post-Norm 的高性能与 Pre-Norm 的稳定性"杂交"。
- **QK-Norm 用柯西-施瓦茨**:点积上界 = 模长乘积,归一化 Q、K 模长即锁死 logits 上界,治 1e4 量级爆炸——数学不变式直接变工程 trick。

## 面试考点(Pre-Norm 为何主流、RMSNorm 相比 LN 的优势等)

1. **为何 NLP 不用 BN?** 变长+Padding 扭曲统计;小 micro-batch 噪声大;Token 跨样本无对应关系,破坏独立性。
2. **Pre-Norm 为何成为主流?** 高速公路恒等路径使梯度稳定,可去 Warm-up,训超深大模型成本低;GPT-2/3、LLaMA、PaLM 均采用。代价是深层贡献衰减(深度诅咒)、性能上限略低于 Post-Norm。
3. **Post-Norm 性能为何略高却不用?** 每层输入标准化、充分利用非线性表达,理论上限高;但梯度穿 LN 易爆/消,必须 Warm-up,大模型训练成本不可接受。
4. **RMSNorm 相比 LN 的优势?** (a)去 centering,单次扫描省内存带宽(推理快 10%-40%);(b)去 β 减参数、简化算子;(c)易 Kernel Fusion,在 FlashAttention/vLLM 中深度优化;(d)实验表明重缩放不变性才是 LN 成功主因,中心化可省。
5. **Gemma 的 RMSNorm 为何写成 `(1+γ)`?** γ≈0 初始化时输出≈输入,近似恒等,利于深网初始信号传播,保留 Pre-Norm 直通特性。
6. **DeepNorm 如何把网络做到 1000 层?** α>1 放大主干 + β 按层规模缩放子层权重(FFN 全权重 + W_V/W_O,W_Q/W_K 不缩放),使每层更新期望有界;α/β 公式按 Enc/Dec/Enc-Dec 三种架构分别给出。
7. **什么是注意力熵坍塌?如何治?** 训练中 Q·K^T 可达 1e4,Softmax 饱和趋 one-hot,梯度消失。治法:QK-Norm(锁模长乘积上界)、QKV-Norm、Softmax Capping(截断到 [-30,30])。
8. **Sandwich-Norm 解决什么?** 图像 token 数值敏感,Pre-Norm 残差输出仍会溢出;在残差分支末端再加 LN 二次归一化,使 CogView 在 FP16 稳定训练。
9. **LN 的两个不变性?** 重中心化 LN(x+δ)=LN(x);重缩放 LN(λx)=LN(x),后者隐式 LR 衰减防权重膨胀。
10. **Minimind RMSNorm 为何先 float32 再 type_as?** fp16 下 x² 求和易溢出,upcast 到 fp32 保数值稳定,再回原精度省显存。

## 批判性批注

- **"内部协变量偏移"叙事过时**:现代研究(如 Santurkar et al. 2018)对 ICS 是否真是 BN 成功主因存疑,归一化真正作用更可能是平滑 Loss 曲面、降低 Lipschitz 常数。文中把 ICS 当唯一动因略简化,但作为教学引入可接受。
- **"RMSNorm 表达力弱于 LN"未量化**:文中说 LN 理论上限略高、RMSNorm 表达力略弱,但未给出 PPL/下游任务对照实验数据,仅以"消融发现重缩放才是主因"立论,属经验性结论而非严格证明。LLaMA 系实证已表明差异可忽略。
- **"Post-Norm 性能上限略高"缺乏机制解释**:仅归因于"每层输入标准化、充分利用非线性",未区分是优化景观更优还是最终收敛点更优,二者不能混为一谈。
- **DeepNorm 的 α/β 公式偏 cargo-cult**:公式来自理论推导但文中未展示推导,读者易当成"查表背参数";且 1000 层实验在多语言翻译这种任务上,BLEU+5 是否能推广到通用 LLM 未验证,DeepNet 后续在主流 LLM 中并未普及,说明其工程价值有限。
- **QK-Norm 与 Softmax Capping 的副作用未讨论**:截断会改变注意力分布形状,Capping 到 [-30,30] 在长上下文/稀疏注意力下可能损失表达力;文中只谈稳定不谈代价。
- **Sandwich-Norm / NormFormer 适用边界模糊**:都是特定场景(图像生成、1.3B 研究)产物,文中列为"专用变体"但未说明何时该用、何时过度,实践者易误以为"越多 Norm 越好"。
- **Minimind eps=1e-5 与 LLaMA 官方 1e-6 不一致**:文中代码注释未点出这一差异,作为教学实现可,但若读者照搬到生产,eps 选择会影响数值稳定性,应标注。
- **"性能上限略高/略低"量级未给**:Post vs Pre 的 BLEU/PPL 差距通常 <1%,文中用"略"一笔带过,易让读者高估该差异的重要性,从而错误选择 Post-Norm。
- **缺少对 DiT/AdaLN 等新趋势提及**:作为"前沿架构"章节,未涉及 diffusion transformer 中的 adaptive LN 等正在兴起的归一化设计,前沿性略有欠缺。

## 篇内小思维导图(缩进树)

```
归一化技术
├── 解决的问题
│   ├── 内部协变量偏移(ICS)
│   └── 深层梯度爆/消、Loss 曲面不平滑
├── 按"沿哪个维统计"分
│   ├── BN(CV, batch 维)→ NLP 失效(变长/小 batch/Token 独立)
│   ├── LN(NLP, 特征维)→ 双不变性(re-center / re-scale)
│   └── GN(分组, 介于两者)
├── 按架构位置分
│   ├── Post-Norm(BERT/原 Transformer)→ 性能略高, 需 Warm-up, 梯度脆
│   └── Pre-Norm(GPT-2/3, LLaMA)→ 高速公路恒等路径, 主流, 但深度诅咒
├── LN 的效率简化
│   └── RMSNorm(去 centering, 去 β)→ LLaMA/Gemma/Mistral 标配
│       ├── 单次扫描省内存带宽(推理快 10-40%)
│       └── Gemma 修正: (1+γ) 近似恒等初始化
├── 超深/超大场景专用
│   ├── DeepNorm(α 放大主干 + β 缩放子层, 1000 层 DeepNet)
│   ├── QK-Norm(治注意力熵坍塌, ViT-22B)
│   │   └── 变体: QKV-Norm, Softmax Capping(Mistral)
│   ├── Sandwich-Norm(CogView 图像, 二次归一化)
│   └── NormFormer(三重过归一化, 加速收敛)
└── Minimind 实践
    └── RMSNorm + Pre-Norm
        ├── weight=ones, 无 β
        ├── torch.rsqrt 加速
        └── float32 计算 → type_as 回原精度(数值稳定)
```
