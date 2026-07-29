# 基石:Embedding 与位置编码

## 一句话精炼

本文以"分布语义学→位置编码演进→RoPE 复数旋转→NTK/YaRN 频域外推"为主线,讲透了 LLM 如何把离散符号、线性时序与高维几何三者缝合:Embedding 解决"意义"的度量,RoPE 用旋转诱导出相对位置的平移不变性,YaRN 则用频域分段插值加熵温度修正,把上下文窗口从 4k 一路撑到 128k+。

## 核心概念

- **Token Embedding** — 把离散 Token ID 经一个 $V \times D$ 查表矩阵映射成稠密连续向量,是分布语义学("You shall know a word by the company it keeps")的数学实现,训练中由反向传播把共现词推向邻近。
- **分布语义假设** — Firth 名言:词义由其伴随词决定;Embedding 空间因此涌现线性子结构(如 $\vec{King}-\vec{Man}+\vec{Woman}\approx\vec{Queen}$),代数上对齐了性别、时态等抽象维度。
- **词袋模型局限** — 纯 Embedding 位置无关;"张三打了李四"与"李四打了张三"在词袋视角下表示一致,故必须显式注入位置。
- **绝对位置编码 (APE, Sinusoidal)** — Vaswani 用不同频率正余弦波为每个位置生成"指纹",多尺度时钟、可线性变换推断相对位置;直接加到 Embedding 上(加法注入)。
- **可学习 APE** — GPT-2/BERT 用 $L_{max}\times D$ 可学习矩阵,灵活但无法外推,越界即失效。
- **相对位置编码 (RPE) / 偏置相加法** — Shaw/T5 把位置信息从 Embedding 移到 Attention Score 上,加一个表示 $(i-j)$ 的可学习偏置;T5 用对数分桶,近处精确远处共享,契合人类认知。
- **ALiBi** — 在 Score 矩阵上减 $m\cdot|i-j|$,无参、外推之王,但强制线性衰减限制长依赖且不兼容 KV Cache 优化。
- **RoPE (旋转位置编码)** — 苏剑林/RoFormer:对 Q、K 在复平面乘 $e^{im\theta}$ 做绝对位置旋转,内积自然只剩 $(m-n)$,即"以绝对旋转诱导相对位置";语义正交(保模长)、KV Cache 完美兼容、自然长程衰减。
- **多级时钟隐喻** — RoPE 各维度是转速不同的指针:低维如秒针转得快,敏锐感知紧邻;高维如时针转得慢,保持长距离相位连贯。
- **长程衰减** — 相对距离增大时高频分量相位差随机化,内积期望趋于 0,局部性自然涌现,无需硬编码。
- **外推性故障** — 推理长度超过训练长度时旋转角度进入 OOD 区域,Softmax 熵崩塌,PPL 爆炸。
- **线性内插 (PI)** — 把位置 $m$ 替换为 $m/s$ 把窗外世界压回窗内,解 PPL 但损失高频分辨率,短文本性能下降。
- **NTK-Aware Scaling** — 基于 NTK 频谱偏差(网络先学低频、难学高频),对不同频率维度差异化缩放:高频不插值保局部精度,低频强插值获全局视野;通过改 base 实现,可 zero-shot 外推。
- **动态 NTK** — 推理时按 $s=\max(1, L_{current}/L_{train})$ 动态缩放,短输入不缩放,长输入平滑介入。
- **YaRN** — 集大成者:NTK-by-parts 三频段斜坡函数(高频外推、低频内插、中频过渡)+ 熵温度缩放 $\sqrt{1/t}=0.1\ln(s)+1$ 修正分布漂移;0.1% 微调数据即可 4k→128k。
- **Lost in the Middle** — 插值后 Attention 分布熵漂移导致长文本中间信息检索失效,YaRN 的温度缩放正是治此。
- **Llama 3 高 Theta** — base 从 1e4 提到 5e5,拉长低频波长,预训练即强迫学习超长依赖,为 1M 微调铺路。
- **Mistral SWA + RoPE** — 滑动窗口注意力每层只看 $W$ 个 Token,多层堆叠使有效感受野线性增长,RoPE 保证全局相对位置一致传递。
- **M-RoPE (Qwen2-VL/PaliGemma)** — 将 Embedding 切三段分别旋转时间/高度/宽度,实现 1D 文本→2D 图像→3D 视频的时空统一建模。
- **DeepSeek-V2 解耦 RoPE** — MLA 低秩压缩与 RoPE 冲突,解耦为 Content Vector(语义压缩、不加 RoPE)与 RoPE Vector(位置不压缩、直接旋转),Score 相加,证明 RoPE 极强架构兼容性。

## 关键公式

### 正弦位置编码 (Vaswani)
$$PE_{(pos, 2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad PE_{(pos, 2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
关键性质:$\vec{PE}_{pos+k}$ 可表为 $\vec{PE}_{pos}$ 的线性函数,理论上允许模型学习相对位置。

### ALiBi
$$\text{Score}_{i,j}=\mathbf{q}_i\cdot\mathbf{k}_j - m\cdot|i-j|$$
$m$ 为 head 特定斜率,线性衰减强归纳偏置。

### RoPE 复数形式 (二维)
$$f(\boldsymbol{q},m)=\boldsymbol{q}\cdot e^{im\theta}=r_q e^{i(\theta_q+m\theta)},\quad f(\boldsymbol{k},n)=r_k e^{i(\theta_k+n\theta)}$$
Hermitian 内积取实部:
$$\langle f(\boldsymbol{q},m),f(\boldsymbol{k},n)\rangle=\text{Re}\left[\boldsymbol{q}\boldsymbol{k}^*\cdot e^{i(m-n)\theta}\right]$$
关键结论:最终内积仅以 $(m-n)$ 出现,即平移不变性——绝对旋转诱导相对位置。

### RoPE 分块旋转矩阵 (多维)
$$\mathbf{R}_{\Theta,m}=\begin{pmatrix}\cos m\theta_0&-\sin m\theta_0&&\\ \sin m\theta_0&\cos m\theta_0&&\\ &&\cos m\theta_1&-\sin m\theta_1&\\ &&\sin m\theta_1&\cos m\theta_1&\\ &&&&\ddots\end{pmatrix}$$
频率设定 $\theta_j=\text{base}^{-2j/d}$,$\text{base}=10000$;低维高频、高维低频。

### 线性内插 (PI)
$$f(\boldsymbol{q},m)=\boldsymbol{q}e^{i\frac{m}{s}\theta},\quad \Delta\theta_{new}=\Delta\theta_{original}/s$$
解 PPL 但牺牲高频分辨率。

### NTK-Aware Base Change
$$b'=b\cdot s^{\frac{d}{d-2}}$$
等效于随维度 $d$ 增加,缩放因子从 1 平滑过渡到 $s$。

### YaRN NTK-by-parts 斜坡函数
$$r=\frac{L_{train}}{\lambda_d},\quad \gamma(d)=\text{clamp}\left(\frac{r-\alpha}{\beta-\alpha},0,1\right)$$
$$h(\theta_d)=(1-\gamma(d))\frac{\theta_d}{s}+\gamma(d)\theta_d$$
高频段($\gamma=1$)不插值,低频段($\gamma=0$)线性插值,中频过渡。

### YaRN 温度缩放
$$\text{Attention}(Q,K)=\text{Softmax}\left(\frac{QK^T}{t\sqrt{d_k}}\right),\quad \sqrt{\frac{1}{t}}=0.1\ln(s)+1$$
$s=1$ 时无变化,$s$ 大时 logits 数值升高、softmax 更尖锐,修正分布漂移。

## 关键算法/流程

### RoPE 复数旋转的等价实数实现 (rotate\_half 技巧)
对向量 $\mathbf{x}=[x_0,x_1,\dots,x_{d-1}]$,把后半部分取反交换得 $\text{rotate\_half}(\mathbf{x})=[-x_{d/2},\dots,-x_{d-1},x_0,\dots,x_{d/2-1}]$,则复数旋转 $x\cdot e^{i\theta}$ 等价为:
$$R_\theta(\mathbf{x})=\mathbf{x}\odot\cos\theta+\text{rotate\_half}(\mathbf{x})\odot\sin\theta$$
这样把分块对角矩阵乘法化成逐元素乘加,GPU 友好。

### precompute\_freqs\_cis 流程 (minimind)
1. 频率向量 $\text{freqs}_i=1/\text{base}^{2i/d}$,$i=0,2,\dots,d-2$;
2. 若启用 YaRN 且 $end/orig\_max>1$:用 $\text{inv\_dim}(b)=\frac{d\log(orig\_max/(b\cdot2\pi))}{2\log(\text{base})}$ 算出低/高频边界 low/high,得斜坡 $\gamma$,应用 $\text{freqs}\leftarrow\text{freqs}\cdot(1-\gamma+\gamma/factor)$;
3. 位置索引 $t=[0,\dots,end-1]$,外积 $\text{freqs}=\text{outer}(t,\text{freqs})$ 得 $[end, d/2]$;
4. cos/sin 各复制一倍拼到 $dim$ 维,乘 attn\_factor(温度缩放)返回。

### apply\_rotary\_pos\_emb 流程
1. 定义 rotate\_half(后半取反前移);
2. cos/sin 在 unsqueeze\_dim 上插维以广播到 q/k 的 [batch, seq, heads, head\_dim];
3. q\_embed = q*cos + rotate\_half(q)*sin;k\_embed 同理;返回。

### YaRN 整体方案
NTK-by-parts(频域分段斜坡混合) + Temperature Scaling(熵修正) → PPL 与 Passkey Retrieval 双优,0.1% 微调 4k→128k。

## 源码要点

涉及 minimind `model/model_minimind.py` 两函数:

- **`precompute_freqs_cis(dim, end, rope_base, rope_scaling)`**
  - `freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))` —— 只用偶数索引生成 $d/2$ 个频率,符合 RoPE 二维分块。
  - YaRN 分支:`inv_dim(b)` 由波长 $\lambda=2\pi/f$ 反推维度索引;`low/high` 划定快慢频边界;`ramp = clamp((arange(d/2)-low)/(high-low), 0, 1)` 即 $\gamma(d)$;`freqs = freqs * (1 - ramp + ramp / factor)` 实现 $h(\theta)=(1-\gamma)\theta/s+\gamma\theta$。
  - `attn_factor` 直接乘到 cos/sin 上,对应温度缩放(代码里默认 1.0,由配置注入)。
  - `torch.cat([cos, cos], dim=-1)` 把 $d/2$ 复制到 $d$,配合 rotate\_half 的"后半取反前移"布局。

- **`apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)`**
  - `rotate_half(x) = torch.cat((-x[..., d//2:], x[..., :d//2]), dim=-1)` —— `[a,b,c,d]→[-c,-d,a,b]`,实现复数旋转的虚部操作。
  - `q_embed = (q * cos) + (rotate_half(q) * sin)` —— 等价 $x\cdot e^{i\theta}=x\cos\theta+\text{rotate\_half}(x)\sin\theta$。
  - cos/sin 在 `unsqueeze_dim` 上插维以广播匹配 q/k 形状,默认 dim=1 对应 `[batch, seq, heads, head_dim]`。
  - `position_ids` 参数未使用(cos/sin 已含位置),保留以兼容 HF 接口。

## 作者独到见解/类比

"几何与时空的折叠"指三层含义:

1. **几何折叠** — Embedding 把离散符号折叠进高维连续空间(语义几何化),RoPE 又把线性时序折叠进同一空间的旋转(位置几何化);加法 APE 是"平移",RoPE 是"旋转",作者明确断言"几何优于代数":旋转比平移更符合点积注意力的几何直观,让相对位置不是学出来的特征而是数学必然涌现的性质。
2. **时空折叠** — 时序(1D 文本位置)被折进几何(向量旋转),再经 M-RoPE 折叠到 2D 图像、3D 视频的时空统一;DeepSeek 解耦 RoPE 则把"位置几何"与"语义几何"显式拆开又相加,证明 RoPE 是可插拔的独立几何模块。
3. **频域视角的觉醒** — NTK/YaRN 把"位置编码外推"重新解读为信号处理问题:低频维度波长极长、训练时连半圈都没转完,故外推必崩;高频已转无数圈,故直接外推无妨。作者称这是"从矩阵运算上升到频域分析"的胜利。

其他独到点:
- 把正弦编码、RoPE、ALiBi 都统一进"多级时钟/密码锁指针"隐喻——指针转速随维度递减,秒针敏近、时针保远。
- 把 YaRN 的温度缩放拔高为"信息密度与注意力分布的热力学平衡",不只是显存问题。
- 预言未来位置编码可能"根据内容自适应调节时钟转速",或随线性 Attention/SSM(Mamba)复兴与状态方程融合。

## 面试考点

- **为什么 RoPE 内积只剩 $(m-n)$?** —— 复数 Hermitian 内积取实部,$e^{im\theta}\cdot e^{-in\theta}=e^{i(m-n)\theta}$,位置项相消,只剩相对距离;这就是平移不变性。
- **RoPE 为何要切 $d/2$ 个二维子空间?** —— 复数是 2D,推广到多维用"分而治之",每个子空间配不同 $\theta_j$,形成多尺度频率。
- **高频不插值、低频强插值的依据?** —— NTK 频谱偏差+波长分析:高频波长短,训练内已转多圈,模型对各相位相对关系学得好,外推安全;低频波长极长,训练内只转一点,外推即 OOD。
- **PI 与 NTK 的区别?** —— PI 对所有维度统一除 $s$,损失高频分辨率;NTK 通过改 base 实现非线性,高频近不缩放、低频强缩放,zero-shot 更优。
- **YaRN 比 NTK 多了什么?** —— NTK-by-parts 精细化三频段分段(取代粗略 base change)+ 温度缩放修正 Attention 熵漂移,治 Lost in the Middle。
- **温度系数 $t$ 的经验式?** —— $\sqrt{1/t}=0.1\ln(s)+1$,升 logits 使 softmax 尖锐,保持对关键信息聚焦。
- **RoPE vs ALiBi 对比** —— RoPE 乘法旋转、语义正交保模长、KV Cache 完美兼容、自然长程衰减;ALiBi 偏置减法、强制线性衰减、KV Cache 不友好但外推极强。
- **Llama 3 为何把 base 提到 5e5?** —— 拉长低频波长,预训练即学超长依赖,衰减曲线平缓,为 1M 扩展铺数学基础。
- **DeepSeek-V2 为何要解耦 RoPE?** —— MLA 低秩压缩会破坏旋转的语义空间,故拆 Content(压缩、不加 RoPE)与 RoPE Vector(不压缩、直接旋转),Score 相加。
- **rotate\_half 为何等价复数旋转?** —— `[a,b,c,d]→[-c,-d,a,b]` 实现了实部/虚部交换与取反,使 $x\cos\theta+\text{rotate\_half}(x)\sin\theta$ 等价 $x\cdot e^{i\theta}$。

## 批判性批注

- **"RoPE 理论上支持无限长度"与"外推必崩"的张力未充分解释** —— 文中既称 RoPE 频率连续可延展,又承认超训练长度即 PPL 爆炸,但未点明根因是"非线性 Softmax 对未见旋转角度的敏感",只归为 OOD 略显笼统;其实 RoPE 数学上确实周期可延,崩点在 Attention 的非线性放大。
- **"分布语义=Embedding 公理"的循环风险** —— 把 Firth 假设当作"公理化",但共现统计→几何邻近的证明依赖 Skip-gram/CBOW 的特定目标,现代 LLM 的 Embedding 是端到端任务驱动的,未必严格对应分布语义;作者未区分预训练词向量与 LLM 端到端 Embedding。
- **YaRN 温度公式是经验拟合而非理论推导** —— $\sqrt{1/t}=0.1\ln(s)+1$ 作者也承认是"大量实验拟合",文中却冠以"热力学修正""熵理论"的深刻命名,理论包装偏重,实际是工程调参。
- **M-RoPE "三段各 d/3" 缺少消融依据** —— 作者称"极其优雅",但为何均分三段而非按模态重要性加权,未给实验支撑;时空维度其实不对称(时间帧数远少于空间像素),均分可能次优。
- **DeepSeek 解耦 RoPE 的代价被淡化** —— 只讲"Score 相加"的优势,未提解耦带来的额外参数与潜在 Content/Position 信息割裂风险。
- **ALiBi"无法融入 KV Cache"说法略绝对** —— ALiBi 的 bias 是 query-key 距离函数,推理时仍可在 score 上加偏置,KV Cache 可存 Key,真正不友好的是 bias 依赖动态距离;表述可更精确。
- **minimind 代码的 `attn_factor` 默认 1.0** —— 即默认未启用温度缩放,YaRN 的关键一环在示例代码里其实是空转,读者照抄会漏掉熵修正。
- **未涉及 RoPE 的已知缺陷** —— 如"长距离旋转相位混叠""低频维度旋转不足导致外推失效"已提,但"高频维度在 NTK 下仍可能 OOD 的边界情形"未充分讨论。

## 篇内小思维导图

```
Embedding 与位置编码
├── Embedding (语义几何)
│   ├── 离散 ID → 稠密向量 (V×D 查表)
│   ├── 分布语义假设 (Firth) → 共现驱动邻近
│   ├── 线性子结构 (King-Man+Woman≈Queen)
│   └── 局限:位置无关 (词袋无法区分语序)
├── 位置编码演进
│   ├── APE 绝对
│   │   ├── 正弦 (多尺度时钟, 加法注入, 可线性变换)
│   │   └── 可学习 (GPT-2/BERT, 无法外推)
│   ├── RPE 相对
│   │   ├── 偏置相加 (Shaw/T5 对数分桶)
│   │   └── ALiBi (Score-=m|i-j|, 外推之王但限长依赖)
│   └── RoPE 旋转 ← 现代主流
├── RoPE
│   ├── 复数域: q·e^{imθ} → 内积只剩 (m-n) 平移不变
│   ├── 分块对角矩阵 (d/2 个 2D 子空间, θ_j=base^{-2j/d})
│   ├── 多级时钟隐喻 (秒针敏近/时针保远)
│   ├── 长程自然衰减 + KV Cache 完美
│   └── 外推故障 → 需 NTK/YaRN
├── 长上下文扩展
│   ├── 线性内插 PI (m/s, 解PPL但损高频分辨率)
│   ├── NTK-Aware (base change, 高频不插/低频强插, zero-shot)
│   ├── 动态 NTK (按 L_current 动态 s)
│   └── YaRN ← 集大成
│       ├── NTK-by-parts (三频段斜坡 γ)
│       └── 温度缩放 (√(1/t)=0.1 ln(s)+1, 修正熵漂移)
└── 工程实践 (2024-2026)
    ├── Llama 3 高Theta (base=5e5, 预训即学超长依赖)
    ├── Mistral SWA+RoPE (感受野线性增长)
    ├── M-RoPE (时间/高/宽 三段, 多模态时空统一)
    └── DeepSeek-V2 解耦RoPE (Content压缩+RoPE不压缩, MLA兼容)
```
