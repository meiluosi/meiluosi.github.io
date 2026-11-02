# 前沿基础模型技术研究室（Frontier Foundation Model Research）栏目规划与索引台账

规划日期：2026-11-01。本栏目定位为持续维护的在线交互专著（**MONOGRAPH IV**）与三维研究知识库，第一阶段深度拆解 **DeepSeek** 全系技术演进，建立统一的架构、后训练与基础设施比较标尺，后续横向扩展至 OpenAI、Anthropic、Google DeepMind、Meta、Qwen 等团队。

---

## 一、栏目定位与三大核心追问

不同于按论文发布时间逐篇摘要的读书笔记，本栏目**以模型技术演进为主线，以核心研究问题为组织方式，以论文、开源源码和可复现实验作为证据**，还原一个前沿基础模型团队为什么会形成今天的技术体系。

无论研究哪家实验室，最终不仅要知道它使用了 MLA、MoE、GRPO、mHC、Engram、CED 等技术名词，更要回答三个深层问题：

1. **为什么这样设计？（Why This Design?）**
   当时遇到了什么具体瓶颈（显存、带宽、通信、训练稳定性、奖励欺骗）？备选方案是什么？为什么放弃了其他路线？
2. **不同技术如何协同？（How Do Subsystems Co-Design?）**
   模型结构、训练算法、推理系统与底层 AI 基础设施（算子内核、通信原语、存储调度）之间是如何联合优化的？
3. **哪些是真正具有普遍意义的发现？（What Generalizes Beyond Scale?）**
   如果脱离特定实验室的 GPU 集群规模与硬件条件（例如降维到单卡或 MacBook Air M4 16GB），哪些机制仍然有效，哪些结论高度依赖工程规模？

---

## 二、DeepSeek 六条核心技术研究线（六个协同视角）

截至 2026 年 11 月，DeepSeek 的技术演进已经从传统的大模型架构优化，发展到对**计算、存储与 Agent 工作负载的联合设计**（从 V2/V3/R1 演进至 V3.2、mHC、Engram、DualPath 以及采用 Causal Encoder–Decoder 不对称计算与跨层 KV 复用的 V4 / V4.1-Flash）。

我们用以下六条相互交叉的研究线覆盖其完整技术体系（前三条为核心重心，后三条提供关键系统与场景约束）：

| 研究线编号 | 研究线名称 | 核心关注问题 | 与其他线的交叉关系 |
| :--- | :--- | :--- | :--- |
| **Line 01** | **基础模型架构（Architecture）** | 如何在不引爆激活参数与 KV Cache 的前提下扩展模型容量与上下文？（DeepSeekMoE、MLA、MTP、mHC、Engram、CED、CSA2） | 直接受限于 Line 03 的显存带宽与通信开销，并决定 Line 02 的吞吐上限 |
| **Line 02** | **后训练与推理强化学习（Post-Training & RL）** | 复杂推理与工具交互能力如何从 RL 中涌现与受控？（GRPO、R1-Zero、冷启动 SFT、过程验证 Math-V2、Agent RL） | 依赖 Line 01/03 提供的长序列采样效率与 Line 04 的形式化/数学验证环境 |
| **Line 03** | **训练与推理系统效率（Systems & Efficiency）** | 算法如何与 GPU 内核、FP8 低精度、流水线并行、KV 存储带宽协同？（FlashMLA、DeepGEMM、DeepEP、DualPipe、DualPath） | 将 Line 01 的理论复杂度转化为真实硬件上的 MFU 与低延迟服务 |
| **Line 04** | **代码与形式化数学专项（Coder & Prover）** | 代码与形式化证明数据如何反哺通用基座的逻辑推理与验证器训练？（DeepSeek-Coder、DeepSeek-Prover、Math 系列） | 为 Line 02 提供天然可验证的奖励信号（编译器/证明器反馈） |
| **Line 05** | **多模态与视觉文档理解（Multimodal & OCR）** | 视觉理解与生成表征应当解耦还是统一？（DeepSeek-VL、Janus 解耦架构、DeepSeek-OCR 1/2） | 检验 Line 01 稀疏架构在多模态 Token 流下的扩展性 |
| **Line 06** | **Agent 工作负载与工程底座（Agent & Harness）** | 多轮长轨迹工具调用如何重塑注意力稀疏化与缓存调度？（V3.2 Agent RL、DualPath 存储带宽优化、DeepSeek Harness） | 将单次问答推向长周期状态维护，倒逼 V4 / V4.1 的架构重构 |

---

## 三、四阶段技术演进阅读路线

按照**问题引出与技术演进顺序**（而非严格发表日期）分为四个阶段推进，系统论文在对应机制处穿插精读：

### 第一阶段（01 · 基础）：如何建立高效基础模型（2023–2024）
* **演进链路**：DeepSeek LLM → DeepSeekMoE → DeepSeekMath → DeepSeek-V2 → DeepSeek-V3
* **核心追问**：
  * MoE、MLA、GRPO 各自为了解决什么痛点而诞生？
  * 为什么 V2 是关键转折点（MLA + DeepSeekMoE 首次合流）？
  * V3 如何通过**无辅助损失负载均衡（Auxiliary-Loss-Free Load Balancing）**、**多 Token 预测（MTP）**与 **FP8 混合精度训练**将超大规模稀疏模型低成本稳定训出？

### 第二阶段（02 · 推理）：如何发展复杂推理与验证能力（2025）
* **演进链路**：DeepSeek-R1-Zero → DeepSeek-R1 → R1 蒸馏模型 → DeepSeekMath-V2 → DeepSeek-V3.2
* **核心追问**：
  * 纯强化学习（R1-Zero）为何能自发涌现自我反思与长思维链？SFT 冷启动究竟修补了什么缺陷？
  * 奖励函数如何防止被投机（Reward Hacking）？如何从只看最终答案（Outcome Reward）走向可验证推理过程（DeepSeekMath-V2）？
  * 如何把稀疏注意力、强化学习与多步工具调用 Agent 任务结合（DeepSeek-V3.2）？

### 第三阶段（03 · 架构）：如何重新设计模型连接、记忆与不对称计算（2025–2026）
* **演进链路**：mHC → Engram → DeepSeek-V4 → DeepSeek-V4.1-Flash
* **核心追问**：
  * **mHC（Manifold-Constrained Hyper-Connections）**：能否突破传统残差连接的表达瓶颈，同时用流形约束守住深层训练稳定性？
  * **Engram**：静态世界知识的记忆检索与动态神经计算能否在架构层解耦？
  * **DeepSeek-V4 & V4.1-Flash**：如何实现经济可行的百万 Token 上下文？为什么输入阶段（Prompt Encoding）和输出阶段（Autoregressive Decoding）应当采用不对称计算结构（CED 因果编解码器、CSA2 与跨层 KV 复用）？

### 第四阶段（04 · 系统）：从模型算法扩展到完整 AI 基础设施
* **演进链路**：FlashMLA → DeepGEMM → DeepEP → DualPipe → DualPath → DeepInfer / DeepSeek Harness
* **核心追问**：
  * 算法创新如何落到 SM 核心、Tensor Core、NVLink/RDMA 通信与 SSD/HBM 多级缓存调度上？
  * 在 Agent 多轮长上下文推理中，**DualPath** 如何突破 KV Cache 的存储 I/O 带宽瓶颈？

---

## 四、首批 12 项核心文献清单

| 顺序 | 论文 / 技术报告 | 演进阶段 | 最核心的研究问题 |
| :---: | :--- | :---: | :--- |
| **01** | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | 01 基础 | 细粒度专家分割与共享专家隔离，如何让专家真正形成分工？ |
| **02** | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 01 基础 | 如何通过大规模数学语料挖掘与 GRPO（消去 Critic）提升推理并节省显存？ |
| **03** | [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2) | 01 基础 | MLA（多头潜在注意力）为什么能在大幅压缩 KV Cache 的同时不伤性能？ |
| **04** | [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | 01 基础 | 如何用无辅助损失负载均衡、MTP 与 FP8 将 671B 稀疏大模型稳定低成本训出？ |
| **05** | [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) | 02 推理 | 复杂推理能力如何在纯 RL（R1-Zero）与多阶段冷启动 RL（R1）中发展与蒸馏？ |
| **06** | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | 02 推理 | 如何将动态稀疏注意力（DSA）与面向 Agent 工具环境的大规模强化学习结合？ |
| **07** | [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) | 02 推理 | 如何从结果答案奖励转向可自验证的数学定理证明与过程验证？ |
| **08** | [mHC](https://arxiv.org/abs/2512.24880) | 03 架构 | 流形约束超连接能否在改造残差通道容量的同时保持深层训练稳定？ |
| **09** | [Engram](https://github.com/deepseek-ai/Engram) | 03 架构 | 静态知识的条件记忆查询（$O(1)$ 查找）与动态神经计算能否显式分离？ |
| **10** | [DeepSeek-V4](https://arxiv.org/abs/2606.19348) | 03 架构 | 如何通过混合稀疏注意力与记忆机制实现经济可行的百万 Token 上下文？ |
| **11** | [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) | 03 架构 | 为什么输入与输出应采用不对称计算结构（CED 架构、CSA2 与跨层 KV 复用）？ |
| **12** | [DualPath](https://arxiv.org/abs/2602.21548) | 04 系统 | Agent 密集型推理如何通过双路径加载突破 KV Cache 存储带宽瓶颈？ |

> **官方核验入口**：
> - [DeepSeek Transparency Center](https://www.deepseek.com/en/transparency/) — 官方模型版本、技术报告与透明度说明
> - [DeepSeek 官方 GitHub](https://github.com/deepseek-ai) — 开源权重、内核实现、通信库与工具链源码

---

## 五、单篇精读规范：研究工程五步法与四项固定交付

每篇核心论文/专题的拆解均严格执行**研究工程五步法**：

| 步骤 | 研究动作 | 以 DeepSeek-V2 的 MLA 为例 |
| :---: | :--- | :--- |
| **① 原始设计** | 还原历史痛点与最初解法 | 为什么传统 MHA/GQA 存在 KV Cache 显存墙？MLA 最初怎样用低秩联合压缩解决？ |
| **② 数学原理** | 推导公式、张量维度与源码映射 | 低秩上下投影矩阵、矩阵吸收（Matrix Absorption）、解耦 RoPE（Decoupled RoPE）及源码逐行对应 |
| **③ 设计局限** | 剖析代价、边界与硬件约束 | 压缩带来的信息瓶颈、推理阶段矩阵吸收前后的计算/显存权衡、不同 GPU 算术强度限制 |
| **④ 改进思想** | 独立提出替代方案并与后续演进对照 | 思考能否跨层共享 KV 或结合稀疏选择？再与 V3.2 DSA 及 V4.1 跨层 KV 复用对比 |
| **⑤ 实验验证** | 本机可控规模算例或缩模消融实验 | 在 MacBook Air M4 (16GB) 上对比 MHA / GQA / MLA 的 KV 缓存占用、吞吐拐点与重建误差 |

每篇固定交付**四项标准化成果**：
* **A. 技术讲解（Mechanism Walkthrough）**：从物理直觉 → 数学公式 → 张量形状流 → 开源源码对应，配套 Svelte 5 可交互认知沙盒。
* **B. 设计决策还原（Design Trade-off Reconstruction）**：用表格整理“当时遇到的问题 → 备选方案（含被放弃路线） → 最终选择 → 付出的代价”。
* **C. 四级证据审查（Four-Tier Evidence Audit）**：
  * `[E1 论文报告]`：论文作者汇报的实验指标与消融结论；
  * `[E2 源码确认]`：能从官方开源仓库或公开算子代码中直接核实的机制；
  * `[E3 第三方复现]`：社区独立复现或公开基准确认的证据；
  * `[E4 本机实验 / 待验假设]`：我们在 M4 16GB 小规模实验上验证的规律，或受限于集群规模尚待验证的推论。
* **D. 可复现实验或思想实验（Runnable Probe / Thought Experiment）**：能本机跑的给脚本与实测曲线（如 MLA 显存/算术强度分析、MoE 路由负载均衡模拟、GRPO 小模型消融）；依赖千卡集群的机制则构建精确的解析成本模型。

---

## 六、首批四个启动单元（Phase 1 Kickoff Units）

不按死板天数推进，而按研究问题闭环推进，先攻克前四个基础单元：

1. **单元 01：建立 DeepSeek 技术演进全景图（第一课）**
   * **题目**：《DeepSeek 技术演进：从普通 Transformer 到 V4.1，究竟改变了什么？》
   * **任务**：梳理 V1 → V2 → V3 → R1 → V3.2 → V4 → V4.1 的架构与训练路线，解释每一代继承与替换了什么。
   * **交付**：技术演进全景图 + 历代关键设计决策对照表 + 架构瓶颈转移交互沙盒。
2. **单元 02：深入 MoE——为什么模型不再全部激活？**
   * **任务**：从 Dense FFN、传统 GShard/Switch MoE、DeepSeekMoE（细粒度 + 共享专家）到 V3 无辅助损失负载均衡，研究计算稀疏化。
   * **交付**：数学原理推导 + 小规模专家路由与负载偏置动态更新实验。
3. **单元 03：深入 MLA——为什么可以只保存压缩表示？**
   * **任务**：从 MHA、MQA、GQA 进入 MLA，彻底算清低秩压缩、矩阵吸收、解耦 RoPE 与 KV Cache 显存及推理带宽的关系。
   * **交付**：张量级计算拆解 + MHA/GQA/MLA 缓存与延迟对比实验。
4. **单元 04：还原 DeepSeek-V3 的完整训练链路**
   * **任务**：把前面学过的机制组合起来，串通预训练数据、MTP、FP8 混合精度、DualPipe 流水线并行、SFT 与 RL 部署全链路。
   * **交付**：全链路协同架构图 + 训练与推理系统设计权衡报告。

---

## 七、三维研究知识库索引（Three-Index Knowledge Base）

为避免十几篇笔记沦为孤立文章，全系列维护以下三张动态关联索引表（每完成一篇即更新对应状态与链接）：

### 索引 1：核心问题索引（Problem Index — 跨公司横向比较基准）

| 问题 ID | 核心研究问题 | DeepSeek 的演进解法 | 后续横向对比锚点（OpenAI / Anthropic / Qwen / DeepMind） |
| :--- | :--- | :--- | :--- |
| **Q-01** | **如何突破自回归推理的 KV Cache 显存与 I/O 带宽墙？** | GQA (V1 67B) → **MLA 低秩压缩 + 解耦 RoPE** (V2/V3) → **DSA 动态稀疏注意力** (V3.2) → **DualPath 存储双路径** → **CED + 跨层 KV 复用** (V4.1-Flash) | 对照 GQA + 滑动窗口、Linear Attention / Hybrid Mamba、TPU 多级缓存方案 |
| **Q-02** | **如何在扩大总参数容量的同时控制单 Token 激活成本与通信开销？** | Dense → **DeepSeekMoE（细粒度专家 + 共享专家）** (V2) → **无辅助损失偏置均衡 + 节点受限路由 + DeepEP** (V3) → **Engram 条件记忆分离** | 对照 Qwen-MoE、Mixtral 粗粒度 Top-2、Switch Transformer |
| **Q-03** | **深层与超宽稀疏网络如何保持训练稳定性并榨干低精度算力？** | 标准 Pre-Norm BF16 → **细粒度 Tile/Block 级 FP8 + DualPipe** (V3) → **mHC 流形约束超连接** | 对照 Meta/Nvidia FP8 配方、μP 超参迁移、Peri-LN / QK-Norm |
| **Q-04** | **复杂多步推理能力如何从强化学习中涌现，且不依赖昂贵的 Critic 或过程标注？** | **GRPO 组内相对优势** (DeepSeekMath) → **R1-Zero 纯规则 RL 涌现 + R1 多阶段冷启动与蒸馏** → **Math-V2 自我验证与过程证明** | 对照 OpenAI o1/o3 PRM/MCTS 路线猜想、VinePPO、RLOO、PRIME |
| **Q-05** | **预训练单步预测目标如何提升样本效率，并衔接推理加速？** | 标准 Next-Token Prediction → **MTP（多 Token 预测）兼顾训练信号密度与投机解码** (V3) → **CED 输入输出不对称计算** (V4.1) | 对照 Meta Multi-Token Prediction、Medusa、Eagle 投机采样 |
| **Q-06** | **长周期 Agent 工具交互如何与模型架构和训练回路协同？** | SFT 工具格式 → **V3.2 稀疏注意力 + 大规模合成 Agent 环境 RL** → **DualPath 解决多轮长上下文 KV 重载瓶颈** | 对照 Anthropic Computer Use / MCP、OpenAI Deep Research Agent 训练范式 |

### 索引 2：核心技术索引（Technique Index）

| 技术标识 | 技术名称 | 所属维度 | 首次引入 / 关键演进载体 | 核心机制一句话总结 | 站内精读状态 |
| :--- | :--- | :---: | :--- | :--- | :---: |
| `TECH-MLA` | Multi-head Latent Attention | 架构 / 推理 | DeepSeek-V2 → V3 → FlashMLA | 将 KV 联合低秩压缩为潜向量缓存，推理时通过矩阵吸收避免显式展开，并用解耦小头承载 RoPE | ⏳ 单元 03 |
| `TECH-MOE` | DeepSeekMoE & Aux-Loss-Free Balancing | 架构 / 训练 | DeepSeekMoE → V2 → V3 | 细粒度切分专家以提升组合表达力、隔离共享专家兜底通用知识，并用动态偏置替代破坏主损失的辅助负载均衡损失 | ⏳ 单元 02 |
| `TECH-MTP` | Multi-Token Prediction | 架构 / 推理 | DeepSeek-V3 | 沿深度串联预测未来多个位置并保持完整因果链，训练期加密监督信号，推理期直接充当投机解码草稿头 | ⏳ 单元 04 |
| `TECH-FP8` | Fine-Grained Mixed Precision FP8 | 系统 / 训练 | DeepSeek-V3 / DeepGEMM | 激活按 $1 \times 128$ tile、权重按 $128 \times 128$ block 在线量化，配合 Tensor Core 高精度累加消除离群值崩塌 | ⏳ 单元 04 |
| `TECH-GRPO` | Group Relative Policy Optimization | 后训练 / RL | DeepSeekMath → R1 | 对同一提示采样一组输出，用组内标准化奖励直接构造优势估计，省去与策略同规模的 Value Critic 网络 | ⏳ 第二阶段 |
| `TECH-R1RL` | Cold-Start Reasoning RL & Distillation | 后训练 / 推理 | DeepSeek-R1-Zero / R1 | 验证纯规则奖励可激发自反思长思维链，并通过少量可读冷启动数据与两阶段 RL + SFT 蒸馏实现推理能力迁移 | ⏳ 第二阶段 |
| `TECH-MHC` | Manifold-Constrained Hyper-Connections | 架构 / 稳定性 | mHC (2025.12) | 将多流超连接转移矩阵投影回双随机流形（Birkhoff 多胞形），在拓宽残差通道的同时守住恒等映射稳定性 | ⏳ 第三阶段 |
| `TECH-ENGRAM` | Conditional Memory via Scalable Lookup | 架构 / 记忆 | Engram (2026.01) | 引入基于现代哈希 N-gram 的 $O(1)$ 条件记忆查找模块，将静态事实存储从动态 FFN 计算中解耦 | ⏳ 第三阶段 |
| `TECH-CED` | Causal Encoder–Decoder & Cross-Layer KV | 架构 / 推理 | DeepSeek-V4.1-Flash | 对输入编码与自回归输出采用不对称计算结构，配合跨层 KV 复用进一步削减长文本首字延迟与解码带宽 | ⏳ 第三阶段 |
| `TECH-DUAL` | DualPipe & DualPath Co-Design | 系统 / 基础设施 | V3 / DualPath (2026.02) | 训练端用双向流水线重叠计算与跨节点 All-to-All 通信；推理端用存储到解码引擎的双路径调度打满全网带宽 | ⏳ 第四阶段 |

### 索引 3：文献与证据台账索引（Paper & Evidence Index）

| 编号 | 文献简称 | 年份/月份 | 一手源码 / 报告状态 | 关联问题 ID | 对应文章路径（随进度更新） |
| :---: | :--- | :---: | :--- | :--- | :--- |
| `P-00` | DeepSeek 技术演进全景 | 2023–2026 | 综合基线 | Q-01 ~ Q-06 | ⏳ 第一课（待发布） |
| `P-01` | DeepSeekMoE (arXiv:2401.06066) | 2024.01 | 论文 + 开源权重 | Q-02 | ⏳ 单元 02（待发布） |
| `P-02` | DeepSeekMath (arXiv:2402.03300) | 2024.02 | 论文 + 开源权重 | Q-04 | ⏳ 第二阶段（待发布） |
| `P-03` | DeepSeek-V2 (arXiv:2405.04434) | 2024.05 | 论文 + 开源仓库 | Q-01, Q-02 | ⏳ 单元 03（待发布） |
| `P-04` | DeepSeek-V3 (arXiv:2412.19437) | 2024.12 | 论文 + 开源推理/算子代码 | Q-01, Q-02, Q-03, Q-05 | ⏳ 单元 04（待发布） |
| `P-05` | DeepSeek-R1 (arXiv:2501.12948) | 2025.01 | 论文 + 开源权重 | Q-04 | ⏳ 第二阶段（待发布） |
| `P-06` | DeepSeek-V3.2 (arXiv:2512.02556) | 2025.12 | 论文 + 开源仓库 | Q-01, Q-06 | ⏳ 第二阶段（待发布） |
| `P-07` | DeepSeekMath-V2 (arXiv:2511.22570) | 2025.11 | 论文 + 开源权重 | Q-04 | ⏳ 第二阶段（待发布） |
| `P-08` | mHC (arXiv:2512.24880) | 2025.12 | 论文 | Q-03 | ⏳ 第三阶段（待发布） |
| `P-09` | Engram (arXiv:2601.07372) | 2026.01 | 论文 + 开源仓库 | Q-02 | ⏳ 第三阶段（待发布） |
| `P-10` | DeepSeek-V4 | 2026 | 待按四级证据核验边界 | Q-01, Q-02 | ⏳ 第三阶段（待发布） |
| `P-11` | DeepSeek-V4.1-Flash | 2026 | 待按四级证据核验边界 | Q-01, Q-05 | ⏳ 第三阶段（待发布） |
| `P-12` | DualPath (arXiv:2602.21548) | 2026.02 | 论文 | Q-01, Q-06 | ⏳ 第四阶段（待发布） |
