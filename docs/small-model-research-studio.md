# 研究工作室与小模型全链路项目

2026-10-04，用户确认推进首页、研究岛 2.0、互动文章、独立探索空间和项目视觉，并指定核心内容为大模型全链路训练。本文是本轮范围与实施规划。

## 一、项目拆分

- **网站**：继续使用 Astro/Svelte/Three.js，承载研究岛、互动导读、文章与实验报告。
- **small-model-lab**：独立训练项目，可迁移交付目录 [`projects/small-model-lab/`](../projects/small-model-lab/README.md)。它可整体复制到任意新仓库。先前的外部规划目录保留，当前可运行实现以此目录为准。数据与检查点不进入博客，不依赖博客构建。GitHub 仓库和权重发布由后续实际发布任务处理。
- 两者通过可导出的报告关联。网站当前展示规划与教学计算，真实结果只从已完成 run 导入。

## 二、设备与规模

2026-10-04 读取 `system_profiler`：MacBook Air（Mac16,12）、Apple M4、10 核 CPU（4 性能 + 6 能效）、8 核 GPU、16 GB 统一内存。早期 `df -h` 显示约 12 GiB 可用空间；稍后 MPS 运行中 `shutil.disk_usage` 记录约 37.83 GiB。两份快照的时点与口径不同，扩展前应重新读取，不能当作持续可用容量。设备序列号等标识不写入项目。小规模基线的实测后端、运行环境和限制见 [训练验证记录](./small-model-training-validation.md)；尚无 30M、100M 或 0.5B 的完整训练测量。

| 档位 | 定位 | 启用条件 |
|---|---|---|
| tiny smoke | 首次验证真实反传、检查点与全链路 | 极小模型、自写格式样本、限步数；只证明工程可运行 |
| 约 30M | 首个正式实验目标 | tiny 闭环通过，先短上下文、micro batch 1 测量 |
| 约 100M | 对照容量/预算 | 30M 工程闭环稳定，并有资源记录 |
| 约 0.5B | 规模扩展 | 短跑峰值内存与持续吞吐满足本机预算 |
| 现成 Qwen2.5-0.5B | LoRA/量化 LoRA 应用对照 | 单独目录、数据与评测记录，不算从零预训练 |

16 bytes/参数估算只计 FP32 权重、梯度、Adam 的两份状态。30M/100M/500M 对应约 0.45/1.49/7.45 GiB；激活、框架、系统、DPO 参考模型等另算。0.5B 的推理/LoRA可行性不能替代全参数预训练评估。

资源测量记录：预热设置、稳定 tokens/s、持续运行速率、模型参数量、精度、micro batch、累积步数、上下文、峰值内存及检查点大小。时间由 token 预算/实测速率推算，并加上验证和保存成本。第一阶段采用有上限的工程小跑，正式训练预算在测量后选择。

扩展闸门：每档先做短跑，再做持续运行测量；记录热身和稳态、CPU/MPS 后端、系统 memory pressure/swap 及每阶段的额外内存。预留系统空间，不靠 swap 勉强维持。DPO 必须重新测量冻结参考的开销。0.5B 的 FP32 模型与 Adam 检查点按约 12 bytes/参数估算，514.86M 单份就约 5.75 GiB，保留两份已接近 11.51 GiB。先重新读取目标磁盘容量，明确数据、临时文件、两份检查点与导出副本的总预算并留余量，再考虑扩展；早期 12 GiB 或稍后的 37.83 GiB 快照都不能替代启用前检查。

## 三、技术路线与参考

主线先用 **PyTorch + MPS**，保留 CPU 小配置和未来 CUDA 路线。第一版采用易于逐项检查的 GPT 式 decoder、共享输入输出 embedding、可训练位置表示、GELU 前馈、LayerNorm；避免第一版同时维护两种架构或两套训练后端。RMSNorm/RoPE/SwiGLU/GQA作为后续受控改动。

分词器从 8K byte-level BPE 开始，训练时记录实际词表大小。只用 train split 拟合。预训练按 next-token loss；SFT 只对 assistant token 计算 loss；DPO 使用 SFT checkpoint 作为冻结参考。

| 一手参考（2026-10-04 核对） | 采用的思路 | 使用边界 |
|---|---|---|
| [MiniMind](https://github.com/jingyaogong/minimind) | 中文小模型、数据格式、预训练与后训练组织 | 当前主线 64M，旧版本26M/104M；“2小时”指特定3090 SFT 1 epoch，不是Mac全链路耗时 |
| [nanochat](https://github.com/karpathy/nanochat) | tokenizer、统一流水线、检查点、评测与推理 | 主性能案例基于多张H100；CPU/MPS教育配置需单独阅读，不能据此承诺本机训练时间 |
| [LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch) | 模型原理、PyTorch实现、SFT及DPO补充材料 | 一些微调示例加载现成权重，不能把所有示例视为同一个从零训练实验 |
| [MLX LM](https://github.com/ml-explore/mlx-lm) | 现成模型推理、LoRA与量化 | 不将现成微调入口当作完整随机初始化训练框架 |
| [MLX transformer_lm](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) | 原生MLX训练循环 | 需要另补数据、后训练和统一评测；暂不作为第二套主线 |
| [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) | 后端可用性与Metal执行 | 算子、dtype与吞吐要在实际环境检查 |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 中文现成基座对照 | 官方卡片0.49B；与从随机初始化的小模型分开报告 |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 小模型规模、训练与数据组织参考 | 主要面向英语，官方预训练token预算远大于本机工程实验 |

引用实现时固定上游commit和许可证。MiniMind/LLMs-from-scratch采用Apache-2.0，nanochat/MLX LM/MLX examples采用MIT；代码许可不覆盖第三方语料，LLMs-from-scratch代码许可也不等于书籍和插图可复制许可。所有采用的数据另建来源清单。

## 四、训练项目里程碑

| 阶段 | 产物 | 完成条件 |
|---|---|---|
| T0 范围与骨架 | README、配置、预算工具、数据处理入口、报告约定 | 项目可独立搬迁，现有命令与未实现阶段明确 |
| T1 数据与分词 | 数据清单、去重、文档group划分、冻结BPE、统计报告 | 无已知跨集合重复；tokenizer只拟合train；记录哈希与特殊token |
| T2 随机初始化预训练 | 模型、训练循环、恢复检查点、资源记录 | 约30M工程小跑可保存恢复；记录实际loss和固定采样 |
| T3 SFT | assistant-only mask、checkpoint、固定提示输出 | prompt/padding不参与loss；可比较预训练与SFT |
| T4 DPO | 冻结参考、偏好数据、DPO checkpoint | 相同prompt配对，参考无梯度；检查长度偏置和退化 |
| T5 统一评测 | 每阶段JSON报告与曲线 | 同一保留集/模板/解码；ppl使用相同tokenizer；不得以训练集分数代替泛化 |
| T6 推理与发布 | 原生CLI、模型卡、数据卡、复现文档 | 新进程加载完整产物；未来格式转换另行验证；代码与大文件分开 |
| T7 扩展实验 | 100M/0.5B或MLX基座对照 | 根据资源与任务目标选择；每轮改变可归因因素 |

工程小跑用于验证流水线；正式模型能力需要更合适的语料与训练预算。先完成T1–T6最小闭环，再展开模型规模和后训练算法对照。GRPO/奖励模型等列作DPO后的研究扩展。

## 五、网站工作包

1. **入口与视觉**：主按钮进入`/world/`；直接阅读入口；首页展示全链路核心作品；项目使用统一SVG视觉。
2. **研究岛2.0**：三个可辨识装置、层次与柔和接触阴影、相机推进、移动反馈、复位、轻量/完整画质；手机点选站点。
3. **内容连接**：每站统一案例、实验、阅读路线；模型站进入`/lab/model-pipeline/`。
4. **旗舰导读**：七阶段交互、输入输出、模型与token预算、数据隔离示例；明确规划值和实际结果的区别。
5. **文章更新**：中英原1B计划重写为小模型全链路；保留旧sourceId/URL，更新日期、可见标题、案例和路线。
6. **实际结果的呈现**：待训练项目产生run后接入真实曲线、模型回答、数据统计与阶段对比。这部分不能先填示例结果伪装完成。
7. **运行体验**：保留离屏暂停、reduced-motion与无WebGL入口，核对键盘/窄屏；具体画质预算依据后续设备测量。

## 六、报告约定

每个实际run应包含：`run_id`、代码revision、配置哈希、数据manifest哈希、tokenizer哈希、设备/后端版本、种子、阶段与父检查点、实际训练token数、wall time、吞吐、峰值内存、评测集版本、解码参数、逐题结果和聚合指标。未测字段用null并记录原因，不能用零或模拟值表示。

完整恢复还要保存optimizer/scheduler、RNG及数据位置。跨后端的逐位复现不作默认承诺，记录后端和容差。

## 七、本轮与后续的分界

本轮推进网站体验、核心文章、互动规划工具，以及独立训练项目骨架。本轮用极小配置验证真实预训练、SFT、DPO 与评测/部署链路，具体是否通过及指标以训练验证记录为准。正式语料选择、30M 及以上训练和能力评测属于后续研究，不因 smoke 通过而标记完成。
