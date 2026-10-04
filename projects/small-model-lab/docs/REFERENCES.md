# 一手开源参考与采用边界

2026-10-04通过官方仓库/文档核查。以下是设计参考；**本项目没有复制或vendoring这些上游实现**。模型、训练、mask、检查点及测试为本项目的独立教育实现；预算/精确去重工具延续本地已有规划骨架。未来如移植代码，应先锁定commit并保留对应NOTICE/LICENSE。

| 官方来源 | 本轮核实和采用 | 不能据此推断 |
|---|---|---|
| [MiniMind README](https://github.com/jingyaogong/minimind/blob/master/README.md) / [仓库](https://github.com/jingyaogong/minimind) | 当前minimind-3 64M，旧版本26M/104M；覆盖预训练、SFT和偏好训练，借鉴阶段划分；Apache-2.0 | README的“2小时”明确为单3090 SFT 1 epoch，不是Mac全链路时间 |
| [nanochat](https://github.com/karpathy/nanochat) / [CPU/MPS入口](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh) | tokenizer到部署的统一结构；README主案例8×H100，CPU/MPS教育运行缩小模型；MIT | 不把H100成绩或项目标题价格移植到M4，也不把缩小配置当能力基线 |
| [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) / [DPO补充材料](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/04_preference-tuning-with-dpo) | 从模型与mask细节理解PyTorch；代码Apache-2.0 | 某些微调示例加载已有权重，不代表所有章节是一条随机初始化实验；代码许可不覆盖书籍/插图 |
| [MLX LM](https://github.com/ml-explore/mlx-lm) | 官方明确聚焦Apple Silicon推理及微调，包括低秩/全参数微调与量化；MIT | 现成模型LoRA/量化可运行不等于0.5B随机初始化全参数训练可行 |
| [MLX transformer_lm](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) | 原生MLX训练循环参考；MIT | 仍需接数据、后训练、恢复和统一评测；本轮没有同时实现第二套后端 |
| [PyTorch MPS官方说明](https://docs.pytorch.org/docs/main/notes/mps.html) / [官方安装](https://pytorch.org/get-started/locally/) | 检查is_built/is_available并将张量/模型移至mps；本轮实际FP32训练验证 | 可用性不等于完整算子、持续速度或峰值内存保证 |
| [DPO原论文](https://arxiv.org/abs/2305.18290) | 冻结参考与chosen/rejected序列概率差的目标定义 | DPO训练loss下降不等于通用能力提高 |
| [Hugging Face tokenizers官方文档](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html) | BPE、ByteLevel及训练器API；本轮实际版本0.22.1 | 目标词表大小不保证小语料能拟合到该大小 |

版本记录：本项目实测依赖固定在 `requirements-tested.txt`；训练报告包含源码内容SHA-256、配置和数据哈希。上述仓库内容以访问当日为准，远端HEAD固定尝试因GitHub连接失败未取得，故上游commit记录为**未核实**，未编造SHA或称其为永久快照。若之后真正引入上游代码，应在引入时完成commit和许可证锁定，而不是依赖这些浮动参考链接。

依赖许可与数据许可独立：PyTorch、tokenizers依各自官方分发许可；MIT/Apache-2.0代码许可不赋予任何第三方语料或模型权重的额外权限。fixture为自写CC0数据，未来外部语料需另建来源清单。
