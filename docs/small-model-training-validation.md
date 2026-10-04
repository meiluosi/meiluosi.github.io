# 小模型训练项目验证记录

2026-10-04，交付目录为 [`projects/small-model-lab`](../projects/small-model-lab/README.md)。本项目可以整体复制为独立GitHub仓库；相邻旧规划目录没有被修改。本轮未创建远端仓库或发布权重。

## 实际运行了什么

- 138,560参数的随机初始化GPT式decoder：2层、64宽、4头、128上下文、512词表；train-only byte-level BPE。
- M4实际MPS执行预训练12步、SFT8步、DPO8步，保存后恢复预训练至14步。每阶段均为真实反向传播和AdamW更新；DPO参考固定为SFT检查点。
- 预训练、SFT、DPO在相同保留合成文本上计算NLL/PPL；后训练另输出单题completion指标。数据仅是16条短文与8条各类后训练fixture，所有指标只是工程验证。
- 独立MPS进程使用 `generate --chat --max-new-tokens 4` 成功载入DPO检查点与内嵌tokenizer。原样结果见报告，没有修饰成模型能力成果。
- 10项CPU测试通过：因果性、真实更新、assistant/EOS/padding mask、变长token加权累积、DPO梯度及冻结参考、分组去重、跨阶段隔离、manifest篡改拒绝、资源参数校验、CPU完全一致恢复及新进程加载。
- 完整目录复制到 `/private/tmp` 后，在新工作目录执行CPU smoke成功，三阶段和恢复14步均通过，证明无需网站或旧外部骨架。

## 证据

- [完整MPS真实记录](../projects/small-model-lab/reports/mac-mps-smoke.json)：逐步loss、token计数、时间范围、源码/配置/数据/tokenizer/检查点hash、每阶段固定fixture评测、恢复和生成。
- [新进程MPS输出](../projects/small-model-lab/reports/mac-mps-fresh-process.json)。
- [独立目录CPU记录](../projects/small-model-lab/reports/portable-copy.json)。
- [验证清单](../projects/small-model-lab/reports/validation.json)。
- [资源方案](../projects/small-model-lab/docs/PLAN.md)、[一手资料核查](../projects/small-model-lab/docs/REFERENCES.md)、[模型卡](../projects/small-model-lab/docs/MODEL_CARD.md)。

正式归档的MPS运行中，预训练/SFT/DPO/恢复的训练环分别约2.019/0.749/0.853/0.189秒，分别处理716/178/427/87个输入token。仅为不到4秒的短跑，不含准备、评测和保存；首轮运行曾更慢，说明kernel缓存和启动影响显著。这些数字不能用于估计30M或0.5B的耗时。

MPS更新后采样的分配内存不是瞬时峰值，`mps_true_peak_bytes`保留null。RSS为同一进程生命周期最高值，会包含之前阶段；不能把它当某阶段独占内存，也不能与统一内存的GPU数字直接相加。温度、交换、持续吞吐和正式能力评测均未测。

## 硬件和环境

本机实读为MacBook Air Mac16,12 / Apple M4 / CPU10核（4+6）/ GPU8核 / 16 GiB统一内存。早期`df -h`约12 GiB可用，MPS运行进程的`shutil.disk_usage`约37.83 GiB；两种读取方式和时点不同，差异来源未单独测定，不能宣布唯一或恒定磁盘容量。每次训练前重新检查并取保守预算。

系统Python3.14.7未用于训练；本次复用Python3.10.18、PyTorch2.14.0和tokenizers0.22.1环境。受限进程曾显示MPS不可用，正常权限下探测及实际反传成功；此差异属于执行环境可见性，不是芯片不支持。

## 完成范围与下一步

数据清洗/分组、BPE、随机初始化预训练、assistant-only SFT、冻结参考DPO、保留集统计、周期检查点/恢复、greedy本地CLI均已有可运行实现。每JSONL上限32MiB、驻内存文档分块、常量学习率，适合小规模起步。streaming/packing、近重复与外部评测去污、正式能力评测、混合精度、量化和网络服务尚未实现。

约30M（目标词表下29,676,544参数）是下一轮实际语料与持续资源测量起点；约100M（91,740,672参数）与约0.5B（514,859,520参数）均未训练，必须逐档测量后决定。旧1B默认目标已撤换。0.5B的FP32 Adam状态本身约7.672GiB，DPO参考、激活、检查点副本和系统另算，16GiB设备不能仅凭静态参数估算判定可行。
