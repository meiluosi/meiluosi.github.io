# 第二卷：预训练基础与小模型实验 审校记录

日期：2026-10-20。对应六篇 `2026-10-20-llm-training-pretrain-*`。这是来源与验收记录，不是新增模型训练报告。

## 正文组织

从第一卷的中文技术语料场景继续，先确定分词、预测目标，再跟踪前向计算、batch 和更新，最后设计可解释实验。每篇有可核对例子、误用边界和检查问题。共用概念网络由正文生成，没有为凑数量插入虚构训练结果。

## 来源与引用范围

| 正文 | 一手来源 | 使用范围与约束 |
| --- | --- | --- |
| 01 分词 | [BPE 论文](https://aclanthology.org/P16-1162/)、[SentencePiece](https://aclanthology.org/D18-2012/)、[HF BPE 教程](https://huggingface.co/learn/llm-course/en/chapter6/5)、[Tokenizers pipeline](https://huggingface.co/docs/tokenizers/pipeline) | 区分子词算法、实现工具、规范化、预切分与编号；自写中文合并例子不代表实际分词器输出 |
| 02 目标 | [GPT-2 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/gpt2)、[CrossEntropyLoss 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)、Transformer 第 3.1 节 | 标签内部错位是指定接口约定；交叉熵算例不含类别权重和标签平滑 |
| 03 前向 | [Transformer 第 3 节](https://arxiv.org/html/1706.03762v7#S3)、[RoFormer 第 3 节](https://arxiv.org/html/2104.09864v5#S3)、[SDPA 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)、[FlashAttention](https://arxiv.org/abs/2205.14135) | 教学结构为 decoder-only pre-norm，不冒充原始 encoder-decoder post-norm；RoPE 等式不证明长上下文外推能力；高效算子不等于删减连接 |
| 04 batch | [Data Collator 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/data_collator) | 动态补齐与标签处理；独立打包的遮罩、标签和位置规则是正文明确提出的设计，不声称该通用 collator 自动实现它们 |
| 05 优化 | [Adam](https://arxiv.org/abs/1412.6980)、[AdamW](https://arxiv.org/abs/1711.05101)、[AdamW 2.9 接口](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html)、[AMP 2.9](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html)、[范数裁剪](https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm_.html) | 区分裸梯度下降例子与 Adam 更新；缩放还原、累积和裁剪顺序；不推荐无实测的统一学习率 |
| 06 实验 | [PyTorch 检查点教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[复现说明 2.9](https://docs.pytorch.org/docs/2.9/notes/randomness.html)、仓库既有验证记录 | 权重加载不等于过程恢复；相同种子不保证跨环境逐位一致；历史短跑仅证明工程范围 |

公开文档固定版本用于核对语义，不宣称当前网站环境安装了对应训练依赖。未获取成功的 HF language_modeling 页面未作为正文依据。

## 工程记录的身份

第 06 篇引用 `docs/small-model-training-validation.md`，已核对本地文件、`projects/small-model-lab/reports/validation.json`、`configs/tiny-smoke.json` 与 `model.py`。对应远端跟踪提交 `5784b2b2e2e04d7f4a66c5db0c3df0eba43d973d` 包含该文档，正文使用固定提交链接，避免误用不存在的 main 分支。

该记录日期为 2026-10-04，包含 138,560 参数模型的短程工程验证，不是本轮重跑。30M/100M/0.5B、持续性能与模型能力均不能由该短跑推出。本轮没有新增训练结果。

## 教学算例的独立复核

首批已复核中文 BPE 合并频次、28→20→15 的长度变化、负对数概率与 token 加权平均。

后四篇的可重复检查位于 [check_examples.py](./validation/pretrain-volume-2026-10-20/check_examples.py)，已运行结果位于 [examples.json](./validation/pretrain-volume-2026-10-20/examples.json)。只依赖 Python 标准库，运行后打印结果，不修改项目文件。

覆盖：

- 注意力权重 `(0.5, 0.25, 0.25)` 与输出 `(1.5, 1.5)`。
- 二维 RoPE 点积的相对旋转等式。
- B 文档位置 5 仅允许读取位置 4、5；独立打包保留五个有效目标。
- 三种学习率的单参数梯度下降序列。
- `(3,4)` 裁剪到范数 2 后为 `(1.2,1.6)`。
- 32,768 个名义位置与假设 75% 下的 24,576 个有效目标。

这些检查验证算术与设计语义，不验证一个训练框架已经正确实现它们。正文中的模型对齐、打包等价、正式对照和恢复实验均按设计呈现。

## 最终页面验收

`pnpm check` 为 0 errors、0 warnings、4 条既有 hints；`pnpm type-check` 通过；完整构建生成 109 页。六篇 167 个站内链接属性及目标锚点通过。1440px、320px 共 12 个页面状态没有横向溢出、公式错误、脚本错误或失效章节锚点。概念网络分支切换正常；关键术语已在正文显式标记。

- [浏览器结构检查](./validation/pretrain-volume-2026-10-20/browser.json)
- [内部链接检查](./validation/pretrain-volume-2026-10-20/internal-links.json)
- [桌面与手机截图](./screenshots/pretrain-volume-2026-10-20/)

构建和页面验证支持的是内容渲染正确，不是训练设计已经得到实验验证。


## 2026-10-23 最终交叉审校补充

逐篇矩阵见 `llm-training-series-final-audit.md` 第二卷章节。修正 softmax 加法平移与尺度的英文混淆、注意力系数与向量贡献的双语解释、英文嵌入示例的同 ID 前提、吞吐计数及曲线表中的证据越界。第二卷已有记录的来源范围仍适用；本轮另读固定版本 SDPA 与 AMP 文档确认布尔 mask、评估 dropout、缩放还原和累积顺序。历史 138560 参数短跑与 JSON 原记录核对一致，未在本轮重跑。
