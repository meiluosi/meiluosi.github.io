# 第三卷：监督微调 SFT 审校记录

日期：2026-10-20。范围为六篇，正文已全部完成。贯穿任务是从中文实验材料中提取 `model`、`learning_rate`、`steps`。Cedar 模型与记录为自写教学样例，不是实际训练结果。

## 前三篇的来源与边界

| 文章 | 来源 | 支持的内容与限制 |
| --- | --- | --- |
| 01 任务示范 | [InstructGPT 第 3 节](https://arxiv.org/html/2203.02155v1#S3)、[FLAN](https://arxiv.org/abs/2109.01652) | 区分 SFT、奖励建模、PPO；多任务指令训练的研究方向。未把 RLHF 最终效果归因于 SFT，未把英文分布效果转写成本地中文任务结论 |
| 02 消息与模板 | [Transformers 4.57.1 chat_templating](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating) | 消息序列化、模板与模型配套、生成提示、避免重复特殊 token。教学尖括号标记不是实际模型模板 |
| 03 监督范围 | [TRL 0.23.1 SFT Trainer](https://huggingface.co/docs/trl/v0.23.1/en/sft_trainer)、[Transformers 4.57.1 模板接口](https://huggingface.co/docs/transformers/v4.57.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template) | assistant-only 与 completion-only 不等义；模板需支持生成区域 mask；实际标签需要核对。引用版本只用于说明接口，不宣称已安装/运行对应环境 |

InstructGPT 方法章节已实际读取：包括示范、排序反馈与 PPO 的分阶段定义，以及 SFT 单独模型的描述。后续涉及该论文具体效果时需保持阶段与评测分布限制，不能只引用摘要中的最终偏好结果。

后续备选入口已找到 [LIMA](https://arxiv.org/abs/2305.11206) 与 [Self-Instruct](https://arxiv.org/abs/2212.10560)，尚不作为未完成篇目的已审校证据；写作前继续读取方法与限制。

## 本批核心技术检查

- SFT 单条回答求和目标与多样本归一化分开说明，不把两种平均混为一谈。
- 系统消息不被描述为对所有基础模型自动有效；输入材料中的指令样式文本仍需任务覆盖和测试。
- 首个回答 token 的标签位于答案位置，但由前一个输出位置预测。
- prompt 标签忽略不表示提示不可见，也不表示提示相关表示或参数不参与反向传播。
- 多轮错误历史可以作为上下文，但是否监督是独立决定；拆分与完整对话混用可能重复加权。
- 结束标记按实际生成协议处理，不统一等同于 EOS 或 padding。
- 截断后同时检查有效目标数和回答完整性，不能只检查是否有非忽略标签。

## 已执行的算例验证

[check_examples.py](./validation/sft-volume-2026-10-20/check_examples.py)直接读取正文 JSON 和 Python 代码块，检查外层消息、内层答案、字段数值与 `null`、首个回答 token 和结束目标配对。另以错误的输出位置 mask 构造反例，确认会漏掉首个回答目标。

结果见 [examples.json](./validation/sft-volume-2026-10-20/examples.json)。该检查只验证自写教学格式和位置规则，没有使用真实分词器，也没有训练模型；不能代替正文提出的多轮、截断、packing 与任务评价。

## 前三篇页面验收

`pnpm check` 为 0 errors、0 warnings、4 条既有 hints；`pnpm type-check` 通过；完整构建生成 112 页。三篇共 73 个站内链接属性与目标锚点通过。1440px 和 320px 六个页面状态无横向溢出、KaTeX 错误、脚本错误或失效章节锚点；实际切换了消息篇概念网络分支。

- [浏览器检查](./validation/sft-volume-2026-10-20/browser-first-three.json)
- [站内链接检查](./validation/sft-volume-2026-10-20/internal-links-first-three.json)
- [截图](./screenshots/sft-volume-2026-10-20/)，包含正文顶部、公式、概念网络和手机代码块。

第二卷最后一篇已补充第三卷入口。页面验收不等于完成正式 SFT 实验，后三篇尚待写作和验收。

## 后三篇来源与审校

| 文章 | 一手来源 | 核对范围 |
| --- | --- | --- |
| 04 指令数据 | [Self-Instruct 方法与过滤](https://arxiv.org/html/2212.10560v2#S2) | 生成指令/实例、相似性与无效记录过滤；没有把启发式过滤当成答案正确性证明。覆盖表、字段审核和来源分组为本卷提取任务的实验设计 |
| 05 参数化选择 | [LoRA](https://arxiv.org/html/2106.09685v2#S4)、[QLoRA](https://arxiv.org/html/2305.14314v1#S3)、[PEFT 0.17.0 配置](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora) | 低秩增量而非基础权重低秩压缩；A 随机/B 为零、标准 alpha/r 与其他缩放区别；量化存储和计算类型区别；总内存不能按可训练参数比例推算 |
| 06 效果评价 | [MT-Bench 与 Chatbot Arena](https://arxiv.org/abs/2306.05685)、[IFEval](https://arxiv.org/abs/2311.07911) | 模型评审偏差、可验证指令约束的评价方向；不搬用其分数，不将格式检查等同事实正确。字段/整条/配对计数为自写设计 |

后三篇没有加入未经运行的训练结果。LIMA 页面曾未能获取全文，本卷没有引用其具体实验结论。

### 独立数值检查

[check_adaptation_evaluation.py](./validation/sft-volume-2026-10-20/check_adaptation_evaluation.py)已运行，结果在 [adaptation-evaluation.json](./validation/sft-volume-2026-10-20/adaptation-evaluation.json)。验证内容包括：

- 秩一增量矩阵与分支/直接乘法输出均为 `(4,2)`。
- 4096×4096 矩阵使用 r=8：16,777,216 对 65,536，新增参数比例为 0.390625%。
- 十亿权重理想编码的 4-bit/16-bit 存储约 0.466/1.863 GiB，不含运行时开销。
- 配对计数 66/12/4/18 对应旧 70%、新 78%，修复 12、退化 4。
- 全部填 null 的例子可得 200/300 字段正确，但已知字段均未提取。
- Python 布尔值不能按宽松整数判断接受为步数。

前三篇正文代码与 JSON 检查再次通过。上述复核不代表全参数、LoRA 或 QLoRA 的实际资源/能力对照已经运行。

## 整卷最终验收

六篇完成，`pnpm check` 为 0 errors、0 warnings、4 条既有 hints，`pnpm type-check` 通过，完整构建生成 115 页。171 个站内链接属性及目标锚点检查通过；1440px 和 320px 共 12 个页面状态没有横向溢出、KaTeX 错误、脚本错误或失效章节锚点。概念网络分支操作通过，已查看手机公式和代码截图。

- [整卷浏览器检查](./validation/sft-volume-2026-10-20/browser-complete.json)
- [整卷站内链接检查](./validation/sft-volume-2026-10-20/internal-links-complete.json)

所有方法比较仍按实验设计呈现；本卷没有执行正式 SFT/LoRA/QLoRA 训练，也没有据此宣称某方案胜出。
