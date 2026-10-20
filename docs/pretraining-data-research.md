# 大模型训练研究笔记 第一卷

资料核对日期：2026-10-15。交付为六篇中文博客正文与本维护记录；不包含全量数据下载、真实语料加工、模型训练、远端提交或部署。

## 文章与范围

| 顺序 | 源文件 | 内容 |
| --- | --- | --- |
| 01 | [语料地图](../src/content/posts/2026-10-15-llm-training-data-01-map.md) | 中英文、多语言与合成语料的路线、版本和选型 |
| 02 | [清洗](../src/content/posts/2026-10-15-llm-training-data-02-cleaning.md) | 抽取、规范化、语言识别、规则及误删审计 |
| 03 | [质量](../src/content/posts/2026-10-15-llm-training-data-03-quality.md) | 参考分类、教育评分、困惑度、DSIR 与分布改变 |
| 04 | [去重](../src/content/posts/2026-10-15-llm-training-data-04-dedup.md) | 哈希、shingle、MinHash、LSH、重复组、评测污染 |
| 05 | [混合](../src/content/posts/2026-10-15-llm-training-data-05-mixture.md) | 文档与 token 权重、温度、覆盖率、DoReMi、训练交接 |
| 06 | [构建](../src/content/posts/2026-10-15-llm-training-data-06-build.md) | manifest、血缘、处理依赖、分组、分词、验收与恢复 |

栏目使用现有 `category` 和共同标签“大模型训练研究笔记”；另有共同标签“预训练数据”。第一篇含卷内目录，各篇有上下文链接。采用现有 Markdown、Mermaid、KaTeX 和代码块，不新增渲染器或前端依赖。

## 研究方法

参考用户指定的 [research-learning](https://github.com/meiluosi/research-learning/blob/main/skills/research-learning/SKILL.md) 及其 `references/claim-status.md`、`references/handoff.md`：重建设计、保留适用边界，区分来源报告、推导、工程建议和待验证问题。用 `pages:write-page` 的写作审校原则核对正文，目标仍为现有博客仓库。

论文读取以 HTML 全文中的相关方法、实验及限制章节为主，数据集身份与规模使用发布方数据卡。搜索摘要只用于发现入口。未下载第三方完整语料或复制论文图表；文中网页、短语、比例和记录均为明确标注的自写教学示例。

## 一手资料与具体使用位置

| 来源 | 阅读位置与用途 | 关联文章 |
| --- | --- | --- |
| [Common Crawl](https://commoncrawl.org/get-started) | 文件格式与获取说明，区分 WARC/WAT/WET | 01、02 |
| [C4 数据卡](https://huggingface.co/datasets/allenai/c4) | 语言与配置入口，避免 C4/mC4 混淆 | 01 |
| [RefinedWeb 数据卡](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/blob/main/README.md) | Dataset Structure 与公开提取部分；968M 网页、500–650GT（依 tokenizer） | 01 |
| [FineWeb 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | What is it 与 Changelog；原始 15T、当前说明超过 18.5T | 01 |
| [FineWeb 论文 v1](https://arxiv.org/html/2406.17557v1) | §3.1–3.4、§4、§6；抽取实验、教育评分与限制 | 02、资料交叉核对 |
| [FineWeb 官方技术报告](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) | 全局与逐批次去重实验 | 04 |
| [FineWeb-Edu 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 派生关系、教育评分与首发规模；不将新抓取规模套用旧实验 | 01、03 |
| [The Pile 论文](https://arxiv.org/abs/2101.00027) | 摘要中的规模与组成，作为历史路线，不支持当前可访问性承诺 | 01 |
| [Dolma 论文](https://arxiv.org/abs/2402.00159) | 原始 3T 英文多来源语料及开放工具的范围 | 01 |
| [RedPajama 仓库](https://github.com/togethercomputer/RedPajama-Data) | V2 的 annotated/deduplicated head_middle 表与语言范围 | 01 |
| [SlimPajama 发布](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) | 2023-06-09；初代来源、627B 规模与跨来源去重，区分预期与结果 | 01 |
| [DCLM 论文 v1](https://arxiv.org/html/2406.11794v1) | §3、§4.4、Table 5；基准与参考分类器 | 01、03 |
| [DCLM 仓库](https://github.com/mlfoundations/dclm) | 2025-09 的 CORE/EXTENDED 基线修正，v1/v2 不直接比较 | 01 |
| [SkyPile-150B 数据卡](https://huggingface.co/datasets/Skywork/SkyPile-150B) | Dataset Summary；规模与纯文本字段 | 01 |
| [万卷 1.0](https://github.com/opendatalab/WanJuan1.0) | 文本来源与多模态发布边界，不引用总量作为文本量 | 01 |
| [ChineseWebText 2.0 仓库](https://github.com/CASIA-LM/ChineseWebText-2.0) | 3.8TB 发布范围及工具入口 | 01 |
| [ChineseWebText 2.0 论文 v1](https://arxiv.org/html/2411.19668v1) | §3.1–3.5；上游来源、质量/领域/毒性分开处理 | 01、03 |
| [CCI3.0-HQ 论文 v1](https://arxiv.org/html/2410.18505v1) | §2、§3.1；500GB、Qwen2 标注、冻结 BGE-M3、0.5B/100B 设置 | 01、03 |
| [OpenCSG V2 数据卡](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu-v2) | 188M 条、约 420B token 的自报口径；不补造 tokenizer | 01 |
| [OpenCSG V3 数据卡](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V3) | 当前明确为教育 SFT，与 V2 正文不同 | 01 |
| [IndustryCorpus2](https://huggingface.co/datasets/BAAI/IndustryCorpus2) | 行业组织入口，不引用动态仓库体积 | 01 |
| [Matrix 数据卡](https://huggingface.co/datasets/m-a-p/Matrix/blob/main/README.md) | 4.69T 的中英混合池与组成，不等同于网页子集 | 01 |
| [悟道入口](https://github.com/BAAI-WuDao/Data) | 官方历史入口，不宣称本次取得完整文件 | 01 |
| [OSCAR](https://oscar-project.org/) | 多语言语料系列入口 | 01 |
| [FineWeb2 仓库](https://github.com/huggingface/fineweb-2) | 语言/文字系统配置 | 01 |
| [FineWeb2 论文 v1](https://arxiv.org/html/2506.20920v1) | §4.2，语言识别与阈值；注意部分过滤器仍使用固定阈值，不能写成全部规则均逐语言独立优化 | 02 |
| [Nemotron-CC 发布](https://developer.nvidia.com/blog/announcing-nemotron-cc-a-trillion-token-english-language-dataset-for-llm-pretraining/) | 原始发布 6.3T，其中 1.9T 合成 | 01 |
| [Nemotron-CC 论文 v1](https://arxiv.org/html/2412.02595v1) | §2 分类器组合与合成增强，不复述跨模型比较为因果证明 | 03 |
| [Nemotron 当前配方](https://docs.nvidia.com/nemotron/nightly/nemotron/data/curation/nemotron-cc.html) | v2 处理入口；nightly 文档是动态来源，记录访问日期 | 01 |
| [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | PDF 来源扩展入口；未引用其动态规模或性能数值 | 01 |
| [C4 分析](https://aclanthology.org/2021.emnlp-main.98/) | Dodge 等，EMNLP 2021；过滤与代表性 | 02 |
| [DSIR 全文 v3](https://arxiv.org/html/2302.03169v3) | 特征空间中的重要性重采样，不等同于事实质量判断 | 03 |
| [IR 教材](https://nlp.stanford.edu/IR-book/html/htmledition/near-duplicates-and-shingling-1.html) | shingle、Jaccard、MinHash 的定义与性质 | 04 |
| [去重论文](https://aclanthology.org/2022.acl-long.577/) | Lee 等，ACL 2022；去重与记忆、评测的关联 | 04 |
| [DoReMi 全文 v3](https://arxiv.org/html/2305.10429v3) | §2、§6；参考模型、超额损失、平均域权重和迁移边界 | 05 |
| [DataTrove](https://github.com/huggingface/datatrove) | README 组件说明及 examples/fineweb.py | 02 |
| [Datasets loading](https://huggingface.co/docs/datasets/loading) | 固定仓库版本与加载对象 | 06 |
| [Datasets streaming](https://huggingface.co/docs/datasets/stream) | 流式读取、缓冲区打乱、分片、take/skip | 06 |

## 编辑核对中处理的风险

- RefinedWeb 规模回到数据卡正文与 metadata 核对，采用 968M 网页和 tokenizer 相关的 500–650B tokens。
- RedPajama-V2 采用其官方表格的 30.4T annotated/deduplicated head_middle 口径，不混用其他论文的重分词结果。
- FineWeb 首发与当前数据卡规模分开；FineWeb-Edu 首发研究值明确限定版本。
- CCI3.0-HQ v1 不同段落的标注数量表述并不完全统一，正文不以这些数值构建比较结论。
- OpenCSG 名称相近但用途不同的版本分别记录；不把 V3 SFT 直接归入 V2 预训练正文。
- SlimPajama 数据卡初次读取失败，改读发布方官方文章取得规模与来源；不使用第三方转载补空白。
- 公式例子均标为教学计算或明确假设下的推导；没有本地模型质量或真实语料统计的虚构结果。
- 稳定分组示例仅处理已合并的 group_id，不把代码称为完整去重管道。

## 验收记录

验收日期：2026-10-16。六篇正文合计约 2.3 万中文字，引用 39 个不同的一手资料页面。

- `pnpm check`、`pnpm type-check`、`pnpm build` 均通过。Astro 检查无错误、无警告，保留 4 条已有提示；构建中的已有语法高亮及包体提示不影响生成。
- 构建时将可选 `AI_SUMMARY_API_KEY` 设为空，避免触发外部自动摘要。Pagefind 共索引 70 个页面。
- 六篇生成页面的 190 个站内链接均能定位到构建产物，卷内目录和前后篇链接可达。
- 真实浏览器检查六篇在 1440 和 320 像素宽度下的阅读表现，共 12 个页面状态；均无整页横向溢出、公式渲染错误或页面脚本异常，Mermaid 图正常生成。
- 执行第六篇 Python 教学示例，检查规范化、哈希、空值拒绝、稳定分组；10,000 个分组在正序和逆序处理时分区一致。JSON 示例可解析，LSH 概率表与混合采样算例核算一致。
- 生成的知识图谱保留全部原有节点及内容，并增加本卷文章和分类／标签；图标和 LQIP 产物未变化。
- 阅读截图保存在 [验收截图目录](./screenshots/pretraining-data-2026-10-16/)。文章已写入本地博客，尚未提交或部署。
- 截图复核发现 Mermaid HTML 节点文字显示异常，四张图已改用 SVG 文字标签，并重新构建、检查。

这些检查验证的是正文示例和网站呈现，不代表已经构建真实训练语料或测得模型训练收益。后续实验应独立记录数据版本、预算、消融与原始结果。


## 2026-10-20 读者友好改写

六篇正文已按研究工程学习顺序重写：具体问题与朴素做法、直觉与数字示例、必要公式、失败边界、改进方法、可比较的实验问题。贯穿中文技术模型的数据准备场景，明确它是教学设定。

保留原 URL、发布时间、卷序、主要数据集、引用与核心算法。规模信息明确沿用 2026-10-15 核对的历史版本，不宣称本次重新更新所有项目。第六篇 JSON 与 Python 示例保留原实现；新增数字例子为文中假设下的推导，不作为真实数据统计或训练结果。

后续主题见 [栏目规划](./llm-training-series-plan.md)。第二卷补齐预训练基础，第三至第六卷聚焦后训练，第七卷整理评测与工程方法；评测贯穿所有卷次。

本次验收：`pnpm check`（0 错误、0 警告）、`pnpm type-check` 和完整构建通过。六篇约 2.1 万中文字，39 个不同的一手资料链接全部保留，JSON 与 Python 代码块与改写前一致。181 个站内链接属性可达。

真实浏览器检查六篇在 1440px 与 320px 下的 12 个页面状态：无整页横向溢出、KaTeX 错误、页面脚本错误或目录／概念节点失效锚点；四张 Mermaid 图正常渲染，概念网络分支切换正常。截图见 [改写验收目录](./screenshots/data-volume-rewrite-2026-10-20/)。本轮仅修改本地内容，未提交或推送。
