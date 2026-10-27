<!-- 由各批research.json和index-annotations.json生成；历史数据保留快照，当前修订写入注释层。 -->

# 全局路线覆盖与缺口

截至2026-10-28，20条路线均有指定范围的审校序列。共150条批次节点记录，按稳定ID去重为144个节点；162条批次来源记录涉及153个来源ID。相同来源的版本／阅读范围继续分别保存，不以去重篇数代表阅读深度。

第一轮106个候选中，81个有同ID后续笔记；25个没有同ID匹配，其中1个有同作品章节部分覆盖，24个尚无对应作品的阅读映射。相关但不同的作品不计入原作覆盖。已有笔记也可能仅覆盖摘要或部分章节，不能写成全文完成。候选在某路线上的关联不代表该路线的全部解释已审校。

[统一入口](index.html) · [共同问题索引](questions-index.md) · [可编辑覆盖图](coverage.svg) · [汇总数据](research-index.json)

| 路线 | 最新路线叙事 | 去重关联节点 | 首轮候选同ID笔记／总数 | 当前缺口 |
| --- | --- | ---: | ---: | --- |
| [行为表现与操作性路线](round-three-b/atlas.html?route=behavioral) | [B批](round-three-b/routes/behavioral.md) | [8](route-coverage/behavioral.md) | 4／5 | 未读取Block 1981全文及Hernández-Orallo–Dowe 2010交互测验正文；本批跨任务问题由ARC与HELM覆盖。；不追踪2022之后的排行榜与最新协议；HELM只按2022 v1，ARC只按2019 v2记录。 |
| [心理测量与能力结构路线](round-three-b/atlas.html?route=psychometric) | [B批](round-three-b/routes/psychometric.md) | [11](route-coverage/psychometric.md) | 6／7 | Meredith 1993原文正文未取得；不变性的方法证据使用Jak–Jorgensen 2017，不能冒称补齐1993原文。；Thurstone 1938、Horn的关键扩展原作、2012完整CHC书章尚待阅读；1963和1993的公开OCR需用原版扫描复核图表。；Gardner首版与后出版本未校勘，2006回应未读；不根据单一大学样本裁决全部多元智能理论。 |
| [学习效率与泛化路线](round-three-a/atlas.html?route=learning-efficiency) | [A批](round-three-a/routes/learning-efficiency.md) | [18](route-coverage/learning-efficiency.md) | 5／5 | 本批已将系统性争论、SCAN／MLC、捷径学习及问题卡对应；更广泛的现代泛化机制与评测仍可继续扩展。；Baird复现范围限定为1995六状态预测例，不能替代组合泛化实验或一般强化学习收敛结论。 |
| [理性、适应与资源约束路线](round-three-b/atlas.html?route=rational-resource) | [B批](round-three-b/routes/rational-resource.md) | [7](route-coverage/rational-resource.md) | 5／6 | 未通读Hutter 2005专著或后续AIXI证明；2000稿的保证只按所读条件表述。；Chater等1997直接批评获取失败；本批启发式边界由1996原文的协议和作者自限支持。；NARS限1995论文中的3.0；资源理性仅覆盖指定目标文章、两篇评论与部分回应，不代表当前全部研究。 |
| [符号、逻辑、搜索与组合路线](round-three-a/atlas.html?route=symbolic-logic) | [A批](round-three-a/routes/symbolic-logic.md) | [9](route-coverage/symbolic-logic.md) | 6／7 | Logic Theorist、GPS、框架问题的独立原文及其他规划分支仍可继续拓展；本批不声称穷尽。；A*扩展节点最优性及后续更正未复核；STRIPS期刊版尚未与会议版逐项对照。 |
| [连接主义、统计学习与表征路线](round-three-a/atlas.html?route=connectionist) | [A批](round-three-a/routes/connectionist.md) | [10](route-coverage/connectionist.md) | 6／9 | 十个节点覆盖本批指定转折，不代表完整连接主义史。；C批已选读Hebb的1988重印所列片段；1949原书完整论证与版本校勘、感知机专著及Minsky–Papert 1969仍待补。不能用1958选段替代收敛定理。；张量积表示的网络实现、递归操作及Fodor–McLaughlin 1990后续批评尚未完成原文审校。；Transformer、规模规律和现代语言模型分支尚未展开；本批现代训练节点覆盖监督视觉及程序归纳，不声称覆盖全部现代模型。 |
| [概率推断、因果模型与世界模型路线](round-three-a/atlas.html?route=causal-world-models) | [A批](round-three-a/routes/causal-world-models.md) | [10](route-coverage/causal-world-models.md) | 6／7 | Pearl 1988专著、因果发现与演算完备性的后续发展尚未独立展开；此处不声称覆盖全部因果史。；PlaNet补充材料与训练代码、World Models后续版本及更广泛世界模型成果留作扩展；本文的作者实验数字均未计作本项目复现。 |
| [控制论、强化学习与主动推断路线](round-three-a/atlas.html?route=cybernetics-rl) | [A批](round-three-a/routes/cybernetics-rl.md) | [12](route-coverage/cybernetics-rl.md) | 7／8 | Wiener原版与所读再版未逐字对勘，Bellman 1957专著、AlphaZero等仍可后续拓展；现有材料不替代这些版本。；MuZero已核对2019v1和2020期刊正文的所列段落及表1；期刊网页缺失公式、补充材料与全套图形未逐项校勘。 |
| [认知架构与多机制协作路线](round-three-a/atlas.html?route=cognitive-architectures) | [A批](round-three-a/routes/cognitive-architectures.md) | [6](route-coverage/cognitive-architectures.md) | 2／5 | Newell 1990原书及ACT* 1983尚未建立本批原典笔记；现有节点不替代这些版本。；标准模型的其他直接异议、图2逐格比较与2017年后的修订仍待扩展；本批没有全面重跑认知架构。 |
| [神经科学路线](round-three-c/atlas.html?route=neuroscience) | [C批](round-three-c/routes/neuroscience.md) | [7](route-coverage/neuroscience.md) | 5／6 | Marr多层解释虽在早期叙事中提及，尚无Vision原书独立笔记；连接组仅选读Schlegel指定范围，未审查全部FlyWire伴随论文。；Hebb仅选集重印；LTP仅摘要与354页讨论图像，原始记录和完整统计未重验；缺长期学习行为与突触干预的联合证据。；2018网络理论为观点文章，2023/2026预测检验留后续；本批不作截至2026年的完整综述。；第四轮补读Barbey的测量与病损方法、Jonas的任务限制；未独立复算影像和处理器代码。Barbey作者名单的当前更正见索引修订记录。 |
| [发展、建构与开放式学习路线](round-three-c/atlas.html?route=developmental-openended) | [C批](round-three-c/routes/developmental-openended.md) | [6](route-coverage/developmental-openended.md) | 6／6 | Piaget1936法文初版受访问验证限制，实际采用1952英译后续印次；未逐条核对婴儿观察记录和法英异文。；核心知识所引每个实验和跨文化证据未单独复核；新颖性、任务生成不等于递归自我改进或无限增长。；长期持续学习、灾难遗忘、开放性度量及POET后续版本仍待补读。；第四轮第二阶段选读Smith等1999实验1与6，明确年龄／等待时长及试次层次；短时效应不作为长期发展已识别的证据。Thelen等1984原刊未取得。 |
| [具身、生态、生成与动力系统路线](round-three-a/atlas.html?route=embodied-dynamical) | [A批](round-three-a/routes/embodied-dynamical.md) | [12](route-coverage/embodied-dynamical.md) | 6／6 | Brooks已核对实验范围与第二层实现的版本变化；期刊图中全部连线及硬件行为未独立复核。；Thelen与Smith仍限于第1章和第4章指定范围；Beer的全部相图推导和原始模拟未复核。C批已完成生物与发展的指定范围；其余长期学习和发展证据见C批缺口。；第四轮第二阶段新增Smith等1999实验1／6及2007互动理论、2009知觉交互的指定范围选读；原实验数据与长期发展机制尚未独立复核。 |
| [生命、自维持与演化路线](round-three-c/atlas.html?route=autopoiesis-evolution) | [C批](round-three-c/routes/autopoiesis-evolution.md) | [6](route-coverage/autopoiesis-evolution.md) | 4／5 | 此前C批为重印与转录选读；本轮原刊指定页的补核见下项。其余物种与演化经验材料未由本轮新增。；黏菌只核对指定实验正文；分子机制、跨实验室复现和Levin各项形态干预原始研究未全面审查。；Levin2022 TAME、独立批评和生命定义的其他传统仍为明确后续队列。；第四轮第二阶段取得1974原刊扫描，核对187–189、192–193页及§9成分参与例外；新增2005适应性定义与注8。原模拟、全部算法及生命—智能—体验的桥接仍未验证。 |
| [社会文化与语言路线](round-three-d/atlas.html?route=sociocultural-language) | [D批](round-three-d/routes/sociocultural-language.md) | [7](route-coverage/sociocultural-language.md) | 2／5 | Boyd与Richerson 1985专著、Tomasello 1999及Henrich 2015专著未选读。；Vygotsky网页混合译本未与俄文校勘；Derex 2013本次仅读摘要，方法依2014争论不得反充原文全文。；非人类累积文化的原始实验、文化吸引理论与语言多样性仍需补读。 |
| [集体、分布式与延展心智路线](round-three-d/atlas.html?route=collective-extended) | [D批](round-three-d/routes/collective-extended.md) | [8](route-coverage/collective-extended.md) | 3／5 | Hayek的机制论证不覆盖制度比较的全部实证；不把价格协调当成市场完美定理。；Hutchins 1995专著未读；本批新增的是驾驶舱论文作者稿。Clark回复为2006仓储稿，未逐字对照2010书章。；第四轮已补读Woolley补充材料指定页、Bates方法／局限、Riedl 2021多样本分析和2022更正；未复算数据或完成群体文献系统综述。动物集群与长期组织协作仍待补。；第四轮第二阶段补读延展认知的支持、反方与回复，核对驾驶舱任务转换和知觉交互的机会校正指标；未裁决统一的认知构成判据。 |
| [信息、压缩、复杂系统与物理路线](round-three-e/atlas.html?route=information-physics) | [E批](round-three-e/routes/information-physics.md) | [6](route-coverage/information-physics.md) | 5／6 | Rissanen 1978最小描述长度、Landauer 1961物理耗散尚未纳入正文选读。；Solomonoff Part II及后来的收敛定理未读；信息瓶颈的现代深度学习争议不由1999稿代替。；临界性后续统计与跨尺度干预、因果熵补充材料及独立复现待补。 |
| [语义、理解与心灵哲学路线](round-three-e/atlas.html?route=philosophy-of-mind) | [E批](round-three-e/routes/philosophy-of-mind.md) | [6](route-coverage/philosophy-of-mind.md) | 4／5 | Dennett 1987书未整本读；本批改以已核对的1981论文为时间节点。；Putnam后期功能主义修正、Searle作者答复以及其他语义理论待系统比较。；现代多模态、行动和社会互动系统须另行实测；这里不裁决任何在用模型是否理解。 |
| [意识、情感与自我路线](round-three-e/atlas.html?route=consciousness-self) | [E批](round-three-e/routes/consciousness-self.md) | [7](route-coverage/consciousness-self.md) | 4／6 | Baars 1988、Damasio 1994书、情感与自我的其他路线未系统选读。；IIT 2004与2.0／3.0／4.0版本变迁仍缺逐项对照，不能把2025实验倒算成对2004所有命题的检验。；COGITATE补充材料、理论提出者的完整回应、后续独立研究未纳入本批。 |
| [智慧、价值与跨文化路线](round-three-d/atlas.html?route=wisdom-values) | [D批](round-three-d/routes/wisdom-values.md) | [7](route-coverage/wisdom-values.md) | 4／5 | 古典文本均为选读；未做希腊、梵文及汉籍写本校勘，也不代表传统全貌。；Sternberg 1998仅摘要已读，正文获取未成功；当前作者网站另列，不能倒填1998版措辞。；跨文化证据限所读日本与美国样本；儒道佛之外的伊斯兰、非洲及原住民传统待补。 |
| [形而上学、非经典计算与量子路线](round-three-e/atlas.html?route=metaphysics-quantum) | [E批](round-three-e/routes/metaphysics-quantum.md) | [6](route-coverage/metaphysics-quantum.md) | 3／5 | Penrose 1989著作与2014 Orch OR修订未全文选读；现代微管实验及其独立检验留待补读。；Hagan按2002发表年锚定，实际阅读2000 v1，未与期刊版逐字校对。；泛心论组合问题的反驳文献与非经典计算其他分支尚未系统纳入。 |

数量按路线分别去重，多路线节点不能再次相加。图中日期取该节点在该路线最近一次记录的日期口径；不确定古典文本单列。

## 第一轮没有同ID后续笔记的候选

### 行为表现与操作性路线

- **汇集不同智能定义**（e-legg-definitions-2007）：回查被收录定义的原始语境，避免把汇编当作统一共识。 当前映射：[e-legg-hutter-2007](round-three-b/notes/e-legg-hutter-2007.md)：定义汇编与形式化通用智能论文不同。

### 心理测量与能力结构路线

- **多种基本能力**（e-thurstone-1938）：旋转、任务选择与样本怎样影响得到的能力结构？ 当前映射：[e-thurstone-1934](round-three-b/notes/e-thurstone-1934.md)：1934论文不能替代1938专著。

### 学习效率与泛化路线

本路线首轮候选均已有后续笔记；阅读和版本缺口仍见上表。

### 理性、适应与资源约束路线

- **通用序贯决策的形式化**（e-hutter-2005）：不可计算性、先验与效用设定如何影响工程可实现性？ 当前映射：[e-hutter-2000](round-three-b/notes/e-hutter-2000.md)：2000稿不能替代2005专著和后续证明。

### 符号、逻辑、搜索与组合路线

- **搜索与自我对弈相互改进**（e-silver-2017）：规则已知、任务封闭和训练预算限定什么；不等同于通用自我改进。 当前映射：尚未建立对应作品的后续阅读映射；相关主题出现不计为原作阅读。

### 连接主义、统计学习与表征路线

- **感知机能力边界**（e-minsky-papert-1969）：限制适用于哪些模型，不能外推为所有神经网络都不可行。 当前映射：尚未建立对应作品的后续阅读映射；相关主题出现不计为原作阅读。
- **以注意力组织序列建模**（e-vaswani-2017）：原始翻译证据与后来通用模型表现需分别考证。 当前映射：尚未建立对应作品的后续阅读映射；相关主题出现不计为原作阅读。
- **规模与损失的经验关系**（e-kaplan-2020）：损失规律的区间、数据分布与能力外推边界是什么？ 当前映射：尚未建立对应作品的后续阅读映射；相关主题出现不计为原作阅读。

### 概率推断、因果模型与世界模型路线

- **概率图中的不确定性推理**（e-pearl-1988）：概率依赖本身不能自动给出干预因果关系。 当前映射：[e-pearl-1982](round-three-a/notes/e-pearl-1982.md)：早期贝叶斯网络论文不能替代1988专著。

### 控制论、强化学习与主动推断路线

- **搜索与自我对弈相互改进**（e-silver-2017）：规则已知、任务封闭和训练预算限定什么；不等同于通用自我改进。 当前映射：尚未建立对应作品的后续阅读映射；相关主题出现不计为原作阅读。

### 认知架构与多机制协作路线

- **认知架构与知识使用**（e-anderson-1983）：须区分 ACT* 与后续 ACT-R 版本，不将名称倒置到同一年。 当前映射：[e-anderson-2004](round-three-a/notes/e-anderson-2004.md)：2004架构论文不能替代ACT*原书。
- **多机制协作的心智解释**（e-minsky-1986）：解释性比喻怎样转为可检验的机制与实现？ 当前映射：尚未建立对应作品的后续阅读映射；相关主题出现不计为原作阅读。
- **跨任务的统一认知理论**（e-newell-1990）：统一框架在哪些任务形成了可失败的预测？ 当前映射：[e-soar-1987](round-three-a/notes/e-soar-1987.md)：Soar论文不等于1990统一认知专著。

### 神经科学路线

- **解释计算的多个层次**（e-marr-1982）：视觉案例如何外推到其他认知功能，层次之间怎样约束？ 当前映射：早期路线提到多层解释，但无Marr原书的独立阅读节点；不由概念提及补写为已读。

### 发展、建构与开放式学习路线

本路线首轮候选均已有后续笔记；阅读和版本缺口仍见上表。

### 具身、生态、生成与动力系统路线

本路线首轮候选均已有后续笔记；阅读和版本缺口仍见上表。

### 生命、自维持与演化路线

- **比较不同基底上的认知**（e-levin-2022）：实验结果、框架定义和关于心智的广泛推断分别支持什么？ 当前映射：[e-levin-2019](round-three-c/notes/e-levin-2019.md)：2019文章不能替代2022 TAME版本。

### 社会文化与语言路线

- **文化传递的演化模型**（e-boyd-richerson-1985）：传递规则的经验依据及模型与历史解释的距离是什么？ 当前映射：尚未建立对应作品的后续阅读映射；相关主题出现不计为原作阅读。
- **文化积累与人类认知**（e-tomasello-1999）：物种比较和文化比较的证据如何支持特定机制？ 当前映射：[e-tomasello-2005](round-three-d/notes/e-tomasello-2005.md)：共享意图论文与文化起源专著是不同作品。
- **累积文化与认知生态**（e-henrich-2015）：案例选择、替代解释和不同文化的覆盖需核对。 当前映射：[e-henrich-2004](round-three-d/notes/e-henrich-2004.md)：塔斯马尼亚模型论文与2015专著是不同作品。

### 集体、分布式与延展心智路线

- **把认知分析单位扩展到系统**（e-hutchins-1995）：系统边界怎样划定，外部支持与认知构成如何区分？ 当前映射：[e-hutchins-cockpit-1995](round-three-d/notes/e-hutchins-cockpit-1995.md)：同年驾驶舱论文与Cognition in the Wild专著不同。
- **局部信息与群体方向**（e-couzin-2005）：模型演示、动物实测与人类集体智能之间不能直接类推。 当前映射：尚未建立对应作品的后续阅读映射；相关主题出现不计为原作阅读。

### 信息、压缩、复杂系统与物理路线

- **用描述长度选择模型**（e-rissanen-1978）：编码选择与任务目标如何影响所谓最短解释？ 当前映射：尚未建立对应作品的后续阅读映射；相关主题出现不计为原作阅读。

### 语义、理解与心灵哲学路线

- **意向立场与预测**（e-dennett-1987）：预测策略的有效性与内在理解的本体判断如何区分？ 当前映射：[e-dennett-1981](round-three-e/notes/e-dennett-1981.md)：1981论文为相关入口，未核对其与1987书内章节异文；不替代原书。

### 意识、情感与自我路线

- **全局工作空间**（e-baars-1988）：意识通达、报告能力和现象体验分别怎样定义和测量？ 当前映射：[e-dehaene-changeux-2011](round-three-e/notes/e-dehaene-changeux-2011.md)：后来的工作空间综述不能替代Baars原书。
- **情感与决策**（e-damasio-1994）：躯体标记假说、病例证据与后续实验争论需要分别读取。 当前映射：[e-bechara-1997](round-three-e/notes/e-bechara-1997.md)：1997实验不能替代1994专著论证。

### 智慧、价值与跨文化路线

- **概念、缘起与智慧的原典入口**（e-nagarjuna-mmk）：古典成书年代未核；出版页摘要疑似串书，不能用于解释原典内容，须对照实际译本。 当前映射：[e-nagarjuna-24](round-three-d/notes/e-nagarjuna-24.md)：D批选读汉译《中论》第24品；只构成该作品的章节部分覆盖，不等于梵文或全书已读。

### 形而上学、非经典计算与量子路线

- **非算法心智与物理问题**（e-penrose-1989）：论证、物理假说与实验事实逐层分开。 当前映射：[e-hameroff-penrose-1996](round-three-e/notes/e-hameroff-penrose-1996.md)：联合物理假说论文不能替代1989著作。
- **Orch OR 的修订与回应**（e-hameroff-penrose-2014）：核对与 1996 版本的变化及反方回应；作者综述不能替代独立验证。 当前映射：[e-hameroff-penrose-1996](round-three-e/notes/e-hameroff-penrose-1996.md)：1996版本不能替代2014修订和回应。

## 第四轮第一阶段

已进入[从表现到机制的证据审计](round-four/index.html)：六条条件性结论、十份来源记录、六个比较案例与精确方法核验。原始历史节点数量保持原口径，综合来源不重复叠加。

[覆盖与书目修订记录](round-four/index-audit.md)保留25个候选的逐项映射、过时缺口修订及Barbey 2012作者更正（Chad Forbes）。第二阶段[系统边界与发展](round-four/boundaries/index.html)新增九份选读、27条定位主张、六条结论与8项方法检查；[阅读增量](round-four/boundaries/reading-updates.md)单列。后续按[阶段计划](round-four/plan.md)推进意义与意识、综合修订与网站联动；新经验实验尚未执行。
