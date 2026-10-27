# 智能问题演进史 · 第一轮

2026-10-23 建档。**本轮完成的是候选全景与文献工作台；深读、历史关系考证和正式解读尚待第二轮。**

## 阅读入口

- [研究与维护方案](../intelligence-history-plan.md)：长期目标、四轮交付、证据标准和编辑约定。
- [打开可交互时间线](atlas.html)：本地浏览器选择路线、点击节点查看问题与来源；无网络依赖。
- [20 条路线完整底稿](round-one.md)：核心问题、候选转折、已有材料、检索词与缺口。
- [文献队列](reading-queue.md)：逐项来源身份、阅读范围、版本和下一步。
- [可编辑数据](atlas.json) · [Mermaid 源图](timeline.mmd) · [SVG 总图](timeline.svg) · [来源访问记录](source-checks.json)。

![候选全景](timeline.svg)

## 本轮范围与状态

- 20 条路线、106 个独立事件、106 份来源、9 个编辑性跨路线比较问题。
- 103 个现代节点，另列 3 个古典文本入口；跨路线重复展示不会重复计算事件。
- 仅定位 89 条；书目核对 5 条；摘要/内容提要 10 条；指定片段 2 条；正文深读 0 条。
- 采用探索性检索，覆盖截至本轮已发现的候选，不声称系统综述、历史穷尽或截至 2026 年的最新文献完备。
- 全部事件的主张状态都是 candidate；来源网页的存在、摘要阅读和结论成立分别记录。

## 路线盘点

| 路线 | 候选节点（可交叉归属） | 本项目相关材料入口 | 下一轮 |
| --- | ---: | ---: | --- |
| [01 行为表现与操作性路线](round-one.md#behavioral) | 5 | 2 | 按问题补齐 |
| [02 心理测量与能力结构路线](round-one.md#psychometric) | 7 | 1 | 按问题补齐 |
| [03 学习效率与泛化路线](round-one.md#learning-efficiency) | 5 | 3 | 样板深读 |
| [04 理性、适应与资源约束路线](round-one.md#rational-resource) | 6 | 2 | 按问题补齐 |
| [05 符号、逻辑、搜索与组合路线](round-one.md#symbolic-logic) | 7 | 2 | 按问题补齐 |
| [06 连接主义、统计学习与表征路线](round-one.md#connectionist) | 9 | 3 | 按问题补齐 |
| [07 概率推断、因果模型与世界模型路线](round-one.md#causal-world-models) | 7 | 3 | 按问题补齐 |
| [08 控制论、强化学习与主动推断路线](round-one.md#cybernetics-rl) | 8 | 3 | 样板深读 |
| [09 认知架构与多机制协作路线](round-one.md#cognitive-architectures) | 5 | 1 | 按问题补齐 |
| [10 神经科学路线](round-one.md#neuroscience) | 6 | 2 | 按问题补齐 |
| [11 发展、建构与开放式学习路线](round-one.md#developmental-openended) | 6 | 2 | 按问题补齐 |
| [12 具身、生态、生成与动力系统路线](round-one.md#embodied-dynamical) | 6 | 3 | 样板深读 |
| [13 生命、自维持与演化路线](round-one.md#autopoiesis-evolution) | 5 | 0 | 按问题补齐 |
| [14 社会文化与语言路线](round-one.md#sociocultural-language) | 5 | 0 | 按问题补齐 |
| [15 集体、分布式与延展心智路线](round-one.md#collective-extended) | 5 | 2 | 按问题补齐 |
| [16 信息、压缩、复杂系统与物理路线](round-one.md#information-physics) | 6 | 1 | 按问题补齐 |
| [17 语义、理解与心灵哲学路线](round-one.md#philosophy-of-mind) | 5 | 1 | 按问题补齐 |
| [18 意识、情感与自我路线](round-one.md#consciousness-self) | 6 | 0 | 按问题补齐 |
| [19 智慧、价值与跨文化路线](round-one.md#wisdom-values) | 5 | 1 | 按问题补齐 |
| [20 形而上学、非经典计算与量子路线](round-one.md#metaphysics-quantum) | 5 | 0 | 按问题补齐 |

盘点列为人工选择的相关材料入口，不是已读论文数或文章总量。大模型七卷、果蝇连接组、Strandbeest、强化学习和因果推断已有积累；社会文化、意识、生命自主性及非经典路线缺少专门解读。现有路线匹配器通常只返回首个命中，本轮未据它自动判定全部归属。 本轮人工关联 26 个不同的项目材料入口。当前文献架有 8 项：minimind、nanochat、mcts-survey、alphazero、causal-remix、ylearn、transformer、react；其中 alphazero 与 transformer 分别对应本底稿的 silver-2017 和 vaswani-2017。其他条目保留为项目理解入口，尚未完成逐一的书目映射。

## 图的编辑与重生成

编辑 atlas.json 中的来源、事件、路线或比较问题，再从项目根目录执行：

```sh
node scripts/generate-intelligence-history.mjs
node scripts/generate-intelligence-history.mjs --check
```

生成器检查 ID、路线映射、来源引用、日期范围和本地材料路径，并重建本目录的底稿、文献队列、SVG、Mermaid 和离线阅读图。--check 不写文件，同时检查生成文件是否与数据一致。修改数据后重生成；手工编辑生成图会在下次生成时被覆盖。

SVG 横轴为等比例现代出版年，区间日期取中点绘制并保留完整标签；古典文本独立列出。圆点编号对应路线底稿中的节点顺序，同年节点会错开纵向位置。横线只作坐标参考，未绘制历史继承关系。Mermaid 版本是按路线组织的顺序图，间距不表示时间长短。

## 进入第二轮的具体任务

1. 学习效率与泛化：补充迁移/组合泛化的反例，将先验、经验、计算预算与评测协议分开，扩展为 8–12 个可审校节点。
2. 控制论与强化学习：补 Bellman、策略梯度及主动推断原始算法；分别重建分支，再考证模型、奖励与自维持的联系。
3. 具身与生态：补 Gibson 原始研究和被动行走；区分机器人演示、动力系统模型、哲学论证和形态计算度量。
4. 每个样板节点补原文定位、论证/推导、反方材料和证据边界；已有工程文章用作理解入口，不替代历史原文。

## 已发现的资料风险

- Binet–Simon 的卷年/刊年、信息瓶颈的会议/预印本年尚待核；图中以范围保留。
- Piaget、Vygotsky、Wiener、Marr 等的原作、译本或再版日期需要分开。
- 中观译本出版页的检索摘要疑似串书，未将摘要作为理论依据。
- 若干出版入口返回拒绝访问、验证码、重定向错误或空白壳，保留为仅定位；访问问题见来源记录。
- 尚未识别的历史 dot 文献批次仍独立待查，本轮集合不冒充该批文献。

## 验收记录

见 [validation.md](validation.md)。
