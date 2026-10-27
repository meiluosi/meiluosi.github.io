<!-- 由 research.json 生成；修改数据后重生成。B批已验收；不代表完整路线史。 -->

# 1996 · 快速节俭启发式的生态检验

[理性、适应与资源约束路线](../routes/rational-resource.md) · [文献夹](../literature.md)

**原始问题：** 少用信息的决策规则在什么环境下能保持准确？

**方法与机制：** Take The Best按线索有效性顺序搜索，遇到首个可区分线索即停止；结合识别规则。

**成立条件：** 线索顺序与识别的有效性来自指定环境；不能省略这些知识的来源。

## 主张与原文定位

- **原文主张**：规则明确区分搜索、停止和选择；不把所有线索加权相加。
  - 依据：[Reasoning the Fast and Frugal Way: Models of Bounded Rationality](https://web.mit.edu/curhan/www/docs/Articles/biases/Gigerenzer_Goldstein_Reasoning%20Fast%20and%20Frugal.pdf)；653–654，算法步骤。独立验证：未做独立复现。

- **原文主张**：作者在83座德国城市、9个生态线索的模拟中比较准确率与检索量；本研究不新增人的过程验证。
  - 依据：[Reasoning the Fast and Frugal Way: Models of Bounded Rationality](https://web.mit.edu/curhan/www/docs/Articles/biases/Gigerenzer_Goldstein_Reasoning%20Fast%20and%20Frugal.pdf)；654、656–658。独立验证：未做独立复现。

- **原文主张**：回归权重使用该环境全部目标值，并非通常训练／测试拆分；作者承认以城市排名回归可略优。
  - 依据：[Reasoning the Fast and Frugal Way: Models of Bounded Rationality](https://web.mit.edu/curhan/www/docs/Articles/biases/Gigerenzer_Goldstein_Reasoning%20Fast%20and%20Frugal.pdf)；658及脚注4。独立验证：未做独立复现。

## 评价视角

| 项目 | 条件与边界 |
| --- | --- |
| 评价对象／构念 | 特定环境中的推断效率 |
| 任务与环境 | 城市人口二选一；不同知识缺失状态 |
| 样本／适用主体 | 模拟个体；识别有效性参考26名学生先导 |
| 输出指标 | 正确比例、检索线索数 |
| 信度与不确定性 | 模拟重复不替代新环境外推；原数据未复算。 |
| 效度与解释范围 | 环境、缺失编码、竞争方法目标与信息权限影响比较 |
| 先验与经验 | 线索顺序与识别有效性是关键前提 |
| 计算与时间预算 | 检索成本；排序获取成本未完整计入 |

## 阅读记录

### Reasoning the Fast and Frugal Way: Models of Bounded Rationality

Gerd Gigerenzer；Daniel G. Goldstein，1996。Psychological Review 103(4):650–669原刊PDF。

- 实读：650–654、656–659、665；重点为算法、模拟协议、658脚注4及限制。
- 未覆盖：未复算附录数据或逐图校勘曲线；尚未阅读Chater等1997批评正文。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[Reasoning the Fast and Frugal Way: Models of Bounded Rationality](https://web.mit.edu/curhan/www/docs/Articles/biases/Gigerenzer_Goldstein_Reasoning%20Fast%20and%20Frugal.pdf)；DOI：10.1037/0033-295X.103.4.650
- SHA-256：`07d75b55058d015f884e3b1eef32c778709b561666aa00e8b4dd6a1df2a5cebc`

## 边界与下一问

查找线索数不是端到端时间；不能将此比赛写成所有任务、所有竞争算法上的优势。

保留相同学习信息并改变线索结构，优势是否保持？
