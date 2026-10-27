<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2017 · 不同架构能共享哪些可检验约束？

[认知架构与多机制协作路线](../routes/cognitive-architectures.md) · [文献夹](../literature.md)

**原始问题：** 怎样从多个认知架构提取共同假设，支持累积研究？

**方法与机制：** 以ACT-R、Soar与Sigma比较模块、记忆、学习和感知动作；工作记忆连接组件，程序记忆选择行动。

**成立条件：** 目标是类人心智；提案基于有限架构的综合，标准模型本身不是已经实现的单一架构。

## 主张与原文定位

- **原文主张**：模型把模块内及模块间并行，与认知周期中的单次有意行动选择结合。
  - 依据：[A Standard Model of the Mind: Toward a Common Computational Framework across Artificial Intelligence, Cognitive Science, Neuroscience, and Robotics](https://www.researchgate.net/publication/322123676_A_Standard_Model_of_the_Mind_Toward_a_Common_Computational_Framework_across_Artificial_Intelligence_Cognitive_Science_Neuroscience_and_Robotics)；作者稿§4.1；§5表1 A。独立验证：未做独立复现。

- **原文主张**：长期记忆包含符号结构与量化元数据；程序学习至少区分规则组合和调节选择的强化学习。
  - 依据：[A Standard Model of the Mind: Toward a Common Computational Framework across Artificial Intelligence, Cognitive Science, Neuroscience, and Robotics](https://www.researchgate.net/publication/322123676_A_Standard_Model_of_the_Mind_Toward_a_Common_Computational_Framework_across_Artificial_Intelligence_Cognitive_Science_Neuroscience_and_Robotics)；作者稿§4.2–4.3；表1 B–C。独立验证：未做独立复现。

- **原文主张**：作者明确列出元认知、情感、心象和跨模块通信等未解决内容，并计划引入更多架构。
  - 依据：[A Standard Model of the Mind: Toward a Common Computational Framework across Artificial Intelligence, Cognitive Science, Neuroscience, and Robotics](https://www.researchgate.net/publication/322123676_A_Standard_Model_of_the_Mind_Toward_a_Common_Computational_Framework_across_Artificial_Intelligence_Cognitive_Science_Neuroscience_and_Robotics)；作者稿§5，表2说明之后。独立验证：未做独立复现。

## 阅读记录

### A Standard Model of the Mind: Toward a Common Computational Framework across Artificial Intelligence, Cognitive Science, Neuroscience, and Robotics

John E. Laird；Christian Lebiere；Paul S. Rosenbloom，2017。期刊AI Magazine 38(4):13–26，2017-12-28；实读为John E. Laird于2018-02-09上传作者稿的网页全文转录，作者稿页码不同于期刊排版

- 实读：作者稿标题、摘要；§3三种架构、§4.1–4.4、§5 Summary及表1文字；重点核对学习机制、组件通信和未解决问题。原网页全文定位，不是检索摘要
- 未覆盖：§1–2未完整读；图1–4和表2颜色编码未视觉核验，未据其逐格下结论；没有核对作者稿与期刊版逐字差异，也未复现实验
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[A Standard Model of the Mind: Toward a Common Computational Framework across Artificial Intelligence, Cognitive Science, Neuroscience, and Robotics](https://www.researchgate.net/publication/322123676_A_Standard_Model_of_the_Mind_Toward_a_Common_Computational_Framework_across_Artificial_Intelligence_Cognitive_Science_Neuroscience_and_Robotics)；DOI：10.1609/aimag.v38i4.2744；[书目核对](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2744/0)
- SHA-256：无：未取得可校验的本地PDF；实际阅读载体与范围见本条来源记录。

## 边界与下一问

作者列出元认知、情感、心象、跨模块直接通信及学习等缺口；理论一致、实现具备和经验验证需要分别判断。

固定模块通信与时间假设后，哪些跨任务结果会使这些约束失败？
