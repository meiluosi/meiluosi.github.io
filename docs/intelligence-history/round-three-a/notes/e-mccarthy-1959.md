<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1959 · 程序怎样通过接受陈述改变自己的行为？

[符号、逻辑、搜索与组合路线](../routes/symbolic-logic.md) · [文献夹](../literature.md)

**原始问题：** 怎样把知识和启发式表达为系统可以推理的对象？

**方法与机制：** Advice Taker拟用逻辑陈述表达知识、程序和启发式；选择相关前提后进行即时推导，并执行导出的行动命令。

**成立条件：** 这是系统提案，原文尚未完成启发式形式化；逻辑规则可靠不等于知识前提真实。实读后期重排本，回顾脚注与原论证分开。

## 主张与原文定位

- **原文主张**：预期优势是通过补充陈述改善行为，且将尽可能多的启发式放入可表述的语言。
  - 依据：[Programs with Common Sense](https://www.cs.rit.edu/~rlaz/files/mccarthy1959.pdf)；PDF2、4。独立验证：未做独立复现。

- **原文主张**：作者将关键智能作用放在选择前提的程序，并明确不能对全部知识同时进行即时推导。
  - 依据：[Programs with Common Sense](https://www.cs.rit.edu/~rlaz/files/mccarthy1959.pdf)；PDF5，§2第3点。独立验证：未做独立复现。

- **原文主张**：随文讨论追问相关性与现实约束；McCarthy回应称例子有意简化，输出是行动命令，不能靠推导直接到达机场。
  - 依据：[Programs with Common Sense](https://www.cs.rit.edu/~rlaz/files/mccarthy1959.pdf)；PDF11–12及14–15，Bar-Hillel批评和作者回应。独立验证：未做独立复现。

## 阅读记录

### Programs with Common Sense

John McCarthy，1959。15页后期重排本；题首页标1959，PDF2含1996回顾脚注，不能当作1959原始版面；1958会议报告另记

- 实读：PDF1–15正文及随文讨论均已读；PDF8式17和行动推导已视觉核对
- 未覆盖：未与1959会议录扫描本逐字校对；示例全部推导和相关性选择程序未实现或形式验证
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Programs with Common Sense](https://www.cs.rit.edu/~rlaz/files/mccarthy1959.pdf)
- SHA-256：`04b439b126f87e51fc4c85b8db114fddeeafc7f4b2234e1b8f3b474d6ea9b79c`

## 边界与下一问

机场例用于说明推理结构，不是现实导航的完整实现；Bar-Hillel直接质疑相关前提选择、时间因素与模型复杂度。

当记忆规模增加或现实条件变化时，如何选择相关事实并修订假设？
