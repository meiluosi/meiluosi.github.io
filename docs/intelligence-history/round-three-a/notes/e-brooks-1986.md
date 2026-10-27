<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1985备忘录／1986期刊 · 分层控制从模拟推进到实机

[具身、生态、生成与动力系统路线](../routes/embodied-dynamical.md) · [文献夹](../literature.md)

**原始问题：** 怎样增加机器人能力，同时保留已能运行的底层行为？

**方法与机制：** 按行为能力构建异步模块；高层通过抑制或替换信息流影响低层，低层继续运行。

**成立条件：** 依赖预设连接与人工设计模块；新增层需调试真实的交互接口。

## 主张与原文定位

- **原文主张**：1985备忘录结论明确列出尚缺真实机器人演示和完整第二层实现。
  - 依据：[A Robust Layered Control System for a Mobile Robot](https://jmvidal.cse.sc.edu/library/brooks86a.pdf)；§7，印刷19–20页／PDF20–21。独立验证：未做独立复现。

- **原文主张**：1986期刊版增加实验室与机房中的实机报告，并注明视觉模块暂由声呐走廊检测替代。
  - 依据：[A Robust Layered Control System for a Mobile Robot](https://www.scribd.com/document/544347085/A-Robust-Layered-Control-System-for-a-Mobile-Robot)；§IV.C，20–21页；§V.B，21–22页。独立验证：未做独立复现。

- **原文主张**：期刊报告的避障经验带有限定：移动障碍刻意缓慢；快速物体或拥挤环境仍可能造成碰撞。
  - 依据：[A Robust Layered Control System for a Mobile Robot](https://www.scribd.com/document/544347085/A-Robust-Layered-Control-System-for-a-Mobile-Robot)；§IV.A，19页。独立验证：未做独立复现。

## 阅读记录

### A Robust Layered Control System for a Mobile Robot

Rodney A. Brooks，1985。A Robust Layered Control System for a Mobile Robot；Rodney Brooks；实读MIT AI Memo864,1985-09；对应1986期刊节点，已完成所列章节的1985／1986版本对照，未做全篇逐页校勘

- 实读：第二轮封面、§2.2、图3及§7；本批补读扫描本PDF20–22（印刷19–21页），结论与实现限制，附录仅核对范围。
- 未覆盖：所列范围之外的全部正文、附录图表、代码与实验未逐项核验；版本对照不是全篇复现。
- 状态：selected-sections；核对日期：2026-10-26
- 来源：[A Robust Layered Control System for a Mobile Robot](https://jmvidal.cse.sc.edu/library/brooks86a.pdf)
- SHA-256：`39ecc5feed013f9101fcdde9ceff9b337f4acabcfd3c2aadcb26dd892b15871f`
- 阅读记录延续自第二轮；本批复核范围另列。

### A Robust Layered Control System for a Mobile Robot

Rodney A. Brooks，1986。IEEE Journal of Robotics and Automation 2(1), 14–23, March 1986；实读公开页面所载期刊正文转录，非页面AI概述。首页修订日期1986-02-03。

- 实读：期刊14–15页及16页开头正文；19页§IV.A与20–23页§IV.C、§V、§VI。按段落核对模拟、实机、视觉替代与结论。
- 未覆盖：未取得可校验的本地期刊PDF；17–18页未完整阅读；图像、图中连线及OCR破损数字不据此复核；页面AI摘要与问答不作依据。
- 状态：selected-sections；核对日期：2026-10-26
- 来源：[A Robust Layered Control System for a Mobile Robot](https://www.scribd.com/document/544347085/A-Robust-Layered-Control-System-for-a-Mobile-Robot)；DOI：10.1109/JRA.1986.1087032
- SHA-256：无：未取得可校验的本地PDF；实际阅读载体与范围见本条来源记录。

## 边界与下一问

期刊版实机报告不代表完整视觉探索系统或一般碰撞安全保证；高层视觉模块仍未按计划实现，快速障碍与拥挤环境仍可能失败。

怎样分别测试低层避障、目标趋近及视觉探索，在不同扰动下识别失效边界？


[版本对照与未覆盖范围](../versions/brooks-1985-1986.md)
