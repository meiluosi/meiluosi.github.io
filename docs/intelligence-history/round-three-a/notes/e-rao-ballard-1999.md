<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1999 · 层间反馈能解释哪些视觉反应？

[概率推断、因果模型与世界模型路线](../routes/causal-world-models.md) · [连接主义、统计学习与表征路线](../routes/connectionist.md) · [文献夹](../literature.md)

**原始问题：** 局部视觉响应为何受更大范围的情境与高层反馈影响？

**方法与机制：** 高层向低层传递预测，低层向上传递残差；在生成模型、噪声与先验下优化隐状态，并以较慢时间尺度学习连接权重。

**成立条件：** 模型拟合静态自然图像；最大后验目标包含方差加权误差及状态／权重先验，不仅是无条件误差最小化。

## 主张与原文定位

- **原文主张**：模型的前馈信号是预测残差，反馈携带预测；神经活动与突触学习分别由生成模型目标的梯度更新。
  - 依据：[Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-classical Receptive-field Effects](https://homes.cs.washington.edu/~rao/Rao-Ballard-NN-1999.pdf)；80页图1；85–86页Methods，式1–7。独立验证：未做独立复现。

- **原文主张**：图5模拟的32个误差单元中，28个在有反馈时呈现超过50%的端点抑制，取消反馈后剩5个；这些是作者模型结果。
  - 依据：[Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-classical Receptive-field Effects](https://homes.cs.washington.edu/~rao/Rao-Ballard-NN-1999.pdf)；83–84页，图5及相邻正文。独立验证：未做独立复现。

- **原文主张**：作者承认侧向作用也可能解释部分现象，并说明本文结果均来自静态图像，时序预测是后续扩展方向。
  - 依据：[Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-classical Receptive-field Effects](https://homes.cs.washington.edu/~rao/Rao-Ballard-NN-1999.pdf)；84页Discussion；86页Methods末。独立验证：未做独立复现。

## 阅读记录

### Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-classical Receptive-field Effects

Rajesh P. N. Rao；Dana H. Ballard，1999。Nature Neuroscience 2:79–87，1999年1月；1998为收稿／接受年

- 实读：全文79–87页；80页图1、84页图5及86页式3–9已视觉核对。
- 未覆盖：未复现模型，未独立重读所引神经生理实验；模型与神经回路的对应仍按假说记录。
- 状态：full-text；核对日期：2026-10-26
- 来源：[Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-classical Receptive-field Effects](https://homes.cs.washington.edu/~rao/Rao-Ballard-NN-1999.pdf)；DOI：10.1038/4580
- SHA-256：`5805862c6ba254f3e0f86dbab4bb4ba74490a5bcce3fdaccd91396d567c0bed2`

## 边界与下一问

神经对应是机制假说；侧向相互作用是作者承认的替代解释。正文时序扩展未用于所报告模拟，不能把静态重建直接说成未来预测。

操纵高层反馈与局部侧向联系，能否产生可区分的反应预测？
