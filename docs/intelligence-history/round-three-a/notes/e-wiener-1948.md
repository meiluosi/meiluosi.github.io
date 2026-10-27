<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1948 · 反馈为什么既能调节，也会引起振荡？

[控制论、强化学习与主动推断路线](../routes/cybernetics-rl.md) · [文献夹](../literature.md)

**原始问题：** 发出正确指令，是否足以保证执行器达到目标？

**方法与机制：** 把实际执行结果送回调节环节，与目标及其他感觉信息结合；以铁路信号、恒温器和调速器说明反馈。

**成立条件：** 节点按原作1948年定位；实读1961第二版的原版部分。负反馈效果取决于系统设计，后续线性分析有其模型前提。

## 主张与原文定位

- **原文主张**：铁路信号例要求回传执行器实际状态，不能将命令本身当作已经完成的动作。
  - 依据：[Cybernetics: Or Control and Communication in the Animal and the Machine](https://uberty.org/wp-content/uploads/2015/07/Norbert_Wiener_Cybernetics.pdf)；第IV章，印刷96页。独立验证：未做独立复现。

- **原文主张**：作者明确指出设计不当的恒温器可能引起剧烈振荡；负反馈不是无条件的稳定保证。
  - 依据：[Cybernetics: Or Control and Communication in the Animal and the Machine](https://uberty.org/wp-content/uploads/2015/07/Norbert_Wiener_Cybernetics.pdf)；第IV章，印刷97–98页。独立验证：未做独立复现。

## 阅读记录

### Cybernetics: Or Control and Communication in the Animal and the Machine

Norbert Wiener，1961。1961第二版，1985第四次印刷；Part I标为1948原版部分；不能据此声称逐字核过第一版

- 实读：版权页PDF3、第二版序言PDF4–14；Part I标题PDF17；第IV章PDF114–118／印刷95–99页，95页已视觉核对
- 未覆盖：反馈章后续数学分析未完整读；动物与临床例未回查实验原文；1948第一版未逐字对照
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Cybernetics: Or Control and Communication in the Animal and the Machine](https://uberty.org/wp-content/uploads/2015/07/Norbert_Wiener_Cybernetics.pdf)
- SHA-256：`f733dde90a70fe090b5209ac7ca391e86f3ed4477b0a2ebd7054747d8ed0d857`

## 边界与下一问

临床类比是原作者陈述，未由本批验证；反馈机制本身不构成通用智能充分条件。

怎样区分反馈结构、反馈增益与时延对稳定性的贡献？
