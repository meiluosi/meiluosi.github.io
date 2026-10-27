<!-- 由 research.json 生成；修改数据后重生成。C批已验收；不代表完整路线史。 -->

# 2019 · 环境和解法能否共同扩展？

[发展与开放式学习路线](../routes/developmental-openended.md) · [文献夹](../literature.md)

**原始问题：** 研究者是否必须事先给定全部任务序列？

**方法与机制：** 维护环境—代理对，生成新环境、优化各自解法，并进行跨环境迁移。

**成立条件：** 本版采用二维步行、受限环境参数和固定身体；任务生成器仍有人工先验。

## 主张与原文定位

- **原文主张**：POET并行维护环境和代理，并用跨任务迁移连接不同的探索路径。
  - 依据：[Paired Open-Ended Trailblazer (POET)](https://arxiv.org/html/1901.01753v2)；§3开篇及§4.3迁移讨论。独立验证：未做独立复现。

- **原文主张**：§5承认障碍参数有最大范围、身体固定；更无界的编码属于后续工作。
  - 依据：[Paired Open-Ended Trailblazer (POET)](https://arxiv.org/html/1901.01753v2)；§5第4–5段。独立验证：未做独立复现。

## 证据标准

| 项目 | 条件与边界 |
| --- | --- |
| 证据性质 | 算法演示（方法与边界选读） |
| 分析单位 | 环境—代理种群 |
| 观察／论证依据 | 多路径课程与迁移比较 |
| 需要区分的解释 | 任务数量、新颖性、复杂度及跨任务能力增长。 |

## 阅读记录

### Paired Open-Ended Trailblazer (POET)

Rui Wang；Joel Lehman；Jeff Clune；Kenneth O. Stanley，2019。arXiv:1901.01753v2，2019-01-09；官方HTML。

- 实读：§3开篇及优化／迁移框架；§4.3单路径课程比较片段；§5环境范围和固定身体限制。
- 未覆盖：附录算法细节、全部结果和代码未运行；后续POET版本未覆盖。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[Paired Open-Ended Trailblazer (POET)](https://arxiv.org/html/1901.01753v2)
- SHA-256：`2abba5a131ffff3b7ca6a71cd7eb016f97f144002ccb7f4d8a5b42b23eec637a`

## 边界与下一问

“开放式”在此是研究方向与有限域演示，不能作为已实现无界RSI的证明。

扩展编码后，能力增长如何与任务漂移、难度度量和资源增加区分？
