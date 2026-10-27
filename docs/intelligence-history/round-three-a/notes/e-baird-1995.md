<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1995 · 共享参数怎样破坏表格方法的收敛？

[控制论、强化学习与主动推断路线](../routes/cybernetics-rl.md) · [学习效率与泛化路线](../routes/learning-efficiency.md) · [文献夹](../literature.md)

**原始问题：** 值函数能表示正确答案，为什么迭代仍会发散？

**方法与机制：** 直接法把后继值当作固定目标，残差梯度还对后继值求导；共享权重使更新一个状态同时改变别的状态。

**成立条件：** 采用六状态、零奖励、等频更新、原始特征；主实验参数是本次选择，遵循式3和§5正文求和规则，式7存在符号冲突。

## 主张与原文定位

- **原文主张**：原图1有六个状态和七个权重；末页勘误说明图中系数2曾漏排。
  - 依据：[Residual Algorithms: Reinforcement Learning with Function Approximation](http://leemon.com/papers/1995b.pdf)；PDF2图1及PDF9勘误。独立验证：未做独立复现。

- **本次推导或验证**：式3与式7的直接法梯度符号相反；将式7判为疑似排印错误是本次审校判断，未获作者确认。
  - 依据：[Residual Algorithms: Reinforcement Learning with Function Approximation](http://leemon.com/papers/1995b.pdf)；PDF2式3、PDF4式7；replication/protocol.md 修订A。独立验证：未做独立复现。

- **本次推导或验证**：独立实现中γ=0.99、α=0.01时直接法振荡发散，残差梯度渐近收敛但很慢；γ=0.9的受检步长下直接法稳定。
  - 依据：[Residual Algorithms: Reinforcement Learning with Function Approximation](http://leemon.com/papers/1995b.pdf)；replication/README.md、results.json、traces.csv。独立验证：已完成所标范围，见复现报告。

- **原文主张**：1999年论文图4.2与4.3分别给出七状态螺旋发散和十一状态单调增长，不能与1995六状态例互换。
  - 依据：[Reinforcement Learning Through Gradient Descent](https://www.ri.cmu.edu/pub_files/pub2/baird_leemon_1999_1/baird_leemon_1999_1.pdf)；PDF37–38／印刷35–36页，图4.2、4.3。独立验证：未做独立复现。

## 阅读记录

### Residual Algorithms: Reinforcement Learning with Function Approximation

Leemon Baird，1995。9页作者版，1995-07-15勘误；六状态图1

- 实读：正文§2–5和§9、末页勘误；图1、式3/6/7/8与末页图已视觉核对
- 未覆盖：未完整复核一般残差算法的证明、随机转移推导和全部图5实验
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Residual Algorithms: Reinforcement Learning with Function Approximation](http://leemon.com/papers/1995b.pdf)
- SHA-256：`a32e916f75e4ca626c470d24fe0a3e587c14bd10edfae686fc9edf7498831979`

### Reinforcement Learning Through Gradient Descent

Leemon Baird，1999。博士论文；所读印刷35–37页

- 实读：§4.2，PDF37–39；图4.2与4.3及图说已视觉核对
- 未覆盖：其余章节未读；七状态及十一状态例未数值复现
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Reinforcement Learning Through Gradient Descent](https://www.ri.cmu.edu/pub_files/pub2/baird_leemon_1999_1/baird_leemon_1999_1.pdf)
- SHA-256：`f9651058cf69b741ecc359298e4cf0e6708cfb8e4d6c733eb804f510e3654536`

## 边界与下一问

结果仅覆盖选定线性预测问题；不复现图5完整速度实验，不推广到全部采样、折扣、步长或非线性模型。

怎样把投影、采样分布和目标的变化分别隔离，说明稳定性的来源？

[正式复现报告、结果与图](../replication/README.md)
