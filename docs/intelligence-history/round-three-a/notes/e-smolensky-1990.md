<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1990 · 分布式向量怎样保存谁处于哪个角色？

[连接主义、统计学习与表征路线](../routes/connectionist.md) · [符号、逻辑、搜索与组合路线](../routes/symbolic-logic.md) · [文献夹](../literature.md)

**原始问题：** 把多个对象的特征叠加以后，怎样保留对象与角色的绑定并取回它们？

**方法与机制：** 先把结构分解为填充值与角色，以外积表示每个绑定，再求和；通过作用于角色侧的向量取回填充值。

**成立条件：** 精确解绑定理要求单值角色及当前绑定角色向量线性独立；直接用角色向量查询与对偶向量查询的条件不同。

## 主张与原文定位

- **原文主张**：结构可表示为填充值向量与角色向量外积之和；角色和填充值均可采用分布式表示。
  - 依据：[Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/Smolensky-90-AI-Tensor-Product-Variable-Binding-and-the-Representation-of-Symbolic-Structures-in-Connectionist-Systems.pdf)；§2.2.3–2.2.4，173–176页。独立验证：未做独立复现。

- **原文主张**：单值角色的向量线性独立时，对偶向量支持精确解绑；自寻址查询对非正交角色一般会混入其他填充值。
  - 依据：[Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/Smolensky-90-AI-Tensor-Product-Variable-Binding-and-the-Representation-of-Symbolic-Structures-in-Connectionist-Systems.pdf)；§3.1，186–187页，定理3.1与3.3。独立验证：未做独立复现。

- **原文主张**：作者明确说本研究始于对Fodor与Pylyshyn的回应，同时在结论中承认绑定单元增长和结构处理仍有困难。
  - 依据：[Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/Smolensky-90-AI-Tensor-Product-Variable-Binding-and-the-Representation-of-Symbolic-Structures-in-Connectionist-Systems.pdf)；§1.2，165页；§4，212–213页。独立验证：未做独立复现。

## 阅读记录

### Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems

Paul Smolensky，1990。Artificial Intelligence 46(1–2):159–216；58页期刊版，作者任职机构公开文件

- 实读：159–176页§1、§2.1–2.2；185–190页§2.4、§3.1–3.2；211页末§3.9引入至213页结论；214–216页参考文献。186–187页精确／自寻址解绑定理及证明已视觉核对。
- 未覆盖：177–184页特殊表示、191–211页大部分实现／递归操作／记忆推导尚未完整读；未实现TPPS或验证学习结果。读取211页结尾不等于审查此前存储定理证明。
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/Smolensky-90-AI-Tensor-Product-Variable-Binding-and-the-Representation-of-Symbolic-Structures-in-Connectionist-Systems.pdf)；DOI：10.1016/0004-3702(90)90007-M；[书目核对](https://www.sciencedirect.com/science/article/pii/000437029090007M)
- SHA-256：`e6460dcb065a8d8781a9be6bc66c2b207dc7908881b97dec759b2b68cb6cadcb`

## 边界与下一问

该表示不能独自保证学习成功或一般系统性；作者保留维度增长、递归处理与记忆存储限制。数学存在的操作不等于网络已经学会执行。

怎样控制角色编码、容量和训练经验，区分精确结构操作与近似串扰？
