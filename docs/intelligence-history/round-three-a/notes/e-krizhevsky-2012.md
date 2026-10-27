<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2012 · 深层视觉模型怎样在大数据上训练并控制过拟合？

[连接主义、统计学习与表征路线](../routes/connectionist.md) · [文献夹](../literature.md)

**原始问题：** 如何让大容量卷积网络在有限计算与大量标注图像下取得有效识别结果？

**方法与机制：** 组合卷积先验、ReLU、双GPU训练、数据增广、dropout及带动量的随机梯度下降。

**成立条件：** 评价依赖指定ImageNet版本、数据划分、测试裁剪和模型组合；不同组件并非都通过完全隔离的因果实验评估。

## 主张与原文定位

- **原文主张**：所读系统使用五层卷积与三层全连接，并通过数据增广和dropout抑制过拟合；训练使用带动量和权重衰减的随机梯度下降。
  - 依据：[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)；§3.5–5，4–7页；图2。独立验证：未做独立复现。

- **原文主张**：PDF表1的ILSVRC-2010错误率为top-1 37.5%、top-5 17.0%；该结果包含十裁剪预测平均。
  - 依据：[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)；7页表1及脚注5；5页§4.1。独立验证：未做独立复现。

- **原文主张**：ILSVRC-2012单CNN的18.2%是验证集top-5错误率；15.3%测试错误率属于七CNN组合，其中两个使用额外数据预训练。
  - 依据：[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)；7页§6、表2。独立验证：未做独立复现。

## 阅读记录

### ImageNet Classification with Deep Convolutional Neural Networks

Alex Krizhevsky；Ilya Sutskever；Geoffrey E. Hinton，2012。NIPS 2012官方9页PDF；同一入口网页摘要的训练数、神经元数和2010错误率与PDF不同，本记录采用固定指纹PDF及其表1–2

- 实读：PDF1–8全部正文与第9页参考文献；图2和表1–2已视觉核对。另打开官方摘要页用于检查版本差异。
- 未覆盖：补充材料、原代码、完整逐层尺寸一致性和训练实验未复核；图1、3、4未逐项视觉核验；网页摘要与PDF差异的编辑历史未确定。
- 状态：full-text；核对日期：2026-10-25
- 来源：[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)；[书目核对](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- SHA-256：`90137160c57217953d5f61857e64ca58e85f06e1b13b4f475c918b1b582b9771`

## 边界与下一问

15.3%来自包含额外预训练的七网络组合，不能写成单网络成绩；分类结果不自动支持通用理解。官网摘要与链接PDF有差异，本文固定采用PDF。

在外观、背景或位置变化后，高分类成绩所依赖的特征是否仍有效？
