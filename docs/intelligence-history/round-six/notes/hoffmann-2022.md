# Training Compute-Optimal Large Language Models

Jordan Hoffmann、Sebastian Borgeaud、Arthur Mensch、Elena Buchatskaya、Trevor Cai、Eliza Rutherford、Diego de Las Casas、Lisa Anne Hendricks、Johannes Welbl、Aidan Clark、Tom Hennigan、Eric Noland、Katie Millican、George van den Driessche、Bogdan Damoc、Aurelia Guy、Simon Osindero、Karen Simonyan、Erich Elsen、Jack W. Rae、Oriol Vinyals、Laurent Sifre · 2022 · [原始载体](https://arxiv.org/pdf/2203.15556v1)

## 版本与范围

arXiv:2203.15556v1，2022-03-29；NeurIPS正式版本书目单独记录。

- 已读：PDF 1–2引言及式1；PDF 7–8 §3.4与表2；PDF 27附录D.3对照。
- 未读／未重验：未重训模型或复算拟合；训练损失及近无限数据条件不涵盖所有推理成本或数据约束。
- 指纹：网页阅读，未下载并计算文件指纹。

## 问题与方案

对固定训练FLOPs，应怎样配置模型大小和训练token数量？

用三类拟合与多个训练时长估计最优前沿，并将Chinchilla与Gopher等比较。

前提：估计目标是给定训练计算下的语言模型损失，依赖数据与训练配置；不等于全生命周期成本最优。

限制：三种拟合给出相近但不完全相同的指数；未在本项目重训或重拟合。

下一问：不同参数定义、模型规模范围及推理成本会怎样影响结论？

## 指定原文主张

<a id="e-hoffmann-2022-r6c1"></a>

### e-hoffmann-2022-r6c1

作者训练400余个模型，用三种方法估算计算最优模型与token规模；表2的三个参数指数分别为0.50、0.49、0.46。

定位：PDF 2 §1；PDF 8，表2。这是原文报告，未独立复算。

<a id="e-hoffmann-2022-r6c2"></a>

### e-hoffmann-2022-r6c2

作者报告Chinchilla采用70B参数与约1.4万亿token，并在所测条件下比较Gopher。

定位：PDF 2，引言末段；PDF 27，附录D.3。这是原文报告，未独立复算。
