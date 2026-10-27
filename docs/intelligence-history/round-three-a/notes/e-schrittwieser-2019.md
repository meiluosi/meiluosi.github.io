<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2019 v1／2020期刊 · MuZero的版本、训练信息与泛化边界

[控制论、强化学习与主动推断路线](../routes/cybernetics-rl.md) · [概率推断、因果模型与世界模型路线](../routes/causal-world-models.md) · [文献夹](../literature.md)

**原始问题：** 规划是否必须重建每个像素或真实状态？

**方法与机制：** 表示、潜在动力学和预测网络联合学习奖励、价值及策略；在潜在状态上树搜索，搜索结果反过来提供训练目标。

**成立条件：** 所比较版本的动力学均为确定性；根节点可用动作是环境提供的信息，训练仍依赖交互和搜索。

## 主张与原文定位

- **原文主张**：2019 v1已经报告Reanalyze及其重搜旧轨迹的目标更新方案。
  - 依据：[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/pdf/1911.08265v1)；§4，PDF5–6与表1；Appendix H，15页。独立验证：未做独立复现。

- **原文主张**：期刊明确说明每个环境分别训练，共享模型和新环境零样本测试留待后续。
  - 依据：[Mastering Atari, Go, chess and shogi by planning with a learned model](https://go.gale.com/ps/i.do?asid=1de10613&id=GALE%7CA649646063&it=r&p=HRCA&u=googlescholar&v=2.1)；Methods / Training，末段。独立验证：未做独立复现。

- **原文主张**：期刊保留根节点动作可用性信息，并将其作为随观察提供的标准问题设定。
  - 依据：[Mastering Atari, Go, chess and shogi by planning with a learned model](https://go.gale.com/ps/i.do?asid=1de10613&id=GALE%7CA649646063&it=r&p=HRCA&u=googlescholar&v=2.1)；Methods / Comparison to AlphaZero，第(2)项。独立验证：未做独立复现。

## 阅读记录

### Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

Julian Schrittwieser；Ioannis Antonoglou；Thomas Hubert；Karen Simonyan；Laurent Sifre；Simon Schmitt；Arthur Guez；Edward Lockhart；Demis Hassabis；Thore Graepel；Timothy Lillicrap；David Silver，2019。Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model；Schrittwieser等；2019首版预印本，后有2020期刊版

- 实读：第二轮PDF2、4–5；本批核对v1首页及§2–4、表1（PDF2–6）、Appendix A（11页）、G/H/I（15–16页）；14页架构说明亦读。
- 未覆盖：所列范围之外的全部正文、附录图表、代码与实验未逐项核验；版本对照不是全篇复现。
- 状态：selected-sections；核对日期：2026-10-26
- 来源：[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/pdf/1911.08265v1)
- SHA-256：`971eb94abb7a8d9afc2fe758a06dba2f6dabbbcfb0a5b52c2d84c2011513ee3b`
- 阅读记录延续自第二轮；本批复核范围另列。

### Mastering Atari, Go, chess and shogi by planning with a learned model

Julian Schrittwieser；Ioannis Antonoglou；Thomas Hubert；Karen Simonyan；Laurent Sifre；Simon Schmitt；Arthur Guez；Edward Lockhart；Demis Hassabis；Thore Graepel；Timothy Lillicrap；David Silver，2020。Nature 588, 604–609；在线发表2020-12-23，期号日期12-24；实读Gale数据库的期刊正文HTML。

- 实读：Introduction至Conclusions；Table 1；Methods各小节，重点Comparison to AlphaZero、Training、MuZero Reanalyze及Evaluation。正文文字和表格可读。
- 未覆盖：HTML以Formula omitted替代部分公式；未逐式对照，未取得期刊PDF；补充图表、原始数据和代码未重新核对。
- 状态：selected-sections；核对日期：2026-10-26
- 来源：[Mastering Atari, Go, chess and shogi by planning with a learned model](https://go.gale.com/ps/i.do?asid=1de10613&id=GALE%7CA649646063&it=r&p=HRCA&u=googlescholar&v=2.1)；DOI：10.1038/s41586-020-03051-4
- SHA-256：无：未取得可校验的本地PDF；实际阅读载体与范围见本条来源记录。

## 边界与下一问

Reanalyze已存在于2019 v1，不能算作期刊新增。期刊更明确地限定逐环境训练；本文实验不能证明跨环境零样本泛化。部分公式和补充材料未完成版本核对。

换奖励或改变环境后，任务相关模型保留的信息是否还足够？


[版本对照与未覆盖范围](../versions/muzero-2019-2020.md)
