# 材料版本与差异

| 材料 | 已锁定版本或位置 | 实际观察与边界 |
| --- | --- | --- |
| 论文 | [Riedl 等，2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166150/)，正文的方法与样本 | 22 项研究、1,356 个团队、5,279 人；PMC载体可能保留更正前的正文数字，应同时读更正。 |
| 正式更正 | [PNAS 2022 更正](https://pmc.ncbi.nlm.nih.gov/articles/PMC9171794/) | AVE由44%更正为19.6%；作者还更正／澄清模型拟合报告。不能把44%沿用为当前结果。 |
| 作者仓库 | [QuantifyingCI](https://github.com/riedlc/QuantifyingCI/tree/e565b5b17aff883141100364cd6481ea4375bc00)，提交 `e565b5b17aff883141100364cd6481ea4375bc00`，2021-04-01 | 早于2022更正。说明文件写1,357团队，当前CSV有1,356行和1,356个不同团队ID；差一原因未知。仓库说明写的主脚本名与实际 `R_CODE/00-RUN_ALL.R` 的连接符不同。 |
| 数据文件 | `DATA/full.csv`，SHA-256 `7c5fe80ac43c13949bce47d221f0538f89992611c31baa2862258b039657ab34` | 与论文正文的团队数和人数之和一致。团队完整排除流转及作者为何写1,357，材料中尚未查明；不要凭空解释。 |
| 作者模型 | `R_CODE/helper_fitMetaAnalysis.R`，SHA-256 `cb557b20bea3fb908cdfaf842b30ceca76eeff51e3a6b22cecadc3311ce8c0a5` | `tssem1` 随机效应与 `tssem2`；按研究计算相关矩阵，按配置中的各研究样本量拟合。仓库未包含已拟合模型文件。 |
| 作者旧提取脚本 | `R_CODE/helper_extract.wls.R`，SHA-256 `843eaf6aaa234c10f1c9819aac7c5ef7eb57fca4f15b5b9b5e767b6f9b9e360f` | 对模型 implied S 的对应行求均值并标为 AVE；我们不据此推断这段代码就是更正所称软件错误的原因，也不把它当作更正后的实现。 |
| 作者数据入口 | [Harvard Dataverse V1](https://doi.org/10.7910/DVN/RAKAOG) | 论文指向此存档；本次实际计算使用上述仓库CSV，尚未完成两处文件的逐字节一致性验证。仓库未发现明确的许可证文件；本项目不再分发原始数据。 |
| 补充材料 | 正文列出的 SI Appendix Table S4 | 本次直接下载受到访问限制；是否与更正后版本一致、逐表细节仍待核对。 |

作者顶层脚本会安装软件包并修改工作目录；本次不直接运行该脚本。锁定提交并不自动锁定其统计包版本或更新后的作者更正代码。
