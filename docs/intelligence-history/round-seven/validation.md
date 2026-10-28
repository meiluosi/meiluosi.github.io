# 第七轮验收与复算记录

2026-10-28。本轮完成数据及版本核对，并尝试运行原作者代码中的三任务和八任务两阶段模型；三任务载荷平方均值四舍五入到19.6%，八任务为21.0466%，不一致且出现警告。未将本轮记为整篇论文复现。

## 环境与执行

- 数据与代码：作者 [2021-04-01 提交](https://github.com/riedlc/QuantifyingCI/tree/e565b5b17aff883141100364cd6481ea4375bc00)，CSV SHA-256 `7c5fe80ac43c13949bce47d221f0538f89992611c31baa2862258b039657ab34`；其他指纹见 [versions.md](versions.md)。
- 临时隔离环境：macOS ARM64，R 4.5.3，OpenMx 2.22.11，metaSEM 1.5.0，SLSQP。作者2021年原始包版本尚未确认。首次从源码安装失败于缺少编译器，补齐隔离编译器、依赖后安装成功；未修改系统 R。
- 先运行 `python scripts/verify-intelligence-collective-data.py <锁定CSV>`；该脚本核对指纹并仅输出聚合描述。再使用 `Rscript scripts/attempt-intelligence-collective-model.R <锁定仓库目录> three <绝对输出路径>`，将 `three` 换为 `eight` 可运行八任务配置。脚本仅调用作者的配置及经检查的辅助模型脚本，把用于显示的 `screenreg` 换为无操作；不运行会自行安装软件包和更改工作目录的顶层脚本。模型文件写入指定的临时路径，不放进本站。
- `runRerun=FALSE` 与原作者顶层默认一致。两次运行各模型得到的结果一致到本记录展示的精度；两阶段返回状态0。八任务第一阶段报告 Hessian 无法求逆、使用广义逆，标准误可能不可靠；状态0不抹除该警告。
- 在相关结构分析、单位潜变量方差和单位观测方差的模型口径下，本站对载荷平方求平均；计算值、全部载荷、拟合指标、软件版本和更正对照见 [model-results.json](evidence/model-results.json)。三任务目标模型自由度为0，不能借拟合指数宣称模型已得到独立验证。
- 现有差异不能归因于某一种版本或算法：作者更正后的 SI 表格、本次仓库 CSV 与 Dataverse 文件的逐字节对应、作者原始包版本、团队排除流转都未完全核实。对八任务应先取得材料再比较，不靠调参使数字吻合。

## 网站与结构

- 原始数据未提交，报告只保存团队数和汇总结果；脚本可用锁定CSV重算。
- 结构审计验证数据结果、三／八任务载荷平方均值、网页与可编辑图，旧结论和研究队列标记了本轮进度；第四轮9项“未执行”是当时历史快照，不覆盖当前状态。
- `pnpm check`、`pnpm type-check`、`AI_SUMMARY_API_KEY= pnpm build` 和桌面、390像素移动端、关闭 JavaScript 的浏览器检查在最终构建后执行，详情见 [browser-check.json](evidence/browser-check.json)。
