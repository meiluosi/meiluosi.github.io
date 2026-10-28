# 第六轮验收记录

2026-10-28。完成的是本轮选定范围，而非24项全部通读：历史24项无对应作品映射的候选中，4项增补指定原典片段，20项仍待映射；另读2项竞争解释。1项原有部分映射的候选独立计数。

- 结构审计核对24项历史基线、6个新增来源与节点、12条原文定位、4条关系及3条旧结论补充；20条路线现有150个不同节点、177次节点出现。审计记录见 [structure-check.json](evidence/structure-check.json)。
- 生成器的 `--check` 均通过；第四轮网站结构审计和全路线时间线审计通过，历史报告各保留了本轮之前的快照。
- `pnpm check`：204个文件，0错误、0警告；`pnpm type-check` 通过；`AI_SUMMARY_API_KEY= pnpm build` 生成309个页面。
- 浏览器测试检查6个新增节点、定位主张与原结论跳转，以及桌面、390像素移动端和关闭 JavaScript 的阅读；见 [browser-check.json](evidence/browser-check.json)。
- 本机默认 Python 的 XML 扩展存在动态库不匹配；审计改用现有的 `/opt/miniconda3/bin/python` 并通过。外部原文可用性与作者的原始实验不在结构审计范围内。
