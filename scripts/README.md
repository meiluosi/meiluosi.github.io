# scripts/ 说明

本目录有两类性质完全不同的脚本。**修改前先分清是哪一类。**

## 1. 长期维护脚本(可修改、可重复运行)

| 脚本 | 用途 | 在 `pnpm build` 中 | 在 CI 中 |
|------|------|:---:|:---:|
| `generate-icons.js` | 生成 `src/constants/icons.ts` | ✅ | ✅(经 build) |
| `generate-lqips.ts` | 生成图片 LQIP `src/constants/lqips.json` | ✅ | ✅ |
| `generate-summaries.ts` | 生成文章摘要 | ✅ | ✅ |
| `validate-frontmatter.js` | 校验 frontmatter 可解析 | ✅ | ✅ |
| `subset-fonts.ts` | 构建后字体子集化 | ✅ | ✅ |
| `audit-internal-links.js` | 扫描 `dist/` 的站内死链 | ❌ | ✅(build 之后) |
| `report-bundle-size.js` | 构建产物体积基线（只读，`pnpm report:bundle`） | ❌ | 仅 perf-baseline |
| `report-missing-translations.js` | 缺英文版文章统计（只读，`pnpm report:translations`） | ❌ | ❌ |
| `new-post.js` | 脚手架新文章(`pnpm new-post`) | ❌ | ❌ |
| `generate-research-covers.mjs` | 生成研究封面 | ❌ | ❌ |
| `quarantine-bad-posts.mjs` | 隔离有问题的文章 | ❌ | ❌ |

## 2. 研究轮次脚本(冻结,**不要移动、重命名或修改**)

名称含 `intelligence-*`、`replicate-*`、`attempt-*` 的脚本(`generate-`/`check-`/`audit-`/`verify-`)属于
`docs/intelligence-history/` 各研究轮次的**可复现证据链**:

- 脚本用 `import.meta.dirname` 相对路径定位 `docs/` 下的数据。
- 部分验收结果(如 `research.json`、`method-check/results.json`)记录了脚本自身的 SHA256,
  并且文档中的复现命令写死了 `scripts/xxx` 路径。
- 移动或改动脚本会使这些哈希与复现命令失效。

因此它们保持原位。需要复核某一轮时,按对应 `docs/intelligence-history/<round>/README.md`
中的命令,**从仓库根目录**运行。

### 新增研究轮次时的约定

- 不要复制上一轮脚本改名;先考虑把共性抽到 `scripts/lib/`。
- 新脚本在文件头注释说明:输入、输出、是否会写入 `docs/`、是否可重复运行。
- 验收完成后,在 `docs/` 对应轮次的 README 里登记脚本与哈希,之后视为冻结。

## 通用要求

- 一律从仓库根目录运行:`node scripts/xxx.js`。
- 校验类脚本用非零退出码表示失败,才能接入 CI。
