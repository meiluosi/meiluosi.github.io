# ADR 0001: 部署主线

- 状态:Accepted
- 日期:2026-11-01

## 背景

仓库同时存在三套部署相关配置:

| 文件 | 平台 | 现状 |
|------|------|------|
| `.github/workflows/deploy.yml` | GitHub Pages | 唯一自动化的发布流程,push 到 `master` 触发 |
| `wrangler.jsonc` + `@astrojs/cloudflare` | Cloudflare | 手动部署,静态资源目录 `dist` |
| `vercel.json` | Vercel | 含安全响应头与缓存策略,无自动流程 |

三者并存导致:不清楚哪个是线上真实来源;响应头、缓存策略只在 Vercel 配置里生效;改部署相关代码时无从判断要兼容哪些平台。

## 决策

1. **主线:GitHub Pages**,通过 `deploy.yml` 发布。这是仓库内唯一自动化、可审计的发布路径。
2. **`vercel.json` 与 `wrangler.jsonc` 降级为备用/预览配置**:保留,不删除,但不作为验收依据,也不保证与主线同步。
3. 新功能不得依赖仅在备用平台才有的能力(如自定义响应头、边缘函数),除非先更新本 ADR。
4. 发布前必须通过 `.github/workflows/ci.yml` 中的全部检查。

## 后果

- 需要自定义响应头的诉求(CSP、缓存)在 GitHub Pages 上无法实现,若成为硬需求,应另起 ADR 评估迁移。
- 若日后确认不再使用 Vercel/Cloudflare,应删除对应配置与依赖(如 `@astrojs/cloudflare`),并将本 ADR 标注为 Superseded。

## 待确认

- `astro.config.mjs` 的 adapter 与 `site`/`base` 是否完全符合 GitHub Pages(本 ADR 未改动构建配置)。
