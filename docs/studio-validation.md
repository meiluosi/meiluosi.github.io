# RSI 研究工作室：验收与性能记录

检查日期：2026-10-04。基于当前工作区已有实现继续补齐；此记录只描述本轮实际检查，不把前轮截图或规划当作实测。

## 本轮实现

- 首页以 SVG 研究岛先呈现，3D 由访客显式启动；首页按钮直达独立研究岛或阅读路线。
- 首屏后的模型专题含可操作的参数状态预算，三条首次访问路线分别连接模型、搜索、因果实验及对应阅读。
- 研究岛增加路线、到达/靠近反馈、收起/展开内容、镜头跟随、暂停动态、转向/海岸边界修复与 44px 触控目标。
- 工作台实际执行简化 BPE、next-token NLL、SFT mask、DPO 单对公式、token 加权评测和部署产物核对。数据和概率明确为教学输入。
- 三个项目使用一致的模型/棋盘/因果 SVG，并有 1200×630 PNG 分享封面。`node scripts/generate-research-covers.mjs` 可以离线重新生成；案例页 OG 与 Twitter 图片使用对应封面。
- 保留旧 1B 文章与案例 URL，更新可见标题、规划内容、档案和阅读路线。训练项目使用 tiny 基线，30M 为首个正式实验目标，100M 与 0.5B 均需资源测量。

## 实际浏览器验证

Codex 内置 Chromium，本地 Astro dev；桌面 1440×960、窄屏 320×780，中英文切换。

另用 `pnpm preview --host 127.0.0.1 --port 4326` 检查生产构建：桌面首页入口与静态岛正常，研究岛完整画质可加载，页面宽度为1440且无横向溢出，控制台抽查无 warning/error。`studio-home-production.jpg` 与 `studio-island-production-desktop.jpg` 为生产预览截图，无开发工具栏。

| 场景 | 检查结果 |
|---|---|
| 首页入口 | “进入研究岛”导航成功；静态岛保留站点入口；“在此启动3D”加载场景；启动按钮可操作 |
| 首页小预算 | 30M / 100M / 0.5B 分别显示 0.45 / 1.49 / 7.45 GiB；可见后水合与英文切换有效 |
| 三站内容 | 模型/搜索/因果面板分别连接实验、案例和阅读路线；模型站自动行驶后出现“已到达” |
| 场景控制 | 完整/轻量切换、暂停动态、总览；面板可收起观看场景；Escape 关闭后焦点回到触发站点 |
| 窄屏 | 首页、研究岛、工作台中英文 documentWidth = 320；无页面横向溢出；长内容面板内部滚动，关闭按钮可见 |
| BPE | 合并数 0 与 2；输入 snow 显示未知字符 n → `<unk>`；说明字符教学 BPE 与训练项目字节 BPE 的区别 |
| SFT | assistant-only 为 3 个有效目标 / NLL 0.611；关闭后为 6 / 1.572，PAD 始终排除（给定概率教学值） |
| 固定评测 | token 加权时 B 更低，按篇平均后 A 更低；两者聚合语义明确 |
| DPO | 默认 Δ = 0、β = 0.1 时显示 ln 2 = 0.6931（公式演示） |
| 预算与分享 | 500M / context512 / batch1 / accumulation16 / 1000更新显示7.45GiB、8,192tokens/更新、约8.19M tokens；复制链接并刷新恢复阶段/预算 |
| 控制台 | 本轮抽查返回零 warning/error；开发热更新会重置当前面板，生产构建无 HMR |

截图位于 `docs/screenshots/studio-*`。窄屏测试属于桌面视口模拟；尚未验证真实手机 GPU、触摸多指、系统 reduced-motion 运行时切换或人为 WebGL context loss。WASD 和离屏/后台暂停的逻辑做代码审核，未完成持续驾驶与真机电量/热量压测。

## 性能策略与测量口径

- 首页 `client:visible` 仅水合岛壳；`create-world`/Three 在显式启动函数内动态导入，构建 HTML 没有渲染器 modulepreload。独立研究岛用 `client:idle` 自动准备。
- 轻量模式禁用动态阴影并限制 DPR；完整模式可开阴影。离屏、后台和手动暂停时按需停止渲染；reduced-motion 禁用持续环境动画并跳过自动驾驶过渡。
- 图形资源、监听器、观察器和动画帧随组件销毁清理。WebGL 不可用时保留静态站点链接；阅读和实验不依赖 3D。
- 全链路工作台保持独立 chunk；首页预算控件不导入完整工作台。新增分享图片只作为社交元数据，无首屏图片请求。
- 包体以生产构建文件字节与 Python gzip 估算记录；gzip 不是对 GitHub Pages 实际传输压缩方式的测量。没有把本地 dev 的加载速度当成线上 LCP/INP/移动端 FPS。

最终构建文件：renderer（含 Three）587,064 bytes / gzip 148,665；岛壳 20,348 / 8,472；工作台 49,462 / 19,559；首页预算 2,005 / 1,262。均不含共用 Svelte runtime。

## 检查命令与静态链接

- `pnpm check`：204 文件，0 errors、0 warnings、4 个既有 hints。
- `pnpm type-check`：通过。
- `AI_SUMMARY_API_KEY='' pnpm build`：通过；Pagefind 两种语言、65篇页面。重定向归档没有 html 的提示为现有行为。
- 生产产物审计：首页/研究岛/实验室/阅读路线/案例共15页面、289内部链接（含中英目标），目标文件与片段均存在。
- 6 项 Node 测试通过：`src/utils/model-pipeline.test.ts` 覆盖 BPE、NLL、mask、DPO 和固定评测聚合。
- `src/components/features/world/navigation.test.ts` 覆盖957条路线与障碍/海岸边界，全部通过。
- 独立目录迁移：把 `projects/small-model-lab` 复制到仓库外临时目录，CPU 下运行 `python scripts/lab.py smoke --output ../portable-run --device cpu`，三阶段与恢复到第14步均通过；不依赖 Astro、博客路径或外部旧训练仓库。记录保存在 `projects/small-model-lab/reports/portable-copy.json`。

受限环境启动 dev 超时、tsx 构建 IPC 被拒；相同本地命令在获准的沙箱外运行成功。未创建部署、GitHub远端、提交或PR。训练实测单独见 [small-model-training-validation.md](./small-model-training-validation.md)。
