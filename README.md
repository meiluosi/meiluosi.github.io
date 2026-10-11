# 枫语 / Feng Yu

我是冯宇（枫语），一名大模型算法工程师。这里记录我对模型学习、搜索反馈与因果评估的探索，也收着阅读、实验和动手造物的过程。

**[访问网站](https://meiluosi.github.io/)** · [阅读文章](https://meiluosi.github.io/notes/) · [GitHub 个人主页](https://github.com/meiluosi) · [English](README.en.md)

[![枫语研究工作室版首页：关于智能学习与改进的问题，以及可探索的研究岛](docs/screenshots/studio-home-production.jpg)](https://meiluosi.github.io/)

*研究工作室版首页截图，2026-10-04。*

## 从一个问题开始

| 入口 | 可以动手观察什么 |
| --- | --- |
| [小模型工作台](https://meiluosi.github.io/lab/model-pipeline/) | 从数据和分词出发，理解预训练、SFT、DPO、评测与部署怎样相连。配套 [Python 项目](projects/small-model-lab/README.md) 提供可运行的训练链路。 |
| [MCTS 搜索实验](https://meiluosi.github.io/lab/mcts/) | 单步运行选择、扩展、模拟与回传；改变搜索预算和探索权重，观察树与统计量如何变化。 |
| [相关，还是因果？](https://meiluosi.github.io/lab/causal-playground/) | 改变模拟优惠券的分配方式与真实效果，看看选择偏差怎样影响观察到的差异。 |

这些互动页用于理解原理。配套小模型项目已验证 **138,560 参数的 tiny 工程链路**，包含保存、恢复与 CLI 加载；30M 及以上训练仍在规划中。运行记录与边界见 [训练验证文档](docs/small-model-training-validation.md)。

## 本地运行

使用 Node.js 22 和 pnpm 9.14.4，与现有部署工作流保持一致。

```bash
git clone https://github.com/meiluosi/meiluosi.github.io.git
cd meiluosi.github.io
pnpm install
pnpm dev
```

打开 [localhost:4321](http://localhost:4321)。提交网站改动前运行 `pnpm check`、`pnpm type-check` 和 `pnpm build`；构建后可用 `pnpm preview` 预览。

## 维护与文档

| 文档 | 内容 |
| --- | --- |
| [站点使用与维护](docs/site-guide.md) | 命令、目录、配置、写作语法、构建与部署 |
| [架构说明](docs/architecture.md) | 页面、组件与数据如何组织 |
| [开发路线图](docs/roadmap.md) | 已完成工作、进行中的探索与下一步 |
| [研究记录的证据](docs/case-study-evidence.md) | 案例的来源、实现范围与待补充信息 |
| [全部文档](docs/README.md) | 功能规划、调研、验收记录与内容维护入口 |
| [贡献与开发约定](AGENTS.md) | 仓库结构、代码风格与验证要求 |

## 技术与来源

站点使用 **Astro 7、Svelte 5、TypeScript 与 Tailwind CSS 4**，通过 GitHub Pages 发布静态页面。Pagefind 提供搜索，KaTeX 与 Mermaid 支持技术写作，Three.js 用于研究岛。具体依赖与命令见 [package.json](package.json)。

上游仓库为 [klovien/klovien.github.io](https://github.com/klovien/klovien.github.io)。感谢上游作者；项目保留 saicaca、CuteLeaf 的原有版权声明，详见 [LICENSE](LICENSE)。

代码采用 [MIT](LICENSE) 许可；博客内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可，另有注明的内容与第三方资源以各自声明为准。
