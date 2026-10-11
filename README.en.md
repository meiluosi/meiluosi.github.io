# Feng Yu / 枫语

I'm Feng Yu (冯宇), an LLM algorithm engineer. This is where I explore model learning, search feedback, and causal evaluation, and keep notes on reading, experiments, and things I make.

**[Visit the website](https://meiluosi.github.io/)** · [Read the articles](https://meiluosi.github.io/notes/) · [GitHub profile](https://github.com/meiluosi) · [简体中文](README.md)

[![Feng Yu research studio homepage: questions about learning and improvement, alongside an explorable research island](docs/screenshots/studio-home-production.jpg)](https://meiluosi.github.io/)

*Research studio homepage screenshot, October 4, 2026.*

## Start with a question

| Explore | What you can try |
| --- | --- |
| [Small-model workbench](https://meiluosi.github.io/lab/model-pipeline/) | Start with data and tokenization, then follow the connections between pretraining, SFT, DPO, evaluation, and deployment. The accompanying [Python project](projects/small-model-lab/README.md) provides a runnable training pipeline. |
| [MCTS search experiment](https://meiluosi.github.io/lab/mcts/) | Step through selection, expansion, simulation, and backpropagation. Change the search budget and exploration weight to watch the tree and statistics evolve. |
| [Correlation or causation?](https://meiluosi.github.io/lab/causal-playground/) | Adjust simulated coupon allocation and its true effect to see how selection bias changes the observed difference. |

These interactive pages illustrate concepts. The accompanying model project has verified a **138,560-parameter tiny engineering pipeline**, including checkpoint saving, resuming, and CLI loading; training at 30M parameters and above remains planned. See the [training validation notes](docs/small-model-training-validation.md) for run records and limitations.

## Run locally

Use Node.js 22 and pnpm 9.14.4 to match the existing deployment workflow.

```bash
git clone https://github.com/meiluosi/meiluosi.github.io.git
cd meiluosi.github.io
pnpm install
pnpm dev
```

Open [localhost:4321](http://localhost:4321). Before submitting website changes, run `pnpm check`, `pnpm type-check`, and `pnpm build`; use `pnpm preview` to inspect the production build.

## Maintenance and documentation

Guides are in Chinese unless indicated otherwise.

| Document | Contents |
| --- | --- |
| [Site maintenance guide](docs/site-guide.md) | Commands, directories, configuration, writing syntax, builds, and deployment |
| [Architecture](docs/architecture.md) | How pages, components, and data fit together |
| [Roadmap](docs/roadmap.md) | Completed work, ongoing exploration, and next steps |
| [Evidence behind the research notes](docs/case-study-evidence.md) | Sources, implementation scope, and open details |
| [All documentation](docs/README.md) | Planning, research, validation records, and content maintenance |
| [Contributor guidelines](AGENTS.md) (English) | Repository structure, code style, and validation requirements |

## Technology and credits

Built with **Astro 7, Svelte 5, TypeScript, and Tailwind CSS 4**, with static pages deployed to GitHub Pages. Pagefind provides search, KaTeX and Mermaid support technical writing, and Three.js powers the research island. See [package.json](package.json) for dependencies and scripts.

The upstream repository is [klovien/klovien.github.io](https://github.com/klovien/klovien.github.io). Thanks to its authors; this project retains the original copyright notices for saicaca and CuteLeaf in [LICENSE](LICENSE).

Code is licensed under [MIT](LICENSE). Blog content is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), except where otherwise noted; third-party materials retain their respective terms.
