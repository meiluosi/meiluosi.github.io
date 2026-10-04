export type BrandLocale = "zh-CN" | "en";

type BrandDictionary = Record<string, string>;

export const brandTranslations: Record<BrandLocale, BrandDictionary> = {
	"zh-CN": {
		"nav.work": "研究档案",
		"nav.notes": "笔记",
		"nav.about": "关于",
		"nav.contact": "联系",
		"nav.language": "切换到英文",
		"home.hero.title":
			'让智能，<br/><span class="serif">递归地变得更好</span><span class="period">.</span>',
		"home.hero.intro":
			"你好，我是枫语。<br/>我把学习、实验与系统构建，放在通向递归自我改进（RSI）与 AGI 的同一条路径上。",
		"home.hero.explore": "进入研究岛",
		"home.about.heading": "所有尝试，<br/><span>都指向同一个问题。</span>",
		"home.about.question": "智能能否持续理解、评估并改进自身？",
		"home.about.mission":
			"以递归自我改进（RSI）为长期问题，探索通向 AGI 的路径。",
		"home.about.link": "多了解一点关于我",
		"home.now.heading": "把长期问题<br/><span>变成眼前的工作。</span>",
		"home.now.mission":
			"以递归自我改进（RSI）为长期问题，探索通向 AGI 的路径。",
		"home.now.models": "模型学习与系统",
		"home.now.feedback": "反馈与自我改进",
		"home.now.causal": "因果与评估",
		"home.work.link": "查看完整研究档案",
		"home.work.heading": '把想法，变成<span class="serif">实验。</span>',
		"home.work.copy":
			"模型工作流、自我对弈、博弈学习与因果分析。<br/>四条路径，回答同一个长期问题。",
		"home.notes.link": "全部文章",
		"home.notes.heading": '边走，<span class="serif">边记录。</span>',
		"home.notes.copy": "阅读、实验、踩坑，然后把它们写下来。",
		"home.contact.pre": "有想法？有问题？或者只是想打个招呼？",
		"home.contact.heading": '那就<span class="serif">聊聊吧。</span>',
		"home.contact.button": "给我写封邮件",
		"home.footer.back": "回到顶部 ↑",
		"work.hero.title": "让智能，<br/><em>递归地变好。</em>",
		"work.hero.lead":
			"围绕递归自我改进这个长期问题，沿着模型系统、反馈学习与因果评估，逐步积累可检查的研究记录。",
		"work.hero.archive": "先看研究档案 ↓",
		"work.archive.heading": "四次探索，四条路径。",
		"work.archive.intro":
			"从本地模型工作流到自我对弈、博弈学习与因果分析。每个案例都从一个具体问题出发，并继续通向下一次实验。",
		"work.source": "查看原文",
		"work.footer.contact": "有好的问题？来聊聊",
		"notes.hero.title": "边走，<em>边记录。</em>",
		"notes.hero.lead":
			"研究笔记、实验计划和实现记录。它们是朝向 RSI 与 AGI 的不同路径，也是不断修正问题的过程。",
		"notes.start": "从一个问题开始",
		"notes.entryPoints": "四条入口",
		"notes.openThread": "打开这条研究线索",
		"notes.index": "笔记索引",
		"notes.recent": "最近更新",
		"notes.format": "形式",
		"notes.topic": "方向",
		"notes.all": "全部",
		"notes.originalChinese": "中文原文",
		"notes.format.plan": "实验计划",
		"notes.format.code-example": "代码示例",
		"notes.format.application": "应用拆解",
		"notes.format.technical": "技术讲解",
		"notes.format.experiment": "实验方法",
		"notes.format.reflection": "学习与思考",
		"notes.footer": "重新探索方向",
		"about.hero.title": "我是枫语。<br/><em>你好。</em>",
		"about.hero.lead":
			"我想把所有学习、实验与系统构建，放在同一条长期路径上：探索递归自我改进（RSI），并一步步走向 AGI。",
		"about.hero.cta": "联系我",
		"about.question":
			"智能能否理解自己的能力边界，根据反馈找出改进方法，再检验这些改进是否真的有效？这是我理解 RSI 时最关心的问题。",
		"about.context":
			"这里记录与这个问题相关的模型训练与推理、强化学习和因果分析。笔记会区分实验计划、实现记录和应用案例，也会随着新的证据不断修正。",
		"about.toolbox.models": "模型如何学习、推理，<br/>以及如何被有效评估。",
		"about.toolbox.feedback": "反馈怎样改变策略，<br/>并让行为持续变好。",
		"about.toolbox.causal": "哪些干预带来改进，<br/>又如何确认因果关系。",
		"about.principle.0.title": "从问题开始",
		"about.principle.0.body":
			"先把问题定义清楚，再决定该读什么、写什么代码，以及什么证据才算回答。",
		"about.principle.1.title": "理论要落到实践",
		"about.principle.1.body":
			"读论文，也做实验。理解一个方法，最好亲手实现它、观察它在哪些条件下失效。",
		"about.principle.2.title": "把过程讲明白",
		"about.principle.2.body":
			"记录假设、选择和结果，让知识可以被复查、复用，也欢迎别人提出不同看法。",
		"about.footer": "看看我在探索什么",
		"contact.back": "← 回到探索",
		"contact.title": "世界很大，<br/><em>不妨聊聊。</em>",
		"contact.intro":
			"如果你也在做有意思的事，或者有一个值得认真讨论的问题，欢迎来信。我会尽量亲自回复。",
		"contact.email": "写信给我",
		"contact.explore": "继续探索 ↓",
		"footer.work": "探索方向",
		"footer.notes": "笔记",
		"footer.contact": "联系",
		"post.published": "发布于",
		"post.updated": "更新于",
		"post.wordsMeta": "{words} 字",
		"post.readMeta": "{minutes} 分钟阅读",
		"post.related": "相关文章",
		"post.smartRecommend": "智能推荐",
		"post.random": "随机文章",
		"post.randomRecommend": "随机推荐",
		"post.citationTitle": "引用本文",
		"post.copyCitation": "复制 BibTeX",
		"post.comments": "评论区",
		"post.commentsSubtitle": "分享你的想法，与大家交流讨论",
		"post.allNotes": "所有笔记",
	},
	en: {
		"nav.work": "WORK",
		"nav.notes": "NOTES",
		"nav.about": "ABOUT",
		"nav.contact": "LET’S TALK",
		"nav.language": "切换到中文",
		"home.hero.title":
			'Make intelligence,<br/><span class="serif">better by recursion</span><span class="period">.</span>',
		"home.hero.intro":
			"Hi, I’m Feng Yu.<br/>I put learning, experiments, and system building on one long path toward recursive self-improvement (RSI) and AGI.",
		"home.hero.explore": "Enter the research island",
		"home.about.heading":
			"Every attempt<br/><span>points to one question.</span>",
		"home.about.question":
			"Can intelligence understand, evaluate, and improve itself continuously?",
		"home.about.mission":
			"RSI is the long-term question: a path toward AGI through systems that can learn, receive feedback, and improve.",
		"home.about.link": "A little more about me",
		"home.now.heading":
			"Turn the long view<br/><span>into today’s work.</span>",
		"home.now.mission":
			"RSI is the long-term question: a path toward AGI through systems that can learn, receive feedback, and improve.",
		"home.now.models": "Learning & systems",
		"home.now.feedback": "Feedback & self-improvement",
		"home.now.causal": "Causality & evaluation",
		"home.work.link": "View the full research archive",
		"home.work.heading":
			'Turn ideas into <span class="serif">experiments.</span>',
		"home.work.copy":
			"Model workflows, self-play, game learning, and causal analysis.<br/>Four paths toward one long-term question.",
		"home.notes.link": "All notes",
		"home.notes.heading":
			'Keep moving,<br/><span class="serif">keep recording.</span>',
		"home.notes.copy": "Read, experiment, get stuck, and write the trail down.",
		"home.contact.pre": "Have an idea, a question, or just want to say hello?",
		"home.contact.heading": 'Let’s<span class="serif"> talk.</span>',
		"home.contact.button": "Write me an email",
		"home.footer.back": "BACK TO TOP ↑",
		"work.hero.title": "Make intelligence,<br/><em>better by recursion.</em>",
		"work.hero.lead":
			"I follow recursive self-improvement through model systems, feedback learning, and causal evaluation—building records that can be checked.",
		"work.hero.archive": "START WITH THE ARCHIVE ↓",
		"work.archive.heading": "Four explorations, four paths.",
		"work.archive.intro":
			"From local model workflows to self-play, game learning, and causal analysis. Each case begins with a concrete question and opens into the next experiment.",
		"work.source": "READ THE NOTE",
		"work.footer.contact": "Have a good question? Let’s talk",
		"notes.hero.title": "Keep moving, <em>keep recording.</em>",
		"notes.hero.lead":
			"Research notes, experiment plans, and implementation records. Different paths toward RSI and AGI, revised as the evidence changes.",
		"notes.start": "START WITH A QUESTION",
		"notes.entryPoints": "FOUR ENTRY POINTS",
		"notes.openThread": "OPEN THIS THREAD",
		"notes.index": "NOTEBOOK INDEX",
		"notes.recent": "RECENT FIRST",
		"notes.format": "FORM",
		"notes.topic": "THREAD",
		"notes.all": "ALL",
		"notes.originalChinese": "CHINESE ORIGINAL",
		"notes.format.plan": "EXPERIMENT PLAN",
		"notes.format.code-example": "CODE EXAMPLE",
		"notes.format.application": "APPLICATION",
		"notes.format.technical": "TECHNICAL",
		"notes.format.experiment": "EXPERIMENT",
		"notes.format.reflection": "REFLECTION",
		"notes.footer": "Explore the research threads",
		"about.hero.title": "I’m Feng Yu.<br/><em>Hello.</em>",
		"about.hero.lead":
			"I want to put learning, experiments, and system building on one long path: exploring recursive self-improvement (RSI), one step toward AGI at a time.",
		"about.hero.cta": "Get in touch",
		"about.question":
			"Can intelligence understand its own limits, use feedback to find better strategies, and test whether those improvements are real? That is the question I keep returning to when I think about RSI.",
		"about.context":
			"This site records work around model training and inference, reinforcement learning, and causal analysis. Notes distinguish plans, implementations, and applications, and change when new evidence appears.",
		"about.toolbox.models":
			"How models learn and reason,<br/>and how to evaluate them well.",
		"about.toolbox.feedback":
			"How feedback changes policies<br/>and keeps behavior improving.",
		"about.toolbox.causal":
			"Which interventions help,<br/>and how to establish causality.",
		"about.principle.0.title": "Start with the question",
		"about.principle.0.body":
			"Define the question before deciding what to read, what to code, and what would count as an answer.",
		"about.principle.1.title": "Put theory into practice",
		"about.principle.1.body":
			"Read papers, then run experiments. A method becomes clearer when you implement it and see where it fails.",
		"about.principle.2.title": "Make the process legible",
		"about.principle.2.body":
			"Record assumptions, choices, and results so the work can be checked, reused, and challenged.",
		"about.footer": "See what I’m exploring",
		"contact.back": "← BACK TO EXPLORING",
		"contact.title": "The world is big,<br/><em>let’s talk.</em>",
		"contact.intro":
			"If you are making something interesting, or have a question worth taking seriously, send a note. I try to reply personally.",
		"contact.email": "WRITE AN EMAIL",
		"contact.explore": "KEEP EXPLORING ↓",
		"footer.work": "RESEARCH",
		"footer.notes": "NOTES",
		"footer.contact": "CONTACT",
		"post.published": "Published",
		"post.updated": "Updated",
		"post.wordsMeta": "{words} words",
		"post.readMeta": "{minutes} min read",
		"post.related": "Related posts",
		"post.smartRecommend": "FOR YOU",
		"post.random": "More to explore",
		"post.randomRecommend": "RANDOM PICKS",
		"post.citationTitle": "Cite this article",
		"post.copyCitation": "Copy BibTeX",
		"post.comments": "Discussion",
		"post.commentsSubtitle": "Share a thought or join the conversation",
		"post.allNotes": "All notes",
	},
};

export function getBrandTranslation(
	locale: BrandLocale,
	key: string,
): string | undefined {
	return brandTranslations[locale][key];
}
