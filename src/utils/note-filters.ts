/** Synchronous filtering with short, interruptible visual feedback. */
export function mountNoteFilters(): () => void {
	const root = document.querySelector<HTMLElement>(".notes-page");
	if (!root) return () => {};
	const formats = [
		...root.querySelectorAll<HTMLButtonElement>("[data-note-format-filter]"),
	];
	const topics = [
		...root.querySelectorAll<HTMLButtonElement>("[data-note-topic-filter]"),
	];
	const entries = [
		...root.querySelectorAll<HTMLElement>(
			"[data-note-format][data-note-category]",
		),
	];
	const status = root.querySelector<HTMLElement>("[data-note-results]");
	const empty = root.querySelector<HTMLElement>("[data-note-empty]");
	const reset = root.querySelector<HTMLButtonElement>("[data-note-reset]");
	const input = root.querySelector<HTMLInputElement>("[data-note-search]");
	const clear = root.querySelector<HTMLButtonElement>("[data-note-clear]");
	type SearchRecord = { id: string; zh: string; en: string };
	let index: Map<string, SearchRecord> | null = null;
	let loading = false;
	let failed = false;
	const params = new URLSearchParams(location.search);
	if (input) input.value = (params.get("q") ?? "").slice(0, 200);
	const preference = matchMedia("(prefers-reduced-motion: reduce)");
	const events = new AbortController();
	const active = new Map<HTMLElement, Animation>();
	let format =
		formats.find((button) => button.ariaPressed === "true")?.dataset
			.noteFormatFilter ?? "all";
	let topic =
		topics.find((button) => button.ariaPressed === "true")?.dataset
			.noteTopicFilter ?? "all";
	format = formats.some(
		(button) => button.dataset.noteFormatFilter === params.get("format"),
	)
		? params.get("format")!
		: "all";
	topic = topics.some(
		(button) => button.dataset.noteTopicFilter === params.get("topic"),
	)
		? params.get("topic")!
		: "all";
	const syncButtons = () => {
		for (const button of formats)
			button.ariaPressed = String(button.dataset.noteFormatFilter === format);
		for (const button of topics)
			button.ariaPressed = String(button.dataset.noteTopicFilter === topic);
	};
	syncButtons();
	let count = entries.filter((entry) => !entry.hidden).length;
	const cancelAnimations = () => {
		for (const animation of active.values()) animation.cancel();
		active.clear();
	};
	const describe = () => {
		if (!status) return;
		status.textContent = document.documentElement.lang.startsWith("en")
			? loading
				? "Searching…"
				: failed
					? "Search could not load. Type again to retry, or clear to browse."
					: `${count} of ${entries.length} notes`
			: loading
				? "正在搜索…"
				: failed
					? "搜索加载失败，请重新输入重试，或清空后浏览。"
					: `显示 ${count} / ${entries.length} 篇笔记`;
	};
	const update = (animate = true) => {
		const query = input?.value.trim() ?? "";
		const words = query.toLowerCase().split(/\s+/).filter(Boolean);
		const url = new URL(location.href);
		for (const [key, value] of [
			["q", query],
			["format", format === "all" ? "" : format],
			["topic", topic === "all" ? "" : topic],
		]) {
			if (value) url.searchParams.set(key, value);
			else url.searchParams.delete(key);
		}
		history.replaceState(history.state, "", url);
		const before = new Map(
			entries
				.filter((entry) => !entry.hidden)
				.map((entry) => [entry, entry.getBoundingClientRect()]),
		);
		cancelAnimations();
		count = 0;
		for (const entry of entries) {
			const record = index?.get(entry.dataset.noteId ?? "");
			const searchable = record
				? `${record.zh} ${record.en}`.toLowerCase()
				: "";
			entry.hidden =
				(format !== "all" && entry.dataset.noteFormat !== format) ||
				(topic !== "all" && entry.dataset.noteCategory !== topic) ||
				(words.length > 0 &&
					(!record || !words.every((word) => searchable.includes(word))));
			const excerpt = entry.querySelector<HTMLElement>("[data-note-excerpt]");
			if (excerpt) {
				excerpt.hidden = entry.hidden || !query;
				if (!excerpt.hidden && record) {
					const preferred = document.documentElement.lang.startsWith("en")
						? record.en
						: record.zh;
					const text = words.some((word) =>
						preferred.toLowerCase().includes(word),
					)
						? preferred
						: `${record.zh} ${record.en}`;
					const position = Math.max(
						0,
						text.toLowerCase().indexOf(words[0]) - 35,
					);
					excerpt.textContent = `${position ? "…" : ""}${text.slice(position, position + 180)}${text.length > position + 180 ? "…" : ""}`;
				}
			}
			if (!entry.hidden) count++;
		}
		if (empty) empty.hidden = count !== 0 || loading || failed;
		describe();
		if (
			!animate ||
			preference.matches ||
			typeof Element.prototype.animate !== "function"
		)
			return;
		let order = 0;
		for (const entry of entries) {
			if (entry.hidden || entry.contains(document.activeElement)) continue;
			const rect = entry.getBoundingClientRect();
			if (rect.bottom <= 0 || rect.top >= innerHeight) continue;
			const previous = before.get(entry);
			const offset = previous
				? Math.max(-72, Math.min(72, previous.top - rect.top))
				: 12;
			if (previous && Math.abs(offset) < 1) continue;
			const animation = entry.animate(
				[
					{ opacity: previous ? 1 : 0, transform: `translateY(${offset}px)` },
					{ opacity: 1, transform: "translateY(0)" },
				],
				{
					duration: 240,
					delay: Math.min(order++, 3) * 20,
					easing: "cubic-bezier(.22,1,.36,1)",
					fill: "backwards",
				},
			);
			active.set(entry, animation);
			animation.onfinish = () => {
				if (active.get(entry) === animation) active.delete(entry);
			};
		}
	};
	let timer: ReturnType<typeof setTimeout>;
	const search = async () => {
		if (input?.value.trim() && !index && !loading) {
			loading = true;
			failed = false;
			update(false);
			try {
				const response = await fetch("/api/note-search.json", {
					signal: events.signal,
				});
				if (!response.ok) throw new Error("Search unavailable");
				const records: SearchRecord[] = await response.json();
				index = new Map(records.map((record) => [record.id, record]));
			} catch {
				if (!events.signal.aborted) failed = true;
			} finally {
				loading = false;
			}
		}
		if (events.signal.aborted) return;
		if (!input?.value.trim()) failed = false;
		update(false);
	};
	input?.addEventListener(
		"input",
		() => {
			clearTimeout(timer);
			timer = setTimeout(() => void search(), 150);
		},
		{ signal: events.signal },
	);
	clear?.addEventListener(
		"click",
		() => {
			if (input) {
				input.value = "";
				input.focus();
			}
			void search();
		},
		{ signal: events.signal },
	);
	root.addEventListener(
		"click",
		(event) => {
			if (
				!(event.target instanceof Element) ||
				!event.target.closest("a.entry, a.trail-card")
			)
				return;
			try {
				sessionStorage.setItem(
					"feng-yu-notes-return",
					JSON.stringify({
						url: location.pathname + location.search,
						y: scrollY,
					}),
				);
			} catch {
				/* Browsing works without storage. */
			}
		},
		{ signal: events.signal },
	);
	for (const button of [...formats, ...topics]) {
		button.addEventListener(
			"click",
			() => {
				const isFormat = button.dataset.noteFormatFilter !== undefined;
				if (isFormat) format = button.dataset.noteFormatFilter ?? "all";
				else topic = button.dataset.noteTopicFilter ?? "all";
				for (const peer of isFormat ? formats : topics)
					peer.ariaPressed = String(peer === button);
				update();
			},
			{ signal: events.signal },
		);
	}
	reset?.addEventListener(
		"click",
		() => {
			format = topic = "all";
			if (input) input.value = "";
			failed = false;
			for (const button of formats)
				button.ariaPressed = String(button.dataset.noteFormatFilter === "all");
			for (const button of topics)
				button.ariaPressed = String(button.dataset.noteTopicFilter === "all");
			// The reset button disappears with the empty state, so return focus to a stable control.
			formats[0]?.focus({ preventScroll: true });
			update();
		},
		{ signal: events.signal },
	);
	root.addEventListener("focusin", () => cancelAnimations(), {
		signal: events.signal,
	});
	document.addEventListener("site-locale-change", () => update(false), {
		signal: events.signal,
	});
	preference.addEventListener(
		"change",
		() => {
			if (preference.matches) cancelAnimations();
		},
		{ signal: events.signal },
	);
	update(false);
	void search().then(() => {
		// Explicit return links restore the list; browser Back keeps native restoration.
		if (!new URLSearchParams(location.search).has("return")) return;
		try {
			const saved = JSON.parse(
				sessionStorage.getItem("feng-yu-notes-return") ?? "null",
			);
			const current = new URL(location.href);
			current.searchParams.delete("return");
			history.replaceState(history.state, "", current);
			if (
				saved?.url === current.pathname + current.search &&
				Number.isFinite(saved.y)
			)
				requestAnimationFrame(() =>
					scrollTo({ top: saved.y, behavior: "instant" }),
				);
		} catch {
			/* Ignore expired or unavailable history. */
		}
	});
	return () => {
		clearTimeout(timer);
		cancelAnimations();
		events.abort();
	};
}
