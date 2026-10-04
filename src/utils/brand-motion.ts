/** One-shot entrances. Content stays readable without JavaScript or animation APIs. */
export function mountBrandMotion(restored = false): () => void {
	const root = document.querySelector<HTMLElement>("[data-brand-motion]");
	if (!root || typeof Element.prototype.animate !== "function") return () => {};

	const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
	const active = new Map<HTMLElement, Animation>();
	const pending = new Set<HTMLElement>();
	let observer: IntersectionObserver | undefined;
	const events = new AbortController();
	const easing = "cubic-bezier(0.22, 1, 0.36, 1)";

	const stop = (element: HTMLElement) => {
		active.get(element)?.cancel();
		active.delete(element);
		pending.delete(element);
		observer?.unobserve(element);
	};
	const stopAll = () => {
		observer?.disconnect();
		pending.clear();
		for (const animation of active.values()) animation.cancel();
		active.clear();
	};
	const enter = (element: HTMLElement, delay = 0) => {
		element.closest(".now-threads")?.classList.add("threads-revealed");
		if (preference.matches || element.contains(document.activeElement)) return;
		const animation = element.animate(
			[
				{ opacity: 0, transform: "translate3d(0, 18px, 0)" },
				{ opacity: 1, transform: "translate3d(0, 0, 0)" },
			],
			{ duration: 620, delay, easing, fill: "backwards" },
		);
		active.set(element, animation);
		animation.onfinish = () => {
			if (active.get(element) === animation) active.delete(element);
		};
	};

	if (!preference.matches) {
		const threads = root.querySelector<HTMLElement>(".now-threads");
		if (threads && threads.getBoundingClientRect().top < window.innerHeight) {
			threads.classList.add("threads-revealed");
		}
		// A deep link or restored reading position must never replay the hero.
		if (!restored && !location.hash && window.scrollY < 24) {
			root.querySelectorAll<HTMLElement>("[data-motion-intro]").forEach((element, index) => {
				if (element.getBoundingClientRect().top < window.innerHeight) {
					enter(element, Math.min(index, 4) * 65);
				}
			});
		}
		if ("IntersectionObserver" in window) {
			observer = new IntersectionObserver((entries) => {
				let order = 0;
				for (const entry of entries) {
					if (!entry.isIntersecting) continue;
					const element = entry.target as HTMLElement;
					if (!pending.has(element)) continue;
					stop(element);
					enter(element, Math.min(order++, 3) * 55);
				}
			}, { threshold: 0, rootMargin: "0px 0px -24px 0px" });
			for (const element of root.querySelectorAll<HTMLElement>("[data-motion-reveal]")) {
				// Do not move content already visible when the page is mounted.
				if (element.getBoundingClientRect().top >= window.innerHeight && !element.hidden) {
					pending.add(element);
					observer.observe(element);
				}
			}
		}
	}

	const revealTarget = (target: Element) => {
		for (const element of new Set([...pending, ...active.keys()])) {
			if (element.contains(target) || target.contains(element)) stop(element);
		}
	};
	root.addEventListener("focusin", (event) => {
		if (event.target instanceof Element) revealTarget(event.target);
	}, { signal: events.signal });
	window.addEventListener("hashchange", () => {
		try {
			const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
			if (target) revealTarget(target);
		} catch { /* Malformed external fragments do not affect navigation. */ }
	}, { signal: events.signal });
	preference.addEventListener("change", () => {
		if (preference.matches) stopAll();
	}, { signal: events.signal });

	return () => {
		events.abort();
		stopAll();
	};
}
