/**
 * 仅在「可见」时运行的逐帧循环。
 *
 * 目标元素离开视口、或标签页被隐藏时暂停 requestAnimationFrame，
 * 回到可见状态后自动恢复。用于 canvas / 动画类岛，避免离屏空转耗电。
 *
 * 返回清理函数，组件卸载时调用。
 */
export function startVisibleLoop(
	target: Element | null | undefined,
	tick: () => void,
): () => void {
	let frameId = 0;
	let inView = true;
	let disposed = false;

	const loop = (): void => {
		frameId = 0;
		if (disposed || !inView || document.hidden) return;
		tick();
		frameId = requestAnimationFrame(loop);
	};

	const resume = (): void => {
		if (!disposed && inView && !document.hidden && frameId === 0) {
			frameId = requestAnimationFrame(loop);
		}
	};

	let observer: IntersectionObserver | undefined;
	if (target && typeof IntersectionObserver !== "undefined") {
		observer = new IntersectionObserver(
			(entries) => {
				for (const entry of entries) inView = entry.isIntersecting;
				resume();
			},
			{ rootMargin: "120px" },
		);
		observer.observe(target);
	}

	document.addEventListener("visibilitychange", resume);
	resume();

	return () => {
		disposed = true;
		if (frameId) cancelAnimationFrame(frameId);
		frameId = 0;
		observer?.disconnect();
		document.removeEventListener("visibilitychange", resume);
	};
}
