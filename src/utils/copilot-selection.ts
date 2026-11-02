/** Resolve an article selection into the text and position of the quick action. */
export function getCopilotSelection(
	selection: Selection | null,
	container: HTMLElement | null,
	scrollX: number,
	scrollY: number,
): { text: string; top: number; left: number } | null {
	if (!selection || selection.isCollapsed || selection.rangeCount === 0)
		return null;
	const text = selection.toString().trim();
	if (text.length < 2 || text.length > 500) return null;
	const range = selection.getRangeAt(0);
	if (!container?.contains(range.commonAncestorContainer)) return null;
	const rect = range.getBoundingClientRect();
	return {
		text,
		top: rect.top + scrollY - 38,
		left: Math.max(16, rect.left + scrollX + rect.width / 2 - 40),
	};
}
