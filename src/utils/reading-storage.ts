const STORAGE_KEY = "feng-yu-reading-v1";
export function readSaved(): string[] {
	const value = localStorage.getItem(STORAGE_KEY) ?? "[]";
	let raw: unknown;
	try {
		raw = JSON.parse(value);
	} catch {
		return [];
	}
	if (!Array.isArray(raw)) return [];
	return [
		...new Set(
			raw.filter(
				(id): id is string => typeof id === "string" && id.length < 300,
			),
		),
	].slice(0, 500);
}
export function toggleSaved(id: string): boolean {
	const saved = readSaved();
	const next = saved.includes(id)
		? saved.filter((item) => item !== id)
		: [id, ...saved];
	localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
	window.dispatchEvent(new Event("reading-saved-change"));
	return next.includes(id);
}
