<script lang="ts">
import { onMount } from "svelte";
import { type BrandLocale, getBrandTranslation } from "@/i18n/brand";

let locale = $state<BrandLocale>("zh-CN");

function readPreferredLocale(): BrandLocale {
	const requested = new URLSearchParams(window.location.search).get("locale");
	if (requested === "en" || requested === "zh-CN") return requested;

	// A translated article lives at an /en/ route; keep its interface language
	// aligned with the article even when an older preference was saved.
	const routeLanguage = document.querySelector<HTMLElement>(
		"#swup-container[lang]",
	)?.lang;
	if (routeLanguage?.toLowerCase().startsWith("en")) return "en";

	let saved: string | null = null;
	try {
		saved = window.localStorage?.getItem("feng-yu-locale") ?? null;
	} catch {
		// Storage can be disabled in privacy modes; the page can still switch in memory.
	}
	if (saved === "en" || saved === "zh-CN") return saved;
	if (routeLanguage?.toLowerCase().startsWith("zh")) return "zh-CN";
	return navigator.language.toLowerCase().startsWith("en") ? "en" : "zh-CN";
}

function savePreferredLocale(next: BrandLocale) {
	try {
		window.localStorage?.setItem("feng-yu-locale", next);
	} catch {
		// Keep the current choice for this page when storage is unavailable.
	}
}

function applyLocale(next: BrandLocale) {
	const route = document.querySelector<HTMLElement>(
		"[data-locale-zh-url], [data-locale-en-url]",
	);
	const counterpart =
		route?.dataset[next === "en" ? "localeEnUrl" : "localeZhUrl"];
	if (
		counterpart &&
		new URL(counterpart, document.baseURI).pathname !== window.location.pathname
	) {
		locale = next;
		savePreferredLocale(next);
		const destination = new URL(counterpart, document.baseURI);
		destination.searchParams.set("locale", next);
		window.location.assign(destination);
		return;
	}

	locale = next;
	document.documentElement.lang = next === "en" ? "en" : "zh-CN";
	savePreferredLocale(next);
	for (const element of document.querySelectorAll<HTMLElement>("[data-i18n]")) {
		const key = element.dataset.i18n;
		const value =
			element.dataset[`i18n${next === "en" ? "En" : "Zh"}`] ||
			(key ? getBrandTranslation(next, key) : undefined);
		if (!value) continue;
		if (element.dataset.i18nHtml === "true") {
			element.innerHTML = value;
		} else {
			element.textContent = value;
		}
	}
	for (const element of document.querySelectorAll<HTMLElement>(
		"[data-i18n-aria-label-zh], [data-i18n-aria-label-en]",
	)) {
		const value =
			element.dataset[next === "en" ? "i18nAriaLabelEn" : "i18nAriaLabelZh"];
		if (value) element.setAttribute("aria-label", value);
	}
	for (const element of document.querySelectorAll<HTMLElement>(
		"[data-i18n-alt-zh], [data-i18n-alt-en]",
	)) {
		const value = element.dataset[next === "en" ? "i18nAltEn" : "i18nAltZh"];
		if (value) element.setAttribute("alt", value);
	}
	for (const notice of document.querySelectorAll<HTMLElement>(
		"[data-locale-notice]",
	)) {
		notice.hidden = next !== "en";
	}
	for (const element of document.querySelectorAll<HTMLElement>(
		"[data-hide-in-en]",
	)) {
		element.hidden = next === "en" && element.dataset.hideInEn === "true";
	}
	for (const link of document.querySelectorAll<HTMLAnchorElement>(
		"[data-i18n-href-zh], [data-i18n-href-en]",
	)) {
		const target = link.dataset[next === "en" ? "i18nHrefEn" : "i18nHrefZh"];
		if (target) link.href = target;
	}
	const localizedTitle = document.querySelector<HTMLTitleElement>(
		"title[data-locale-title-zh], title[data-locale-title-en]",
	);
	const titleValue =
		localizedTitle?.dataset[next === "en" ? "localeTitleEn" : "localeTitleZh"];
	if (localizedTitle && titleValue) localizedTitle.textContent = titleValue;
	for (const meta of document.querySelectorAll<HTMLMetaElement>(
		'meta[data-locale-title-zh][data-locale-title-en]',
	)) {
		const value = meta.dataset[next === "en" ? "localeTitleEn" : "localeTitleZh"];
		if (value) meta.content = value;
	}
	const description = document.querySelector<HTMLMetaElement>(
		'meta[name="description"][data-locale-description-zh], meta[name="description"][data-locale-description-en]',
	);
	const descriptionValue =
		description?.dataset[
			next === "en" ? "localeDescriptionEn" : "localeDescriptionZh"
		];
	if (description && descriptionValue) {
		description.content = descriptionValue;
	}
	for (const meta of document.querySelectorAll<HTMLMetaElement>(
		'meta[data-locale-description-zh][data-locale-description-en]',
	)) {
		const value =
			meta.dataset[next === "en" ? "localeDescriptionEn" : "localeDescriptionZh"];
		if (value) meta.content = value;
	}
	document.dispatchEvent(
		new CustomEvent("site-locale-change", { detail: next }),
	);
}

onMount(() => {
	applyLocale(readPreferredLocale());
	const currentUrl = new URL(window.location.href);
	if (currentUrl.searchParams.has("locale")) {
		currentUrl.searchParams.delete("locale");
		window.history.replaceState(null, "", currentUrl);
	}
});
</script>

<button
	type="button"
	class="language-switcher"
	aria-label={locale === "en" ? "Switch to Chinese" : "Switch to English"}
	onclick={() => applyLocale(locale === "en" ? "zh-CN" : "en")}
>
	{locale === "en" ? "中文" : "EN"}
</button>

<style>
	.language-switcher{white-space:nowrap;flex-shrink:0;border:1px solid rgba(37,60,52,.2);border-radius:999px;background:transparent;color:var(--brand-accent);padding:6px 9px;font:8px "JetBrains Mono",monospace;letter-spacing:.08em;cursor:pointer;transition:border-color .2s,color .2s}.language-switcher:hover,.language-switcher:focus-visible{border-color:var(--brand-accent);color:var(--brand-ink)}.language-switcher:focus-visible{outline:2px solid var(--brand-accent);outline-offset:3px}
</style>
