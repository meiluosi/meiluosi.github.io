<script lang="ts">
// A vector overview is available before hydration and never requests the 3D bundle.
let { locale = "zh-CN" }: { locale?: "zh-CN" | "en" } = $props();
</script>

<svg viewBox="45 55 555 350" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
	<title>{locale === "en" ? "A small island with a lamplit desk, a wooden board, and an observation table" : "小岛上的台灯、木棋盘与观察台"}</title>
	<ellipse cx="325" cy="339" rx="246" ry="53" fill="#315844" opacity=".055" />
	<path d="M68 228 91 287 188 335 344 353 485 317 563 263 577 228Z" fill="#315844" />
	<path d="m91 287 97 48 156 18-5-34-146-13Z" fill="#263c35" />
	<path d="m344 353 141-36 78-54-20 7-91 29-113 20Z" fill="#41634f" />
	<ellipse cx="322" cy="223" rx="255" ry="104" fill="#9dbda6" />
	<ellipse cx="322" cy="217" rx="250" ry="100" fill="#d8ddc8" />
	<ellipse cx="322" cy="217" rx="239" ry="92" stroke="#f3ead6" stroke-width="1.5" />
	<path d="M179 192 Q318 116 466 190 Q433 267 339 271 Q215 259 179 192Z" stroke="#f3f6f1" stroke-width="4" />
	<path d="M233 170 245 170 240 177M436 218l-2 10-12-2M271 259l-10-8 12-2" stroke="#a57758" stroke-width="2" stroke-linecap="round" />
	<g transform="translate(168 164)">
		<ellipse cy="41" rx="65" ry="24" fill="#315844" opacity=".12" />
		<path d="M-54 20v14c0 25 108 25 108 0V20" fill="#e2ddcf" />
		<ellipse cy="20" rx="54" ry="21" fill="#f0eadb" />
		<path d="m-42 2 51 13L47-4-4-17Z" fill="#b58c61" />
		<path d="m-42 2 51 13v8L-42 10Zm51 13L47-4v8L9 23Z" fill="#8c684c" />
		{#each [0, 1, 2, 3] as level}
			<g transform={`translate(0 ${-level * 15})`}>
				<path d="m-19-5 28 7 24-12-28-7Z" fill="#a9c3a7" />
				<path d="m-19-5 28 7v9l-28-7ZM9 2l24-12v9L9 11Z" fill="#41634f" />
			</g>
		{/each}
		<path d="m-35-7 6 2v10l-6-2Zm8-5 6 2V0l-6-2Z" fill="#f0eadb" />
		<path d="m28 8 5-2v-13l-5 2Zm9-5 5-2v-21l-5 2Z" fill="#eeb28c" />
		<path d="m-34 8 23 6 14-7-23-6Z" fill="#315844" /><path d="m-32 6 22 6 13-7-22-6Z" fill="#f3ead6" />
		<path d="m-24 6 14 4m-19-1 13 4m-21-4 3-10" stroke="#9e8058" stroke-width="1.3" stroke-linecap="round" />
		<ellipse cx="-31" cy="-2" rx="9" ry="3" fill="#8c684c" /><path d="M-31-2v-40l17-10" stroke="#9e8058" stroke-width="3" stroke-linecap="round" />
		<path d="m-14-56-12 14h23Z" fill="#9e8058" /><ellipse cx="-14" cy="-42" rx="10" ry="2.5" fill="#ffe6ad" />
		<ellipse cx="-12" cy="-1" rx="18" ry="5" fill="#ffe6ad" opacity=".2" />
	</g>
	<g transform="translate(473 152)">
		<ellipse cy="49" rx="63" ry="22" fill="#315844" opacity=".12" />
		<path d="M-53 30v13c0 24 106 24 106 0V30" fill="#e2ddcf" />
		<ellipse cy="30" rx="53" ry="21" fill="#f0eadb" />
		<path d="m-41 20 58 13L43 9-58-14Z" fill="#c8a174" />
		<path d="m-39 15 65 14m-58-21 65 14m-57-21 66 14M-28 17l23-22m-7 26L12 0M4 25l24-22" stroke="#41634f" stroke-width="1.2" />
		<ellipse cx="-11" cy="16" rx="4" ry="2.6" fill="#263c35" /><ellipse cx="6" cy="13" rx="4" ry="2.6" fill="#f3f6f1" />
		<path d="m-8-65-23 32-17 29m17-29 14 25M-8-65l24 32L1-7m15-26L34-2" stroke="#9e8058" stroke-width="3" />
		<g fill="#41634f"><circle cx="-31" cy="-33" r="5" /><circle cx="-48" cy="-4" r="4" /><circle cx="-17" cy="-8" r="4" /><circle cx="1" cy="-7" r="4" /></g>
		<g fill="#eeb28c" stroke="#a57758" stroke-width="1"><circle cx="-8" cy="-65" r="7" /><circle cx="16" cy="-33" r="5" /><circle cx="34" cy="-2" r="4" /></g>
		<ellipse cx="-42" cy="34" rx="10" ry="5" fill="#8c684c" /><ellipse cx="-42" cy="32" rx="8" ry="3" fill="#f3ead6" />
	</g>
	<g transform="translate(334 254)">
		<ellipse cy="32" rx="66" ry="24" fill="#315844" opacity=".12" />
		<path d="M-52 17v14c0 24 104 24 104 0V17" fill="#e2ddcf" />
		<ellipse cy="17" rx="52" ry="21" fill="#f0eadb" />
		<path d="m-39 10 66 13 22-12-66-13Z" fill="#b58c61" /><path d="m-31 9 52 10 18-10-53-10Z" fill="#f3ead6" />
		<path d="m-8 6 25 5m-29-1 21 4M-29 6l3 12" stroke="#9e8058" stroke-width="1.4" stroke-linecap="round" />
		<path d="m-3-72-29 50 66 6-37-56Z" stroke="#587989" stroke-width="4" stroke-linejoin="round" />
		<path d="m-29-37-3 15 14-6m35 15 17-3-11-12m-4-13 9 2-1-9" fill="#587989" />
		<circle cx="-3" cy="-72" r="11" fill="#587989" /><circle cx="-32" cy="-22" r="11" fill="#eeb28c" /><circle cx="34" cy="-16" r="11" fill="#587989" />
		<ellipse cx="-32" cy="-22" rx="17" ry="20" stroke="#a57758" stroke-width="2" transform="rotate(25 -32 -22)" />
	</g>
	<g transform="translate(306 177)">
		<ellipse cy="16" rx="25" ry="10" fill="#315844" opacity=".1" />
		<path d="m-23 0 25 9 24-13-24-7Z" fill="#f3c7ac" /><path d="m-23 0 25 9v11l-25-9ZM2 9 26-4V7L2 20Z" fill="#c89573" />
		<path d="m-17-12 21 6 14-8-21-6Z" fill="#41634f" /><path d="m-17-12 21 6V8l-21-6ZM4-6l14-8v14L4 8Z" fill="#315844" />
		<path d="m7-4 8-5v8L7 4Z" fill="#f0eadb" />
		<g fill="#263c35"><ellipse cx="-16" cy="11" rx="5" ry="6" /><ellipse cx="10" cy="14" rx="5" ry="6" /><ellipse cx="23" cy="7" rx="4" ry="5" /></g>
		<g fill="#f0eadb"><ellipse cx="-16" cy="11" rx="2" ry="3" /><ellipse cx="10" cy="14" rx="2" ry="3" /></g>
	</g>
</svg>

<style>
	svg { display: block; width: 100%; max-width: 680px; height: auto; overflow: visible; }
</style>
