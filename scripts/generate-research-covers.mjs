import fs from "node:fs/promises";
import sharp from "sharp";
const dir = new URL("../public/assets/images/brand/", import.meta.url);
const assetPath = (name) => new URL(name, dir);
const panels = [
	[
		"model",
		"FROM DATA TO MODEL",
		"DATA / TOKENIZE / PRETRAIN / ALIGN / EVALUATE / DEPLOY",
		`<path d="M95 110H405M170 110V35H315V110"/><rect x="30" y="68" width="62" height="85" rx="10" fill="#f5f7f0"/><path d="M47 88H74M47 105H70M47 122H66"/><rect x="145" y="55" width="86" height="112" rx="15" fill="#315746"/><g fill="#dce7d9" stroke="none">${[0, 1, 2].flatMap((y) => [0, 1, 2].map((x) => `<circle cx="${164 + x * 24}" cy="${79 + y * 29}" r="5"/>`)).join("")}</g><rect x="280" y="67" width="74" height="88" rx="13" fill="#ecc2a5"/><path d="M299 110l13 14 23-35"/><circle cx="413" cy="110" r="16" fill="#315746"/><circle cx="268" cy="35" r="8" fill="#ecc2a5"/>`,
	],
	[
		"search",
		"SEARCH INTO LEARNING",
		"SELECT / EXPAND / SIMULATE / BACK UP",
		`<rect x="10" y="5" width="195" height="195" rx="10" fill="#f5f7f0"/>${[0, 1, 2, 3, 4, 5, 6].map((i) => `<path d="M28 ${23 + i * 26}H185M${28 + i * 26} 23V180" opacity=".4"/>`).join("")}<g fill="#315746"><circle cx="80" cy="75" r="10"/><circle cx="106" cy="101" r="10"/><circle cx="80" cy="127" r="10"/></g><g fill="#ecc2a5"><circle cx="106" cy="75" r="10"/><circle cx="132" cy="101" r="10"/></g><path d="M320 20L260 95L238 175M320 20L393 95M260 95L306 175"/><path d="M320 20L260 95L306 175" stroke-width="5"/>${[
			[320, 20],
			[260, 95],
			[393, 95],
			[238, 175],
			[306, 175],
		]
			.map(
				([x, y], i) =>
					`<circle cx="${x}" cy="${y}" r="${i === 0 ? 16 : 12}" fill="${i === 4 ? "#ecc2a5" : "#315746"}"/>`,
			)
			.join("")}`,
	],
	[
		"causal",
		"QUESTION THE IMPROVEMENT",
		"ASSIGNMENT / CONFOUNDING / INTERVENTION / EVALUATION",
		`<path d="M220 25L70 165M220 25L370 165M95 165H340"/><circle cx="220" cy="25" r="26" fill="#dce7d9"/><circle cx="70" cy="165" r="35" fill="#315746"/><circle cx="370" cy="165" r="35" fill="#ecc2a5"/><path d="M56 165H84M70 151V179" stroke="#f5f7f0"/><path d="M355 173L367 155L383 167M210 25H230M320 155L339 165L320 175"/>`,
	],
];
for (const [k, title, sub, diagram] of panels) {
	const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630"><rect width="1200" height="630" fill="#f3f6f1"/><rect x="26" y="26" width="1148" height="578" rx="20" fill="none" stroke="#cdd8cc"/><text x="68" y="86" font-family="Arial,sans-serif" font-size="20" letter-spacing="4" fill="#41634f">FENG YU / RESEARCH STUDIO</text><text x="68" y="185" font-family="Arial,sans-serif" font-size="47" fill="#253c34">${title}</text><text x="68" y="228" font-family="Arial,sans-serif" font-size="15" letter-spacing="1.5" fill="#647368">${sub}</text><g transform="translate(640 285) scale(1.1)" stroke="#41634f" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none">${diagram}</g><text x="68" y="470" font-family="Georgia,serif" font-size="65" font-style="italic" fill="#41634f">RSI → AGI</text><text x="68" y="550" font-family="Arial,sans-serif" font-size="16" letter-spacing="2" fill="#647368">EXPLORE / EXPERIMENT / UNDERSTAND</text></svg>`;
	await fs.writeFile(assetPath(`research-${k}.svg`), svg);
	await sharp(Buffer.from(svg))
		.png()
		.toFile(assetPath(`research-${k}.png`).pathname);
}
