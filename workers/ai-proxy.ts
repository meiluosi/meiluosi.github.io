/**
 * Cloudflare Worker — AI API 代理
 *
 * 功能：
 * 1. 代理 OpenAI 兼容的 Chat Completions API 请求
 * 2. 按 IP 限流，防止滥用
 * 3. 隐藏 API Key，保证安全
 *
 * 部署步骤：
 * 1. cd workers/ai-proxy
 * 2. pnpm install (如果需要)
 * 3. npx wrangler deploy
 * 4. 将返回的 URL 填入 src/config/aiChatConfig.ts 的 apiEndpoint
 *
 * 或者直接在 Cloudflare Dashboard 中创建 Worker，粘贴此文件内容。
 */

// ========== 配置区 ==========
// 在这里设置你的 API 信息（部署后这些值会作为 Worker 环境变量/secret）

// 上游 API 地址（OpenAI 兼容格式）
const UPSTREAM_API = "https://api.deepseek.com/v1/chat/completions";
// API Key — 建议通过 wrangler secret put API_KEY 设置
// 如果直接部署，修改这里的值（不推荐，建议用 secret）
const API_KEY = "YOUR_API_KEY_HERE";

// 限流配置
const RATE_LIMIT = {
	windowMs: 60 * 60 * 1000, // 时间窗口：1 小时
	maxRequests: 30, // 每窗口最大请求数
};

// 允许的来源（你的博客域名）
const ALLOWED_ORIGINS = [
	"https://meiluoshi.github.io",
	"http://localhost:4321",
	"http://localhost:3000",
];

// ========== 限流存储 (内存) ==========
// 注意：Worker 内存不持久，重启后清零。如需持久化，使用 KV 或 D1
const rateLimitMap = new Map<string, { count: number; resetAt: number }>();

function isRateLimited(ip: string): boolean {
	const now = Date.now();
	const record = rateLimitMap.get(ip);

	if (!record || now > record.resetAt) {
		rateLimitMap.set(ip, { count: 1, resetAt: now + RATE_LIMIT.windowMs });
		return false;
	}

	if (record.count >= RATE_LIMIT.maxRequests) {
		return true;
	}

	record.count++;
	return false;
}

// ========== CORS 处理 ==========
function corsHeaders(origin: string): Record<string, string> {
	const isAllowed = ALLOWED_ORIGINS.includes(origin) || ALLOWED_ORIGINS.includes("*");
	return {
		"Access-Control-Allow-Origin": isAllowed ? origin : ALLOWED_ORIGINS[0],
		"Access-Control-Allow-Methods": "POST, OPTIONS",
		"Access-Control-Allow-Headers": "Content-Type, Authorization",
		"Access-Control-Max-Age": "86400",
	};
}

// ========== Worker 入口 ==========
export default {
	async fetch(request: Request): Promise<Response> {
		const origin = request.headers.get("Origin") || "";
		const headers = corsHeaders(origin);

		// 处理 CORS 预检请求
		if (request.method === "OPTIONS") {
			return new Response(null, { status: 204, headers });
		}

		// 只接受 POST 请求
		if (request.method !== "POST") {
			return new Response(JSON.stringify({ error: "Method not allowed" }), {
				status: 405,
				headers: { ...headers, "Content-Type": "application/json" },
			});
		}

		// 限流检查
		const ip = request.headers.get("CF-Connecting-IP") || "unknown";
		if (isRateLimited(ip)) {
			return new Response(
				JSON.stringify({
					error: "请求过于频繁，请稍后再试",
					code: "rate_limited",
				}),
				{
					status: 429,
					headers: { ...headers, "Content-Type": "application/json" },
				},
			);
		}

		// 解析请求
		let body: any;
		try {
			body = await request.json();
		} catch {
			return new Response(JSON.stringify({ error: "Invalid JSON" }), {
				status: 400,
				headers: { ...headers, "Content-Type": "application/json" },
			});
		}

		// 验证必填字段
		if (!body.messages || !Array.isArray(body.messages)) {
			return new Response(
				JSON.stringify({ error: "messages is required" }),
				{
					status: 400,
					headers: { ...headers, "Content-Type": "application/json" },
				},
			);
		}

		// 构造上游请求
		const upstreamBody = {
			model: body.model || "deepseek-chat",
			messages: body.messages,
			max_tokens: Math.min(body.max_tokens || 1024, 4096),
			temperature: body.temperature ?? 0.7,
			stream: body.stream ?? false,
		};

		// 获取 API Key（优先从环境变量/secret）
		// 在 Cloudflare Worker 中，可以通过 env 或全局变量获取
		// @ts-ignore - Worker 环境变量
		const apiKey = (typeof API_KEY_SECRET !== "undefined" ? API_KEY_SECRET : null) || API_KEY;

		// 转发请求到上游 API
		const upstreamResponse = await fetch(UPSTREAM_API, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${apiKey}`,
			},
			body: JSON.stringify(upstreamBody),
		});

		// 流式响应透传
		if (upstreamBody.stream && upstreamResponse.ok) {
			return new Response(upstreamResponse.body, {
				status: upstreamResponse.status,
				headers: {
					...headers,
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
				},
			});
		}

		// 非流式响应
		const responseBody = await upstreamResponse.text();
		return new Response(responseBody, {
			status: upstreamResponse.status,
			headers: {
				...headers,
				"Content-Type": "application/json",
			},
		});
	},
};