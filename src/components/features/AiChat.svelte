<script lang="ts">
	import { onMount } from "svelte";
	import { aiChatConfig } from "@/config/aiChatConfig";

	interface Message {
		role: "user" | "assistant" | "system";
		content: string;
	}

	interface Props {
		class?: string;
	}

	let { class: className = "" }: Props = $props();

	// 状态
	let isOpen = $state(false);
	let messages = $state<Message[]>([]);
	let inputText = $state("");
	let isLoading = $state(false);
	let chatContainer: HTMLDivElement | undefined = $state();
	let inputRef: HTMLInputElement | undefined = $state();

	// 初始化欢迎消息
	onMount(() => {
		messages = [
			{
				role: "assistant",
				content: aiChatConfig.welcomeMessage,
			},
		];
	});

	function toggleChat() {
		isOpen = !isOpen;
		if (isOpen) {
			// 打开后聚焦输入框
			setTimeout(() => inputRef?.focus(), 100);
		}
	}

	function scrollToBottom() {
		if (chatContainer) {
			setTimeout(() => {
				chatContainer!.scrollTop = chatContainer!.scrollHeight;
			}, 50);
		}
	}

	async function sendMessage() {
		const text = inputText.trim();
		if (!text || isLoading) return;

		// 添加用户消息
		messages = [...messages, { role: "user", content: text }];
		inputText = "";
		isLoading = true;
		scrollToBottom();

		// 构造请求
		const apiMessages = [
			{ role: "system", content: aiChatConfig.systemPrompt },
			...messages,
		];

		try {
			const response = await fetch(aiChatConfig.apiEndpoint, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					model: aiChatConfig.model,
					messages: apiMessages,
					max_tokens: aiChatConfig.maxTokens,
					temperature: aiChatConfig.temperature,
					stream: false,
				}),
			});

			if (!response.ok) {
				const err = await response.json().catch(() => ({}));
				throw new Error(
					(err as any).error || `请求失败 (${response.status})`,
				);
			}

			const data = await response.json();
			const reply = data.choices?.[0]?.message?.content || "抱歉，我没有理解你的问题...";

			messages = [...messages, { role: "assistant", content: reply }];
		} catch (e: any) {
			messages = [
				...messages,
				{
					role: "assistant",
					content: `😅 抱歉，出了点问题：${e.message || "网络错误"}`,
				},
			];
		} finally {
			isLoading = false;
			scrollToBottom();
		}
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			sendMessage();
		}
	}

	// 快捷问题
	const quickQuestions = [
		"介绍一下枫语博客",
		"什么是强化学习？",
		"推荐一篇因果推断的文章",
		"LLM 是怎么训练的？",
	];
</script>

{#if aiChatConfig.enable}
	<!-- 触发按钮 -->
	<button
		class="ai-chat-trigger {className}"
		class:active={isOpen}
		onclick={toggleChat}
		title={aiChatConfig.triggerHint}
		aria-label={isOpen ? "关闭对话" : "打开对话"}
	>
		{#if isOpen}
			<svg
				xmlns="http://www.w3.org/2000/svg"
				width="24"
				height="24"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				stroke-width="2"
				stroke-linecap="round"
				stroke-linejoin="round"
			>
				<line x1="18" y1="6" x2="6" y2="18"></line>
				<line x1="6" y1="6" x2="18" y2="18"></line>
			</svg>
		{:else}
			<svg
				xmlns="http://www.w3.org/2000/svg"
				width="24"
				height="24"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				stroke-width="2"
				stroke-linecap="round"
				stroke-linejoin="round"
			>
				<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
				<line x1="9" y1="10" x2="15" y2="10"></line>
				<line x1="12" y1="7" x2="12" y2="13"></line>
			</svg>
		{/if}
	</button>

	<!-- 对话窗口 -->
	{#if isOpen}
		<div class="ai-chat-panel">
			<!-- 标题栏 -->
			<div class="ai-chat-header">
				<span class="ai-chat-title">{aiChatConfig.title}</span>
				<button
					class="ai-chat-close"
					onclick={toggleChat}
					aria-label="关闭对话"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="18"
						height="18"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
					>
						<line x1="18" y1="6" x2="6" y2="18"></line>
						<line x1="6" y1="6" x2="18" y2="18"></line>
					</svg>
				</button>
			</div>

			<!-- 消息列表 -->
			<div class="ai-chat-messages" bind:this={chatContainer}>
				{#each messages as msg}
					<div
						class="ai-chat-message"
						class:user={msg.role === "user"}
						class:assistant={msg.role === "assistant"}
					>
						<div class="ai-chat-bubble">
							{msg.content}
						</div>
					</div>
				{/each}

				{#if isLoading}
					<div class="ai-chat-message assistant">
						<div class="ai-chat-bubble ai-chat-typing">
							<span class="dot"></span>
							<span class="dot"></span>
							<span class="dot"></span>
						</div>
					</div>
				{/if}

				<!-- 快捷问题（仅首次对话时显示） -->
				{#if messages.length <= 1 && !isLoading}
					<div class="ai-chat-quick">
						{#each quickQuestions as q}
							<button
								class="ai-chat-quick-btn"
								onclick={() => {
									inputText = q;
									sendMessage();
								}}
							>
								{q}
							</button>
						{/each}
					</div>
				{/if}
			</div>

			<!-- 输入框 -->
			<div class="ai-chat-input-area">
				<input
					type="text"
					bind:value={inputText}
					bind:this={inputRef}
					placeholder={aiChatConfig.placeholder}
					onkeydown={handleKeyDown}
					disabled={isLoading}
					class="ai-chat-input"
				/>
				<button
					class="ai-chat-send"
					onclick={sendMessage}
					disabled={isLoading || !inputText.trim()}
					aria-label="发送"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="18"
						height="18"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
					>
						<line x1="22" y1="2" x2="11" y2="13"></line>
						<polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
					</svg>
				</button>
			</div>
		</div>
	{/if}
{/if}

<style>
	/* 触发按钮 */
	.ai-chat-trigger {
		position: fixed;
		bottom: 24px;
		right: 24px;
		z-index: 999;
		width: 52px;
		height: 52px;
		border-radius: 50%;
		border: none;
		background: var(--theme-color, hsl(165, 60%, 50%));
		color: #fff;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
		transition: all 0.3s ease;
	}

	.ai-chat-trigger:hover {
		transform: scale(1.1);
		box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
	}

	.ai-chat-trigger.active {
		background: var(--theme-color, hsl(165, 60%, 50%));
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
	}

	/* 对话面板 */
	.ai-chat-panel {
		position: fixed;
		bottom: 88px;
		right: 24px;
		z-index: 998;
		width: 380px;
		max-width: calc(100vw - 48px);
		height: 520px;
		max-height: calc(100vh - 140px);
		background: var(--card-bg, #fff);
		border: 1px solid var(--card-border, #e5e7eb);
		border-radius: 16px;
		box-shadow: 0 8px 40px rgba(0, 0, 0, 0.15);
		display: flex;
		flex-direction: column;
		overflow: hidden;
		animation: aiChatSlideUp 0.3s ease;
	}

	:global(.dark) .ai-chat-panel {
		background: var(--card-bg, #1f2937);
		border-color: var(--card-border, #374151);
		box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
	}

	@keyframes aiChatSlideUp {
		from {
			opacity: 0;
			transform: translateY(16px) scale(0.95);
		}
		to {
			opacity: 1;
			transform: translateY(0) scale(1);
		}
	}

	/* 标题栏 */
	.ai-chat-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 12px 16px;
		border-bottom: 1px solid var(--card-border, #e5e7eb);
		background: var(--theme-color, hsl(165, 60%, 50%));
		color: #fff;
		flex-shrink: 0;
	}

	:global(.dark) .ai-chat-header {
		border-bottom-color: rgba(255, 255, 255, 0.1);
	}

	.ai-chat-title {
		font-size: 14px;
		font-weight: 600;
	}

	.ai-chat-close {
		background: none;
		border: none;
		color: #fff;
		cursor: pointer;
		opacity: 0.8;
		padding: 4px;
		border-radius: 6px;
		transition: opacity 0.2s;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.ai-chat-close:hover {
		opacity: 1;
	}

	/* 消息列表 */
	.ai-chat-messages {
		flex: 1;
		overflow-y: auto;
		padding: 16px;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.ai-chat-message {
		display: flex;
		max-width: 85%;
	}

	.ai-chat-message.user {
		align-self: flex-end;
	}

	.ai-chat-message.assistant {
		align-self: flex-start;
	}

	.ai-chat-bubble {
		padding: 10px 14px;
		border-radius: 14px;
		font-size: 13.5px;
		line-height: 1.6;
		word-break: break-word;
		white-space: pre-wrap;
	}

	.user .ai-chat-bubble {
		background: var(--theme-color, hsl(165, 60%, 50%));
		color: #fff;
		border-bottom-right-radius: 4px;
	}

	.assistant .ai-chat-bubble {
		background: var(--bubble-bg, #f3f4f6);
		color: var(--text-color, #374151);
		border-bottom-left-radius: 4px;
	}

	:global(.dark) .assistant .ai-chat-bubble {
		background: var(--bubble-bg, #374151);
		color: var(--text-color, #e5e7eb);
	}

	/* 打字动画 */
	.ai-chat-typing {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 14px 18px;
	}

	.ai-chat-typing .dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--text-color, #9ca3af);
		animation: aiChatBounce 1.4s infinite ease-in-out both;
	}

	.ai-chat-typing .dot:nth-child(1) {
		animation-delay: -0.32s;
	}
	.ai-chat-typing .dot:nth-child(2) {
		animation-delay: -0.16s;
	}

	@keyframes aiChatBounce {
		0%,
		80%,
		100% {
			transform: scale(0.6);
		}
		40% {
			transform: scale(1);
		}
	}

	/* 快捷问题 */
	.ai-chat-quick {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		margin-top: 8px;
	}

	.ai-chat-quick-btn {
		font-size: 12px;
		padding: 6px 12px;
		border-radius: 20px;
		border: 1px solid var(--theme-color, hsl(165, 60%, 50%));
		background: transparent;
		color: var(--theme-color, hsl(165, 60%, 50%));
		cursor: pointer;
		transition: all 0.2s;
		white-space: nowrap;
	}

	.ai-chat-quick-btn:hover {
		background: var(--theme-color, hsl(165, 60%, 50%));
		color: #fff;
	}

	/* 输入区域 */
	.ai-chat-input-area {
		display: flex;
		align-items: center;
		padding: 12px 16px;
		border-top: 1px solid var(--card-border, #e5e7eb);
		gap: 8px;
		flex-shrink: 0;
	}

	:global(.dark) .ai-chat-input-area {
		border-top-color: rgba(255, 255, 255, 0.08);
	}

	.ai-chat-input {
		flex: 1;
		padding: 10px 14px;
		border: 1px solid var(--card-border, #e5e7eb);
		border-radius: 24px;
		font-size: 13.5px;
		outline: none;
		background: var(--input-bg, #f9fafb);
		color: var(--text-color, #374151);
		transition: border-color 0.2s;
	}

	.ai-chat-input:focus {
		border-color: var(--theme-color, hsl(165, 60%, 50%));
	}

	:global(.dark) .ai-chat-input {
		background: var(--input-bg, #374151);
		color: var(--text-color, #e5e7eb);
		border-color: rgba(255, 255, 255, 0.1);
	}

	.ai-chat-send {
		width: 36px;
		height: 36px;
		border-radius: 50%;
		border: none;
		background: var(--theme-color, hsl(165, 60%, 50%));
		color: #fff;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		transition: all 0.2s;
		flex-shrink: 0;
	}

	.ai-chat-send:hover:not(:disabled) {
		transform: scale(1.1);
	}

	.ai-chat-send:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	/* 响应式 */
	@media (max-width: 640px) {
		.ai-chat-trigger {
			bottom: 16px;
			right: 16px;
			width: 44px;
			height: 44px;
		}

		.ai-chat-panel {
			bottom: 72px;
			right: 8px;
			left: 8px;
			width: auto;
			height: 480px;
			max-height: calc(100vh - 120px);
		}
	}
</style>