(function () {
    if (!window.AIAssistantConfig || window.AIAssistantConfig.enabled === false) {
        return;
    }

    const config = Object.assign({
        endpoint: "",
        model: "gpt-4o-mini",
        temperature: 0.2,
        placeholder: "向 AI 助手提问",
        intro: "您好，我是博客 AI 助手。",
        require_api_key: false,
        persist_conversation: false,
        default_examples: []
    }, window.AIAssistantConfig);

    function byId(id) {
        return document.getElementById(id);
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function formatMessage(text) {
        const escaped = escapeHtml(text);
        const withCodeBlocks = escaped.replace(/```([\s\S]*?)```/g, function (_, code) {
            return '<pre><code>' + code.replace(/\n/g, "<br>") + '</code></pre>';
        });
        return withCodeBlocks.replace(/\n/g, "<br>");
    }

    function appendMessage(container, role, text) {
        const message = document.createElement("div");
        message.className = "ai-assistant-message " + role;
        message.innerHTML = formatMessage(text.trim());
        container.appendChild(message);
        container.scrollTop = container.scrollHeight;
    }

    function loadStoredKey() {
        try {
            return window.localStorage.getItem("ai-assistant-api-key") || "";
        } catch (err) {
            return "";
        }
    }

    function storeKey(value) {
        try {
            if (!value) {
                window.localStorage.removeItem("ai-assistant-api-key");
            } else {
                window.localStorage.setItem("ai-assistant-api-key", value);
            }
        } catch (err) {
            console.warn("AI assistant: unable to persist API key", err);
        }
    }

    function restoreConversation() {
        if (!config.persist_conversation) {
            return [];
        }
        try {
            const raw = window.sessionStorage.getItem("ai-assistant-conversation");
            return raw ? JSON.parse(raw) : [];
        } catch (err) {
            return [];
        }
    }

    function persistConversation(records) {
        if (!config.persist_conversation) {
            return;
        }
        try {
            window.sessionStorage.setItem("ai-assistant-conversation", JSON.stringify(records || []));
        } catch (err) {
            console.warn("AI assistant: unable to persist conversation", err);
        }
    }

    document.addEventListener("DOMContentLoaded", function () {
        const panel = byId("ai-assistant-panel");
        const toggle = byId("ai-assistant-toggle");
        const closeBtn = byId("ai-assistant-close");
        const textarea = byId("ai-assistant-input");
        const sendBtn = byId("ai-assistant-send");
        const status = byId("ai-assistant-status");
        const messages = byId("ai-assistant-messages");
        const hintsContainer = byId("ai-assistant-hints");
        const keyWrapper = byId("ai-assistant-api-wrapper");
        const keyInput = byId("ai-assistant-api-key");

        if (!panel || !toggle || !textarea || !sendBtn || !messages) {
            console.warn("AI assistant: markup missing, skip initialization.");
            return;
        }

        textarea.placeholder = config.placeholder;
        if (config.intro) {
            appendMessage(messages, "ai", config.intro);
        }

        const conversation = restoreConversation();
        if (conversation.length) {
            conversation.forEach(function (item) {
                appendMessage(messages, item.role === "user" ? "user" : "ai", item.content);
            });
        }

        if (config.default_examples && config.default_examples.length && hintsContainer) {
            config.default_examples.forEach(function (example) {
                const button = document.createElement("button");
                button.type = "button";
                button.textContent = example;
                button.addEventListener("click", function () {
                    textarea.value = example;
                    textarea.focus();
                });
                hintsContainer.appendChild(button);
            });
        }

        if (config.require_api_key && keyWrapper && keyInput) {
            keyWrapper.style.display = "flex";
            const stored = loadStoredKey();
            if (stored) {
                keyInput.value = stored;
            }
            keyInput.addEventListener("change", function (event) {
                storeKey(event.target.value.trim());
            });
        } else if (keyWrapper) {
            keyWrapper.style.display = "none";
        }

        function togglePanel(open) {
            if (open === undefined) {
                panel.classList.toggle("open");
            } else if (open) {
                panel.classList.add("open");
            } else {
                panel.classList.remove("open");
            }
            if (panel.classList.contains("open")) {
                textarea.focus();
            }
        }

        toggle.addEventListener("click", function () {
            togglePanel();
        });

        if (closeBtn) {
            closeBtn.addEventListener("click", function () {
                togglePanel(false);
            });
        }

        function setStatus(text) {
            if (status) {
                status.textContent = text || "";
            }
        }

        async function requestAnswer(prompt) {
            if (!config.endpoint) {
                throw new Error("尚未配置 AI 接口，请在 _config.yml 中设置 ai_assistant.endpoint。");
            }

            const payload = {
                model: config.model,
                temperature: config.temperature,
                prompt: prompt,
                conversation: conversation
            };

            const headers = {
                "Content-Type": "application/json"
            };

            if (config.require_api_key && keyInput) {
                const value = keyInput.value.trim();
                if (!value) {
                    throw new Error("请先填写可用的 API Key。");
                }
                headers["Authorization"] = "Bearer " + value;
            }

            const response = await fetch(config.endpoint, {
                method: "POST",
                headers: headers,
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error("调用 AI 接口失败 (" + response.status + ")");
            }

            const data = await response.json();
            if (!data) {
                throw new Error("接口返回为空，请检查服务端响应。");
            }

            if (data.answer) {
                return data.answer;
            }

            if (data.choices && data.choices.length) {
                return data.choices[0].message?.content || "";
            }

            if (data.output) {
                return data.output;
            }

            return JSON.stringify(data, null, 2);
        }

        async function handleSend() {
            const text = textarea.value.trim();
            if (!text) {
                return;
            }

            appendMessage(messages, "user", text);
            conversation.push({ role: "user", content: text });
            persistConversation(conversation);
            textarea.value = "";
            textarea.focus();
            sendBtn.disabled = true;
            setStatus("正在思考...");

            try {
                const answer = await requestAnswer(text);
                const cleanAnswer = answer || "(未获取到内容)";
                appendMessage(messages, "ai", cleanAnswer);
                conversation.push({ role: "assistant", content: cleanAnswer });
                persistConversation(conversation);
                setStatus("");
            } catch (error) {
                console.error("AI assistant error:", error);
                appendMessage(messages, "ai", "⚠️ " + (error.message || "AI 接口调用失败"));
                setStatus("调用失败，请稍后重试");
            } finally {
                sendBtn.disabled = false;
            }
        }

        sendBtn.addEventListener("click", handleSend);
        textarea.addEventListener("keydown", function (event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                handleSend();
            }
        });
    });
})();
