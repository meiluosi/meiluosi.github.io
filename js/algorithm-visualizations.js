(function () {
    document.addEventListener("DOMContentLoaded", function () {
        initRLChart();
        initDUCGStepper();
        initNeuralNetworkExplorer();
    });

    /* ------------------ 强化学习训练曲线 ------------------ */
    function initRLChart() {
        const canvas = document.getElementById("rl-training-chart");
        const playBtn = document.getElementById("rl-play");
        const episodeLabel = document.getElementById("rl-episode");

        if (!canvas || !playBtn) {
            return;
        }

        const maxEpisode = 200;
        const labels = Array.from({ length: maxEpisode + 1 }, (_, idx) => idx);
        const baseline = labels.map((ep) => {
            const progress = ep / maxEpisode;
            const value = 40 + 60 / (1 + Math.exp(-8 * (progress - 0.5)));
            const noise = (Math.random() - 0.5) * 5 * (1 - Math.min(progress * 1.2, 1));
            return Math.max(0, Math.min(100, value + noise));
        });

        const chart = new Chart(canvas.getContext("2d"), {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "平均回报",
                        data: new Array(labels.length).fill(null),
                        borderColor: "#f5576c",
                        borderWidth: 3,
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                animation: false,
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "训练迭代"
                        }
                    },
                    y: {
                        suggestedMin: 0,
                        suggestedMax: 110,
                        title: {
                            display: true,
                            text: "平均回报"
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });

        let timer = null;
        let currentEpisode = 0;

        function resetChart() {
            chart.data.datasets[0].data = new Array(labels.length).fill(null);
            chart.update();
            currentEpisode = 0;
            updateEpisodeLabel();
            playBtn.textContent = "播放动画";
        }

        function updateEpisodeLabel() {
            if (episodeLabel) {
                episodeLabel.textContent = "迭代：" + currentEpisode + " / " + maxEpisode;
            }
        }

        function step() {
            if (currentEpisode > maxEpisode) {
                clearInterval(timer);
                timer = null;
                playBtn.textContent = "重新播放";
                return;
            }
            chart.data.datasets[0].data[currentEpisode] = baseline[currentEpisode];
            chart.update("none");
            updateEpisodeLabel();
            currentEpisode += 2;
        }

        playBtn.addEventListener("click", function () {
            if (timer) {
                clearInterval(timer);
                timer = null;
                playBtn.textContent = "继续";
                return;
            }

            if (currentEpisode > maxEpisode) {
                resetChart();
            }

            playBtn.textContent = "暂停";
            timer = setInterval(step, 200);
        });

        resetChart();
    }

    /* ------------------ DUCG 推理流程 ------------------ */
    function initDUCGStepper() {
        const container = document.getElementById("ducg-step");
        const prevBtn = document.getElementById("ducg-prev");
        const nextBtn = document.getElementById("ducg-next");

        if (!container || !prevBtn || !nextBtn) {
            return;
        }

        const steps = [
            {
                title: "证据建模",
                description: "整理案件事实与证据节点，构建包含 62 个事件变量的 DUCG 图，设置证据置信度区间。",
                confidence: 0.62
            },
            {
                title: "因果推演",
                description: "通过证据依赖关系向上推理，更新被告行为节点的后验概率，并对关键节点进行敏感度分析。",
                confidence: 0.74
            },
            {
                title: "法律条款映射",
                description: "将推理结果映射到相应法律条款，评估行为构成要件，筛选满足置信阈值的条款。",
                confidence: 0.81
            },
            {
                title: "判决建议生成",
                description: "综合证据冲突与置信度，生成判决建议，并附带可以支撑判决的证据链可视化。",
                confidence: 0.87
            }
        ];

        let index = 0;

        function renderStep() {
            const step = steps[index];
            const progressPercent = Math.round((index + 1) / steps.length * 100);
            container.innerHTML = `
                <h4 style="margin-top:0;">阶段 ${index + 1} · ${step.title}</h4>
                <p style="line-height:1.7; margin-bottom:12px;">${step.description}</p>
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="flex:1; height:10px; background:rgba(185, 80, 214, 0.15); border-radius:999px; overflow:hidden;">
                        <div style="width:${progressPercent}%; height:100%; background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);"></div>
                    </div>
                    <span style="font-weight:600; color:#b950d6;">完成度 ${progressPercent}%</span>
                </div>
                <p style="margin:12px 0 0; color:#b950d6; font-weight:600;">当前置信度：${Math.round(step.confidence * 100)}%</p>
            `;

            prevBtn.disabled = index === 0;
            nextBtn.disabled = index === steps.length - 1;
        }

        prevBtn.addEventListener("click", function () {
            if (index > 0) {
                index -= 1;
                renderStep();
            }
        });

        nextBtn.addEventListener("click", function () {
            if (index < steps.length - 1) {
                index += 1;
                renderStep();
            }
        });

        renderStep();
    }

    /* ------------------ 神经网络结构探索 ------------------ */
    function initNeuralNetworkExplorer() {
        const depthSlider = document.getElementById("nn-depth");
        const widthSlider = document.getElementById("nn-width");
        const depthLabel = document.getElementById("nn-depth-label");
        const widthLabel = document.getElementById("nn-width-label");
        const container = document.getElementById("nn-visual");
        const summary = document.getElementById("nn-summary");

        if (!depthSlider || !widthSlider || !container || !summary) {
            return;
        }

        const inputNodes = 6;
        const outputNodes = 2;

        function renderNetwork() {
            const hiddenLayers = parseInt(depthSlider.value, 10);
            const hiddenWidth = parseInt(widthSlider.value, 10);
            depthLabel.textContent = hiddenLayers;
            widthLabel.textContent = hiddenWidth;

            const totalLayers = hiddenLayers + 2;
            const layerSizes = [inputNodes];
            for (let i = 0; i < hiddenLayers; i += 1) {
                layerSizes.push(hiddenWidth);
            }
            layerSizes.push(outputNodes);

            container.innerHTML = "";
            const svgNS = "http://www.w3.org/2000/svg";
            const svg = document.createElementNS(svgNS, "svg");
            svg.setAttribute("width", "100%");
            svg.setAttribute("height", "100%");
            svg.setAttribute("viewBox", "0 0 1000 600");
            svg.classList.add("nn-svg");
            container.appendChild(svg);

            const nodePositions = [];
            layerSizes.forEach(function (count, layerIndex) {
                const x = (layerIndex / (layerSizes.length - 1)) * 1000;
                for (let i = 0; i < count; i += 1) {
                    const y = ((i + 1) / (count + 1)) * 600;
                    nodePositions.push({ layer: layerIndex, index: i, x: x, y: y });
                    const node = document.createElement("div");
                    node.className = "nn-node";
                    node.style.position = "absolute";
                    node.style.left = (x / 10) + "%";
                    node.style.top = (y / 6) + "%";
                    node.style.transform = "translate(-50%, -50%)";
                    node.title = "Layer " + layerIndex + " · Node " + (i + 1);
                    container.appendChild(node);
                }
            });

            for (let layer = 0; layer < layerSizes.length - 1; layer += 1) {
                const fromNodes = nodePositions.filter(function (n) { return n.layer === layer; });
                const toNodes = nodePositions.filter(function (n) { return n.layer === layer + 1; });
                fromNodes.forEach(function (from) {
                    toNodes.forEach(function (to) {
                        const line = document.createElementNS(svgNS, "line");
                        line.setAttribute("x1", from.x);
                        line.setAttribute("y1", from.y);
                        line.setAttribute("x2", to.x);
                        line.setAttribute("y2", to.y);
                        line.setAttribute("stroke", "rgba(245, 87, 108, 0.3)");
                        line.setAttribute("stroke-width", "4");
                        svg.appendChild(line);
                    });
                });
            }

            const paramCount = layerSizes.slice(1).reduce(function (acc, curr, idx) {
                const prev = layerSizes[idx];
                return acc + curr * prev + curr; // 权重 + 偏置
            }, 0);

            summary.innerHTML = `
                <strong>结构摘要：</strong><br>
                输入层：${inputNodes} 个节点 · 输出层：${outputNodes} 个节点<br>
                隐藏层：${hiddenLayers} 层，每层 ${hiddenWidth} 个神经元<br>
                估计参数量：${paramCount.toLocaleString()}
            `;
        }

        depthSlider.addEventListener("input", renderNetwork);
        widthSlider.addEventListener("input", renderNetwork);
        window.addEventListener("resize", renderNetwork);
        renderNetwork();
    }
})();
