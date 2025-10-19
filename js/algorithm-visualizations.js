(function () {
    document.addEventListener("DOMContentLoaded", function () {
        initRLVisualization();
        initDUCGNetwork();
        initNeuralNetworkDesigner();
    });

    /* ------------------ 强化学习训练可视化 ------------------ */
    function initRLVisualization() {
        const canvas = document.getElementById("rl-training-chart");
        const playBtn = document.getElementById("rl-play");
        const resetBtn = document.getElementById("rl-reset");
        const speedSlider = document.getElementById("rl-speed");
        const speedLabel = document.getElementById("rl-speed-label");
        const episodeLabel = document.getElementById("rl-episode");
        const statusLabel = document.getElementById("rl-status");

        if (!canvas || !playBtn || !resetBtn) {
            return;
        }

        const maxEpisode = 500;
        let currentEpisode = 0;
        let timer = null;
        let isRunning = false;

        // 生成多条训练曲线数据
        const datasets = [
            {
                label: "Actor Loss",
                data: [],
                borderColor: "#667eea",
                backgroundColor: "rgba(102, 126, 234, 0.1)",
                borderWidth: 2,
                fill: true
            },
            {
                label: "Critic Loss", 
                data: [],
                borderColor: "#f093fb",
                backgroundColor: "rgba(240, 147, 251, 0.1)",
                borderWidth: 2,
                fill: true
            },
            {
                label: "Average Reward",
                data: [],
                borderColor: "#10b981",
                backgroundColor: "rgba(16, 185, 129, 0.1)",
                borderWidth: 3,
                fill: false
            }
        ];

        const chart = new Chart(canvas.getContext("2d"), {
            type: "line",
            data: {
                labels: Array.from({ length: maxEpisode + 1 }, (_, idx) => idx),
                datasets: datasets
            },
            options: {
                animation: false,
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        title: { display: true, text: "训练迭代次数" },
                        grid: { color: 'rgba(0,0,0,0.05)' }
                    },
                    y: {
                        title: { display: true, text: "Loss / Reward" },
                        grid: { color: 'rgba(0,0,0,0.05)' }
                    }
                },
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        backgroundColor: 'rgba(255,255,255,0.95)',
                        titleColor: '#374151',
                        bodyColor: '#374151',
                        borderColor: '#e5e7eb',
                        borderWidth: 1
                    }
                }
            }
        });

        function generateTrainingData(episode) {
            const progress = episode / maxEpisode;
            
            // Actor Loss: 从高开始逐渐下降
            const actorLoss = Math.max(0.01, (2 - progress * 1.8) + Math.sin(episode * 0.1) * 0.1);
            
            // Critic Loss: 波动下降
            const criticLoss = Math.max(0.01, (1.5 - progress * 1.3) + Math.cos(episode * 0.15) * 0.08);
            
            // Average Reward: S 型上升曲线
            const baseReward = 100 / (1 + Math.exp(-8 * (progress - 0.5)));
            const noise = (Math.random() - 0.5) * 8 * (1 - progress * 0.7);
            const avgReward = Math.max(-50, Math.min(100, baseReward + noise));

            return { actorLoss, criticLoss, avgReward };
        }

        function updateDisplay() {
            episodeLabel.textContent = `迭代：${currentEpisode} / ${maxEpisode}`;
            
            if (currentEpisode > 0) {
                const data = generateTrainingData(currentEpisode);
                document.getElementById("current-reward").textContent = data.avgReward.toFixed(1);
                
                const avgData = datasets[2].data.slice(-50);
                const avgReward = avgData.length > 0 ? 
                    avgData.reduce((a, b) => a + b, 0) / avgData.length : 0;
                document.getElementById("average-reward").textContent = avgReward.toFixed(1);
                
                const maxReward = Math.max(...datasets[2].data);
                document.getElementById("max-reward").textContent = maxReward.toFixed(1);
                
                // 动态学习率
                const learningRate = 0.001 * Math.exp(-currentEpisode / 200);
                document.getElementById("learning-rate").textContent = learningRate.toExponential(3);
            }
        }

        function step() {
            if (currentEpisode >= maxEpisode) {
                stopTraining();
                statusLabel.textContent = "训练完成";
                playBtn.textContent = "训练完成";
                return;
            }

            const data = generateTrainingData(currentEpisode);
            datasets[0].data[currentEpisode] = data.actorLoss;
            datasets[1].data[currentEpisode] = data.criticLoss;
            datasets[2].data[currentEpisode] = data.avgReward;

            chart.update("none");
            updateDisplay();
            currentEpisode++;
        }

        function startTraining() {
            if (isRunning) return;
            
            isRunning = true;
            statusLabel.textContent = "训练中...";
            playBtn.textContent = "暂停训练";
            
            const speed = parseInt(speedSlider.value);
            const interval = Math.max(50, 500 - speed * 45);
            
            timer = setInterval(step, interval);
        }

        function stopTraining() {
            if (timer) {
                clearInterval(timer);
                timer = null;
            }
            isRunning = false;
            statusLabel.textContent = "已暂停";
            playBtn.textContent = "继续训练";
        }

        function resetTraining() {
            stopTraining();
            currentEpisode = 0;
            datasets.forEach(dataset => {
                dataset.data = [];
            });
            chart.update();
            updateDisplay();
            statusLabel.textContent = "准备开始";
            playBtn.textContent = "开始训练";
        }

        playBtn.addEventListener("click", function () {
            if (isRunning) {
                stopTraining();
            } else {
                startTraining();
            }
        });

        resetBtn.addEventListener("click", resetTraining);

        speedSlider.addEventListener("input", function () {
            speedLabel.textContent = `${this.value}x`;
            if (isRunning) {
                stopTraining();
                startTraining();
            }
        });

        updateDisplay();
    }

    /* ------------------ DUCG 推理网络可视化 ------------------ */
    function initDUCGNetwork() {
        const networkContainer = document.getElementById("ducg-network");
        const startBtn = document.getElementById("ducg-start");
        const prevBtn = document.getElementById("ducg-prev");
        const nextBtn = document.getElementById("ducg-next");
        const resetBtn = document.getElementById("ducg-reset");
        const speedSlider = document.getElementById("ducg-speed");
        const speedLabel = document.getElementById("ducg-speed-label");
        const statusLabel = document.getElementById("ducg-status");
        const stepInfo = document.getElementById("ducg-step-info");

        if (!networkContainer || !startBtn) {
            return;
        }

        let currentStep = 0;
        let timer = null;
        let isRunning = false;

        // DUCG 网络节点定义
        const nodes = [
            { id: 'evidence1', x: 50, y: 80, label: '证据A', type: 'evidence', active: false, confidence: 0.8 },
            { id: 'evidence2', x: 50, y: 180, label: '证据B', type: 'evidence', active: false, confidence: 0.7 },
            { id: 'evidence3', x: 50, y: 280, label: '证据C', type: 'evidence', active: false, confidence: 0.9 },
            { id: 'event1', x: 200, y: 130, label: '事件1', type: 'event', active: false, confidence: 0 },
            { id: 'event2', x: 200, y: 230, label: '事件2', type: 'event', active: false, confidence: 0 },
            { id: 'causal1', x: 350, y: 100, label: '因果1', type: 'causal', active: false, confidence: 0 },
            { id: 'causal2', x: 350, y: 180, label: '因果2', type: 'causal', active: false, confidence: 0 },
            { id: 'decision', x: 500, y: 140, label: '判决', type: 'decision', active: false, confidence: 0 }
        ];

        // 连接关系定义
        const edges = [
            { from: 'evidence1', to: 'event1', active: false },
            { from: 'evidence2', to: 'event1', active: false },
            { from: 'evidence2', to: 'event2', active: false },
            { from: 'evidence3', to: 'event2', active: false },
            { from: 'event1', to: 'causal1', active: false },
            { from: 'event1', to: 'causal2', active: false },
            { from: 'event2', to: 'causal2', active: false },
            { from: 'causal1', to: 'decision', active: false },
            { from: 'causal2', to: 'decision', active: false }
        ];

        // 推理步骤定义
        const steps = [
            { description: "证据输入阶段", nodes: ['evidence1', 'evidence2', 'evidence3'], edges: [] },
            { description: "证据激活", nodes: ['evidence1'], edges: ['evidence1->event1'] },
            { description: "事件推理", nodes: ['event1'], edges: ['evidence2->event1'] },
            { description: "多证据融合", nodes: ['evidence2', 'event2'], edges: ['evidence2->event2', 'evidence3->event2'] },
            { description: "因果关系建立", nodes: ['causal1'], edges: ['event1->causal1'] },
            { description: "因果传播", nodes: ['causal2'], edges: ['event1->causal2', 'event2->causal2'] },
            { description: "决策推理", nodes: ['decision'], edges: ['causal1->decision', 'causal2->decision'] },
            { description: "推理完成", nodes: [], edges: [] }
        ];

        function getNodeTypeColor(type, active, confidence) {
            const colors = {
                evidence: active ? '#10b981' : '#6b7280',
                event: active ? '#3b82f6' : '#6b7280',
                causal: active ? '#8b5cf6' : '#6b7280',
                decision: active ? '#ef4444' : '#6b7280'
            };
            
            const alpha = Math.max(0.3, confidence);
            const color = colors[type];
            return active ? color : '#9ca3af';
        }

        function drawNetwork() {
            networkContainer.innerHTML = '';

            // 绘制边
            edges.forEach(edge => {
                const fromNode = nodes.find(n => n.id === edge.from);
                const toNode = nodes.find(n => n.id === edge.to);
                
                if (fromNode && toNode) {
                    const edgeElement = document.createElement('div');
                    edgeElement.className = 'ducg-edge';
                    if (edge.active) edgeElement.classList.add('active');
                    
                    const dx = toNode.x - fromNode.x;
                    const dy = toNode.y - fromNode.y;
                    const length = Math.sqrt(dx * dx + dy * dy);
                    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                    
                    edgeElement.style.left = fromNode.x + 'px';
                    edgeElement.style.top = fromNode.y + 'px';
                    edgeElement.style.width = length + 'px';
                    edgeElement.style.transform = `rotate(${angle}deg)`;
                    
                    networkContainer.appendChild(edgeElement);
                }
            });

            // 绘制节点
            nodes.forEach(node => {
                const nodeElement = document.createElement('div');
                nodeElement.className = 'ducg-node';
                nodeElement.textContent = node.label;
                nodeElement.style.left = (node.x - 40) + 'px';
                nodeElement.style.top = (node.y - 40) + 'px';
                nodeElement.style.backgroundColor = getNodeTypeColor(node.type, node.active, node.confidence);
                
                if (node.active) {
                    nodeElement.style.transform = 'scale(1.1)';
                }
                
                nodeElement.addEventListener('click', () => {
                    showNodeInfo(node);
                });
                
                networkContainer.appendChild(nodeElement);
            });
        }

        function showNodeInfo(node) {
            const info = `节点: ${node.label}\n类型: ${node.type}\n置信度: ${(node.confidence * 100).toFixed(1)}%`;
            alert(info);
        }

        function executeStep(stepIndex) {
            if (stepIndex >= steps.length) return;

            const step = steps[stepIndex];
            
            // 重置所有节点和边
            nodes.forEach(node => {
                node.active = false;
                if (node.type === 'evidence') {
                    node.confidence = Math.random() * 0.3 + 0.7; // 0.7-1.0
                }
            });
            edges.forEach(edge => edge.active = false);

            // 激活当前步骤的节点
            step.nodes.forEach(nodeId => {
                const node = nodes.find(n => n.id === nodeId);
                if (node) {
                    node.active = true;
                    if (node.type !== 'evidence') {
                        node.confidence = Math.random() * 0.4 + 0.5; // 0.5-0.9
                    }
                }
            });

            // 激活当前步骤的边
            step.edges.forEach(edgeDesc => {
                const [fromId, toId] = edgeDesc.split('->');
                const edge = edges.find(e => e.from === fromId && e.to === toId);
                if (edge) edge.active = true;
            });

            drawNetwork();
            updateStepInfo(stepIndex, step.description);
        }

        function updateStepInfo(stepIndex, description) {
            stepInfo.textContent = `步骤：${stepIndex + 1} / ${steps.length}`;
            document.getElementById("current-stage").textContent = description;
            document.getElementById("active-nodes").textContent = nodes.filter(n => n.active).length;
            
            const avgConfidence = nodes.filter(n => n.active).reduce((sum, n) => sum + n.confidence, 0) / 
                                nodes.filter(n => n.active).length || 0;
            document.getElementById("confidence-level").textContent = (avgConfidence * 100).toFixed(1) + '%';
            document.getElementById("evidence-count").textContent = nodes.filter(n => n.type === 'evidence' && n.active).length;
            document.getElementById("inference-depth").textContent = stepIndex + 1;
        }

        function startInference() {
            if (isRunning) {
                stopInference();
                return;
            }

            isRunning = true;
            statusLabel.textContent = "推理中...";
            startBtn.textContent = "暂停推理";
            
            const speed = parseInt(speedSlider.value);
            const interval = (6 - speed) * 500;
            
            timer = setInterval(() => {
                if (currentStep >= steps.length - 1) {
                    stopInference();
                    statusLabel.textContent = "推理完成";
                    return;
                }
                nextStep();
            }, interval);
        }

        function stopInference() {
            if (timer) {
                clearInterval(timer);
                timer = null;
            }
            isRunning = false;
            statusLabel.textContent = "已暂停";
            startBtn.textContent = "继续推理";
        }

        function nextStep() {
            if (currentStep < steps.length - 1) {
                currentStep++;
                executeStep(currentStep);
            }
        }

        function prevStep() {
            if (currentStep > 0) {
                currentStep--;
                executeStep(currentStep);
            }
        }

        function resetInference() {
            stopInference();
            currentStep = 0;
            executeStep(0);
            statusLabel.textContent = "准备推理";
            startBtn.textContent = "开始推理";
        }

        // 事件监听
        startBtn.addEventListener("click", startInference);
        prevBtn.addEventListener("click", prevStep);
        nextBtn.addEventListener("click", nextStep);
        resetBtn.addEventListener("click", resetInference);
        
        speedSlider.addEventListener("input", function () {
            speedLabel.textContent = `${this.value}x`;
        });

        // 初始化
        resetInference();
    }

    /* ------------------ 神经网络架构设计器 ------------------ */
    function initNeuralNetworkDesigner() {
        const canvas = document.getElementById("nn-canvas");
        const depthSlider = document.getElementById("nn-depth");
        const widthSlider = document.getElementById("nn-width");
        const activationSelect = document.getElementById("nn-activation");
        const forwardBtn = document.getElementById("nn-forward");
        const randomizeBtn = document.getElementById("nn-randomize");
        const depthLabel = document.getElementById("nn-depth-label");
        const widthLabel = document.getElementById("nn-width-label");
        const statusLabel = document.getElementById("nn-status");

        if (!canvas || !depthSlider || !widthSlider) {
            return;
        }

        const ctx = canvas.getContext("2d");
        let animationId = null;
        let forwardPropAnimation = false;

        // 网络参数
        const inputNodes = 4;
        const outputNodes = 3;
        let weights = [];
        let biases = [];

        function initializeWeights(layerSizes) {
            weights = [];
            biases = [];
            
            for (let i = 1; i < layerSizes.length; i++) {
                const prevSize = layerSizes[i - 1];
                const currSize = layerSizes[i];
                
                // Xavier/Glorot 初始化
                const limit = Math.sqrt(6 / (prevSize + currSize));
                const layerWeights = [];
                
                for (let j = 0; j < currSize; j++) {
                    const nodeWeights = [];
                    for (let k = 0; k < prevSize; k++) {
                        nodeWeights.push((Math.random() * 2 - 1) * limit);
                    }
                    layerWeights.push(nodeWeights);
                }
                
                weights.push(layerWeights);
                
                // 偏置初始化为0
                const layerBiases = new Array(currSize).fill(0);
                biases.push(layerBiases);
            }
        }

        function getActivationFunction(name) {
            switch(name) {
                case 'relu':
                    return x => Math.max(0, x);
                case 'sigmoid':
                    return x => 1 / (1 + Math.exp(-x));
                case 'tanh':
                    return x => Math.tanh(x);
                default:
                    return x => x;
            }
        }

        function drawNetwork() {
            const hiddenLayers = parseInt(depthSlider.value);
            const hiddenWidth = parseInt(widthSlider.value);
            
            depthLabel.textContent = hiddenLayers;
            widthLabel.textContent = hiddenWidth;

            const layerSizes = [inputNodes];
            for (let i = 0; i < hiddenLayers; i++) {
                layerSizes.push(hiddenWidth);
            }
            layerSizes.push(outputNodes);

            // 重新初始化权重
            initializeWeights(layerSizes);

            // 清空画布
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // 设置画布尺寸
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;

            const layerSpacing = canvas.width / (layerSizes.length + 1);
            const nodePositions = [];

            // 计算节点位置
            layerSizes.forEach((count, layerIndex) => {
                const x = layerSpacing * (layerIndex + 1);
                const nodeSpacing = canvas.height / (count + 1);
                
                for (let i = 0; i < count; i++) {
                    const y = nodeSpacing * (i + 1);
                    nodePositions.push({
                        layer: layerIndex,
                        index: i,
                        x: x,
                        y: y,
                        activation: 0,
                        size: 8
                    });
                }
            });

            // 绘制连接线
            ctx.strokeStyle = 'rgba(102, 126, 234, 0.3)';
            ctx.lineWidth = 1;
            
            for (let layer = 0; layer < layerSizes.length - 1; layer++) {
                const fromNodes = nodePositions.filter(n => n.layer === layer);
                const toNodes = nodePositions.filter(n => n.layer === layer + 1);
                
                fromNodes.forEach((from, fromIdx) => {
                    toNodes.forEach((to, toIdx) => {
                        ctx.beginPath();
                        ctx.moveTo(from.x, from.y);
                        ctx.lineTo(to.x, to.y);
                        
                        // 根据权重调整线条颜色和粗细
                        if (weights[layer] && weights[layer][toIdx] && weights[layer][toIdx][fromIdx] !== undefined) {
                            const weight = weights[layer][toIdx][fromIdx];
                            const intensity = Math.min(1, Math.abs(weight) * 2);
                            const color = weight > 0 ? 
                                `rgba(16, 185, 129, ${intensity})` : 
                                `rgba(239, 68, 68, ${intensity})`;
                            ctx.strokeStyle = color;
                            ctx.lineWidth = intensity * 2 + 0.5;
                        }
                        
                        ctx.stroke();
                    });
                });
            }

            // 绘制节点
            nodePositions.forEach(node => {
                const intensity = forwardPropAnimation ? node.activation : 0.5;
                const color = getNodeColor(node.layer, layerSizes.length, intensity);
                
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.size, 0, 2 * Math.PI);
                ctx.fill();
                
                // 节点边框
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // 层标签
                if (node.index === 0) {
                    ctx.fillStyle = '#374151';
                    ctx.font = '12px Arial';
                    ctx.textAlign = 'center';
                    
                    let layerName = '';
                    if (node.layer === 0) layerName = '输入层';
                    else if (node.layer === layerSizes.length - 1) layerName = '输出层';
                    else layerName = `隐藏层 ${node.layer}`;
                    
                    ctx.fillText(layerName, node.x, node.y - node.size - 15);
                }
            });

            updateNetworkInfo(layerSizes);
            return nodePositions;
        }

        function getNodeColor(layer, totalLayers, activation) {
            const colors = [
                '#10b981', // 输入层 - 绿色
                '#3b82f6', // 隐藏层 - 蓝色
                '#ef4444'  // 输出层 - 红色
            ];
            
            let colorIndex = 1; // 默认隐藏层
            if (layer === 0) colorIndex = 0; // 输入层
            if (layer === totalLayers - 1) colorIndex = 2; // 输出层
            
            const baseColor = colors[colorIndex];
            // 根据激活值调整透明度
            const alpha = 0.3 + activation * 0.7;
            
            // 解析颜色并添加透明度
            if (baseColor.startsWith('#')) {
                const r = parseInt(baseColor.slice(1, 3), 16);
                const g = parseInt(baseColor.slice(3, 5), 16);
                const b = parseInt(baseColor.slice(5, 7), 16);
                return `rgba(${r}, ${g}, ${b}, ${alpha})`;
            }
            
            return baseColor;
        }

        function animateForwardPropagation() {
            const nodePositions = drawNetwork();
            forwardPropAnimation = true;
            statusLabel.textContent = "前向传播中...";
            
            let step = 0;
            const totalSteps = nodePositions[nodePositions.length - 1].layer + 1;
            
            function propagateStep() {
                if (step >= totalSteps) {
                    forwardPropAnimation = false;
                    statusLabel.textContent = "设计模式";
                    return;
                }
                
                // 激活当前层的所有节点
                nodePositions.forEach(node => {
                    if (node.layer === step) {
                        node.activation = Math.random() * 0.5 + 0.5; // 0.5-1.0
                    } else if (node.layer < step) {
                        node.activation = 0.3; // 已处理的层保持低激活
                    } else {
                        node.activation = 0; // 未处理的层无激活
                    }
                });
                
                drawNetwork();
                step++;
                
                setTimeout(propagateStep, 800);
            }
            
            propagateStep();
        }

        function randomizeWeights() {
            const hiddenLayers = parseInt(depthSlider.value);
            const hiddenWidth = parseInt(widthSlider.value);
            
            const layerSizes = [inputNodes];
            for (let i = 0; i < hiddenLayers; i++) {
                layerSizes.push(hiddenWidth);
            }
            layerSizes.push(outputNodes);
            
            initializeWeights(layerSizes);
            drawNetwork();
            statusLabel.textContent = "权重已随机化";
            
            setTimeout(() => {
                statusLabel.textContent = "设计模式";
            }, 1500);
        }

        function updateNetworkInfo(layerSizes) {
            // 计算总参数量
            let totalParams = 0;
            for (let i = 1; i < layerSizes.length; i++) {
                totalParams += layerSizes[i] * layerSizes[i - 1] + layerSizes[i]; // 权重 + 偏置
            }
            
            document.getElementById("total-params").textContent = totalParams.toLocaleString();
            document.getElementById("total-layers").textContent = layerSizes.length;
            
            // 计算复杂度 (简化为 O(n²))
            const maxLayerSize = Math.max(...layerSizes);
            document.getElementById("computation-cost").textContent = `O(${maxLayerSize}²)`;
            
            // 估算内存占用 (假设每个参数4字节)
            const memoryKB = (totalParams * 4 / 1024).toFixed(1);
            document.getElementById("memory-usage").textContent = `${memoryKB} KB`;
        }

        // 事件监听器
        depthSlider.addEventListener("input", drawNetwork);
        widthSlider.addEventListener("input", drawNetwork);
        activationSelect.addEventListener("change", drawNetwork);
        forwardBtn.addEventListener("click", animateForwardPropagation);
        randomizeBtn.addEventListener("click", randomizeWeights);

        // 窗口大小变化时重绘
        window.addEventListener("resize", () => {
            setTimeout(drawNetwork, 100);
        });

        // 初始化
        drawNetwork();
    }
})();
