const state = {
    agents: {},
    macro: {},
    links: [],
    history: {
        months: [],
        gdp: [],
        gini: []
    },
    selectedAgentId: null
};

const AGENT_COLORS = {
    'government': '#f85149',
    'bank': '#d29922',
    'economy': '#ffffff',
    'supervisor': '#a371f7',
    'firm': '#58a6ff',
    'employee': '#3fb950'
};

const isServedByFastAPI = window.location.port === '8000';
const BACKEND_HTTP = isServedByFastAPI ? '' : 'http://127.0.0.1:8000';
const BACKEND_WS_HOST = isServedByFastAPI ? window.location.host : '127.0.0.1:8000';

// 1. KHOI TAO CHARTS (CHART.JS)
let macroChart = null;
const chartCanvas = document.getElementById('macroSeriesChart');
if (chartCanvas && typeof Chart !== 'undefined') {
    const ctx = chartCanvas.getContext('2d');
    macroChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'GDP ($)',
                    data: [],
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    yAxisID: 'y',
                    borderWidth: 1.5,
                    tension: 0.2,
                    pointRadius: 0
                },
                {
                    label: 'Gini Index',
                    data: [],
                    borderColor: '#f85149',
                    borderDash: [3, 3],
                    yAxisID: 'y1',
                    borderWidth: 1.5,
                    tension: 0.2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { grid: { color: '#1e2533' }, ticks: { color: '#8b949e', font: { size: 10 } } },
                y: { position: 'left', grid: { color: '#1e2533' }, ticks: { color: '#58a6ff', font: { size: 10 } } },
                y1: { position: 'right', min: 0, max: 1, grid: { drawOnChartArea: false }, ticks: { color: '#f85149', font: { size: 10 } } }
            },
            plugins: {
                legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 10 } } }
            }
        }
    });
}

// 2. BO CUC CO DINH THOANG MAT (DETERMINISTIC FIXED LAYOUT)
function assignFixedCoordinates(agentId, type) {
    // 4 The che vi mo o trung tam hang tren cung
    if (type === 'government') return { fx: -120, fy: -160 };
    if (type === 'bank')       return { fx: -40,  fy: -160 };
    if (type === 'economy')    return { fx: 40,   fy: -160 };
    if (type === 'supervisor') return { fx: 120,  fy: -160 };

    // 5 Doanh nghiep dan hang ngang o giua
    if (type === 'firm') {
        const idx = parseInt(agentId.split('_')[1], 10) || 0;
        return { fx: (idx - 2) * 80, fy: -60 };
    }

    // 50 Nguoi lao dong chia thanh luoi 5 hang x 10 cot ben duoi
    if (type === 'employee') {
        const idx = parseInt(agentId.split('_')[1], 10) || 0;
        const col = idx % 10;
        const row = Math.floor(idx / 10);
        return {
            fx: (col - 4.5) * 42,
            fy: 20 + row * 45
        };
    }

    return { fx: 0, fy: 0 };
}

// 3. KHOI TAO NETWORK GRAPH VOI NHAN TEXT & TOA DO KHOA
let graph = null;
const graphContainer = document.getElementById('topology-graph');
if (graphContainer && typeof ForceGraph !== 'undefined') {
    graph = ForceGraph()(graphContainer)
        .nodeId('id')
        .nodeRelSize(5)
        .linkColor(link => {
            if (link.type === 'HIRE' || link.type === 'WAGE_PAID') return 'rgba(88, 166, 255, 0.7)';
            if (link.type === 'LOAN_DISBURSED') return 'rgba(210, 153, 34, 0.8)';
            if (link.type === 'PENALTY_ENFORCED') return 'rgba(163, 113, 247, 0.9)';
            return 'rgba(255, 255, 255, 0.3)';
        })
        .linkWidth(link => ['LOAN_DISBURSED', 'PENALTY_ENFORCED'].includes(link.type) ? 2.5 : 1.5)
        .linkDirectionalParticles(2)
        .linkDirectionalParticleSpeed(0.03)
        .onNodeClick(node => inspectAgent(node.id))
        .nodeCanvasObject((node, ctx, globalScale) => {
            const label = node.shortName;
            const fontSize = 10 / globalScale;
            ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;

            // Ve hat Node
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI, false);
            ctx.fillStyle = AGENT_COLORS[node.type] || '#888';
            ctx.fill();
            ctx.lineWidth = 1.5 / globalScale;
            ctx.strokeStyle = '#ffffff';
            ctx.stroke();

            // Ve Nhan (ID) ngay duoi hat
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillStyle = '#e6edf3';
            ctx.fillText(label, node.x, node.y + node.radius + 2);
        });

    function resizeGraph() {
        if (graph && graphContainer) {
            graph.width(graphContainer.clientWidth);
            graph.height(graphContainer.clientHeight);
        }
    }
    window.addEventListener('resize', resizeGraph);
    setTimeout(resizeGraph, 150);
}

// 4. CAP NHAT GIAO DIEN & CANH KET NOI TAM THOI
function updateDashboardUI(payload) {
    const monthEl = document.getElementById('metric-month');
    if (monthEl) monthEl.innerText = payload.timestep;

    const m = payload.macro;
    if (m) {
        const setVal = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.innerText = text;
        };

        setVal('val-gdp', `$${Math.round(m.gdp).toLocaleString()}`);
        setVal('val-gini', (m.gini || 0).toFixed(3));
        setVal('val-inflation', `${((m.inflation || 0) * 100).toFixed(2)}%`);
        setVal('val-cost', `$${(m.living_cost || 0).toFixed(1)}`);
        setVal('val-pop', `${m.active_employees || 0} / ${m.active_firms || 0}F`);
        setVal('val-bank', `$${Math.round(m.bank_reserves).toLocaleString()} (NPL: $${Math.round(m.npl)})`);

        if (macroChart) {
            state.history.months.push(payload.timestep);
            state.history.gdp.push(m.gdp);
            state.history.gini.push(m.gini);

            if (state.history.months.length > 50) {
                state.history.months.shift();
                state.history.gdp.shift();
                state.history.gini.shift();
            }

            macroChart.data.labels = state.history.months;
            macroChart.data.datasets[0].data = state.history.gdp;
            macroChart.data.datasets[1].data = state.history.gini;
            macroChart.update();
        }
    }

    // Xu ly Nodes kem toa do khoa co dinh
    state.agents = payload.agents || {};
    const nodes = Object.values(state.agents).map(a => {
        const coords = assignFixedCoordinates(a.agent_id, a.type);
        let shortName = a.agent_id.toUpperCase();
        let radius = 5;

        if (a.type === 'employee') {
            shortName = `E${a.agent_id.split('_')[1]}`;
            radius = 4;
        } else if (a.type === 'firm') {
            shortName = `F${a.agent_id.split('_')[1]}`;
            radius = 7;
        } else {
            radius = 9;
        }

        return {
            id: a.agent_id,
            type: a.type,
            shortName: shortName,
            radius: radius,
            cash: a.cash || a.treasury || a.reserves || 0,
            status: a.status,
            fx: coords.fx,
            fy: coords.fy
        };
    });

    // Xu ly Canh tuong tac: Chi song trong 1 chu ky roi tu huy
    if (payload.events && payload.events.length > 0) {
        payload.events.forEach(e => {
            appendLog(e);
            if (e.source && e.target && e.source !== 'MARKET' && e.target !== 'MARKET') {
                state.links.push({
                    source: e.source,
                    target: e.target,
                    type: e.type,
                    ttl: 1 // Tu dong bien mat ngay sau 1 chu ky de chong roi mat
                });
            }
        });
    }

    // Giam thoi gian ton tai va loc bo cac lien ket het han
    state.links.forEach(l => l.ttl--);
    state.links = state.links.filter(l => l.ttl >= 0);

    if (graph) {
        graph.graphData({ nodes, links: state.links });
    }

    if (state.selectedAgentId && state.agents[state.selectedAgentId]) {
        inspectAgent(state.selectedAgentId);
    }
}

function appendLog(e) {
    const stream = document.getElementById('log-stream-container');
    if (!stream) return;

    const entry = document.createElement('div');
    entry.className = `log-entry ${e.type}`;
    entry.innerText = `[M${e.timestep}] ${e.type}: ${e.source} -> ${e.target} | ${JSON.stringify(e.payload)}`;
    stream.prepend(entry);

    if (stream.children.length > 80) {
        stream.removeChild(stream.lastChild);
    }
    const countEl = document.getElementById('log-count');
    if (countEl) countEl.innerText = `${stream.children.length} events`;
}

function inspectAgent(agentId) {
    state.selectedAgentId = agentId;
    const a = state.agents[agentId];
    const container = document.getElementById('inspector-content');
    if (!container) return;

    if (!a) {
        container.innerHTML = `<span class="text-muted">Agent ${agentId} is no longer active.</span>`;
        return;
    }

    let detailHtml = `<b>${a.agent_id.toUpperCase()}</b> [${a.type}] Status: <b>${a.status}</b> | `;
    if (a.type === 'employee') {
        detailHtml += `Cash: <b>$${(a.cash || 0).toFixed(1)}</b> | Energy: <b>${(a.energy || 0).toFixed(2)}</b> | Skill: <b>${a.skill_level}</b> | Employer: <b>${a.employed_by || 'Unemployed'}</b> | Debt: <b>$${a.debt || 0}</b>`;
    } else if (a.type === 'firm') {
        detailHtml += `Cash: <b>$${(a.cash || 0).toFixed(1)}</b> | Capital: <b>$${(a.capital_stock || 0).toFixed(1)}</b> | Employees: <b>${a.headcount}</b> | Debt: <b>$${a.debt || 0}</b>`;
    } else if (a.type === 'government') {
        detailHtml += `Treasury: <b>$${Math.round(a.treasury || 0).toLocaleString()}</b> | Worker Tax: <b>${((a.tax_rate_worker || 0) * 100).toFixed(1)}%</b> | Firm Tax: <b>${((a.tax_rate_firm || 0) * 100).toFixed(1)}%</b>`;
    } else if (a.type === 'bank') {
        detailHtml += `Reserves: <b>$${Math.round(a.reserves || 0).toLocaleString()}</b> | Lending Rate: <b>${((a.lending_rate || 0) * 100).toFixed(2)}%</b> | Total Loans: <b>$${Math.round(a.total_loans || 0).toLocaleString()}</b>`;
    } else {
        detailHtml += JSON.stringify(a);
    }
    container.innerHTML = detailHtml;
}

// 5. DIEU KHIEN REST API & GAN SU KIEN NUT BAM
async function sendControl(action, value = null) {
    try {
        const res = await fetch(`${BACKEND_HTTP}/api/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, value })
        });
        return await res.json();
    } catch (err) {
        console.error('[FE] Failed to dispatch control:', err);
    }
}

const bindClick = (id, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', fn);
};

bindClick('btn-play', () => sendControl('PLAY'));
bindClick('btn-pause', () => sendControl('PAUSE'));
bindClick('btn-step', () => sendControl('STEP'));
bindClick('btn-reset', () => {
    state.history.months = [];
    state.history.gdp = [];
    state.history.gini = [];
    const stream = document.getElementById('log-stream-container');
    if (stream) stream.innerHTML = '';
    sendControl('RESET');
});

const speedSlider = document.getElementById('speed-slider');
if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
        const val = e.target.value;
        const display = document.getElementById('speed-display');
        if (display) display.innerText = `${val}ms`;
        sendControl('SET_SPEED', val / 1000.0);
    });
}

// 6. NAP DANH SACH CHECKPOINTS TU BACKEND
async function initCheckpointsDropdown() {
    const select = document.getElementById('checkpoint-select');
    if (!select) return;

    try {
        const res = await fetch(`${BACKEND_HTTP}/api/checkpoints`);
        const data = await res.json();

        select.innerHTML = '';

        const heuristicOpt = document.createElement('option');
        heuristicOpt.value = 'heuristic';
        heuristicOpt.innerText = 'Live Model (Heuristic)';
        select.appendChild(heuristicOpt);

        if (data.checkpoints && data.checkpoints.length > 0) {
            data.checkpoints.forEach(cp => {
                const opt = document.createElement('option');
                opt.value = cp;
                opt.innerText = `Policy: ${cp}`;
                select.appendChild(opt);
            });
        }

        if (data.active) {
            select.value = data.active;
        }

        select.addEventListener('change', async (e) => {
            console.log(`[FE] Switching policy checkpoint to: ${e.target.value}`);
            await sendControl('LOAD_CHECKPOINT', e.target.value);
            state.history.months = [];
            state.history.gdp = [];
            state.history.gini = [];
            const stream = document.getElementById('log-stream-container');
            if (stream) stream.innerHTML = '';
            await sendControl('RESET');
        });
    } catch (err) {
        console.warn('[FE] Failed to load checkpoints list:', err);
    }
}

// 7. KET NOI WEBSOCKET
let ws = null;
function initWebSocket() {
    const wsUrl = `ws://${BACKEND_WS_HOST}/ws/stream`;
    console.log(`[FE] Connecting to WebSocket: ${wsUrl}`);

    const dot = document.getElementById('connection-dot');
    const label = document.getElementById('connection-status');

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('[FE] WebSocket connection established successfully.');
            if (dot) dot.classList.add('connected');
            if (label) label.innerText = 'ONLINE';
        };

        ws.onmessage = (event) => {
            const payload = JSON.parse(event.data);
            if (payload.type === 'SIMULATION_TICK' || payload.type === 'INIT_STATE' || payload.type === 'SIMULATION_RESET') {
                updateDashboardUI(payload);
            }
        };

        ws.onclose = () => {
            if (dot) dot.classList.remove('connected');
            if (label) label.innerText = 'DISCONNECTED';
            setTimeout(initWebSocket, 2000);
        };

        ws.onerror = (err) => {
            console.warn('[FE] WebSocket connection error:', err);
            ws.close();
        };
    } catch (e) {
        console.error('[FE] Initialization error:', e);
        setTimeout(initWebSocket, 2000);
    }
}

// Khoi dong toan bo cac tien trinh giao dien
initCheckpointsDropdown();
initWebSocket();