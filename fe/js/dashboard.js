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

// 1. KHOI TAO CHARTS (AN TOAN VOI NULL CHECK)
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

// 2. KHOI TAO NETWORK FORCE GRAPH
let graph = null;
const graphContainer = document.getElementById('topology-graph');
if (graphContainer && typeof ForceGraph !== 'undefined') {
    graph = ForceGraph()(graphContainer)
        .nodeId('id')
        .nodeLabel(node => `${node.id} (${node.type.toUpperCase()})\nCash: $${Math.round(node.cash || 0)}`)
        .nodeColor(node => AGENT_COLORS[node.type] || '#888')
        .nodeRelSize(node => ['government', 'bank', 'economy', 'supervisor'].includes(node.type) ? 8 : 4)
        .linkColor(link => {
            if (link.type === 'HIRE' || link.type === 'WAGE_PAID') return 'rgba(88, 166, 255, 0.4)';
            if (link.type === 'LOAN_DISBURSED') return 'rgba(210, 153, 34, 0.6)';
            if (link.type === 'PENALTY_ENFORCED') return 'rgba(163, 113, 247, 0.6)';
            return 'rgba(255, 255, 255, 0.15)';
        })
        .linkWidth(link => ['LOAN_DISBURSED', 'PENALTY_ENFORCED'].includes(link.type) ? 2 : 1)
        .linkDirectionalParticles(link => link.type === 'WAGE_PAID' ? 2 : 0)
        .linkDirectionalParticleSpeed(0.01)
        .onNodeClick(node => inspectAgent(node.id));

    // Kiem tra an toan truoc khi kich hoat cac luc keo cua D3
    if (typeof d3 !== 'undefined') {
        graph.d3Force('charge', d3.forceManyBody().strength(-40))
             .d3Force('radial', d3.forceRadial(180, 0, 0).strength(0.05));
    }

    function resizeGraph() {
        if (graph && graphContainer) {
            graph.width(graphContainer.clientWidth);
            graph.height(graphContainer.clientHeight);
        }
    }
    window.addEventListener('resize', resizeGraph);
    setTimeout(resizeGraph, 100);
}

// 3. CAP NHAT GIAO DIEN
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

    state.agents = payload.agents || {};
    const nodes = Object.values(state.agents).map(a => ({
        id: a.agent_id,
        type: a.type,
        cash: a.cash || a.treasury || a.reserves || 0,
        status: a.status
    }));

    if (payload.events && payload.events.length > 0) {
        payload.events.forEach(e => {
            appendLog(e);
            if (e.source && e.target && e.source !== 'MARKET' && e.target !== 'MARKET') {
                const existing = state.links.find(l => l.source.id === e.source && l.target.id === e.target && l.type === e.type);
                if (!existing) {
                    state.links.push({ source: e.source, target: e.target, type: e.type, expire: 5 });
                }
            }
        });
    }

    state.links.forEach(l => l.expire--);
    state.links = state.links.filter(l => l.expire > 0);

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

// 4. DIEU KHIEN REST API
async function sendControl(action, value = null) {
    try {
        await fetch(`${BACKEND_HTTP}/api/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, value })
        });
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

// 5. KET NOI WEBSOCKET
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

initWebSocket();