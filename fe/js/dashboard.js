// Khoi tao trang thai toan cuc
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

// Bang mau tac tu
const AGENT_COLORS = {
    'government': '#f85149',
    'bank': '#d29922',
    'economy': '#ffffff',
    'supervisor': '#a371f7',
    'firm': '#58a6ff',
    'employee': '#3fb950'
};

// 1. KHOI TAO CHARTS (CHART.JS)
const ctx = document.getElementById('macroSeriesChart').getContext('2d');
const macroChart = new Chart(ctx, {
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

// 2. KHOI TAO NETWORK FORCE GRAPH
const graphContainer = document.getElementById('topology-graph');
const graph = ForceGraph()(graphContainer)
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
    .onNodeClick(node => inspectAgent(node.id))
    .d3Force('charge', d3.forceManyBody().strength(-40))
    .d3Force('radial', d3.forceRadial(180, 0, 0).strength(0.05));

function resizeGraph() {
    graph.width(graphContainer.clientWidth);
    graph.height(graphContainer.clientHeight);
}
window.addEventListener('resize', resizeGraph);
setTimeout(resizeGraph, 100);

// 3. CAP NHAT GIAO DIEN
function updateDashboardUI(payload) {
    document.getElementById('metric-month').innerText = payload.timestep;
    
    const m = payload.macro;
    if (m) {
        document.getElementById('val-gdp').innerText = `$${Math.round(m.gdp).toLocaleString()}`;
        document.getElementById('val-gini').innerText = (m.gini || 0).toFixed(3);
        document.getElementById('val-inflation').innerText = `${((m.inflation || 0) * 100).toFixed(2)}%`;
        document.getElementById('val-cost').innerText = `$${(m.living_cost || 0).toFixed(1)}`;
        document.getElementById('val-pop').innerText = `${m.active_employees || 0} / ${m.active_firms || 0}F`;
        document.getElementById('val-bank').innerText = `$${Math.round(m.bank_reserves).toLocaleString()} (NPL: $${Math.round(m.npl)})`;

        // Update Series
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

    // Cap nhat Agent Nodes
    state.agents = payload.agents || {};
    const nodes = Object.values(state.agents).map(a => ({
        id: a.agent_id,
        type: a.type,
        cash: a.cash || a.treasury || a.reserves || 0,
        status: a.status
    }));

    // Cap nhat Links tu Events
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

    // Giam thoi gian ton tai cua cac ket noi hieu ung
    state.links.forEach(l => l.expire--);
    state.links = state.links.filter(l => l.expire > 0);

    graph.graphData({ nodes, links: state.links });

    // Refresh Inspector neu dang inspect 1 agent
    if (state.selectedAgentId && state.agents[state.selectedAgentId]) {
        inspectAgent(state.selectedAgentId);
    }
}

function appendLog(e) {
    const stream = document.getElementById('log-stream-container');
    const entry = document.createElement('div');
    entry.className = `log-entry ${e.type}`;
    entry.innerText = `[M${e.timestep}] ${e.type}: ${e.source} -> ${e.target} | ${JSON.stringify(e.payload)}`;
    stream.prepend(entry);
    
    if (stream.children.length > 80) {
        stream.removeChild(stream.lastChild);
    }
    document.getElementById('log-count').innerText = `${stream.children.length} events`;
}

function inspectAgent(agentId) {
    state.selectedAgentId = agentId;
    const a = state.agents[agentId];
    const container = document.getElementById('inspector-content');
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
        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, value })
        });
    } catch (err) {
        console.error('Failed to dispatch control:', err);
    }
}

document.getElementById('btn-play').addEventListener('click', () => sendControl('PLAY'));
document.getElementById('btn-pause').addEventListener('click', () => sendControl('PAUSE'));
document.getElementById('btn-step').addEventListener('click', () => sendControl('STEP'));
document.getElementById('btn-reset').addEventListener('click', () => {
    state.history.months = [];
    state.history.gdp = [];
    state.history.gini = [];
    document.getElementById('log-stream-container').innerHTML = '';
    sendControl('RESET');
});

document.getElementById('speed-slider').addEventListener('input', (e) => {
    const val = e.target.value;
    document.getElementById('speed-display').innerText = `${val}ms`;
    sendControl('SET_SPEED', val / 1000.0);
});

// 5. WEBSOCKET CONNECTION
function initWebSocket() {
    const ws = new WebSocket(`ws://${window.location.host}/ws/stream`);
    const dot = document.getElementById('connection-dot');
    const label = document.getElementById('connection-status');

    ws.onopen = () => {
        dot.classList.add('connected');
        label.innerText = 'ONLINE';
    };

    ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === 'SIMULATION_TICK' || payload.type === 'INIT_STATE' || payload.type === 'SIMULATION_RESET') {
            updateDashboardUI(payload);
        }
    };

    ws.onclose = () => {
        dot.classList.remove('connected');
        label.innerText = 'OFFLINE';
        setTimeout(initWebSocket, 2000);
    };
}

initWebSocket();