// ==============================================================================
// 1. GLOBAL STATE & THEME CONSTANTS
// ==============================================================================
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

// ==============================================================================
// 2. REST API & CONTROLS BINDING (MODULE ĐỘC LẬP)
// ==============================================================================
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

function initControlButtons() {
    const bind = (id, fn) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', fn);
    };

    bind('btn-play', () => sendControl('PLAY'));
    bind('btn-pause', () => sendControl('PAUSE'));
    bind('btn-step', () => sendControl('STEP'));
    bind('btn-reset', () => {
        state.history.months = [];
        state.history.gdp = [];
        state.history.gini = [];
        if (macroChart) {
            macroChart.data.labels = [];
            macroChart.data.datasets[0].data = [];
            macroChart.data.datasets[1].data = [];
            macroChart.update();
        }
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
}

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
            if (macroChart) {
                macroChart.data.labels = [];
                macroChart.data.datasets[0].data = [];
                macroChart.data.datasets[1].data = [];
                macroChart.update();
            }
            const stream = document.getElementById('log-stream-container');
            if (stream) stream.innerHTML = '';
            await sendControl('RESET');
        });
    } catch (err) {
        console.warn('[FE] Failed to load checkpoints list:', err);
    }
}

// ==============================================================================
// 3. CHART HISTORICAL SERIES (MODULE ĐỘC LẬP)
// ==============================================================================
let macroChart = null;
function initMacroChart() {
    const chartCanvas = document.getElementById('macroSeriesChart');
    if (!chartCanvas || typeof Chart === 'undefined') return;

    try {
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
                        borderWidth: 2,
                        tension: 0.1,
                        pointRadius: 0
                    },
                    {
                        label: 'Gini Index',
                        data: [],
                        borderColor: '#f85149',
                        borderDash: [3, 3],
                        yAxisID: 'y1',
                        borderWidth: 2,
                        tension: 0.1,
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

        // Gan su kien cho cac nut dieu huong bieu do
        const btnAll = document.getElementById('btn-chart-all');
        if (btnAll) {
            btnAll.addEventListener('click', () => {
                if (!macroChart) return;
                macroChart.data.labels = state.history.months;
                macroChart.data.datasets[0].data = state.history.gdp;
                macroChart.data.datasets[1].data = state.history.gini;
                macroChart.update();
            });
        }

        const btnLast100 = document.getElementById('btn-chart-last100');
        if (btnLast100) {
            btnLast100.addEventListener('click', () => {
                if (!macroChart) return;
                const len = state.history.months.length;
                const start = Math.max(0, len - 100);
                macroChart.data.labels = state.history.months.slice(start);
                macroChart.data.datasets[0].data = state.history.gdp.slice(start);
                macroChart.data.datasets[1].data = state.history.gini.slice(start);
                macroChart.update();
            });
        }

        const btnResetZoom = document.getElementById('btn-chart-reset-zoom');
        if (btnResetZoom) {
            btnResetZoom.addEventListener('click', () => {
                if (macroChart && typeof macroChart.resetZoom === 'function') {
                    macroChart.resetZoom();
                }
            });
        }
    } catch (e) {
        console.error('[FE] Error initializing Macro Chart:', e);
    }
}

// ==============================================================================
// 4. SPATIAL FIXED TOPOLOGY GRAPH (VÔ HIỆU HÓA PHYSICS, TỌA ĐỘ CỐ ĐỊNH)
// ==============================================================================
function getFixedCoordinates(agentId, type) {
    const id = (agentId || '').toLowerCase();
    const t = (type || '').toLowerCase();

    // 4 The che vi mo: Hang tren cung (y = -190)
    if (t.includes('gov') || id.startsWith('gov'))   return { x: -210, y: -190 };
    if (t.includes('bank') || id.startsWith('bank')) return { x: -70,  y: -190 };
    if (t.includes('eco') || id.startsWith('eco'))   return { x: 70,   y: -190 };
    if (t.includes('sup') || id.startsWith('sup'))   return { x: 210,  y: -190 };

    // 5 Doanh nghiep: Hang giua (y = -80)
    if (t.includes('firm') || id.startsWith('firm')) {
        const idx = parseInt(id.replace(/\D/g, ''), 10) || 0;
        return { x: (idx - 2) * 110, y: -80 };
    }

    // 50 Nguoi lao dong: Luoi 5 hang x 10 cot (y = 30 den 230)
    if (t.includes('emp') || id.startsWith('emp')) {
        const idx = parseInt(id.replace(/\D/g, ''), 10) || 0;
        const col = idx % 10;
        const row = Math.floor(idx / 10);
        return {
            x: (col - 4.5) * 52,
            y: 30 + row * 50
        };
    }

    return { x: 0, y: 0 };
}

let graph = null;
function initTopologyGraph() {
    const graphContainer = document.getElementById('topology-graph');
    if (!graphContainer || typeof ForceGraph === 'undefined') return;

    try {
        graph = ForceGraph()(graphContainer)
            .nodeId('id')
            // VO HIEU HOA TOAN BO LUC VAT LY DE TRONG KHONG BI ROI
            .d3Force('charge', null)
            .d3Force('center', null)
            .d3Force('link', null)
            .d3Force('radial', null)
            .d3VelocityDecay(1)
            .linkColor(link => {
                if (link.type === 'HIRE' || link.type === 'WAGE_PAID') return 'rgba(88, 166, 255, 0.95)';
                if (link.type === 'LOAN_DISBURSED') return 'rgba(210, 153, 34, 0.95)';
                if (link.type === 'PENALTY_ENFORCED') return 'rgba(163, 113, 247, 0.95)';
                return 'rgba(255, 255, 255, 0.7)';
            })
            .linkWidth(link => ['LOAN_DISBURSED', 'PENALTY_ENFORCED'].includes(link.type) ? 3 : 2)
            .linkDirectionalParticles(3)
            .linkDirectionalParticleSpeed(0.05)
            .onNodeClick(node => inspectAgent(node.id))
            .nodeCanvasObject((node, ctx, globalScale) => {
                const r = node.radius || 10;
                const nx = (typeof node.x === 'number') ? node.x : 0;
                const ny = (typeof node.y === 'number') ? node.y : 0;

                // 1. Ve hinh cau Node
                ctx.beginPath();
                ctx.arc(nx, ny, r, 0, 2 * Math.PI, false);
                ctx.fillStyle = AGENT_COLORS[node.type] || '#58a6ff';
                ctx.fill();
                ctx.lineWidth = 2.5 / globalScale;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();

                // 2. Ve Text ID ro net
                const fontSize = Math.max(9, 13 / globalScale);
                ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = '#ffffff';
                ctx.fillText(node.shortName || node.id, nx, ny);
            });

        function resizeGraph() {
            if (graph && graphContainer) {
                graph.width(graphContainer.clientWidth);
                graph.height(graphContainer.clientHeight);
            }
        }
        window.addEventListener('resize', resizeGraph);
        setTimeout(resizeGraph, 200);
    } catch (e) {
        console.error('[FE] Error initializing ForceGraph:', e);
    }
}

// ==============================================================================
// 5. UPDATE CYCLE & TELEMETRY STREAM
// ==============================================================================
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

        // Luu tru toan bo dong lich su
        state.history.months.push(payload.timestep);
        state.history.gdp.push(m.gdp);
        state.history.gini.push(m.gini);

        if (macroChart) {
            macroChart.data.labels = state.history.months;
            macroChart.data.datasets[0].data = state.history.gdp;
            macroChart.data.datasets[1].data = state.history.gini;
            macroChart.update('none');
        }
    }

    // Xu ly danh sach Nodes co dinh
    state.agents = payload.agents || {};
    const existingNodeIds = new Set(Object.keys(state.agents));

    const nodes = Object.values(state.agents).map(a => {
        const coords = getFixedCoordinates(a.agent_id, a.type);
        let shortName = a.agent_id.substring(0, 3).toUpperCase();
        let radius = 10;

        if (a.type === 'employee') {
            const num = a.agent_id.replace(/\D/g, '');
            shortName = `E${num}`;
            radius = 8.5;
        } else if (a.type === 'firm') {
            const num = a.agent_id.replace(/\D/g, '');
            shortName = `F${num}`;
            radius = 13;
        } else {
            radius = 16;
            if (a.type === 'government') shortName = 'GOV';
            if (a.type === 'bank') shortName = 'BANK';
            if (a.type === 'economy') shortName = 'ECO';
            if (a.type === 'supervisor') shortName = 'SUP';
        }

        return {
            id: a.agent_id,
            type: a.type,
            shortName: shortName,
            radius: radius,
            cash: a.cash || a.treasury || a.reserves || 0,
            status: a.status,
            // Khoa cung ca x, y va fx, fy
            x: coords.x,
            y: coords.y,
            fx: coords.x,
            fy: coords.y
        };
    });

    // Xu ly cac duong noi tuong tac: Chi giu dung 1 nhip roi tu huy
    const currentTickLinks = [];
    if (payload.events && payload.events.length > 0) {
        payload.events.forEach(e => {
            appendLog(e);
            // CHI TAO DUONG NOI NEU CA HAI TAC TU DEU DANG CON HOAT DONG (Tranh sinh dummy node)
            if (e.source && e.target && existingNodeIds.has(e.source) && existingNodeIds.has(e.target)) {
                currentTickLinks.push({
                    source: e.source,
                    target: e.target,
                    type: e.type
                });
            }
        });
    }

    if (graph) {
        graph.graphData({ nodes, links: currentTickLinks });
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

// ==============================================================================
// 6. WEBSOCKET CONNECTION
// ==============================================================================
let ws = null;
function initWebSocket() {
    const wsUrl = `ws://${BACKEND_WS_HOST}/ws/stream`;
    const dot = document.getElementById('connection-dot');
    const label = document.getElementById('connection-status');

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('[FE] WebSocket connected.');
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
            console.warn('[FE] WebSocket error:', err);
            ws.close();
        };
    } catch (e) {
        console.error('[FE] WebSocket init error:', e);
        setTimeout(initWebSocket, 2000);
    }
}

// KHOI DONG TOAN BO HE THONG TREN TRINH DUYET
initControlButtons();
initCheckpointsDropdown();
initMacroChart();
initTopologyGraph();
initWebSocket();