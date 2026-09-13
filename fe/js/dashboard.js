// ==============================================================================
// 1. GLOBAL STATE & THEME CONSTANTS
// ==============================================================================
const state = {
    agents: {},
    macro: {},
    links: [],
    history: { months: [], gdp: [], gini: [] },
    selectedAgentId: null
};

const AGENT_COLORS = {
    'government': '#f85149', 'gov': '#f85149',
    'bank': '#d29922',
    'economy': '#ffffff', 'eco': '#ffffff',
    'supervisor': '#a371f7', 'sup': '#a371f7',
    'firm': '#58a6ff',
    'employee': '#3fb950'
};

const isServedByFastAPI = window.location.port === '8000';
const BACKEND_HTTP = isServedByFastAPI ? '' : 'http://127.0.0.1:8000';
const BACKEND_WS_HOST = isServedByFastAPI ? window.location.host : '127.0.0.1:8000';

// ==============================================================================
// 2. REST API & CONTROLS BINDING
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
        document.getElementById('bulletin-stream-container').innerHTML = '';
        document.getElementById('ledger-stream-container').innerHTML = '';
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

function initPolicySandboxControls() {
    const bindSlider = (sliderId, labelId, suffix, transform = v => v) => {
        const slider = document.getElementById(sliderId);
        const label = document.getElementById(labelId);
        if (slider && label) {
            slider.addEventListener('input', (e) => {
                label.innerText = `${transform(e.target.value)}${suffix}`;
            });
        }
    };

    bindSlider('slider-worker-tax', 'lbl-worker-tax', '%');
    bindSlider('slider-firm-tax', 'lbl-firm-tax', '%');
    bindSlider('slider-lending-rate', 'lbl-lending-rate', '%');
    bindSlider('slider-living-cost', 'lbl-living-cost', '', v => `$${parseFloat(v).toFixed(1)}`);

    const applyBtn = document.getElementById('btn-apply-policy');
    if (applyBtn) {
        applyBtn.addEventListener('click', () => {
            const policyPayload = {
                worker_tax: parseFloat(document.getElementById('slider-worker-tax').value) / 100.0,
                firm_tax: parseFloat(document.getElementById('slider-firm-tax').value) / 100.0,
                lending_rate: parseFloat(document.getElementById('slider-lending-rate').value) / 100.0,
                living_cost: parseFloat(document.getElementById('slider-living-cost').value)
            };
            sendControl('SET_POLICY', policyPayload);
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

        if (data.active) select.value = data.active;

        select.addEventListener('change', async (e) => {
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
            document.getElementById('bulletin-stream-container').innerHTML = '';
            document.getElementById('ledger-stream-container').innerHTML = '';
            await sendControl('RESET');
        });
    } catch (err) {
        console.warn('[FE] Failed to load checkpoints list:', err);
    }
}

// ==============================================================================
// 3. CHART HISTORICAL SERIES
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
                    x: { grid: { color: '#1e2533' }, ticks: { color: '#8b949e', font: { size: 9 }, maxTicksLimit: 10 } },
                    y: { 
                        position: 'left', 
                        grid: { color: '#1e2533' }, 
                        ticks: { color: '#58a6ff', font: { size: 9 }, callback: (v) => `$${Math.round(v).toLocaleString()}` },
                        beginAtZero: true,
                        grace: '10%'
                    },
                    y1: { 
                        position: 'right', 
                        min: 0, 
                        max: 1.0, 
                        grid: { drawOnChartArea: false }, 
                        ticks: { color: '#f85149', font: { size: 9 } } 
                    }
                },
                plugins: {
                    legend: { labels: { color: '#8b949e', boxWidth: 10, font: { size: 10 } } }
                }
            }
        });

        const bindChartBtn = (id, sliceCount) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.addEventListener('click', () => {
                if (!macroChart) return;
                const len = state.history.months.length;
                const start = sliceCount ? Math.max(0, len - sliceCount) : 0;
                macroChart.data.labels = state.history.months.slice(start);
                macroChart.data.datasets[0].data = state.history.gdp.slice(start);
                macroChart.data.datasets[1].data = state.history.gini.slice(start);
                macroChart.update();
            });
        };

        bindChartBtn('btn-chart-all', null);
        bindChartBtn('btn-chart-last100', 100);

        const btnResetZoom = document.getElementById('btn-chart-reset-zoom');
        if (btnResetZoom) {
            btnResetZoom.addEventListener('click', () => {
                if (macroChart && typeof macroChart.resetZoom === 'function') macroChart.resetZoom();
            });
        }
    } catch (e) {
        console.error('[FE] Error initializing Macro Chart:', e);
    }
}

// ==============================================================================
// 4. SPATIAL NETWORK TOPOLOGY GRAPH
// ==============================================================================
let graph = null;
function initTopologyGraph() {
    const graphContainer = document.getElementById('topology-graph');
    if (!graphContainer || typeof ForceGraph === 'undefined') return;

    try {
        graph = ForceGraph()(graphContainer)
            .nodeId('id')
            .d3Force('charge', null)
            .d3Force('center', null)
            .d3Force('link', null)
            .d3VelocityDecay(1)
            .linkColor(link => {
                if (link.type === 'HIRE' || link.type === 'WAGE_PAID') return 'rgba(88, 166, 255, 0.95)';
                if (link.type === 'LOAN_DISBURSED') return 'rgba(210, 153, 34, 0.95)';
                if (link.type === 'PENALTY_ENFORCED') return 'rgba(163, 113, 247, 0.95)';
                return 'rgba(255, 255, 255, 0.7)';
            })
            .linkWidth(link => ['LOAN_DISBURSED', 'PENALTY_ENFORCED'].includes(link.type) ? 3 : 2)
            .linkDirectionalParticles(2)
            .linkDirectionalParticleSpeed(0.04)
            .onNodeClick(node => inspectAgent(node.id))
            .nodeCanvasObject((node, ctx) => {
                const r = node.radius || 8;
                const nx = (typeof node.x === 'number') ? node.x : 0;
                const ny = (typeof node.y === 'number') ? node.y : 0;

                ctx.beginPath();
                ctx.arc(nx, ny, r, 0, 2 * Math.PI, false);
                ctx.fillStyle = AGENT_COLORS[node.type] || '#58a6ff';
                ctx.fill();
                ctx.lineWidth = 1.5;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();

                const fontSize = node.type === 'employee' ? 7 : (node.type === 'firm' ? 9.5 : 10.5);
                ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = (node.type === 'economy' || node.type === 'eco') ? '#0d1117' : '#ffffff';
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

        setVal('val-gdp', `$${Math.round(m.gdp || 0).toLocaleString()}`);
        setVal('val-gini', (m.gini || 0).toFixed(3));
        setVal('val-inflation', `${((m.inflation || 0) * 100).toFixed(2)}%`);
        setVal('val-cost', `$${(m.living_cost || 0).toFixed(1)}`);
        setVal('val-pop', `${m.active_employees || 0} / ${m.active_firms || 0}F`);
        setVal('val-bank', `$${Math.round(m.bank_reserves || 0).toLocaleString()}`);
        setVal('val-npl', `$${Math.round(m.npl || 0).toLocaleString()}`);

        if (m.m2_supply !== undefined) setVal('val-m2', `$${Math.round(m.m2_supply).toLocaleString()}`);
        if (m.treasury !== undefined) setVal('val-treasury', `$${Math.round(m.treasury).toLocaleString()}`);

        const rawGdp = Number(m.gdp) || 0;
        const safeGdp = (payload.timestep <= 0 || rawGdp > 50000) ? 0 : rawGdp;

        state.history.months.push(payload.timestep);
        state.history.gdp.push(safeGdp);
        state.history.gini.push(m.gini || 0);

        if (macroChart) {
            macroChart.data.labels = state.history.months;
            macroChart.data.datasets[0].data = state.history.gdp;
            macroChart.data.datasets[1].data = state.history.gini;
            macroChart.update('none');
        }
    }

    state.agents = payload.agents || {};

    const macroTopNodes = [];
    const bankNodes = [];
    const firmNodes = [];
    const employeeNodes = [];

    Object.values(state.agents).forEach(a => {
        if (a.status && !['ACTIVE', 'INITIALIZED'].includes(a.status)) return;
        const id = a.agent_id.toLowerCase();
        if (id.startsWith('gov') || id.startsWith('eco') || id.startsWith('sup')) macroTopNodes.push(a);
        else if (id.startsWith('bank')) bankNodes.push(a);
        else if (id.startsWith('firm')) firmNodes.push(a);
        else if (id.startsWith('emp')) employeeNodes.push(a);
    });

    const sortById = (arr) => arr.sort((a, b) => parseInt(a.agent_id.replace(/\D/g, ''), 10) - parseInt(b.agent_id.replace(/\D/g, ''), 10));
    sortById(bankNodes);
    sortById(firmNodes);
    sortById(employeeNodes);

    const nodes = [];

    // TANG 1: VI MO (y = -220)
    const topMacroPositions = {
        'gov': { x: -180, y: -220, label: 'GOV', type: 'government' },
        'eco': { x: 0,    y: -220, label: 'ECO', type: 'economy' },
        'sup': { x: 180,  y: -220, label: 'SUP', type: 'supervisor' }
    };
    macroTopNodes.forEach(mAgent => {
        const prefix = mAgent.agent_id.split('_')[0].toLowerCase();
        const pos = topMacroPositions[prefix] || { x: 0, y: -220, label: prefix.toUpperCase(), type: prefix };
        nodes.push({
            id: mAgent.agent_id, type: pos.type, shortName: pos.label,
            radius: 15, x: pos.x, y: pos.y, fx: pos.x, fy: pos.y
        });
    });

    // TANG 2: BANK (y = -140)
    const totalBanks = bankNodes.length;
    bankNodes.forEach((b, idx) => {
        const num = b.agent_id.replace(/\D/g, '');
        const label = totalBanks === 1 ? 'BANK' : `B${num}`;
        const posX = totalBanks > 1 ? (idx - (totalBanks - 1) / 2) * 110 : 0;
        nodes.push({
            id: b.agent_id, type: 'bank', shortName: label,
            radius: 14, x: posX, y: -140, fx: posX, fy: -140
        });
    });

    // TANG 3: FIRMS (y = -60)
    const totalFirms = firmNodes.length;
    const firmCols = Math.min(8, Math.max(5, totalFirms));
    firmNodes.forEach((f, idx) => {
        const num = f.agent_id.replace(/\D/g, '');
        const col = idx % firmCols;
        const row = Math.floor(idx / firmCols);
        const firmsInThisRow = Math.min(firmCols, totalFirms - row * firmCols);
        const posX = (col - (firmsInThisRow - 1) / 2) * 115;
        const posY = -60 + row * 45;
        nodes.push({
            id: f.agent_id, type: 'firm', shortName: `F${num}`,
            radius: 12, x: posX, y: posY, fx: posX, fy: posY
        });
    });

    // TANG 4: EMPLOYEES (y = 35)
    const empCols = 12;
    employeeNodes.forEach((e, idx) => {
        const num = e.agent_id.replace(/\D/g, '');
        const col = idx % empCols;
        const row = Math.floor(idx / empCols);
        const posX = (col - (empCols - 1) / 2) * 52;
        const posY = 35 + row * 45;
        nodes.push({
            id: e.agent_id, type: 'employee', shortName: `E${num}`,
            radius: 8.5, x: posX, y: posY, fx: posX, fy: posY
        });
    });

    // GIAO DICH TIEN TE & EVENT BUS
    const existingNodeIds = new Set(nodes.map(n => n.id));
    const currentTickLinks = [];
    if (payload.events && payload.events.length > 0) {
        payload.events.forEach(e => {
            appendLog(e);
            const isCritical = ['HIRE', 'FIRE', 'WAGE_PAID', 'LOAN_DISBURSED', 'PENALTY_ENFORCED'].includes(e.type);
            if (isCritical && e.source && e.target && existingNodeIds.has(e.source) && existingNodeIds.has(e.target)) {
                currentTickLinks.push({ source: e.source, target: e.target, type: e.type });
            }
        });
    }

    if (graph) graph.graphData({ nodes, links: currentTickLinks });

    // Luon tu dong cap nhat inspector neu dang chon bat ky agent nao
    if (state.selectedAgentId) {
        inspectAgent(state.selectedAgentId);
    }
}

// ==============================================================================
// 6. DUAL-STREAM LOG FORMATTER (BOC TACH CHI TIET THE CHE)
// ==============================================================================
function appendLog(e) {
    const isBulletin = [
        'POLICY_SHOCK', 'AGENT_DIED', 'AGENT_BANKRUPT', 
        'PENALTY_ENFORCED', 'TAX_EVADED', 'AGENT_BORN'
    ].includes(e.type);

    const streamId = isBulletin ? 'bulletin-stream-container' : 'ledger-stream-container';
    const stream = document.getElementById(streamId);
    if (!stream) return;

    let textContent = '';
    const p = e.payload || {};

    // Dinh dang cau van co nghia ro rang cho tung loai bien co
    if (e.type === 'POLICY_SHOCK') {
        textContent = `[M${e.timestep}] CHÍNH SÁCH: Cầm quyền áp đặt Shock: Thuế CN ${p.worker_tax}%, Thuế DN ${p.firm_tax}%, Lãi suất ${p.lending_rate}%, Sàn sống $${p.living_cost}`;
    } else if (e.type === 'TAX_EVADED') {
        const sourceLabel = e.source.toUpperCase();
        if (p.gross !== undefined) {
            textContent = `[M${e.timestep}] GIAN LẬN: ${sourceLabel} giấu ${p.hidden_pct}% thu nhập (Gross $${p.gross}), trốn $${p.amount} tiền thuế!`;
        } else {
            textContent = `[M${e.timestep}] GIAN LẬN: ${sourceLabel} giấu ${p.hidden_pct}% lợi nhuận (Lãi $${p.profit}), trốn $${p.amount} thuế DN!`;
        }
    } else if (e.type === 'PENALTY_ENFORCED') {
        textContent = `[M${e.timestep}] XỬ PHẠT: Thanh tra bắt quả tang ${e.target.toUpperCase()}, phạt $${p.fine} (Thuế trốn: $${p.evaded})!`;
    } else if (e.type === 'AGENT_DIED') {
        textContent = `[M${e.timestep}] KHAI TỬ: ${e.source.toUpperCase()} qua đời lúc ${p.age} tuổi [Lý do: ${p.reason}]. Di sản thu về Kho bạc.`;
    } else if (e.type === 'AGENT_BANKRUPT') {
        textContent = `[M${e.timestep}] VỠ NỢ: ${e.source.toUpperCase()} giải thể do âm vốn, quỵt nợ ngân hàng $${p.bad_debt}!`;
    } else if (e.type === 'AGENT_BORN') {
        textContent = `[M${e.timestep}] SINH MỚI: ${e.target.toUpperCase()} gia nhập xã hội (Vốn mồi $${p.cash}, Kỹ năng: ${p.skill.toFixed(2)})`;
    } else if (e.type === 'WAGE_PAID') {
        textContent = `[M${e.timestep}] LƯƠNG: ${e.source.toUpperCase()} trả lương cho ${e.target.toUpperCase()} số tiền $${p.amount.toFixed(1)}`;
    } else if (e.type === 'LOAN_DISBURSED') {
        textContent = `[M${e.timestep}] TÍN DỤNG: Ngân hàng giải ngân cho ${e.target.toUpperCase()} vay $${p.amount.toFixed(1)}`;
    } else {
        textContent = `[M${e.timestep}] ${e.type}: ${e.source} -> ${e.target} | ${JSON.stringify(p)}`;
    }

    const entry = document.createElement('div');
    entry.className = `log-entry ${e.type}`;
    entry.innerText = textContent;
    stream.prepend(entry);

    if (stream.children.length > 150) {
        stream.removeChild(stream.lastChild);
    }

    const countEl = document.getElementById('log-count');
    if (countEl) {
        const total = document.querySelectorAll('.log-entry').length;
        countEl.innerText = `${total} events`;
    }
}

function initLogTabs() {
    const btnBulletin = document.getElementById('tab-bulletin');
    const btnLedger = document.getElementById('tab-ledger');
    const streamBulletin = document.getElementById('bulletin-stream-container');
    const streamLedger = document.getElementById('ledger-stream-container');

    if (btnBulletin && btnLedger) {
        btnBulletin.addEventListener('click', () => {
            btnBulletin.classList.add('active');
            btnLedger.classList.remove('active');
            streamBulletin.style.display = 'flex';
            streamLedger.style.display = 'none';
        });

        btnLedger.addEventListener('click', () => {
            btnLedger.classList.add('active');
            btnBulletin.classList.remove('active');
            streamBulletin.style.display = 'none';
            streamLedger.style.display = 'flex';
        });
    }
}

// ==============================================================================
// 7. CLEAN SKELETON INSPECTOR (KHONG GHEP CHUOI HTML)
// ==============================================================================
function inspectAgent(agentId) {
    state.selectedAgentId = agentId;
    const a = state.agents[agentId];
    
    const placeholder = document.getElementById('inspector-placeholder');
    const activeView = document.getElementById('inspector-active-view');
    if (!placeholder || !activeView) return;

    // Neu Agent da chet va bi xoa khoi memory
    if (!a) {
        placeholder.style.display = 'none';
        activeView.style.display = 'flex';

        document.getElementById('ins-id').innerText = agentId.toUpperCase();
        document.getElementById('ins-type').innerText = 'DECEASED';
        
        const statusTag = document.getElementById('ins-status');
        statusTag.innerText = '● EXITED / DECEASED';
        statusTag.className = 'agent-status-tag status-dead';

        document.getElementById('lbl-metric-1').innerText = 'STATUS';
        document.getElementById('ins-val-1').innerText = 'Removed from World';
        document.getElementById('ins-val-1').style.color = '#8b949e';

        document.getElementById('col-energy').style.display = 'none';
        document.getElementById('col-metric-3').style.display = 'none';
        document.getElementById('col-metric-4').style.display = 'none';
        document.getElementById('col-metric-5').style.display = 'none';
        return;
    }

    placeholder.style.display = 'none';
    activeView.style.display = 'flex';
    document.getElementById('col-energy').style.display = 'flex';
    document.getElementById('col-metric-3').style.display = 'flex';
    document.getElementById('col-metric-4').style.display = 'flex';
    document.getElementById('col-metric-5').style.display = 'flex';

    const type = a.type || agentId.split('_')[0];
    document.getElementById('ins-id').innerText = a.agent_id.toUpperCase();
    document.getElementById('ins-type').innerText = type;

    const statusTag = document.getElementById('ins-status');
    statusTag.innerText = `● ${a.status}`;
    statusTag.className = `agent-status-tag ${a.status === 'ACTIVE' ? 'status-active' : 'status-dead'}`;

    if (type === 'employee') {
        const energyVal = Math.max(0, a.energy || 0);
        const energyPct = Math.min(100, Math.round(energyVal * 100));
        const skillTier = a.skill_level >= 2.5 ? 'Expert' : (a.skill_level >= 1.7 ? 'Engineer' : (a.skill_level >= 1.0 ? 'Core' : 'General'));

        document.getElementById('lbl-metric-1').innerText = 'LIQUID CASH';
        const val1 = document.getElementById('ins-val-1');
        val1.innerText = `$${(a.cash || 0).toFixed(1)}`;
        val1.style.color = a.cash < 0 ? '#f85149' : '#58a6ff';

        document.getElementById('lbl-metric-2').innerText = `ENERGY: ${energyPct}% (${energyVal.toFixed(2)})`;
        document.getElementById('ins-energy-wrap').style.display = 'block';
        document.getElementById('ins-val-2').style.display = 'none';
        const bar = document.getElementById('ins-energy-bar');
        bar.style.width = `${energyPct}%`;
        bar.style.background = energyPct < 30 ? '#f85149' : (energyPct < 70 ? '#d29922' : '#3fb950');

        document.getElementById('lbl-metric-3').innerText = 'AGE / MAX';
        document.getElementById('ins-val-3').innerText = `${a.age || 20}y / ${a.max_age || 65}y (${skillTier})`;

        document.getElementById('lbl-metric-4').innerText = 'EMPLOYER';
        const val4 = document.getElementById('ins-val-4');
        val4.innerText = a.employed_by || 'Unemployed';
        val4.style.color = a.employed_by ? '#58a6ff' : '#d29922';

        document.getElementById('lbl-metric-5').innerText = 'WAGE / MONTH';
        document.getElementById('ins-val-5').innerText = `$${(a.wage || 0).toFixed(1)}`;
    } else if (type === 'firm') {
        document.getElementById('lbl-metric-1').innerText = 'CASH RESERVES';
        const val1 = document.getElementById('ins-val-1');
        val1.innerText = `$${Math.round(a.cash || 0).toLocaleString()}`;
        val1.style.color = '#58a6ff';

        document.getElementById('lbl-metric-2').innerText = 'CAPITAL STOCK';
        document.getElementById('ins-energy-wrap').style.display = 'none';
        const val2 = document.getElementById('ins-val-2');
        val2.style.display = 'block';
        val2.innerText = `$${Math.round(a.capital_stock || 0).toLocaleString()}`;

        document.getElementById('lbl-metric-3').innerText = 'BANK DEBT';
        document.getElementById('ins-val-3').innerText = `$${Math.round(a.debt || 0).toLocaleString()}`;

        document.getElementById('lbl-metric-4').innerText = 'HEADCOUNT';
        document.getElementById('ins-val-4').innerText = `${a.headcount || 0} workers`;
        document.getElementById('ins-val-4').style.color = '#f0f6fc';

        document.getElementById('lbl-metric-5').innerText = 'LAST NET PROFIT';
        const val5 = document.getElementById('ins-val-5');
        val5.innerText = `$${(a.last_profit || 0).toFixed(1)}`;
        val5.style.color = a.last_profit >= 0 ? '#3fb950' : '#f85149';
    } else if (type === 'government' || type === 'gov') {
        document.getElementById('lbl-metric-1').innerText = 'SOVEREIGN TREASURY';
        const val1 = document.getElementById('ins-val-1');
        val1.innerText = `$${Math.round(a.treasury || 0).toLocaleString()}`;
        val1.style.color = '#f85149';

        document.getElementById('lbl-metric-2').innerText = 'WORKER TAX';
        document.getElementById('ins-energy-wrap').style.display = 'none';
        const val2 = document.getElementById('ins-val-2');
        val2.style.display = 'block';
        val2.innerText = `${((a.tax_rate_worker || 0) * 100).toFixed(1)}%`;

        document.getElementById('lbl-metric-3').innerText = 'FIRM TAX';
        document.getElementById('ins-val-3').innerText = `${((a.tax_rate_firm || 0) * 100).toFixed(1)}%`;

        document.getElementById('col-metric-4').style.display = 'none';
        document.getElementById('col-metric-5').style.display = 'none';
    } else if (type === 'bank') {
        document.getElementById('lbl-metric-1').innerText = 'BANK RESERVES';
        const val1 = document.getElementById('ins-val-1');
        val1.innerText = `$${Math.round(a.reserves || 0).toLocaleString()}`;
        val1.style.color = '#d29922';

        document.getElementById('lbl-metric-2').innerText = 'LENDING RATE';
        document.getElementById('ins-energy-wrap').style.display = 'none';
        const val2 = document.getElementById('ins-val-2');
        val2.style.display = 'block';
        val2.innerText = `${((a.lending_rate || 0) * 100).toFixed(2)}%`;

        document.getElementById('lbl-metric-3').innerText = 'BAD DEBT (NPL)';
        const val3 = document.getElementById('ins-val-3');
        val3.innerText = `$${Math.round(a.non_performing_loans || 0).toLocaleString()}`;
        val3.style.color = '#f85149';

        document.getElementById('col-metric-4').style.display = 'none';
        document.getElementById('col-metric-5').style.display = 'none';
    }
}

// ==============================================================================
// 8. WEBSOCKET CONNECTION
// ==============================================================================
let ws = null;
function initWebSocket() {
    const wsUrl = `ws://${BACKEND_WS_HOST}/ws/stream`;
    const dot = document.getElementById('connection-dot');
    const label = document.getElementById('connection-status');

    try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => {
            if (dot) dot.classList.add('connected');
            if (label) label.innerText = 'ONLINE';
        };
        ws.onmessage = (event) => {
            const payload = JSON.parse(event.data);
            if (['SIMULATION_TICK', 'INIT_STATE', 'SIMULATION_RESET'].includes(payload.type)) {
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

// KHOI DONG DONG BO
initControlButtons();
initCheckpointsDropdown();
initPolicySandboxControls();
initLogTabs();
initMacroChart();
initTopologyGraph();
initWebSocket();