// ==============================================================================
// 1. GLOBAL STATE & THEME CONSTANTS
// ==============================================================================
const state = {
    agents: {},
    macro: {},
    links: [],
    history: { months: [], gdp: [], gini: [] },
    selectedAgentId: null,
    // Log tab "Chi tiết": nhóm event đang được lọc hiển thị ('all' = không lọc)
    logFilter: 'all',
    // Log tab "Xu hướng": snapshot macro của lần digest gần nhất, dùng để tính delta
    trendSnapshot: null
};

// Nhom hoa toan bo EventType (be/core/enums.py) ve 1 trong 7 nhom co dinh --
// dung chung cho ca mau sac (CSS .log-entry.cat-*) lan bo loc chip, thay vi
// to mau/dinh dang rieng cho tung loai event nhu thiet ke cu (rat loang).
const EVENT_CATEGORY = {
    HIRE: 'labor', FIRE: 'labor', WAGE_PAID: 'labor',
    GOODS_PURCHASED: 'market', PRICE_ADJUSTED: 'market',
    TAX_COLLECTED: 'fiscal', TAX_EVADED: 'fiscal', PENALTY_ENFORCED: 'fiscal', AUDIT_CONDUCTED: 'fiscal',
    LOAN_DISBURSED: 'credit', LOAN_REPAID: 'credit', DEFAULT_OCCURRED: 'credit',
    AGENT_BORN: 'lifecycle', AGENT_DIED: 'lifecycle', AGENT_BANKRUPT: 'lifecycle',
    POLICY_SHOCK: 'policy',
    STATE_UPDATE: 'system'
};

const CATEGORY_LABEL = {
    labor: 'LAO ĐỘNG', market: 'THỊ TRƯỜNG', fiscal: 'TÀI KHOÁ',
    credit: 'TÍN DỤNG', lifecycle: 'VÒNG ĐỜI', policy: 'CHÍNH SÁCH', system: 'HỆ THỐNG'
};

// 1 QUY CHUAN DUY NHAT cho moi loai event -- moi nhanh if/else rieng le nhu
// truoc (dan den bug p.hidden_pct khong ton tai) duoc gop thanh 1 bang tra
// cuu duy nhat, moi loai event PHAI duoc xu ly tuong minh (khong con fallback
// JSON.stringify tho cho cac event thuong gap nhu HIRE/FIRE/LOAN_REPAID...).
function describeEvent(e) {
    const p = e.payload || {};
    const src = (e.source || '').toUpperCase();
    const tgt = (e.target || '').toUpperCase();
    switch (e.type) {
        case 'HIRE': return `${src} tuyển ${tgt} — lương $${p.wage ?? 0}`;
        case 'FIRE': return `${src} sa thải ${tgt}`;
        case 'WAGE_PAID': return `${src} trả lương ${tgt}: $${(typeof p.amount === 'number' ? p.amount.toFixed(1) : p.amount ?? 0)}`;
        case 'GOODS_PURCHASED': return `${src} mua hàng từ thị trường: $${p.amount ?? 0}`;
        case 'PRICE_ADJUSTED': return `Thị trường điều chỉnh giá — ${JSON.stringify(p)}`;
        case 'TAX_COLLECTED': return `Kho bạc thu thuế kỳ này: $${p.amount ?? 0}`;
        case 'TAX_EVADED':
            if (p.gross !== undefined) return `${src} khai thiếu thu nhập (gross $${p.gross}), trốn $${p.amount} thuế`;
            return `${src} khai thiếu lợi nhuận (lãi $${p.profit}), trốn $${p.amount} thuế DN`;
        case 'PENALTY_ENFORCED': return `Thanh tra phạt ${tgt}: $${p.fine} (thuế trốn $${p.evaded})`;
        case 'AUDIT_CONDUCTED': return `Thanh tra kiểm toán ${tgt || src}`;
        case 'LOAN_DISBURSED': return `Ngân hàng ${src} giải ngân cho ${tgt}: $${p.amount ?? 0}`;
        case 'LOAN_REPAID': return `${src} trả nợ ngân hàng: $${p.amount ?? 0}`;
        case 'DEFAULT_OCCURRED': return `${src} vỡ nợ — nợ xấu $${p.bad_debt ?? p.amount ?? 0}`;
        case 'AGENT_BORN': {
            const skillTxt = typeof p.skill === 'number' ? p.skill.toFixed(2) : p.skill;
            const parentTxt = p.parent_id ? `, thừa kế từ ${String(p.parent_id).toUpperCase()}` : '';
            return `${tgt} gia nhập xã hội (vốn mồi $${p.cash}, kỹ năng ${skillTxt}${parentTxt})`;
        }
        case 'AGENT_DIED': {
            const debtTxt = (typeof p.bad_debt === 'number' && p.bad_debt > 0) ? `, để lại nợ xấu $${p.bad_debt.toFixed(1)}` : '';
            return `${src} qua đời lúc ${p.age} tuổi [${p.reason ?? 'không rõ lý do'}]. Di sản thu về Kho bạc${debtTxt}`;
        }
        case 'AGENT_BANKRUPT': {
            const recovered = p.recovered !== undefined ? ` (thu hồi $${p.recovered})` : '';
            const badDebtTxt = typeof p.bad_debt === 'number' ? p.bad_debt.toFixed(1) : p.bad_debt;
            return `${src} giải thể — ${tgt || 'ngân hàng'} ghi nhận nợ xấu $${badDebtTxt}${recovered}`;
        }
        case 'POLICY_SHOCK': return `Can thiệp chính sách: Thuế CN ${p.worker_tax}%, Thuế DN ${p.firm_tax}%, Lãi suất ${p.lending_rate}%, Sàn sống $${p.living_cost}`;
        default: return `${src} → ${tgt} | ${JSON.stringify(p)}`;
    }
}

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
// 1B. API KEY (server yêu cầu header X-API-Key cho mọi /api/*, xem be/server.py)
// ==============================================================================
function getApiKey() {
    try {
        return localStorage.getItem('ai_econ_api_key') || '';
    } catch (e) {
        return '';
    }
}

function setApiKey(key) {
    try {
        localStorage.setItem('ai_econ_api_key', key);
    } catch (e) { /* private mode / storage blocked -- ignore, key just won't persist */ }
}

function showToast(message) {
    const banner = document.getElementById('toast-banner');
    const msg = document.getElementById('toast-message');
    if (!banner || !msg) return;
    msg.innerText = message;
    banner.classList.add('visible');
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => banner.classList.remove('visible'), 6000);
}

function initApiKeyInput() {
    const group = document.getElementById('api-key-group');
    const input = document.getElementById('api-key-input');
    if (!input) return;
    input.value = getApiKey();
    if (group) group.classList.toggle('needs-key', !input.value);
    input.addEventListener('change', () => {
        setApiKey(input.value.trim());
        if (group) group.classList.toggle('needs-key', !input.value.trim());
        // Khoá mới chưa được WebSocket hiện tại dùng (token gắn lúc bắt tay) --
        // buộc kết nối lại để áp dụng khoá mới ngay lập tức.
        if (ws) { try { ws.close(); } catch (e) {} }
    });
}

// ==============================================================================
// 2. REST API & CONTROLS BINDING
// ==============================================================================
async function sendControl(action, value = null) {
    try {
        const res = await fetch(`${BACKEND_HTTP}/api/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-API-Key': getApiKey() },
            body: JSON.stringify({ action, value })
        });
        if (res.status === 401) {
            showToast('Invalid or missing API key. Enter the key printed in the server console (top-right field).');
            return null;
        }
        return await res.json();
    } catch (err) {
        console.error('[FE] Failed to dispatch control:', err);
        return null;
    }
}

function resetHistoryAndLogs() {
    state.history.months = [];
    state.history.gdp = [];
    state.history.gini = [];
    state.trendSnapshot = null;
    if (macroChart) {
        macroChart.data.labels = [];
        macroChart.data.datasets[0].data = [];
        macroChart.data.datasets[1].data = [];
        macroChart.update();
    }
    const trendStream = document.getElementById('trend-stream-container');
    const detailStream = document.getElementById('detail-stream-container');
    if (trendStream) trendStream.innerHTML = '<div class="empty-log-hint">Chưa có dữ liệu xu hướng — nhấn RUN để bắt đầu (digest mỗi 12 tháng).</div>';
    if (detailStream) detailStream.innerHTML = '<div class="empty-log-hint">Chưa có sự kiện nào.</div>';
    const countEl = document.getElementById('log-count');
    if (countEl) countEl.innerText = '0 sự kiện';
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
        resetHistoryAndLogs();
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

// ==============================================================================
// 2B. SIMULATION SETUP (quy mô Worker/Firm/Bank -- CONFIGURE_POPULATION)
// ==============================================================================
function initSetupPanel() {
    const btn = document.getElementById('btn-apply-setup');
    if (!btn) return;

    const readInt = (id, fallback) => {
        const el = document.getElementById(id);
        const v = el ? parseInt(el.value, 10) : NaN;
        return Number.isFinite(v) ? v : fallback;
    };

    btn.addEventListener('click', async () => {
        const payload = {
            num_employees: readInt('setup-num-workers', 50),
            num_firms: readInt('setup-num-firms', 5),
            num_banks: readInt('setup-num-banks', 1)
        };

        btn.disabled = true;
        btn.innerText = 'Restarting…';
        try {
            const result = await sendControl('CONFIGURE_POPULATION', payload);
            if (result && result.population_config) {
                // Phản chiếu giá trị thực tế đã được server kẹp biên (clip) trở lại UI
                const cfg = result.population_config;
                const wEl = document.getElementById('setup-num-workers');
                const fEl = document.getElementById('setup-num-firms');
                const bEl = document.getElementById('setup-num-banks');
                if (wEl) wEl.value = cfg.num_employees;
                if (fEl) fEl.value = cfg.num_firms;
                if (bEl) bEl.value = cfg.num_banks;
                resetHistoryAndLogs();
            }
        } finally {
            btn.disabled = false;
            btn.innerText = 'Apply & New Simulation';
        }
    });
}

async function initCheckpointsDropdown() {
    const select = document.getElementById('checkpoint-select');
    if (!select) return;

    try {
        const res = await fetch(`${BACKEND_HTTP}/api/checkpoints`, { headers: { 'X-API-Key': getApiKey() } });
        if (res.status === 401) {
            showToast('Invalid or missing API key -- cannot list checkpoints.');
            return;
        }
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
            resetHistoryAndLogs();
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

        maybeEmitTrendDigest(payload);
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
            appendDetailLog(e);
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
// 6. LOG THEO 2 MUC DICH: XU HUONG (digest dinh ky) + CHI TIET (1 quy chuan)
// ==============================================================================

// --- Tab "Chi tiết": append 1 dong/event, 1 QUY CHUAN DUY NHAT ---
function appendDetailLog(e) {
    const stream = document.getElementById('detail-stream-container');
    if (!stream) return;

    const placeholder = stream.querySelector('.empty-log-hint');
    if (placeholder) placeholder.remove();

    const category = EVENT_CATEGORY[e.type] || 'system';
    const label = CATEGORY_LABEL[category] || 'HỆ THỐNG';
    const textContent = `[M${e.timestep}] ${label} — ${describeEvent(e)}`;

    const entry = document.createElement('div');
    entry.className = `log-entry cat-${category}`;
    entry.dataset.cat = category;
    entry.innerText = textContent;
    if (state.logFilter !== 'all' && state.logFilter !== category) {
        entry.classList.add('hidden-by-filter');
    }
    stream.prepend(entry);

    if (stream.children.length > 200) {
        stream.removeChild(stream.lastChild);
    }

    const countEl = document.getElementById('log-count');
    if (countEl) {
        const total = document.querySelectorAll('#detail-stream-container .log-entry').length;
        countEl.innerText = `${total} sự kiện`;
    }
}

// --- Tab "Xu hướng": digest ĐỊNH KỲ (mỗi 12 tháng = 1 năm mô phỏng), so
// sánh với snapshot lần trước thay vì hiển thị từng event thô ---
const TREND_PERIOD_MONTHS = 12;

function fmtTrendDelta(curr, prev, unit, goodDirection) {
    const diff = curr - prev;
    if (Math.abs(diff) < 1e-9) return `<span class="trend-delta neutral">±0${unit}</span>`;
    const sign = diff > 0 ? '+' : '';
    const isGood = (goodDirection === 'up' && diff > 0) || (goodDirection === 'down' && diff < 0);
    const cls = goodDirection === 'neutral' ? 'neutral' : (isGood ? 'up-good' : 'up-bad');
    return `<span class="trend-delta ${cls}">${sign}${diff.toFixed(1)}${unit}</span>`;
}

function renderTrendDigest(prev, curr) {
    const stream = document.getElementById('trend-stream-container');
    if (!stream) return;
    const placeholder = stream.querySelector('.empty-log-hint');
    if (placeholder) placeholder.remove();

    const year = Math.floor(curr.timestep / TREND_PERIOD_MONTHS);
    const card = document.createElement('div');
    card.className = 'trend-card';
    card.innerHTML = `
        <div class="trend-title">Năm ${year} · Tháng ${curr.timestep}</div>
        <div class="trend-row">
            <span class="trend-metric">GDP $${Math.round(curr.gdp).toLocaleString()} ${fmtTrendDelta(curr.gdp, prev.gdp, '', 'up')}</span>
            <span class="trend-metric">Gini ${curr.gini.toFixed(3)} ${fmtTrendDelta(curr.gini, prev.gini, '', 'down')}</span>
            <span class="trend-metric">Thất nghiệp ${curr.unemployment.toFixed(1)}% ${fmtTrendDelta(curr.unemployment, prev.unemployment, 'đ', 'down')}</span>
            <span class="trend-metric">NPL ${curr.nplRatio.toFixed(1)}% ${fmtTrendDelta(curr.nplRatio, prev.nplRatio, 'đ', 'down')}</span>
            <span class="trend-metric">Firm ${curr.firms} ${fmtTrendDelta(curr.firms, prev.firms, '', 'neutral')}</span>
            <span class="trend-metric">Dân số ${curr.population} ${fmtTrendDelta(curr.population, prev.population, '', 'neutral')}</span>
        </div>`;
    stream.prepend(card);

    if (stream.children.length > 60) {
        stream.removeChild(stream.lastChild);
    }
}

function maybeEmitTrendDigest(payload) {
    if (!payload.timestep || payload.timestep % TREND_PERIOD_MONTHS !== 0) return;
    const m = payload.macro || {};
    const snapshot = {
        timestep: payload.timestep,
        gdp: Number(m.gdp) || 0,
        gini: Number(m.gini) || 0,
        unemployment: Number(m.unemployment_rate_pct) || 0,
        nplRatio: Number(m.npl_ratio_pct) || 0,
        firms: Number(m.active_firms) || 0,
        population: Number(m.active_employees) || 0
    };

    if (state.trendSnapshot) {
        renderTrendDigest(state.trendSnapshot, snapshot);
    }
    state.trendSnapshot = snapshot;
}

function initLogTabs() {
    const btnTrend = document.getElementById('tab-trend');
    const btnDetail = document.getElementById('tab-detail');
    const streamTrend = document.getElementById('trend-stream-container');
    const streamDetail = document.getElementById('detail-stream-container');
    const filterRow = document.getElementById('log-filter-row');

    if (btnTrend && btnDetail) {
        btnTrend.addEventListener('click', () => {
            btnTrend.classList.add('active');
            btnDetail.classList.remove('active');
            streamTrend.style.display = 'flex';
            streamDetail.style.display = 'none';
            if (filterRow) filterRow.style.display = 'none';
        });

        btnDetail.addEventListener('click', () => {
            btnDetail.classList.add('active');
            btnTrend.classList.remove('active');
            streamTrend.style.display = 'none';
            streamDetail.style.display = 'flex';
            if (filterRow) filterRow.style.display = 'flex';
        });
    }

    document.querySelectorAll('.log-filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            document.querySelectorAll('.log-filter-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            const cat = chip.dataset.cat;
            state.logFilter = cat;
            document.querySelectorAll('#detail-stream-container .log-entry').forEach(entry => {
                entry.classList.toggle('hidden-by-filter', cat !== 'all' && entry.dataset.cat !== cat);
            });
        });
    });
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
    // Trình duyệt không cho gắn custom header khi bắt tay WebSocket, nên token
    // được truyền qua query string (be/server.py đọc ?token=... trên endpoint
    // /ws/stream, khác với header X-API-Key dùng cho /api/*).
    const wsUrl = `ws://${BACKEND_WS_HOST}/ws/stream?token=${encodeURIComponent(getApiKey())}`;
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
            // Lưu ý: server từ chối bắt tay (close TRƯỚC accept()) khi token sai
            // -- trình duyệt nhận đây là handshake thất bại (không có close code
            // tuỳ chỉnh để phân biệt với mất mạng thông thường), nên không hiển
            // thị toast riêng ở đây; cảnh báo API key sai đã hiển thị đầy đủ qua
            // các request REST /api/* (xem sendControl()).
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
initApiKeyInput();
initControlButtons();
initCheckpointsDropdown();
initPolicySandboxControls();
initSetupPanel();
initLogTabs();
initMacroChart();
initTopologyGraph();
initWebSocket();