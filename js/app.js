/**
 * app.js — Orchestrator chính v3
 *
 * Hỗ trợ 2 chế độ AI:
 *   'ql'   — Q-Learning (3 state × 3 action, tabular) — logic gốc
 *   'ddpg' — DDPG (state vector 10D, continuous alpha/rho)
 *
 * Switch mode bằng nút "btn-mode-switch" trên header.
 */

import { Graph }          from './graph.js';
import { ACOEngine }      from './aco.js';
import { QLearningAgent, STATE_LABELS, ACTION_PARAMS, STATES } from './qlearning.js';
import { TSPEnvironment } from './environment.js';
import { DDPGAgent }      from './ddpg_agent.js';
import { Renderer }       from './renderer.js';
import { ChartManager }   from './chart_manager.js';
import { HeadlessTrainer } from './trainer.js';
import { Evaluator }       from './evaluator.js';

// =====================================================================
// CONSTANTS
// =====================================================================
const NUM_NODES     = 20;
const NUM_ANTS      = 30;
const MS_PER_GEN    = 120;
const PARTICLE_RATE = 3;

// State feature labels (cho DDPG state vector display)
const STATE_FEATURE_LABELS = [
  'Cost norm', 'Improve rate', 'Stuck norm',
  'Avg phero', 'Std phero', 'Blocked ratio',
  'α norm', 'ρ norm', 'Gen progress', 'Entropy',
];

// =====================================================================
// APP STATE
// =====================================================================
let graph    = null;
let aco      = null;
let renderer = null;
let chartMgr = null;

// Q-Learning mode
let qlAgent  = null;

// DDPG mode
let ddpgAgent = null;
let env       = null;

// Shared simulation state
let mode        = 'ql';       // 'ql' | 'ddpg'
let isRunning   = false;
let isBlockTool = false;
let generation  = 0;
let bestPath    = null;
let bestCost    = Infinity;
let loopHandle  = null;
let isProcessingGen = false; // Ngăn chặn chồng chéo async (chống đơ)

// DDPG train cadence
let genSinceLastTrain = 0;

// =====================================================================
// DOM ELEMENTS
// =====================================================================
const mainCanvas      = document.getElementById('main-canvas');
const btnStartPause   = document.getElementById('btn-start-pause');
const btnBlockTool    = document.getElementById('btn-block-tool');
const btnReset        = document.getElementById('btn-reset');
const btnModeSwitch   = document.getElementById('btn-mode-switch');

// Shared UI
const elGenNum        = document.getElementById('gen-number');
const elBestDist      = document.getElementById('best-dist');
const elAlpha         = document.getElementById('param-alpha');
const elRho           = document.getElementById('param-rho');
const elEpsilon       = document.getElementById('param-epsilon');

// Q-Learning panel
const qlPanel         = document.getElementById('ql-panel');
const elStateLabel    = document.getElementById('state-label');
const elStateBadge    = document.getElementById('state-badge');
const elAction        = document.getElementById('current-action');
const elReward        = document.getElementById('last-reward');
const elQTable        = document.getElementById('qtable-display');

// DDPG panel
const ddpgPanel       = document.getElementById('ddpg-panel');
const elGaugeAlpha    = document.getElementById('gauge-alpha');
const elGaugeRho      = document.getElementById('gauge-rho');
const elValAlpha      = document.getElementById('val-alpha');
const elValRho        = document.getElementById('val-rho');
const elCriticLoss    = document.getElementById('critic-loss');
const elActorLoss     = document.getElementById('actor-loss');
const elBufferFill    = document.getElementById('buffer-fill');
const elOuNoise       = document.getElementById('ou-noise');
const elDdpgReward    = document.getElementById('ddpg-reward');
const elStateVec      = document.getElementById('state-vec-display');

// Header badge
const elAlgoLabel     = document.getElementById('algo-label');

// DOM - New Action Buttons
const btnTrainFast    = document.getElementById('btn-train-fast');
const btnExportModel  = document.getElementById('btn-export-model');
const btnImportModel  = document.getElementById('btn-import-model');
const btnRunTestSuite = document.getElementById('btn-run-test-suite');
const fileImportJson  = document.getElementById('file-import-json');
const fileImportBin   = document.getElementById('file-import-bin');
const btnClearTraffic = document.getElementById('btn-clear-traffic');

// DOM - Test Suite Modal
const testModal       = document.getElementById('test-modal');
const btnCloseModal   = document.getElementById('btn-close-modal');
const testTableBody   = document.getElementById('test-table-body');
let testCompareChart  = null;

// =====================================================================
// INITIALIZATION
// =====================================================================
function init() {
  resizeCanvas();
  if (chartMgr) chartMgr.destroy();

  graph    = new Graph(NUM_NODES, mainCanvas.width, mainCanvas.height);
  aco      = new ACOEngine(graph, { numAnts: NUM_ANTS });
  renderer = new Renderer(mainCanvas);
  chartMgr = new ChartManager('line-chart');

  graph.generate();
  chartMgr.initialize();

  generation = 0;
  bestPath   = null;
  bestCost   = Infinity;
  genSinceLastTrain = 0;

  // Init agents
  qlAgent = new QLearningAgent();

  if (mode === 'ddpg') {
    env = new TSPEnvironment(graph, aco);
    env.reset();
    if (ddpgAgent) ddpgAgent.resetNoise();
  }

  renderer.render(graph, bestPath, { generation, bestCost: 0 });

  if (mode === 'ql') {
    updateUIQL({ state: STATES.IMPROVING, action: 0, params: ACTION_PARAMS[0], reward: 0, epsilon: qlAgent.epsilon });
  } else {
    updateUIDDPG({ alpha: 1.0, rho: 0.1, reward: 0, trainInfo: null, stateVec: null });
  }

  setBlockToolActive(false);
  console.log('[App] Init:', NUM_NODES, 'nodes,', NUM_ANTS, 'ants, mode:', mode);
}

// =====================================================================
// SIMULATION LOOP — DISPATCH
// =====================================================================
async function runGeneration() {
  if (!isRunning || isProcessingGen) return;
  isProcessingGen = true;

  try {
    if (mode === 'ql') {
      runGenerationQL();
    } else {
      await runGenerationDDPG();
    }
  } catch (err) {
    console.error("[runGeneration] Error:", err);
  } finally {
    isProcessingGen = false;
  }
}

// ─────────────────────────────────────────────
// Q-LEARNING LOOP (giữ nguyên logic gốc)
// ─────────────────────────────────────────────
function runGenerationQL() {
  let qResult;
  if (generation === 0) {
    qResult = { action: 0, params: ACTION_PARAMS[0], state: STATES.IMPROVING, reward: 0, epsilon: qlAgent.epsilon };
  } else {
    qResult = qlAgent.step(bestCost);
    aco.setParams(qResult.params.alpha, qResult.params.rho);
    if (qResult.action === 1 && qResult.state === STATES.DEGRADED) {
      graph.resetPheromones();
    }
  }

  const { bestPath: roundBest, bestCost: roundCost } = aco.runIteration();

  if (roundBest && roundCost < bestCost) {
    bestCost = roundCost;
    bestPath = roundBest;
  }
  generation++;

  if (generation % PARTICLE_RATE === 0) renderer.spawnAntParticles(graph, bestPath);

  const displayDist = bestPath ? graph.pathDistance(bestPath) : 0;
  chartMgr.addDataPoint(generation, displayDist || bestCost, qResult.reward);
  renderer.render(graph, bestPath, { generation, bestCost: displayDist || bestCost });
  updateUIQL(qResult);
}

// ─────────────────────────────────────────────
// DDPG LOOP
// ─────────────────────────────────────────────
async function runGenerationDDPG() {
  if (!ddpgAgent || !env) return;

  // 1. Lấy state
  const state = env.getState();

  // 2. Actor chọn action liên tục
  const { rawAction, alpha, rho } = ddpgAgent.selectAction(state);

  // 3. Áp dụng tham số vào ACO
  aco.setParams(alpha, rho);
  if (env.stuckCounter >= 12) graph.resetPheromones(); // Nếu kẹt lâu, reset phero

  // 4. Chạy 1 iteration ACO
  const { bestPath: roundBest, bestCost: roundCost } = aco.runIteration();

  if (roundBest && roundCost < bestCost) {
    bestCost = roundCost;
    bestPath = roundBest;
  }
  generation++;

  // 5. Tính reward và next state từ environment
  const { reward, nextState } = env.step(roundCost, alpha, rho);

  // 7. Lưu transition
  ddpgAgent.remember(state, rawAction, reward, nextState);

  // 8. Train mỗi 2 generation
  let trainInfo = null;
  genSinceLastTrain++;
  if (genSinceLastTrain >= 2) {
    genSinceLastTrain = 0;
    trainInfo = await ddpgAgent.train();
  }

  // 9. Render + UI
  if (generation % PARTICLE_RATE === 0) renderer.spawnAntParticles(graph, bestPath);

  const displayDist = bestPath ? graph.pathDistance(bestPath) : 0;
  chartMgr.addDataPoint(generation, displayDist || bestCost, reward);
  renderer.render(graph, bestPath, { generation, bestCost: displayDist || bestCost });
  updateUIDDPG({ alpha, rho, reward, trainInfo, stateVec: nextState });
}

// =====================================================================
// LOOP CONTROL
// =====================================================================
function startLoop() {
  if (loopHandle) clearInterval(loopHandle);
  loopHandle = setInterval(runGeneration, MS_PER_GEN);
}
function stopLoop() {
  if (loopHandle) { clearInterval(loopHandle); loopHandle = null; }
}

// =====================================================================
// MODE SWITCHING
// =====================================================================
async function switchMode(newMode) {
  stopLoop();
  isRunning = false;
  btnStartPause.textContent = '▶ Bắt đầu';
  btnStartPause.classList.remove('btn-pause');

  mode = newMode;

  if (newMode === 'ddpg') {
    // Khởi tạo DDPG agent nếu chưa có
    if (!ddpgAgent) {
      showToast('⏳ Đang khởi tạo mạng DDPG...');
      ddpgAgent = new DDPGAgent();
      await ddpgAgent.build();
    }
    env = new TSPEnvironment(graph, aco);
    env.reset();
    ddpgAgent.resetNoise();
    genSinceLastTrain = 0;

    // Show DDPG panel, hide QL panel
    if (qlPanel)   qlPanel.classList.add('panel-hidden');
    if (ddpgPanel) ddpgPanel.classList.remove('panel-hidden');
    if (elAlgoLabel) elAlgoLabel.textContent = 'ACO + DDPG';
    btnModeSwitch.classList.remove('mode-ql');
    btnModeSwitch.classList.add('mode-ddpg');
    btnModeSwitch.querySelector('.mode-label').textContent = 'DDPG';
    btnModeSwitch.querySelector('.mode-arrow').textContent = '→ Q-Learning';
    showToast('✅ Đã chuyển sang DDPG Mode');

  } else {
    // QL mode
    qlAgent = new QLearningAgent();

    if (qlPanel)   qlPanel.classList.remove('panel-hidden');
    if (ddpgPanel) ddpgPanel.classList.add('panel-hidden');
    if (elAlgoLabel) elAlgoLabel.textContent = 'ACO + Q-Learning';
    btnModeSwitch.classList.remove('mode-ddpg');
    btnModeSwitch.classList.add('mode-ql');
    btnModeSwitch.querySelector('.mode-label').textContent = 'Q-Learning';
    btnModeSwitch.querySelector('.mode-arrow').textContent = '→ DDPG';
    showToast('✅ Đã chuyển về Q-Learning Mode');
  }

  // Reset simulation state nhưng GIỮ graph
  generation = 0;
  bestPath   = null;
  bestCost   = Infinity;
  chartMgr.reset();

  renderer.render(graph, bestPath, { generation, bestCost: 0 });

  if (elGenNum)   elGenNum.textContent  = '0';
  if (elBestDist) elBestDist.textContent = '—';
}

// =====================================================================
// UI UPDATE — Q-LEARNING
// =====================================================================
function updateUIQL(qResult) {
  const { state, action, params, reward, epsilon } = qResult;

  if (elStateLabel) elStateLabel.textContent = STATE_LABELS[state] ?? 'BÌNH THƯỜNG';
  if (elStateBadge) {
    elStateBadge.className = 'state-badge';
    if      (state === STATES.IMPROVING) elStateBadge.classList.add('state-ok');
    else if (state === STATES.STUCK)     elStateBadge.classList.add('state-stuck');
    else                                 elStateBadge.classList.add('state-alert');
  }

  if (elAlpha)   elAlpha.textContent   = params.alpha.toFixed(2);
  if (elRho)     elRho.textContent     = params.rho.toFixed(2);
  if (elEpsilon) elEpsilon.textContent = (epsilon ?? qlAgent.epsilon).toFixed(3);
  if (elGenNum)  elGenNum.textContent  = generation;

  const displayDist = bestPath ? graph.pathDistance(bestPath) : 0;
  if (elBestDist) {
    elBestDist.textContent = displayDist > 0
      ? displayDist.toFixed(1)
      : (bestCost < Infinity ? bestCost.toFixed(1) : '—');
  }

  if (elAction) elAction.textContent = params.label ?? '—';
  if (elReward) {
    elReward.textContent = reward >= 0 ? `+${reward}` : `${reward}`;
    elReward.className = 'value ' + (reward > 0 ? 'reward-pos' : (reward < 0 ? 'reward-neg' : ''));
  }

  renderQTable();
}

function renderQTable() {
  if (!elQTable || !qlAgent) return;
  const stateNames  = ['S0', 'S1', 'S2'];
  const actionNames = ['A0', 'A1', 'A2'];

  let html = '<table class="qtable"><thead><tr><th></th>';
  for (const a of actionNames) html += `<th>${a}</th>`;
  html += '</tr></thead><tbody>';

  for (let s = 0; s < 3; s++) {
    const isCurrent = s === qlAgent.currentState;
    html += `<tr class="${isCurrent ? 'row-active' : ''}"><td class="qtable-state">${stateNames[s]}</td>`;
    const maxQ = Math.max(...qlAgent.qTable[s]);
    for (let a = 0; a < 3; a++) {
      const q = qlAgent.qTable[s][a];
      const isBest = (Math.abs(q - maxQ) < 0.001) && isCurrent;
      const intensity = Math.min(1, Math.max(0, (q - 0) / 20));
      const color = `rgba(74,158,255,${intensity * 0.5})`;
      html += `<td class="qtable-cell ${isBest ? 'q-best' : ''}" style="background:${color}">${q.toFixed(1)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  elQTable.innerHTML = html;
}

// =====================================================================
// UI UPDATE — DDPG
// =====================================================================
function updateUIDDPG({ alpha, rho, reward, trainInfo, stateVec }) {
  const { ALPHA_MIN, ALPHA_MAX, RHO_MIN, RHO_MAX } = { ALPHA_MIN: 0.3, ALPHA_MAX: 2.5, RHO_MIN: 0.02, RHO_MAX: 0.9 };

  // Gauge alpha
  const alphaPct = ((alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) * 100).toFixed(1);
  if (elGaugeAlpha) elGaugeAlpha.style.width = `${alphaPct}%`;
  if (elValAlpha)   elValAlpha.textContent   = alpha.toFixed(3);

  // Gauge rho
  const rhoPct = ((rho - RHO_MIN) / (RHO_MAX - RHO_MIN) * 100).toFixed(1);
  if (elGaugeRho) elGaugeRho.style.width = `${rhoPct}%`;
  if (elValRho)   elValRho.textContent   = rho.toFixed(3);

  // Training stats
  if (elCriticLoss) {
    elCriticLoss.textContent = trainInfo?.criticLoss != null
      ? trainInfo.criticLoss.toFixed(4) : '—';
  }
  if (elActorLoss) {
    elActorLoss.textContent = trainInfo?.actorLoss != null
      ? trainInfo.actorLoss.toFixed(4) : '—';
  }
  if (elBufferFill && ddpgAgent) {
    const bs = ddpgAgent.bufferSize;
    elBufferFill.textContent = `${bs}/5000`;
    elBufferFill.className = bs >= 200 ? 'value reward-pos' : 'value';
  }
  if (elOuNoise && ddpgAgent) {
    elOuNoise.textContent = ddpgAgent.noiseLevel.toFixed(3);
  }
  if (elDdpgReward) {
    elDdpgReward.textContent = reward >= 0 ? `+${reward}` : `${reward}`;
    elDdpgReward.className = 'value ' + (reward > 0 ? 'reward-pos' : (reward < 0 ? 'reward-neg' : ''));
  }

  // Shared
  if (elAlpha)   elAlpha.textContent  = alpha.toFixed(3);
  if (elRho)     elRho.textContent    = rho.toFixed(3);
  if (elEpsilon) elEpsilon.textContent = ddpgAgent ? ddpgAgent.noiseLevel.toFixed(3) : '—';
  if (elGenNum)  elGenNum.textContent = generation;

  const displayDist = bestPath ? graph.pathDistance(bestPath) : 0;
  if (elBestDist) {
    elBestDist.textContent = displayDist > 0
      ? displayDist.toFixed(1)
      : (bestCost < Infinity ? bestCost.toFixed(1) : '—');
  }

  // State vector mini bars
  if (elStateVec && stateVec) {
    renderStateVector(stateVec);
  }
}

function renderStateVector(stateVec) {
  if (!elStateVec) return;
  let html = '';
  for (let i = 0; i < stateVec.length; i++) {
    const val = stateVec[i];
    const pct = (Math.min(1, Math.max(0, val)) * 100).toFixed(1);
    const color = val > 0.7 ? '#ff6b6b' : val > 0.4 ? '#ffd700' : '#4a9eff';
    html += `
      <div class="sv-cell">
        <div class="sv-label">${STATE_FEATURE_LABELS[i]}</div>
        <div class="sv-bar-track">
          <div class="sv-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <div class="sv-val">${val.toFixed(2)}</div>
      </div>`;
  }
  elStateVec.innerHTML = html;
}

// =====================================================================
// EVENT HANDLERS
// =====================================================================
btnStartPause.addEventListener('click', () => {
  isRunning = !isRunning;
  if (isRunning) {
    btnStartPause.textContent = '⏸ Tạm dừng';
    btnStartPause.classList.add('btn-pause');
    startLoop();
  } else {
    btnStartPause.textContent = '▶ Bắt đầu';
    btnStartPause.classList.remove('btn-pause');
    stopLoop();
  }
});

btnBlockTool.addEventListener('click', () => {
  isBlockTool = !isBlockTool;
  setBlockToolActive(isBlockTool);
});

if (btnClearTraffic) {
  btnClearTraffic.addEventListener('click', () => {
    if (!graph) return;
    graph.clearAllBlockedEdges();
    if (mode === 'ql' && qlAgent) {
      qlAgent.triggerTrafficEvent(); // Reset baseline
    } else if (mode === 'ddpg' && env) {
      env.triggerTrafficEvent(); // Reset baseline
    }
    // Cần reset lại bestCost hiển thị
    bestCost = Infinity;
    showToast('Đã gỡ bỏ toàn bộ tắc đường!');
    renderer.render(graph, bestPath, { generation, bestCost });
  });
}

btnReset.addEventListener('click', () => {
  stopLoop();
  isRunning = false;
  isBlockTool = false;
  btnStartPause.textContent = '▶ Bắt đầu';
  btnStartPause.classList.remove('btn-pause');
  setBlockToolActive(false);
  init();
});

btnModeSwitch.addEventListener('click', async () => {
  const next = mode === 'ql' ? 'ddpg' : 'ql';
  await switchMode(next);
});

// =====================================================================
// NEW ACTIONS: HEADLESS TRAIN & TEST SUITE
// =====================================================================

// 1. Train Fast (Headless)
if (btnTrainFast) {
  btnTrainFast.addEventListener('click', async () => {
    if (mode !== 'ddpg') {
      showToast('Chỉ có thể Train ở chế độ DDPG!');
      return;
    }
    stopLoop();
    btnStartPause.textContent = '▶ Bắt đầu';
    isRunning = false;
    
    showToast('Đang khởi động Headless Trainer...');
    const trainer = new HeadlessTrainer(graph, ddpgAgent);
    
    btnTrainFast.disabled = true;
    btnTrainFast.textContent = 'Đang train... (Xem log)';
    
    await trainer.train(2000, (gen, bCost, aLoss, noise) => {
      // Cập nhật UI nhẹ nhàng để không đơ
      if (elGenNum) elGenNum.textContent = gen;
      if (elBestDist) elBestDist.textContent = bCost.toFixed(1);
    });
    
    showToast('Train hoàn tất!');
    btnTrainFast.disabled = false;
    btnTrainFast.textContent = '⚡ Train DDPG (Fast)';
    
    // Gắn state mới nhất lên bảng hiển thị
    chartMgr.addDataPoint(2000, bestCost, 0);
  });
}

// 2. Export / Import Model
if (btnExportModel) {
  btnExportModel.addEventListener('click', async () => {
    if (mode === 'ddpg' && ddpgAgent) {
      showToast('Đang xuất mô hình...');
      await ddpgAgent.exportWeights();
      showToast('Đã tải xuống json và bin!');
    }
  });
}

if (btnImportModel) {
  btnImportModel.addEventListener('click', () => {
    fileImportJson.click();
  });
}
if (fileImportJson) {
  fileImportJson.addEventListener('change', (e) => {
    if (!e.target.files.length) return;
    // Bắt user chọn nốt file bin
    showToast('Vui lòng chọn tiếp file .bin');
    fileImportBin.click();
  });
}
if (fileImportBin) {
  fileImportBin.addEventListener('change', async (e) => {
    if (!e.target.files.length || !fileImportJson.files.length) return;
    
    if (mode === 'ddpg' && ddpgAgent) {
      showToast('Đang tải mô hình...');
      const success = await ddpgAgent.importWeights(fileImportJson.files[0], fileImportBin.files[0]);
      if (success) showToast('Tải mô hình thành công!');
      else showToast('Lỗi tải mô hình!');
    }
  });
}

// 3. Test Suite
if (btnRunTestSuite) {
  btnRunTestSuite.addEventListener('click', async () => {
    if (!ddpgAgent) return;
    stopLoop();
    isRunning = false;
    btnStartPause.textContent = '▶ Bắt đầu';
    
    showToast('Đang chạy Test Suite... Vui lòng đợi.');
    btnRunTestSuite.disabled = true;
    btnRunTestSuite.textContent = 'Đang đánh giá...';
    
    const evaluator = new Evaluator();
    const results = await evaluator.runTestSuite(ddpgAgent, (msg, pct) => {
      btnRunTestSuite.textContent = `Đang đánh giá... ${Math.round(pct)}%`;
    });
    
    btnRunTestSuite.disabled = false;
    btnRunTestSuite.textContent = '📊 Run Test Suite';
    
    showTestResults(results);
  });
}

if (btnCloseModal) {
  btnCloseModal.addEventListener('click', () => {
    testModal.classList.add('hidden');
  });
}

function showTestResults(results) {
  testModal.classList.remove('hidden');
  testTableBody.innerHTML = '';
  
  const labels = [];
  const qlData = [];
  const ddpgData = [];

  for (const r of results) {
    labels.push(r.id);
    
    const qlErr = ((r.ql.bestCost - r.optimal) / r.optimal * 100);
    const ddpgErr = ((r.ddpg.bestCost - r.optimal) / r.optimal * 100);
    
    qlData.push(qlErr);
    ddpgData.push(ddpgErr);

    const qlErrStr = qlErr < 0.1 ? 'Optimal' : `+${qlErr.toFixed(2)}%`;
    const ddpgErrStr = ddpgErr < 0.1 ? 'Optimal' : `+${ddpgErr.toFixed(2)}%`;
    
    const html = `
      <tr>
        <td>${r.name}</td>
        <td class="td-optimal">${r.optimal.toFixed(1)}</td>
        <td>${r.ql.bestCost.toFixed(1)}</td>
        <td class="${qlErr < 0.1 ? 'td-perfect' : 'td-error'}">${qlErrStr}</td>
        <td>${r.ddpg.bestCost.toFixed(1)}</td>
        <td class="${ddpgErr < 0.1 ? 'td-perfect' : 'td-error'}">${ddpgErrStr}</td>
      </tr>
    `;
    testTableBody.insertAdjacentHTML('beforeend', html);
  }
  
  // Render Chart
  const ctx = document.getElementById('test-compare-chart').getContext('2d');
  if (testCompareChart) testCompareChart.destroy();
  
  testCompareChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'QL-ACO Error %',
          data: qlData,
          backgroundColor: 'rgba(74,158,255,0.7)',
        },
        {
          label: 'DDPG Error %',
          data: ddpgData,
          backgroundColor: 'rgba(167,139,250,0.7)',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, title: { display: true, text: 'Error (%) vs Optimal' } }
      }
    }
  });
}

// Canvas click — tắc đường
mainCanvas.addEventListener('click', (e) => {
  if (!isBlockTool || !graph) return;
  const rect = mainCanvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (mainCanvas.width / rect.width);
  const my = (e.clientY - rect.top)  * (mainCanvas.height / rect.height);

  const edge = renderer.findNearestEdge(graph, mx, my, 18);
  if (edge) {
    const blocked = graph.blockEdge(edge.i, edge.j);
    if (blocked) {
      if (mode === 'ql') {
        qlAgent.triggerTrafficEvent();
      } else if (env) {
        env.triggerTrafficEvent();
      }
      showToast(`⚠️ Tắc đường: cạnh ${edge.i}↔${edge.j} (×${graph.TRAFFIC_MULTIPLIER})`);
      if (bestPath) {
        const affectsPath = bestPath.some((v, k) => {
          if (k >= bestPath.length - 1) return false;
          const a = bestPath[k], b = bestPath[k+1];
          return (a === edge.i && b === edge.j) || (a === edge.j && b === edge.i);
        });
        if (affectsPath) bestCost = Infinity;
      }
    } else {
      showToast('Cạnh này đã bị tắc đường rồi!');
    }
    renderer.render(graph, bestPath, { generation, bestCost });
  }
});

// Canvas hover
mainCanvas.addEventListener('mousemove', (e) => {
  if (!graph || !renderer) return;
  const rect = mainCanvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (mainCanvas.width / rect.width);
  const my = (e.clientY - rect.top)  * (mainCanvas.height / rect.height);
  if (isBlockTool) {
    const edge = renderer.findNearestEdge(graph, mx, my, 15);
    mainCanvas.style.cursor = edge ? 'crosshair' : 'default';
  } else {
    mainCanvas.style.cursor = 'default';
  }
});

// =====================================================================
// HELPERS
// =====================================================================
function setBlockToolActive(active) {
  isBlockTool = active;
  btnBlockTool.classList.toggle('tool-active', active);
  btnBlockTool.textContent = active ? '🚫 Đang chọn cạnh...' : '🚧 Tạo Tắc Đường';
  mainCanvas.style.cursor = active ? 'crosshair' : 'default';
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3000);
}

function resizeCanvas() {
  const container = document.getElementById('canvas-container');
  if (!container) return;
  mainCanvas.width  = container.clientWidth;
  mainCanvas.height = container.clientHeight;
}

window.addEventListener('resize', () => {
  resizeCanvas();
  if (graph) {
    const oldW = graph.canvasWidth, oldH = graph.canvasHeight;
    graph.canvasWidth  = mainCanvas.width;
    graph.canvasHeight = mainCanvas.height;
    const sx = mainCanvas.width / oldW, sy = mainCanvas.height / oldH;
    for (const node of graph.nodes) { node.x *= sx; node.y *= sy; }
    graph._computeDistances();
    renderer.render(graph, bestPath, { generation, bestCost });
  }
});

// =====================================================================
// BOOTSTRAP
// =====================================================================
window.addEventListener('DOMContentLoaded', () => {
  // Bắt đầu với QL mode
  if (qlPanel)   qlPanel.classList.remove('panel-hidden');
  if (ddpgPanel) ddpgPanel.classList.add('panel-hidden');

  init();

  // Animation loop liên tục ngay cả khi pause
  function animLoop() {
    if (!isRunning && graph) {
      renderer.render(graph, bestPath, { generation, bestCost });
    }
    requestAnimationFrame(animLoop);
  }
  requestAnimationFrame(animLoop);
});

// Debug object — có thể truy cập qua console
window.__smartRoute = { getGraph: () => graph, getAco: () => aco, getMode: () => mode,
  getDdpgAgent: () => ddpgAgent, getEnv: () => env };
