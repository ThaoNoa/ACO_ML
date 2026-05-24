/**
 * app.js - Orchestrator chính
 * Kết nối toàn bộ các module lại:
 *   - Quản lý vòng lặp mô phỏng (generation loop)
 *   - Xử lý events từ UI (Start/Pause, tắc đường)
 *   - Cập nhật UI (trạng thái AI, thông số Live)
 *   - Điều phối giữa Q-Learning <-> ACO <-> Renderer
 */

import { Graph }         from './graph.js';
import { ACOEngine }     from './aco.js';
import { QLearningAgent, STATE_LABELS, ACTION_PARAMS, STATES } from './qlearning.js';
import { Renderer }      from './renderer.js';
import { ChartManager }  from './chart_manager.js';

// =====================================================================
// CONSTANTS
// =====================================================================
const NUM_NODES       = 20;
const NUM_ANTS        = 30;
const MS_PER_GEN      = 120;   // Thời gian mỗi generation (ms)
const PARTICLE_RATE   = 3;     // Spawn kiến mỗi N generation

// =====================================================================
// STATE
// =====================================================================
let graph    = null;
let aco      = null;
let agent    = null;
let renderer = null;
let chartMgr = null;

let isRunning     = false;
let isBlockTool   = false;   // Chế độ tạo tắc đường
let generation    = 0;
let bestPath      = null;
let bestCost      = Infinity;
let loopHandle    = null;

// =====================================================================
// DOM ELEMENTS
// =====================================================================
const mainCanvas    = document.getElementById('main-canvas');
const btnStartPause = document.getElementById('btn-start-pause');
const btnBlockTool  = document.getElementById('btn-block-tool');
const btnReset      = document.getElementById('btn-reset');

const elStateLabel  = document.getElementById('state-label');
const elStateBadge  = document.getElementById('state-badge');
const elAlpha       = document.getElementById('param-alpha');
const elRho         = document.getElementById('param-rho');
const elEpsilon     = document.getElementById('param-epsilon');
const elGenNum      = document.getElementById('gen-number');
const elBestDist    = document.getElementById('best-dist');
const elAction      = document.getElementById('current-action');
const elReward      = document.getElementById('last-reward');
const elQTable      = document.getElementById('qtable-display');

// =====================================================================
// INITIALIZATION
// =====================================================================
function init() {
  resizeCanvas();

  // Hủy chart cũ trước khi tạo mới — tránh lỗi "Canvas is already in use" của Chart.js
  if (chartMgr) chartMgr.destroy();

  graph    = new Graph(NUM_NODES, mainCanvas.width, mainCanvas.height);
  aco      = new ACOEngine(graph, { numAnts: NUM_ANTS });
  agent    = new QLearningAgent();
  renderer = new Renderer(mainCanvas);
  chartMgr = new ChartManager('line-chart');

  graph.generate();
  chartMgr.initialize();

  generation = 0;
  bestPath   = null;
  bestCost   = Infinity;

  // Render frame đầu tiên (trống)
  renderer.render(graph, bestPath, { generation, bestCost: 0 });

  updateUI({ state: STATES.IMPROVING, action: 0, params: ACTION_PARAMS[0], reward: 0, epsilon: agent.epsilon });
  setBlockToolActive(false);

  console.log('[App] Initialized with', NUM_NODES, 'nodes,', NUM_ANTS, 'ants');
}

// =====================================================================
// SIMULATION LOOP
// =====================================================================
function runGeneration() {
  if (!isRunning) return;

  // --- 1. Q-Learning: xác định tham số ACO ---
  let qResult;
  if (generation === 0) {
    qResult = { action: 0, params: ACTION_PARAMS[0], state: STATES.IMPROVING, reward: 0, epsilon: agent.epsilon };
  } else {
    qResult = agent.step(bestCost);
    aco.setParams(qResult.params.alpha, qResult.params.rho);

    // Nếu Q-Learning chọn A1 (Explore) và đang degraded, reset pheromone
    if (qResult.action === 1 && qResult.state === STATES.DEGRADED) {
      graph.resetPheromones();
    }
  }

  // --- 2. ACO: chạy một vòng ---
  const { bestPath: roundBest, bestCost: roundCost } = aco.runIteration();

  // --- 3. Cập nhật kỷ lục ---
  if (roundBest && roundCost < bestCost) {
    bestCost = roundCost;
    bestPath = roundBest;
  }

  generation++;

  // --- 4. Spawn particle kiến ---
  if (generation % PARTICLE_RATE === 0) {
    renderer.spawnAntParticles(graph, bestPath);
  }

  // --- 5. Cập nhật Chart ---
  const displayDist = bestPath ? graph.pathDistance(bestPath) : 0;
  chartMgr.addDataPoint(generation, displayDist || bestCost, qResult.reward);

  // --- 6. Render canvas ---
  renderer.render(graph, bestPath, {
    generation,
    bestCost: displayDist || bestCost,
    highlightNode: -1,
  });

  // --- 7. Cập nhật UI panels ---
  updateUI(qResult);
}

function startLoop() {
  if (loopHandle) clearInterval(loopHandle);
  loopHandle = setInterval(runGeneration, MS_PER_GEN);
}

function stopLoop() {
  if (loopHandle) {
    clearInterval(loopHandle);
    loopHandle = null;
  }
}

// =====================================================================
// UI UPDATE
// =====================================================================
function updateUI(qResult) {
  const { state, action, params, reward, epsilon } = qResult;

  // --- State Badge ---
  const stateLabel = STATE_LABELS[state] ?? 'BÌNH THƯỜNG';
  elStateLabel.textContent = stateLabel;

  elStateBadge.className = 'state-badge';
  if (state === STATES.IMPROVING) {
    elStateBadge.classList.add('state-ok');
  } else if (state === STATES.STUCK) {
    elStateBadge.classList.add('state-stuck');
  } else {
    elStateBadge.classList.add('state-alert');
  }

  // --- Live params ---
  elAlpha.textContent   = params.alpha.toFixed(2);
  elRho.textContent     = params.rho.toFixed(2);
  elEpsilon.textContent = (epsilon ?? agent.epsilon).toFixed(3);
  elGenNum.textContent  = generation;

  const displayDist = bestPath ? graph.pathDistance(bestPath) : 0;
  elBestDist.textContent = displayDist > 0 ? displayDist.toFixed(1) : (bestCost < Infinity ? bestCost.toFixed(1) : '—');

  elAction.textContent = params.label ?? '—';
  elReward.textContent = reward >= 0 ? `+${reward}` : `${reward}`;
  elReward.className = 'value ' + (reward > 0 ? 'reward-pos' : (reward < 0 ? 'reward-neg' : ''));

  // --- Q-Table mini display ---
  renderQTable();
}

function renderQTable() {
  if (!elQTable || !agent) return;
  const stateNames  = ['S0', 'S1', 'S2'];
  const actionNames = ['A0', 'A1', 'A2'];

  let html = '<table class="qtable"><thead><tr><th></th>';
  for (const a of actionNames) html += `<th>${a}</th>`;
  html += '</tr></thead><tbody>';

  for (let s = 0; s < 3; s++) {
    const isCurrent = s === agent.currentState;
    html += `<tr class="${isCurrent ? 'row-active' : ''}"><td class="qtable-state">${stateNames[s]}</td>`;
    const maxQ = Math.max(...agent.qTable[s]);
    for (let a = 0; a < 3; a++) {
      const q = agent.qTable[s][a];
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

btnReset.addEventListener('click', () => {
  stopLoop();
  isRunning = false;
  isBlockTool = false;
  btnStartPause.textContent = '▶ Bắt đầu';
  btnStartPause.classList.remove('btn-pause');
  setBlockToolActive(false);
  // init() sẽ tự destroy chart cũ và tạo mới — không cần gọi reset() riêng
  init();
});

// --- Canvas click: tắc đường ---
mainCanvas.addEventListener('click', (e) => {
  if (!isBlockTool || !graph) return;
  const rect = mainCanvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (mainCanvas.width / rect.width);
  const my = (e.clientY - rect.top)  * (mainCanvas.height / rect.height);

  const edge = renderer.findNearestEdge(graph, mx, my, 18);
  if (edge) {
    const blocked = graph.blockEdge(edge.i, edge.j);
    if (blocked) {
      agent.triggerTrafficEvent();
      showToast(`⚠️ Tắc đường: cạnh ${edge.i}↔${edge.j} (×${graph.TRAFFIC_MULTIPLIER})`);
      // Nếu bestPath đi qua cạnh này, reset về tìm đường mới
      if (bestPath) {
        const affectsPath = bestPath.some((v, k) => {
          if (k >= bestPath.length - 1) return false;
          const a = bestPath[k], b = bestPath[k+1];
          return (a === edge.i && b === edge.j) || (a === edge.j && b === edge.i);
        });
        if (affectsPath) {
          bestCost = Infinity; // Force tìm đường mới
        }
      }
    } else {
      showToast('Cạnh này đã bị tắc đường rồi!');
    }
    renderer.render(graph, bestPath, { generation, bestCost });
  }
});

// --- Canvas hover: highlight edge ---
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
  const w = container.clientWidth;
  const h = container.clientHeight;
  mainCanvas.width  = w;
  mainCanvas.height = h;
}

window.addEventListener('resize', () => {
  resizeCanvas();
  if (graph) {
    // Rescale node positions
    const oldW = graph.canvasWidth,  oldH = graph.canvasHeight;
    graph.canvasWidth  = mainCanvas.width;
    graph.canvasHeight = mainCanvas.height;
    const sx = mainCanvas.width / oldW, sy = mainCanvas.height / oldH;
    for (const node of graph.nodes) {
      node.x *= sx;
      node.y *= sy;
    }
    graph._computeDistances();
    renderer.render(graph, bestPath, { generation, bestCost });
  }
});

// =====================================================================
// BOOTSTRAP
// =====================================================================
window.addEventListener('DOMContentLoaded', () => {
  init();
  // Render animation liên tục ngay cả khi pause (để animation không bị đóng băng)
  function animLoop() {
    if (!isRunning && graph) {
      renderer.render(graph, bestPath, { generation, bestCost });
    }
    requestAnimationFrame(animLoop);
  }
  requestAnimationFrame(animLoop);
});
