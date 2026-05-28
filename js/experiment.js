// experiment.js – Chạy thí nghiệm dài hạn, ghi log dữ liệu
import { Graph } from './graph.js';
import { ACOEngine } from './aco.js';
import { QLearningAgent, ACTION_PARAMS } from './qlearning.js';
import { DDPGAgent } from './ddpg_agent.js';
import { TSPEnvironment } from './environment.js';

// Cấu hình thí nghiệm
const CONFIG = {
    numNodes: 20,
    numAnts: 30,
    maxGens: 1000,
    trafficSchedule: [
        { gen: 200, edge: [3, 7] },
        { gen: 400, edge: [5, 12] },
        { gen: 600, edge: [8, 15] }
    ],
    baselineACO: { alpha: 1.0, rho: 0.1, beta: 2.0 }
};

export async function runExperiment(mode = 'ddpg', onProgress = null) {
    // Tạo đồ thị cố định (dùng chung cho các lần chạy để so sánh)
    const graph = new Graph(CONFIG.numNodes, 800, 600);
    graph.generate();
    const originalNodes = graph.nodes.map(n => ({...n}));

    const results = {
        mode: mode,
        generations: [],
        bestCosts: [],
        cumulativeRewards: [],
        alphaHistory: [],
        rhoHistory: [],
        trafficGens: CONFIG.trafficSchedule.map(t => t.gen)
    };

    let cumulativeReward = 0;

    // Hàm clone graph (giữ nguyên topology)
    const cloneGraph = () => {
        const g = new Graph(CONFIG.numNodes, 800, 600);
        g.nodes = originalNodes.map(n => ({...n}));
        g._computeDistances();
        g._initPheromones();
        return g;
    };

    let aco, agent, env;

    if (mode === 'ddpg') {
        agent = new DDPGAgent();
        await agent.build();
    } else if (mode === 'ql') {
        agent = new QLearningAgent();
    } else {
        // Baseline ACO thuần túy
        agent = null;
    }

    const graphRun = cloneGraph();
    aco = new ACOEngine(graphRun, { numAnts: CONFIG.numAnts });

    if (mode === 'ddpg') {
        env = new TSPEnvironment(graphRun, aco);
        env.reset();
        agent.resetNoise();
    } else if (mode === 'ql') {
        // QL không cần env riêng
    }

    let bestPath = null;
    let bestCost = Infinity;
    let prevBest = Infinity;
    let allTimeBest = Infinity;
    let stuckCounter = 0;

    for (let gen = 0; gen <= CONFIG.maxGens; gen++) {
        // Áp dụng tắc đường: chọn cạnh trên best path hiện tại (nếu có)
        for (let t of CONFIG.trafficSchedule) {
            if (gen === t.gen) {
                let blocked = false;
                // Nếu có bestPath (lộ trình tốt nhất hiện tại)
                if (bestPath && bestPath.length >= 2) {
                    // Chọn ngẫu nhiên một cạnh trên bestPath (không trùng với cạnh đã block)
                    const possibleEdges = [];
                    for (let k = 0; k < bestPath.length - 1; k++) {
                        const a = bestPath[k];
                        const b = bestPath[k+1];
                        const key = a < b ? `${a}-${b}` : `${b}-${a}`;
                        if (!graphRun.blockedEdges.has(key)) {
                            possibleEdges.push([a, b]);
                        }
                    }
                    if (possibleEdges.length > 0) {
                        const randomEdge = possibleEdges[Math.floor(Math.random() * possibleEdges.length)];
                        graphRun.blockEdge(randomEdge[0], randomEdge[1]);
                        blocked = true;
                        console.log(`[Traffic] Blocked edge ${randomEdge[0]}-${randomEdge[1]} at gen ${gen}. Current bestCost = ${bestCost}`);
                        console.log(`[Traffic] at gen ${gen} blocked edge ${randomEdge[0]}-${randomEdge[1]} on best path`);
                    }
                }
                // Nếu không có bestPath (gen đầu) thì thôi
                if (!blocked) {
                    console.log(`[Traffic] at gen ${gen} could not block any edge (no best path)`);
                }

                if (mode === 'ql' && agent) agent.triggerTrafficEvent();
                if (mode === 'ddpg' && env) env.triggerTrafficEvent();
            }
        }

        let alpha = 1.0, rho = 0.1;
        let reward = 0;

        // Chọn tham số theo agent
        if (mode === 'ddpg' && agent) {
            const state = env.getState();
            const action = agent.selectAction(state);
            alpha = action.alpha;
            rho = action.rho;
            aco.setParams(alpha, rho);
            if (env.stuckCounter >= 12) graphRun.resetPheromones();
        } else if (mode === 'ql' && agent) {
            if (gen === 0) {
                // gen 0: dùng action mặc định
                alpha = ACTION_PARAMS[0].alpha;
                rho = ACTION_PARAMS[0].rho;
            } else {
                const qRes = agent.step(bestCost);
                alpha = qRes.params.alpha;
                rho = qRes.params.rho;
                if (qRes.action === 1 && qRes.state === 2) graphRun.resetPheromones();
            }
            aco.setParams(alpha, rho);
        } else {
            // Baseline ACO cố định
            aco.setParams(CONFIG.baselineACO.alpha, CONFIG.baselineACO.rho);
        }

        // Chạy một vòng ACO
        const { bestPath: currentBestPath, bestCost: roundCost } = aco.runIteration();
        bestPath = currentBestPath;   // luôn cập nhật lộ trình tốt nhất vòng này
        if (roundCost < bestCost) bestCost = roundCost;

        // Cập nhật best cost toàn cục
        if (roundCost < bestCost) {
            bestCost = roundCost;
            bestPath = newBestPath;   // <--- CẬP NHẬT
        }
        if (roundCost < allTimeBest) allTimeBest = roundCost;

        // Tính reward cho DDPG/QL
        if (mode === 'ddpg' && env) {
            const { reward: r } = env.step(roundCost, alpha, rho);
            reward = r;
            cumulativeReward += reward;
            // Lưu transition và train
            const state = env.getState(); // lấy state trước khi step? Thực tế nên lấy trước. Ở đây demo
            // Để đơn giản, bỏ qua remember/train trong thí nghiệm này
        } else if (mode === 'ql' && agent) {
            // Tính reward giống trong QLearningAgent.computeReward
            if (roundCost < allTimeBest) reward = 10;
            else if (roundCost > prevBest * 1.05) reward = -10;
            else if (stuckCounter >= 10) reward = -5;
            else reward = 1;
            cumulativeReward += reward;
        } else {
            // Baseline: không học, reward = 0
            reward = 0;
        }

        prevBest = roundCost;
        if (roundCost > allTimeBest * 1.02) stuckCounter++;
        else stuckCounter = 0;

        // Lưu dữ liệu
        results.generations.push(gen);
        results.bestCosts.push(bestCost);
        results.cumulativeRewards.push(cumulativeReward);
        results.alphaHistory.push(alpha);
        results.rhoHistory.push(rho);

        if (onProgress && gen % 50 === 0) {
            onProgress(gen, bestCost, cumulativeReward);
            await new Promise(r => setTimeout(r, 0)); // yield
        }
    }

    return results;
}

// Hàm xuất dữ liệu ra file JSON (chạy trong browser, dùng download)
export function exportResults(results, filename = 'experiment_data.json') {
    const dataStr = JSON.stringify(results, null, 2);
    const blob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}