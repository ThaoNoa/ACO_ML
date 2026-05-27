/**
 * evaluator.js — Đánh giá tự động hiệu năng Q-Learning và DDPG
 * Chạy 5 Test Cases, so sánh với optimal distance (Brute-force).
 */

import { TEST_CASES, BruteForceSolver } from './test_suite.js';
import { Graph } from './graph.js';
import { ACOEngine } from './aco.js';
import { QLearningAgent, STATES, ACTION_PARAMS } from './qlearning.js';
import { TSPEnvironment } from './environment.js';

export class Evaluator {
  /**
   * Chạy bộ test suite cho cả QL và DDPG
   * @param {import('./ddpg_agent.js').DDPGAgent} ddpgAgent
   * @param {Function} onProgress - Callback update UI
   * @returns {Promise<Array>} Results
   */
  async runTestSuite(ddpgAgent, onProgress = null) {
    const results = [];
    const MAX_GENS = 40; // 40 gen để có thời gian phản ứng tắc đường

    for (let i = 0; i < TEST_CASES.length; i++) {
      const tc = TEST_CASES[i];
      if (onProgress) onProgress(`Đang chạy ${tc.id}: ${tc.name}...`, (i / TEST_CASES.length) * 100);

      // 1. Khởi tạo đồ thị cố định (dùng chung cho cả BruteForce, QL, DDPG)
      const graph = new Graph(tc.nodes, 800, 600);
      graph.generate(); // Ngẫu nhiên nhưng sẽ được cố định cho bài test này
      
      // 2. Tính Optimal bằng Brute-Force
      const bfResult = BruteForceSolver.solve(graph);
      const optimalDist = bfResult.optimalCost;

      // 3. Chạy Q-Learning
      const qlResult = await this._runAgent(tc, 'ql', graph, null, MAX_GENS);

      // 4. Chạy DDPG
      const ddpgResult = await this._runAgent(tc, 'ddpg', graph, ddpgAgent, MAX_GENS);

      results.push({
        id: tc.id,
        name: tc.name,
        optimal: optimalDist,
        ql: qlResult,
        ddpg: ddpgResult
      });

      // Tránh block UI
      await new Promise(r => setTimeout(r, 50));
    }

    if (onProgress) onProgress('Đánh giá hoàn tất!', 100);
    return results;
  }

  /**
   * Helper chạy mô phỏng headless cho 1 agent
   */
  async _runAgent(tc, mode, baseGraph, ddpgAgent, maxGens) {
    // Clone graph để tránh ảnh hưởng chéo
    const graph = this._cloneGraph(baseGraph);
    const aco = new ACOEngine(graph, { numAnts: 30 }); // Phục hồi 30 kiến để khớp với môi trường lúc train
    
    let env = null;
    let ql = null;
    
    if (mode === 'ql') {
      ql = new QLearningAgent();
    } else {
      env = new TSPEnvironment(graph, aco);
      env.reset();
      ddpgAgent.resetNoise();
    }

    let globalBest = Infinity;
    let stuckCountTotal = 0;
    
    const applyTraffic = (gen) => {
      if (tc.traffic.length === 0) return;
      for (const t of tc.traffic) {
        if ((t.type === 'hard' && gen === 0) || (t.type === 'dynamic' && gen === t.gen)) {
          graph.blockEdge(t.edge[0], t.edge[1]);
          if (mode === 'ql') ql.triggerTrafficEvent();
          else env.triggerTrafficEvent();
        }
      }
    };

    for (let gen = 0; gen < maxGens; gen++) {
      applyTraffic(gen);

      let alpha = 1.0, rho = 0.1;

      if (mode === 'ql') {
        let qRes;
        if (gen === 0) qRes = { params: ACTION_PARAMS[0], action: 0, state: STATES.IMPROVING };
        else qRes = ql.step(globalBest);
        
        alpha = qRes.params.alpha;
        rho = qRes.params.rho;
        aco.setParams(alpha, rho);
        
        if (qRes.action === 1 && qRes.state === STATES.DEGRADED) graph.resetPheromones();
      } else {
        const state = env.getState();
        const actionInfo = ddpgAgent.selectAction(state);
        alpha = actionInfo.alpha;
        rho = actionInfo.rho;
        aco.setParams(alpha, rho);
        
        if (env.stuckCounter >= 12) graph.resetPheromones();
      }

      const { bestCost: roundCost } = aco.runIteration();
      
      if (roundCost < globalBest) {
        globalBest = roundCost;
      }

      if (mode === 'ddpg') {
        env.step(roundCost, alpha, rho); // DDPG test chỉ suy luận, KHÔNG TRAIN
      }

      // Đếm stuck thô (không cải thiện sau 15 gen)
      if (roundCost > globalBest * 1.05) stuckCountTotal++;
    }

    return {
      bestCost: globalBest,
      errorPct: 0, // Tính sau khi có optimal
      stuckCount: stuckCountTotal
    };
  }

  /**
   * Tạo một bản sao đồ thị mới với cùng vị trí node
   */
  _cloneGraph(source) {
    const g = new Graph(source.size, source.canvasWidth, source.canvasHeight, source.padding);
    g.nodes = source.nodes.map(n => ({...n}));
    g._computeDistances();
    g._initPheromones();
    return g;
  }
}
