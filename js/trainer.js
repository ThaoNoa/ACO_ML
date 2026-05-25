/**
 * trainer.js — Headless Training Pipeline
 * Chạy thuật toán DDPG cực nhanh mà không cần render DOM/Canvas
 */

import { ACOEngine }      from './aco.js';
import { TSPEnvironment } from './environment.js';

export class HeadlessTrainer {
  /**
   * @param {import('./graph.js').Graph} graph - Share chung bản đồ với app
   * @param {import('./ddpg_agent.js').DDPGAgent} agent - DDPG Agent
   */
  constructor(graph, agent) {
    this.graph = graph;
    this.agent = agent;
    // Dùng ACOEngine và Environment riêng biệt để không can thiệp state của Main App UI
    this.aco = new ACOEngine(graph, { numAnts: 30 });
    this.env = new TSPEnvironment(graph, this.aco);
  }

  /**
   * Chạy huấn luyện tốc độ cao
   * @param {number} numGens - Số lượng generation cần chạy
   * @param {Function} onProgress - Callback (gen, bestCost, actorLoss, epsilon)
   */
  async train(numGens = 2000, onProgress = null) {
    this.env.reset();
    this.agent.resetNoise();
    this.graph.resetPheromones();

    // Lưu tạm các edge đang bị tắc để phục hồi sau
    const savedBlocked = new Set(this.graph.blockedEdges);
    this.graph.blockedEdges.clear();
    this.graph._computeDistances();

    let genSinceLastTrain = 0;
    
    const startTime = performance.now();

    for (let gen = 0; gen < numGens; gen++) {
      const state = this.env.getState();
      const { rawAction, alpha, rho } = this.agent.selectAction(state);
      this.aco.setParams(alpha, rho);

      if (this.env.stuckCounter >= 12) this.graph.resetPheromones();

      const { bestCost: roundCost } = this.aco.runIteration();
      const { reward, nextState } = this.env.step(roundCost, alpha, rho);
      
      this.agent.remember(state, rawAction, reward, nextState);

      genSinceLastTrain++;
      if (genSinceLastTrain >= 2) {
        genSinceLastTrain = 0;
        await this.agent.train();
      }

      // Cập nhật UI mỗi 50 gen, đồng thời nhả luồng (yield) để Browser không đơ
      if (gen > 0 && gen % 50 === 0) {
        if (onProgress) {
          onProgress(gen, this.env.bestCost, this.agent.lastActorLoss, this.agent.noiseLevel);
        }
        await new Promise(r => setTimeout(r, 0));
      }
    }

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    console.log(`[Trainer] Đã train ${numGens} gens trong ${elapsed}s.`);

    // Phục hồi tắc đường
    for (const edgeKey of savedBlocked) {
      this.graph.blockedEdges.add(edgeKey);
    }
    this.graph._computeDistances();
    this.graph.resetPheromones();

    if (onProgress) {
      onProgress(numGens, this.env.bestCost, this.agent.lastActorLoss, this.agent.noiseLevel);
    }
  }
}
