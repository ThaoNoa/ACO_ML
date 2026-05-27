/**
 * aco.js - ACO Engine (Ant Colony Optimization)
 * Triển khai thuật toán đàn kiến cho bài toán TSP.
 * Module này nhận graph và các tham số (alpha, beta, rho),
 * sau đó chạy một vòng (iteration) với nhiều kiến.
 */

export class ACOEngine {
  /**
   * @param {import('./graph.js').Graph} graph
   * @param {object} config
   */
  constructor(graph, config = {}) {
    this.graph = graph;

    // Tham số mặc định - sẽ bị Q-Learning điều chỉnh
    this.alpha = config.alpha ?? 1.0;   // Trọng số Pheromone
    this.beta  = config.beta  ?? 2.0;   // Trọng số Heuristic phục hồi mức 2.0
    this.rho   = config.rho   ?? 0.1;   // Tốc độ bay hơi
    this.numAnts = config.numAnts ?? 30;

    this.Q = 100; // Hằng số lượng pheromone deposit
  }

  /**
   * Chạy một vòng iteration với toàn bộ đàn kiến
   * @returns {{ bestPath: number[], bestCost: number, allPaths: Array }}
   */
  runIteration() {
    const n = this.graph.size;
    const allSolutions = [];

    // Mỗi kiến xây dựng một lộ trình
    for (let ant = 0; ant < this.numAnts; ant++) {
      const path = this._buildPath();
      if (path) {
        const cost = this.graph.pathCost(path);
        allSolutions.push({ path, cost });
      }
    }

    if (allSolutions.length === 0) {
      return { bestPath: null, bestCost: Infinity, allPaths: [] };
    }

    // Tìm lộ trình tốt nhất trong vòng này
    allSolutions.sort((a, b) => a.cost - b.cost);
    const { path: bestPath, cost: bestCost } = allSolutions[0];

    // Tính lượng pheromone deposit
    // Dùng chiến lược Elitist: kiến tốt nhất deposit nhiều hơn
    const deposits = allSolutions.slice(0, Math.max(1, Math.floor(this.numAnts * 0.3)))
      .map(({ path, cost }) => ({
        path,
        quality: this.Q / cost
      }));

    // Thêm bonus cho kiến tốt nhất
    deposits.push({ path: bestPath, quality: (this.Q * 3) / bestCost });

    // Cập nhật pheromone trên graph
    this.graph.updatePheromones(this.rho, deposits);

    return { bestPath, bestCost, allPaths: allSolutions };
  }

  /**
   * Một con kiến xây dựng lộ trình hoàn chỉnh bằng xác suất
   * @returns {number[] | null} Mảng chỉ số node (bắt đầu và kết thúc tại node 0)
   */
  _buildPath() {
    const n = this.graph.size;
    const startNode = 0; // Luôn bắt đầu từ depot
    const visited = new Uint8Array(n);
    const path = [startNode];
    visited[startNode] = 1;

    for (let step = 0; step < n - 1; step++) {
      const current = path[path.length - 1];
      const next = this._selectNextNode(current, visited);
      if (next === -1) return null; // Không thể đi tiếp
      path.push(next);
      visited[next] = 1;
    }

    // Quay về depot
    path.push(startNode);
    return path;
  }

  /**
   * Chọn node tiếp theo theo xác suất dựa trên pheromone và heuristic
   * @param {number} current - Node hiện tại
   * @param {Uint8Array} visited - Mảng đã thăm
   * @returns {number} - Chỉ số node tiếp theo, hoặc -1 nếu không có
   */
  _selectNextNode(current, visited) {
    const n = this.graph.size;
    const probabilities = new Float64Array(n);
    let total = 0;

    for (let j = 0; j < n; j++) {
      if (visited[j] || j === current) continue;
      const cost = this.graph.costs[current][j];
      if (cost <= 0) continue;

      const pheromone = this.graph.pheromones[current][j];
      const heuristic = 1.0 / cost; // Nghịch đảo khoảng cách

      const prob = Math.pow(pheromone, this.alpha) * Math.pow(heuristic, this.beta);
      probabilities[j] = prob;
      total += prob;
    }

    if (total === 0) {
      // Fallback: chọn ngẫu nhiên node chưa thăm
      const unvisited = [];
      for (let j = 0; j < n; j++) {
        if (!visited[j] && j !== current) unvisited.push(j);
      }
      if (unvisited.length === 0) return -1;
      return unvisited[Math.floor(Math.random() * unvisited.length)];
    }

    // Roulette wheel selection
    let threshold = Math.random() * total;
    for (let j = 0; j < n; j++) {
      if (probabilities[j] === 0) continue;
      threshold -= probabilities[j];
      if (threshold <= 0) return j;
    }

    // Dự phòng: chọn node có xác suất cao nhất
    let best = -1, bestProb = -1;
    for (let j = 0; j < n; j++) {
      if (probabilities[j] > bestProb) {
        bestProb = probabilities[j];
        best = j;
      }
    }
    return best;
  }

  /**
   * Cập nhật tham số từ Q-Learning / DDPG Agent
   * @param {number} alpha - Trọng số pheromone mới
   * @param {number} rho   - Tốc độ bay hơi mới
   */
  setParams(alpha, rho) {
    this.alpha = alpha;
    this.rho   = rho;
  }

  /**
   * Tính thống kê pheromone toàn đồ thị — dùng bởi TSPEnvironment.getState()
   * @returns {{ avg: number, variance: number, entropy: number }}
   */
  getPheromoneStats() {
    const n = this.graph.size;
    const vals = [];

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) vals.push(this.graph.pheromones[i][j]);
      }
    }

    if (vals.length === 0) return { avg: 0, variance: 0, entropy: 0 };

    // Trung bình
    const avg = vals.reduce((s, v) => s + v, 0) / vals.length;

    // Độ lệch chuẩn
    const variance = Math.sqrt(
      vals.reduce((s, v) => s + (v - avg) ** 2, 0) / vals.length
    );

    // Entropy (normalized): -Σ p·log(p) / log(N)
    const total = vals.reduce((s, v) => s + v, 0);
    const entropy = total > 0
      ? -vals.reduce((s, v) => {
          const p = v / total;
          return s + p * Math.log(p + 1e-10);
        }, 0) / Math.log(vals.length + 1e-10)
      : 0;

    return { avg, variance, entropy: Math.min(1, Math.max(0, entropy)) };
  }
}
