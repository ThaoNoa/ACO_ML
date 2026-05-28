/**
 * graph.js - Quản lý đồ thị: nodes, edges, pheromone matrix
 * Module này chịu trách nhiệm lưu trữ và cung cấp dữ liệu về
 * các điểm (thành phố), khoảng cách, và mức độ pheromone.
 */

export class Graph {
  constructor(numNodes, canvasWidth, canvasHeight, padding = 60) {
    this.numNodes = numNodes;
    this.canvasWidth = canvasWidth;
    this.canvasHeight = canvasHeight;
    this.padding = padding;

    this.nodes = [];         // [{x, y, id, label}]
    this.distances = [];     // Ma trận khoảng cách gốc [n x n]
    this.costs = [];         // Ma trận chi phí thực (có thể bị nhân lên do tắc đường)
    this.pheromones = [];    // Ma trận pheromone [n x n]
    this.blockedEdges = new Set(); // Tập hợp các cạnh bị tắc đường "i-j"

    this.PHEROMONE_INIT = 0.1;
    this.PHEROMONE_MIN  = 0.001;
    this.PHEROMONE_MAX  = 10.0;
    this.TRAFFIC_MULTIPLIER = 1000;
  }

  /**
   * Sinh ngẫu nhiên các node trên canvas
   */
  generate() {
    this.nodes = [];
    const margin = this.padding;
    const w = this.canvasWidth  - margin * 2;
    const h = this.canvasHeight - margin * 2;

    // Đặt node depot (kho xuất phát) ở trung tâm
    this.nodes.push({
      id: 0,
      x: this.canvasWidth / 2,
      y: this.canvasHeight / 2,
      label: '🏭',
      isDepot: true
    });

    // Sinh các node còn lại ngẫu nhiên với khoảng cách tối thiểu
    let attempts = 0;
    while (this.nodes.length < this.numNodes && attempts < 10000) {
      attempts++;
      const x = margin + Math.random() * w;
      const y = margin + Math.random() * h;

      // Kiểm tra khoảng cách tối thiểu giữa các node
      let tooClose = false;
      for (const node of this.nodes) {
        const d = Math.hypot(x - node.x, y - node.y);
        if (d < 55) { tooClose = true; break; }
      }
      if (!tooClose) {
        this.nodes.push({
          id: this.nodes.length,
          x, y,
          label: `${this.nodes.length}`,
          isDepot: false
        });
      }
    }

    this._computeDistances();
    this._initPheromones();
    this.blockedEdges.clear();
  }

  /**
   * Tính ma trận khoảng cách Euclidean
   */
  _computeDistances() {
    const n = this.nodes.length;
    this.distances = Array.from({ length: n }, () => new Float64Array(n));
    this.costs     = Array.from({ length: n }, () => new Float64Array(n));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          this.distances[i][j] = 0;
          this.costs[i][j] = 0;
        } else {
          const d = Math.hypot(
            this.nodes[i].x - this.nodes[j].x,
            this.nodes[i].y - this.nodes[j].y
          );
          this.distances[i][j] = d;
          this.costs[i][j]     = d;
        }
      }
    }
  }

  /**
   * Khởi tạo ma trận pheromone với giá trị đều nhau
   */
  _initPheromones() {
    const n = this.nodes.length;
    this.pheromones = Array.from({ length: n }, () =>
      new Float64Array(n).fill(this.PHEROMONE_INIT)
    );
  }

  /**
   * Áp dụng tắc đường lên một cạnh (i, j)
   */
  blockEdge(i, j) {
    const key = i < j ? `${i}-${j}` : `${j}-${i}`;
    if (this.blockedEdges.has(key)) return false; // Đã bị tắc rồi

    this.blockedEdges.add(key);
    this.costs[i][j] = this.distances[i][j] * this.TRAFFIC_MULTIPLIER;
    this.costs[j][i] = this.distances[j][i] * this.TRAFFIC_MULTIPLIER;
    return true;
  }

  /**
   * Gỡ tắc đường một cạnh (i, j)
   */
  unblockEdge(i, j) {
    const key = i < j ? `${i}-${j}` : `${j}-${i}`;
    this.blockedEdges.delete(key);
    this.costs[i][j] = this.distances[i][j];
    this.costs[j][i] = this.distances[j][i];
  }

  /**
   * Xoá toàn bộ tắc đường
   */
  clearAllBlockedEdges() {
    this.blockedEdges.clear();
    const n = this.nodes.length;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        this.costs[i][j] = this.distances[i][j];
      }
    }
  }

  /**
   * Kiểm tra cạnh có bị tắc không
   */
  isEdgeBlocked(i, j) {
    const key = i < j ? `${i}-${j}` : `${j}-${i}`;
    return this.blockedEdges.has(key);
  }

  /**
   * Cập nhật pheromone sau mỗi iteration
   * @param {number} rho - Tốc độ bay hơi (evaporation rate)
   * @param {Array} deposits - [{path, quality}] - lượng pheromone thêm vào
   */
  updatePheromones(rho, deposits) {
    const n = this.nodes.length;
    // Bay hơi
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        this.pheromones[i][j] = Math.max(
          this.PHEROMONE_MIN,
          this.pheromones[i][j] * (1 - rho)
        );
      }
    }
    // Thêm pheromone mới
    for (const { path, quality } of deposits) {
      for (let k = 0; k < path.length - 1; k++) {
        const a = path[k], b = path[k + 1];
        this.pheromones[a][b] = Math.min(
          this.PHEROMONE_MAX,
          this.pheromones[a][b] + quality
        );
        this.pheromones[b][a] = Math.min(
          this.PHEROMONE_MAX,
          this.pheromones[b][a] + quality
        );
      }
    }
  }

  /**
   * Reset pheromone về mức khởi tạo (dùng khi cần khám phá khẩn cấp)
   */
  resetPheromones() {
    const n = this.nodes.length;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        this.pheromones[i][j] = this.PHEROMONE_INIT;
      }
    }
  }

  /**
   * Tính tổng chi phí của một lộ trình
   * @param {number[]} path - mảng chỉ số node (vòng tròn, quay về đầu)
   */
  pathCost(path) {
    let total = 0;
    for (let i = 0; i < path.length - 1; i++) {
      total += this.costs[path[i]][path[i + 1]];
    }
    return total;
  }

  /**
   * Tính tổng khoảng cách thực (không tính tắc đường) của lộ trình
   */
  pathDistance(path) {
    let total = 0;
    for (let i = 0; i < path.length - 1; i++) {
      total += this.distances[path[i]][path[i + 1]];
    }
    return total;
  }

  get size() {
    return this.nodes.length;
  }
}
