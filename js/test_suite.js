/**
 * test_suite.js — Quản lý các Test Case và Brute-force Solver
 *
 * TSP với N=10 có 9! = 362,880 hoán vị (nếu cố định điểm bắt đầu).
 * JavaScript xử lý mức này trong vòng < 50ms.
 */

export const TEST_CASES = [
  {
    id: 'TC1',
    name: 'Normal Map (N=8)',
    nodes: 8,
    traffic: [] // Không tắc đường
  },
  {
    id: 'TC2',
    name: 'Hard Traffic (N=10) - 1 Edge',
    nodes: 10,
    traffic: [{ type: 'hard', edge: [1, 3] }] // Tắc vĩnh viễn cạnh (1,3)
  },
  {
    id: 'TC3',
    name: 'Hard Traffic (N=10) - 2 Edges',
    nodes: 10,
    traffic: [{ type: 'hard', edge: [2, 5] }, { type: 'hard', edge: [4, 7] }]
  },
  {
    id: 'TC4',
    name: 'Dynamic Traffic (N=10) - 1 Edge',
    nodes: 10,
    traffic: [{ type: 'dynamic', gen: 50, edge: [1, 4] }] // Xuất hiện tắc ở gen 50
  },
  {
    id: 'TC5',
    name: 'Dynamic Traffic (N=10) - 2 Edges',
    nodes: 10,
    traffic: [
      { type: 'dynamic', gen: 50, edge: [3, 6] },
      { type: 'dynamic', gen: 100, edge: [2, 8] }
    ]
  }
];

export class BruteForceSolver {
  /**
   * Tính toán đường đi ngắn nhất (Brute-force) của đồ thị.
   * Cố định node 0 làm điểm bắt đầu/kết thúc để giảm số hoán vị còn (N-1)!
   * @param {import('./graph.js').Graph} graph
   * @returns {{ optimalCost: number, optimalPath: number[] }}
   */
  static solve(graph) {
    const n = graph.size;
    const distances = graph.distances;
    let minCost = Infinity;
    let bestPath = null;

    // Các đỉnh cần hoán vị (từ 1 đến n-1)
    const vertices = [];
    for (let i = 1; i < n; i++) vertices.push(i);

    // Thuật toán sinh hoán vị (Heap's Algorithm)
    const c = new Array(n - 1).fill(0);
    
    // Đánh giá hoán vị đầu tiên
    const evaluate = (perm) => {
      let cost = distances[0][perm[0]]; // Từ 0 đến đỉnh đầu tiên
      for (let i = 0; i < perm.length - 1; i++) {
        cost += distances[perm[i]][perm[i + 1]];
        if (cost >= minCost) return; // Pruning
      }
      cost += distances[perm[perm.length - 1]][0]; // Quay về 0

      if (cost < minCost) {
        minCost = cost;
        bestPath = [0, ...perm, 0];
      }
    };

    evaluate(vertices);

    let i = 0;
    while (i < n - 1) {
      if (c[i] < i) {
        if (i % 2 === 0) {
          const temp = vertices[0];
          vertices[0] = vertices[i];
          vertices[i] = temp;
        } else {
          const temp = vertices[c[i]];
          vertices[c[i]] = vertices[i];
          vertices[i] = temp;
        }
        evaluate(vertices);
        c[i] += 1;
        i = 0;
      } else {
        c[i] = 0;
        i += 1;
      }
    }

    return { optimalCost: minCost, optimalPath: bestPath };
  }
}
