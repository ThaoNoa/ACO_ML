/**
 * environment.js — TSPEnvironment
 *
 * Bọc ACOEngine + Graph thành interface chuẩn cho DDPG agent:
 *   - getState()  → Float32Array[10] (state vector liên tục)
 *   - step()      → { nextState, reward }
 *   - reset()     → khởi tạo lại tracking vars
 *
 * State Vector (10 chiều):
 *   [0] normalized_best_cost   — best_cost / reference_cost
 *   [1] improvement_rate       — (prev - curr) / prev  ∈ [-1, 1]
 *   [2] stuck_counter_norm     — stuckCounter / STUCK_THRESHOLD
 *   [3] avg_pheromone          — mean(pheromones) / PHEROMONE_MAX
 *   [4] pheromone_std          — std(pheromones) / PHEROMONE_MAX
 *   [5] blocked_edge_ratio     — blockedEdges / totalEdges
 *   [6] current_alpha_norm     — (alpha - 0.3) / 2.2
 *   [7] current_rho_norm       — (rho - 0.02) / 0.88
 *   [8] generation_progress    — (generation % 100) / 100
 *   [9] pheromone_entropy      — normalized entropy of pheromone distribution
 */

export const DDPG_CONSTANTS = {
  STUCK_THRESHOLD:  15,   // gen không cải thiện → kẹt
  ALPHA_MIN:        0.5,  // đảm bảo kiến luôn tôn trọng pheromone ở mức tối thiểu
  ALPHA_MAX:        2.5,
  RHO_MIN:          0.02,
  RHO_MAX:          0.50, // giới hạn bay hơi tối đa — tránh "xoá sạch" pheromone mỗi bước
  REF_WARMUP_GENS:  10,   // số gen đầu để tính reference_cost
};

export class TSPEnvironment {
  /**
   * @param {import('./graph.js').Graph}      graph
   * @param {import('./aco.js').ACOEngine}    aco
   */
  constructor(graph, aco) {
    this.graph = graph;
    this.aco   = aco;

    this.generation      = 0;
    this.bestCost        = Infinity;
    this.prevBestCost    = Infinity;
    this.allTimeBest     = Infinity;
    this.stuckCounter    = 0;
    this.referenceCost   = null;   // baseline = avg của REF_WARMUP_GENS gen đầu
    this._refCostSum     = 0;
    this._refCostCount   = 0;

    this.currentAlpha    = 1.0;
    this.currentRho      = 0.1;

    this.trafficTriggered = false;

    // Lịch sử reward để debug / chart
    this.rewardHistory   = [];
  }

  /** Reset tracking — gọi khi bắt đầu episode mới */
  reset() {
    this.generation      = 0;
    this.bestCost        = Infinity;
    this.prevBestCost    = Infinity;
    this.allTimeBest     = Infinity;
    this.stuckCounter    = 0;
    this.referenceCost   = null;
    this._refCostSum     = 0;
    this._refCostCount   = 0;
    this.currentAlpha    = 1.0;
    this.currentRho      = 0.1;
    this.trafficTriggered = false;
    this.rewardHistory   = [];
  }

  /**
   * Cập nhật sau mỗi ACO iteration và tính reward.
   * @param {number} roundCost   - best cost vòng này (từ ACO)
   * @param {number} alpha       - alpha đang dùng
   * @param {number} rho         - rho đang dùng
   * @returns {{ reward: number, nextState: Float32Array }}
   */
  step(roundCost, alpha, rho) {
    this.prevBestCost  = this.bestCost;
    this.currentAlpha  = alpha;
    this.currentRho    = rho;
    this.generation++;

    // Cập nhật reference cost trong warmup
    if (this._refCostCount < DDPG_CONSTANTS.REF_WARMUP_GENS && isFinite(roundCost)) {
      this._refCostSum   += roundCost;
      this._refCostCount++;
      if (this._refCostCount === DDPG_CONSTANTS.REF_WARMUP_GENS) {
        this.referenceCost = this._refCostSum / this._refCostCount;
      }
    }

    // Cập nhật best
    if (roundCost < this.bestCost) {
      this.bestCost = roundCost;
    }
    if (roundCost < this.allTimeBest) {
      this.allTimeBest = roundCost;
    }

    // Stuck counter
    if (roundCost < this.allTimeBest + 0.01) {
      this.stuckCounter = 0;
    } else {
      this.stuckCounter++;
    }

    const reward = this._computeReward(roundCost);
    this.rewardHistory.push(reward);

    const nextState = this.getState();
    return { reward, nextState };
  }

  /**
   * Tính reward dựa trên kết quả iteration
   * Thiết kế: 3 thành phần cộng lại
   *   1. Kết quả ACO (improvement/stuck)
   *   2. Sức khoẻ tham số (bell-curve, phạt cả 2 cực đoan)
   *   3. Entropy pheromone (thưởng đa dạng, phạt lock-in)
   */
  _computeReward(roundCost) {
    if (this.trafficTriggered) {
      this.trafficTriggered = false;
      return 0; // Vừa có biến động lớn, nhường cho vòng sau đánh giá dựa trên trạng thái mới
    }

    if (!isFinite(this.prevBestCost) || this.prevBestCost === Infinity) {
      return 1;
    }

    const { ALPHA_MIN, ALPHA_MAX, RHO_MIN, RHO_MAX, STUCK_THRESHOLD } = DDPG_CONSTANTS;
    const improvementRate = (this.prevBestCost - roundCost) / this.prevBestCost;

    let reward = 0;
    
    // Nếu cost vượt quá 3000, chắc chắn kiến đang đi vào đường cấm (do cost x100)
    if (roundCost > 3000) {
      reward -= 20; // Phạt cực nặng để ép phải đổi đường
    } else {
      if      (improvementRate > 0.05)  reward = +20;
      else if (improvementRate > 0)     reward = +10;
      else if (this.stuckCounter >= 5)  reward = -10; // Phạt nặng nếu kẹt quá lâu
      else if (roundCost <= this.allTimeBest * 1.05) reward = +3;
      else    reward = -2;
    }

    // ── 2. Sức khoẻ tham số (bell-curve, đỉnh tại vùng trung tâm) ──
    // Alpha: vùng tốt nhất là [0.8, 1.8] — phạt cả quá thấp lẫn quá cao
    const alphaNorm = (this.currentAlpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN);
    // Bell-curve: đỉnh tại alphaNorm=0.35 (≈ alpha 1.0), giảm dần 2 phía
    const alphaHealth = Math.exp(-Math.pow((alphaNorm - 0.35) / 0.25, 2));

    // Rho: vùng tốt nhất là [0.08, 0.25] — phạt cả quá thấp (lock-in) lẫn quá cao (xoá sạch)
    const rhoNorm = (this.currentRho - RHO_MIN) / (RHO_MAX - RHO_MIN);
    // Bell-curve: đỉnh tại rhoNorm=0.2 (≈ rho 0.10–0.15)
    const rhoHealth = Math.exp(-Math.pow((rhoNorm - 0.20) / 0.25, 2));

    // Điểm sức khoẻ tổng: [-2, +2]
    const paramScore = (alphaHealth + rhoHealth - 1.0) * 2;
    reward += paramScore;

    // ── 3. Thưởng entropy pheromone ───────────────────────────────
    // entropy cao → kiến đang khám phá nhiều đường → tốt
    // entropy thấp → kiến bị "lock-in" 1 đường → cần phá băng
    const { entropy } = this.aco.getPheromoneStats();
    // Thưởng khi entropy trong [0.3, 0.7], phạt khi entropy < 0.15 (lock-in cứng)
    if (entropy < 0.15) {
      reward -= 3; // Lock-in cứng — cần tăng Rho để phá băng
    } else if (entropy > 0.25 && entropy < 0.75) {
      reward += 1; // Vùng lành mạnh
    }

    return reward;
  }

  /**
   * Trả về state vector 10 chiều dưới dạng Float32Array
   */
  getState() {
    const { STUCK_THRESHOLD, ALPHA_MIN, ALPHA_MAX, RHO_MIN, RHO_MAX } = DDPG_CONSTANTS;
    const g = this.graph;
    const n = g.size;
    const totalEdges = n * (n - 1) / 2;

    // --- Feature 0: normalized_best_cost ---
    const refCost = this.referenceCost ?? (isFinite(this.bestCost) ? this.bestCost : 1);
    const normBestCost = isFinite(this.bestCost)
      ? Math.min(3, this.bestCost / Math.max(refCost, 1))
      : 1.5;

    // --- Feature 1: improvement_rate ---
    let improvementRate = 0;
    if (isFinite(this.prevBestCost) && this.prevBestCost > 0) {
      improvementRate = (this.prevBestCost - this.bestCost) / this.prevBestCost;
      improvementRate = Math.max(-1, Math.min(1, improvementRate));
    }

    // --- Feature 2: stuck_counter_norm ---
    const stuckNorm = Math.min(1, this.stuckCounter / STUCK_THRESHOLD);

    // --- Pheromone stats ---
    const { avg, variance: std, entropy } = this.aco.getPheromoneStats();
    const avgPhero = Math.min(1, avg / g.PHEROMONE_MAX);
    const stdPhero = Math.min(1, std / g.PHEROMONE_MAX);

    // --- Feature 5: blocked_edge_ratio ---
    const blockedRatio = totalEdges > 0
      ? Math.min(1, g.blockedEdges.size / totalEdges)
      : 0;

    // --- Feature 6, 7: current params normalized ---
    const alphaNorm = (this.currentAlpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN);
    const rhoNorm   = (this.currentRho   - RHO_MIN)   / (RHO_MAX   - RHO_MIN);

    // --- Feature 8: generation_progress (cyclic 0-100) ---
    const genProgress = (this.generation % 100) / 100;

    // --- Feature 9: pheromone_entropy (normalized) ---
    const normEntropy = Math.min(1, Math.max(0, entropy));

    return new Float32Array([
      normBestCost,
      improvementRate,
      stuckNorm,
      avgPhero,
      stdPhero,
      blockedRatio,
      Math.min(1, Math.max(0, alphaNorm)),
      Math.min(1, Math.max(0, rhoNorm)),
      genProgress,
      normEntropy,
    ]);
  }

  /** Đánh dấu sự kiện tắc đường (từ user click) */
  triggerTrafficEvent() {
    this.trafficTriggered = true;
    this.stuckCounter = 0;
    
    // Reset toàn bộ baseline! Vì bản đồ đã thay đổi (có đường bị tắc),
    // kỷ lục cũ (allTimeBest) không còn thể đạt được nữa.
    // Việc reset này giúp DDPG không bị kẹt ở mức phạt -1 vĩnh viễn.
    this.bestCost = Infinity;
    this.prevBestCost = Infinity;
    this.allTimeBest = Infinity;
  }

  getBestCost()   { return this.bestCost; }
  getGeneration() { return this.generation; }
}
