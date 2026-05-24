/**
 * qlearning.js - Q-Learning Agent
 * Điều chỉnh tham số ACO dựa trên trạng thái hệ thống.
 *
 * State Space (3 trạng thái):
 *   S0: Đường đi đang cải thiện đều đặn
 *   S1: Bị kẹt ≥ 10 vòng không có kỷ lục mới
 *   S2: Đường đi đột ngột tệ đi (khi tắc đường xảy ra)
 *
 * Action Space (3 hành động):
 *   A0: Duy trì   -> Alpha=1.0, Rho=0.10
 *   A1: Khám phá  -> Alpha=0.5, Rho=0.80 (xóa mùi cũ, tìm đường mới)
 *   A2: Khai thác -> Alpha=2.0, Rho=0.05 (bám chặt lộ trình tốt)
 *
 * Reward:
 *   +10: Kỷ lục đường đi giảm
 *    -5: Vẫn kẹt (không kỷ lục mới)
 *   -10: Đường đi dài thêm đáng kể
 */

export const STATES = { IMPROVING: 0, STUCK: 1, DEGRADED: 2 };
export const ACTIONS = {
  MAINTAIN:  0,  // A0
  EXPLORE:   1,  // A1
  EXPLOIT:   2,  // A2
};

// Tham số tương ứng với từng Action
export const ACTION_PARAMS = {
  [ACTIONS.MAINTAIN]: { alpha: 1.0,  rho: 0.10, label: 'Duy trì (A0)'  },
  [ACTIONS.EXPLORE]:  { alpha: 0.5,  rho: 0.80, label: 'Khám phá (A1)' },
  [ACTIONS.EXPLOIT]:  { alpha: 2.0,  rho: 0.05, label: 'Khai thác (A2)'},
};

export const STATE_LABELS = {
  [STATES.IMPROVING]: 'BÌNH THƯỜNG',
  [STATES.STUCK]:     'ĐANG KẸT - TÌM ĐƯỜNG MỚI',
  [STATES.DEGRADED]:  'PHÁT HIỆN TẮC ĐƯỜNG',
};

export class QLearningAgent {
  constructor(config = {}) {
    this.learningRate = config.learningRate ?? 0.3;  // α (lr)
    this.discountFactor = config.discountFactor ?? 0.9;  // γ
    this.epsilon = config.epsilon ?? 0.2;           // Exploration rate ban đầu
    this.epsilonDecay = config.epsilonDecay ?? 0.995;
    this.epsilonMin = config.epsilonMin ?? 0.05;

    const numStates = 3, numActions = 3;
    // Q-Table khởi tạo tối ưu (optimistic initialization để khuyến khích khám phá)
    this.qTable = Array.from({ length: numStates }, () =>
      new Float64Array(numActions).fill(5.0)
    );

    this.currentState = STATES.IMPROVING;
    this.currentAction = ACTIONS.MAINTAIN;
    this.lastAction = ACTIONS.MAINTAIN;

    // Lịch sử để phát hiện trạng thái
    this.stuckCounter = 0;
    this.STUCK_THRESHOLD = 10; // Vòng không cải thiện trước khi vào S1

    this.prevBestCost = Infinity;
    this.allTimeBest = Infinity;

    this.trafficEventTriggered = false; // Cờ khi user tạo tắc đường

    // Lịch sử reward để vẽ
    this.rewardHistory = [];
    this.actionHistory = [];
    this.stateHistory = [];
  }

  /**
   * Xác định trạng thái hiện tại dựa trên lịch sử
   * @param {number} currentCost - Chi phí lộ trình tốt nhất vòng này
   */
  determineState(currentCost) {
    if (this.trafficEventTriggered) {
      // Tắc đường vừa xảy ra
      this.trafficEventTriggered = false;
      return STATES.DEGRADED;
    }

    if (currentCost < this.allTimeBest - 0.01) {
      // Đang cải thiện
      this.stuckCounter = 0;
      this.allTimeBest = currentCost;
      return STATES.IMPROVING;
    }

    this.stuckCounter++;
    if (this.stuckCounter >= this.STUCK_THRESHOLD) {
      return STATES.STUCK;
    }

    // Kiểm tra xem có tệ đi đột ngột không (>20% so với best)
    if (currentCost > this.allTimeBest * 1.2 && this.allTimeBest !== Infinity) {
      return STATES.DEGRADED;
    }

    return STATES.IMPROVING; // Vẫn ổn, chưa kẹt đủ lâu
  }

  /**
   * Chọn action theo epsilon-greedy
   * @param {number} state
   */
  selectAction(state) {
    if (Math.random() < this.epsilon) {
      // Exploration: chọn ngẫu nhiên
      return Math.floor(Math.random() * 3);
    }
    // Exploitation: chọn action có Q-value cao nhất
    const qValues = this.qTable[state];
    let bestAction = 0;
    for (let a = 1; a < qValues.length; a++) {
      if (qValues[a] > qValues[bestAction]) bestAction = a;
    }
    return bestAction;
  }

  /**
   * Tính reward dựa trên kết quả của action vừa thực hiện
   * @param {number} newCost - Chi phí vòng hiện tại
   * @param {number} prevCost - Chi phí vòng trước
   */
  computeReward(newCost, prevCost) {
    if (newCost < this.allTimeBest) {
      return 10; // Kỷ lục mới!
    }
    if (newCost > prevCost * 1.05) {
      return -10; // Đường đi tệ hơn đáng kể
    }
    if (this.stuckCounter >= this.STUCK_THRESHOLD) {
      return -5; // Vẫn kẹt
    }
    return 1; // Duy trì ổn định
  }

  /**
   * Cập nhật Q-Table theo Bellman equation
   * @param {number} state  - Trạng thái hiện tại
   * @param {number} action - Action đã thực hiện
   * @param {number} reward - Phần thưởng nhận được
   * @param {number} nextState - Trạng thái tiếp theo
   */
  updateQTable(state, action, reward, nextState) {
    const currentQ = this.qTable[state][action];
    const maxNextQ = Math.max(...this.qTable[nextState]);

    // Bellman equation
    const newQ = currentQ + this.learningRate * (
      reward + this.discountFactor * maxNextQ - currentQ
    );
    this.qTable[state][action] = newQ;

    // Decay epsilon
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  /**
   * Bước chính của Q-Learning: step() gồm toàn bộ vòng lặp
   * @param {number} currentCost - Chi phí tốt nhất vòng này
   * @returns {{ action, params, state, reward }}
   */
  step(currentCost) {
    const prevCost = this.prevBestCost;

    // 1. Xác định trạng thái mới
    const newState = this.determineState(currentCost);

    // 2. Tính reward cho action trước đó
    const reward = this.computeReward(currentCost, prevCost);

    // 3. Cập nhật Q-Table
    this.updateQTable(this.currentState, this.currentAction, reward, newState);

    // 4. Chọn action mới cho vòng tiếp theo
    const newAction = this.selectAction(newState);

    // 5. Lưu lịch sử
    this.rewardHistory.push(reward);
    this.actionHistory.push(newAction);
    this.stateHistory.push(newState);

    // 6. Cập nhật trạng thái
    this.currentState = newState;
    this.currentAction = newAction;
    this.prevBestCost = currentCost;
    if (currentCost < this.allTimeBest) {
      this.allTimeBest = currentCost;
    }

    return {
      action: newAction,
      params: ACTION_PARAMS[newAction],
      state: newState,
      reward,
      epsilon: this.epsilon,
      qValues: this.qTable[newState].slice(),
    };
  }

  /**
   * Kích hoạt cờ tắc đường (do user bấm nút)
   */
  triggerTrafficEvent() {
    this.trafficEventTriggered = true;
    this.stuckCounter = 0; // Reset stuck counter
  }

  /**
   * Lấy Q-Table dưới dạng object dễ đọc để debug
   */
  getQTableDisplay() {
    const stateNames = ['S0:Improving', 'S1:Stuck', 'S2:Degraded'];
    const actionNames = ['A0:Maintain', 'A1:Explore', 'A2:Exploit'];
    return this.qTable.map((row, s) =>
      actionNames.map((a, ai) => ({
        state: stateNames[s],
        action: a,
        q: row[ai].toFixed(2),
      }))
    ).flat();
  }
}
