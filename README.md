# TỔNG QUAN DỰ ÁN DÀNH CHO DEVELOPER (DEVELOPER GUIDE)

Tài liệu này cung cấp cái nhìn toàn diện về kiến trúc mã nguồn, luồng dữ liệu (data flow), và logic thuật toán của dự án **SmartRoute v3**. Dự án là sự kết hợp giữa thuật toán Đàn kiến (ACO) và Học tăng cường (Q-Learning & DDPG) để giải quyết bài toán Dynamic TSP.

---

## 1. Cấu trúc Thư mục & Vai trò các File

Mã nguồn frontend hoàn toàn dùng Vanilla JavaScript (ES6 Modules) để đạt hiệu năng cao nhất trên trình duyệt. TensorFlow.js được dùng để thiết kế Mạng Neural.

- `index.html`: Chứa cấu trúc UI, Canvas và tích hợp thư viện Chart.js.
- `js/app.js`: File Orchestrator (Nhạc trưởng). Quản lý Game Loop (chạy liên tục qua `requestAnimationFrame`), xử lý DOM events, cập nhật UI và chuyển đổi giữa 2 chế độ (QL / DDPG).
- `js/graph.js`: Quản lý Đồ thị, tạo tọa độ ngẫu nhiên, ma trận khoảng cách, ma trận pheromone và chức năng chặn đường (Block Edge).
- `js/aco.js`: Chứa thuật toán ACO (Logic con kiến bò). Có thể nhận tham số $\alpha, \rho$ động từ bên ngoài đưa vào.
- `js/qlearning.js`: Chứa thuật toán Học tăng cường dạng bảng (Tabular Q-Learning).
- `js/ddpg_agent.js`: Chứa tác tử DDPG (Actor-Critic) dùng TensorFlow.js. Quản lý 4 mạng Neural, Replay Buffer và quá trình Backpropagation.
- `js/environment.js`: Đóng vai trò là Môi trường (Gym-like Environment) cho DDPG. Nó trích xuất 10 thông số từ `graph` + `aco` để tạo thành State, tính toán Reward và phạt AI khi có tắc đường.
- `js/trainer.js`: Chạy DDPG ở chế độ Headless (không render giao diện) để huấn luyện nhanh 2000 vòng, mô phỏng cả tắc đường để AI học.
- `js/evaluator.js`: Chứa Test Suite, chạy so sánh tự động giữa Q-Learning và DDPG.
- `js/renderer.js`: Chịu trách nhiệm vẽ Canvas (đường thẳng, node, xe cộ, hiệu ứng hover, vết mùi).

---

## 2. Luồng Logic Chính (App Loop)

Trái tim của ứng dụng nằm ở vòng lặp `startLoop()` trong `app.js`. Nó chạy mỗi $120ms$ theo luồng sau:
1. **Lấy Trạng thái (Get State):** `env.getState()` lấy 10 thông số bản đồ.
2. **Ra Quyết định (Action):** DDPG Agent / QL Agent dựa vào State để xuất ra $\alpha, \rho$.
3. **Áp dụng (Apply):** `aco.setParams(alpha, rho)`.
4. **Mô phỏng 1 Vòng (Simulate):** Đàn kiến chạy (`aco.runIteration()`), nhả mùi lên `graph`.
5. **Nhận Thưởng (Reward):** Tính toán xem điểm số tối ưu có tốt lên không $\rightarrow$ Cập nhật lại Agent.
6. **Vẽ lại (Render):** `renderer.render(...)` xuất đồ họa ra màn hình.

---

## 3. Core Logic: Thuật toán Đàn kiến (ACO)

Nằm trong `js/aco.js`. Mỗi con kiến trong số `NUM_ANTS = 30` sẽ xây dựng lộ trình dựa trên công thức xác suất chọn node tiếp theo:

$$ P_{ij} = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum [\tau_{ik}]^\alpha \cdot [\eta_{ik}]^\beta} $$

*Các biến quan trọng:*
- `this.alpha`: Trọng số vết mùi $\tau$ (Được AI điều khiển liên tục).
- `this.beta = 2.0`: Trọng số khoảng cách $\eta$ (Cố định). $\eta = 1/d$.
- `this.rho`: Tốc độ bay hơi (Được AI điều khiển liên tục).
- Cập nhật Pheromone: $\tau_{ij} = (1-\rho)\tau_{ij} + \sum \Delta \tau_{ij}$

---

## 4. Core Logic: Môi trường & State Vector (environment.js)

Để Mạng Neural hiểu được bản đồ, `TSPEnvironment` nén toàn bộ đồ thị thành một vector số thực 10 chiều (10-Dimensional Array).

*Các biến trong State (Values từ 0.0 đến 1.0):*
1. `normBestCost`: Chi phí tốt nhất (Chuẩn hóa so với kỷ lục).
2. `improvementRate`: $\%$ cải thiện so với vòng trước.
3. `stuckNorm`: Điểm số phạt nếu nhiều vòng không tìm được đường mới.
4. `avgPhero`: Lượng mùi trung bình của toàn bản đồ.
5. `stdPhero`: Độ phân tán của mùi.
6. `blockedRatio`: Tỷ lệ các đoạn đường đang bị tắc (`graph.blockedEdges.size / totalEdges`).
7. `alphaNorm`: $\alpha$ vòng trước (chuẩn hóa).
8. `rhoNorm`: $\rho$ vòng trước (chuẩn hóa).
9. `genProgress`: Chu kỳ thời gian.
10. `normEntropy`: Độ hỗn loạn của mùi (Nếu bằng 0 tức là đàn kiến đang bám vào 1 đường duy nhất).

*Hàm phần thưởng cốt lõi `_computeReward(roundCost)`:*
- **Phạt tắc đường:** Nếu kiến đâm vào đường tắc (`roundCost > 3000`), trả về luôn `-20` điểm.
- **Thưởng tiến bộ:** Nếu chi phí đường đi giảm, nhận `+10` đến `+20`.
- **Phạt kẹt (Stuck):** Kẹt cứng quá 5 vòng nhận `-10`.

---

## 5. Core Logic: Deep RL (ddpg_agent.js)

Sử dụng thư viện `@tensorflow/tfjs`. DDPG là một thuật toán Off-policy, Actor-Critic.

### 5.1. Kiến trúc Mạng Neural
- **Actor Network (Người ra quyết định):**
  - Input: Tensor 1D (Size 10) $\rightarrow$ Dense (256, ReLU) $\rightarrow$ Dense (256, ReLU) $\rightarrow$ Dense (128, ReLU) $\rightarrow$ Output: Tensor 1D (Size 2, Tanh).
  - Tanh giới hạn đầu ra trong khoảng $[-1, 1]$, sau đó được map thành $\alpha \in [0.5, 2.5]$ và $\rho \in [0.02, 0.5]$.
- **Critic Network (Trọng tài đánh giá):**
  - Input: Nối ghép State (10) + Action từ Actor (2) = Tensor 1D (Size 12).
  - Kiến trúc tương tự Actor $\rightarrow$ Output: 1 Q-Value tuyến tính (Linear).

### 5.2. Luồng Huấn luyện (Backpropagation)
- Dữ liệu lưu vào mảng `ReplayBuffer` (chứa các bộ `<State, Action, Reward, NextState>`).
- Lấy ngẫu nhiên Batch 64 mẫu (Batch Size).
- **Học Critic:** Tính Target Q = Reward + $\gamma \cdot Q'(NextState, \mu'(NextState))$. Dùng Gradient Descent giảm MSE Loss giữa Q dự đoán và Target Q.
- **Học Actor:** Tính Gradient của Critic theo Action, nhân ngược (chain rule) vào Actor để Actor tìm ra Action đẩy Q-value lên mức tối đa. Hàm mất mát của Actor: `-tf.mean(Q)`.
- Cập nhật mạng Target (Polyak Averaging) với $\tau = 0.005$.

### 5.3. OU Noise (Ornstein-Uhlenbeck)
- Là hàm tạo nhiễu ngẫu nhiên có quán tính, giúp AI khám phá môi trường. Thanh "Noise" trên UI hiển thị biên độ của OU Noise. Qua mỗi bước học, nhiễu giảm dần (Decay) để AI ổn định khai thác (Exploitation).

---

## 6. Logic giả lập tắc đường trong lúc Train (`trainer.js`)

Để đảm bảo Actor Network thực sự biết cách đối phó với giao thông động, `HeadlessTrainer` chạy ngầm và tự động ép các tình huống xấu:
- Cứ mỗi 200 Generation: Nó lấy đường đi tối ưu (`bestPath`), trích xuất ngẫu nhiên 1 đoạn thẳng, và gọi hàm `graph.blockEdge(u, v)`.
- Lúc này chi phí đi qua đoạn thẳng đó bị x100. Môi trường sẽ nhận thấy và ném `-20` điểm phạt thẳng vào mặt Actor nếu Actor không chịu nhả $\rho$ để kiến chuyển hướng.
- Sau 50 Generation: Gọi `graph.clearAllBlockedEdges()` để trả lại bản đồ bình thường. Quá trình này lặp lại 10 lần (2000 vòng) giúp AI tôi luyện tính linh hoạt.
