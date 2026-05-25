# SmartRoute v3 — Hệ thống Điều phối Logistics Thích ứng

> **Dynamic TSP** giải quyết bằng **ACO (Ant Colony Optimization)** kết hợp 2 bộ não AI có thể chuyển đổi qua lại:
> - **Q-Learning (Tabular)** — 3 trạng thái × 3 hành động rời rạc
> - **DDPG (Deep Reinforcement Learning)** — điều chỉnh liên tục tham số Alpha/Rho bằng mạng Neural Network

---

## ⚙️ Yêu cầu phần mềm

| Phần mềm | Phiên bản tối thiểu | Link tải |
|---|---|---|
| **Node.js** | v18 trở lên | https://nodejs.org (chọn bản LTS) |
| Trình duyệt | Chrome / Edge / Firefox mới nhất | — |

> **Lưu ý:** Dự án này **không cần Python** để chạy giao diện web nữa (phiên bản cũ dùng `server.py`, phiên bản v3 đã chuyển sang Vite + Node.js). Các file Python (`main.py`, `aco_engine.py`, v.v.) là phiên bản console cũ, vẫn được giữ lại để tham khảo.

---

## 🚀 Cài đặt & Chạy

### Bước 1: Cài Node.js
Tải và cài đặt từ https://nodejs.org (chọn **LTS**). Sau khi cài, khởi động lại máy tính (hoặc ít nhất là khởi động lại terminal/IDE).

Kiểm tra cài đặt thành công:
```powershell
node --version   # Phải ra: v18.x.x hoặc cao hơn
npm --version    # Phải ra: 9.x.x hoặc cao hơn
```

### Bước 2: Mở PowerShell và cấp quyền chạy script (chỉ cần làm 1 lần)
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Bước 3: Cài thư viện
```powershell
cd "E:\Python Projects\Ant_Colony_QL_DQN"
npm install
```
Lệnh này sẽ tải về `@tensorflow/tfjs` (thư viện AI) và `vite` (web server). Chỉ cần chạy **một lần duy nhất**.

### Bước 4: Khởi động
```powershell
npm run dev
```
Mở trình duyệt và truy cập: **http://localhost:5173/**

---

## 🎮 Hướng dẫn sử dụng

| Nút | Chức năng |
|---|---|
| ▶ **Bắt đầu** | Khởi động / tiếp tục mô phỏng |
| ⏸ **Tạm dừng** | Tạm dừng |
| 🚧 **Tạo Tắc Đường** | Click vào để kích hoạt, sau đó click vào cạnh bất kỳ trên bản đồ để chặn đường (chi phí ×100) |
| ❌ **Xoá Tắc Đường** | Gỡ bỏ toàn bộ điểm tắc nghẽn, AI học lại từ đầu |
| 🔄 **Khởi tạo lại** | Tạo bản đồ mới ngẫu nhiên |
| **🧠 Mode Switch** (header) | Chuyển đổi giữa Q-Learning ↔ DDPG |
| ⚡ **Train DDPG (Fast)** | Huấn luyện DDPG 2,000 thế hệ ở chế độ nhanh (không render đồ họa) — **nên làm trước khi chạy DDPG** |
| 💾 **Export** | Lưu trọng số mạng Neural xuống máy tính (2 file: `.json` + `.bin`) |
| 📂 **Import** | Tải lại trọng số đã lưu (chọn cả 2 file cùng lúc) |
| 📊 **Run Test Suite** | Chạy 5 bài test chuẩn và so sánh độ chính xác QL-ACO vs DDPG so với đường tối ưu (Brute-force) |

### Luồng sử dụng DDPG đề xuất:
1. Nhấn **Mode Switch** để chuyển sang DDPG
2. Nhấn **⚡ Train DDPG (Fast)** — đợi 2–5 giây
3. Nhấn **▶ Bắt đầu** để xem kết quả thực chiến
4. Thêm tắc đường bằng **🚧 Tạo Tắc Đường** và quan sát AI phản ứng
5. (Tuỳ chọn) Nhấn **💾 Export** để lưu lại model xịn

---

## 📁 Cấu trúc dự án

```
Ant_Colony_QL_DQN/
│
├── index.html              # Entry point — giao diện web chính
├── package.json            # Cấu hình npm (TF.js + Vite)
├── vite.config.js          # Cấu hình Vite dev server
│
├── css/
│   └── style.css           # Design system (glassmorphism, dark theme)
│
├── js/
│   ├── graph.js            # [DATA]  Đồ thị: nodes, khoảng cách, ma trận pheromone
│   ├── aco.js              # [ALGO]  ACO Engine: đàn kiến, elitist strategy
│   ├── qlearning.js        # [AI-1]  Q-Learning Agent (tabular, 3×3)
│   ├── environment.js      # [AI-2]  Môi trường DDPG: state vector 10D, reward
│   ├── ddpg_agent.js       # [AI-2]  DDPG Agent: Actor-Critic, Replay Buffer, OU Noise
│   ├── trainer.js          # [AI-2]  Headless Trainer: train nhanh không render
│   ├── test_suite.js       # [EVAL]  5 Test Cases cố định + Brute-force solver
│   ├── evaluator.js        # [EVAL]  Chạy test tự động, so sánh QL vs DDPG
│   ├── renderer.js         # [VIEW]  Canvas renderer: bản đồ, glow, particles
│   ├── chart_manager.js    # [VIEW]  Chart.js: biểu đồ hội tụ real-time
│   └── app.js              # [CTRL]  Orchestrator: vòng lặp, events, UI update
│
├── server.py               # (Cũ) HTTP server Python — không cần dùng nữa
├── main.py                 # (Cũ) Console demo ACO+QL bằng Python
└── aco_engine.py           # (Cũ) ACO Engine Python — tham khảo
```

---

## 🧠 Thuật toán

### ACO Engine (`aco.js`)
- **30 kiến** mỗi generation, bắt đầu từ depot (node 0)
- Chọn node kế tiếp: `P(i→j) ∝ τ(i,j)^α × η(i,j)^β`  
  (`τ`: pheromone, `η = 1/cost`: heuristic khoảng cách)
- **Elitist strategy**: top 30% kiến + kiến tốt nhất deposit mạnh hơn
- Bay hơi: `τ *= (1 - ρ)` sau mỗi iteration

### Q-Learning Agent (`qlearning.js`)

| State | Điều kiện |
|---|---|
| S0: IMPROVING | Best distance đang giảm |
| S1: STUCK | ≥10 gen không có kỷ lục mới |
| S2: DEGRADED | Tắc đường hoặc distance tăng đột ngột |

| Action | Alpha | Rho | Chiến lược |
|---|---|---|---|
| A0: Duy trì | 1.0 | 0.10 | Ổn định |
| A1: Khám phá | 0.5 | 0.80 | Xóa mùi cũ, tìm đường mới |
| A2: Khai thác | 2.0 | 0.05 | Bám chặt lộ trình tốt |

### DDPG Agent (`ddpg_agent.js` + `environment.js`)
- **Actor**: `state(10D)` → Dense(256) → Dense(256) → Dense(128) → `tanh(2)` → `{alpha, rho}`
- **Critic**: `[state ∥ action]` → Dense(256) → Dense(256) → Dense(128) → `Q(1)`
- **State vector 10D**: normalized_best_cost, improvement_rate, stuck_counter, avg_pheromone, std_pheromone, blocked_ratio, alpha_norm, rho_norm, gen_progress, pheromone_entropy
- **Action space**: `alpha ∈ [0.5, 2.5]`, `rho ∈ [0.02, 0.5]` (liên tục)
- **Replay Buffer**: 5,000 transitions, batch size 64
- **OU Noise**: khám phá (σ giảm dần từ 0.3 → 0.05)

---

## ✨ Tính năng nổi bật

- ✅ **2 chế độ AI** chuyển đổi nóng (Q-Learning ↔ DDPG) mà không cần reload
- ✅ **Headless Training**: Train DDPG 2,000 gen trong vài giây, không cần render
- ✅ **Export / Import Model**: Lưu và tải lại trọng số Neural Network
- ✅ **Test Suite tự động**: 5 bài test (N=8, N=10) với Brute-force optimal làm chuẩn
- ✅ **Tắc đường real-time**: Click cạnh → đỏ + ×100 chi phí → AI phản ứng ngay
- ✅ **Xoá tắc đường**: Gỡ toàn bộ điểm tắc nghẽn một lần, AI reset baseline
- ✅ **Pheromone động**: Cạnh đậm hơn = pheromone cao hơn
- ✅ **Particle kiến**: Hiệu ứng kiến chạy dọc lộ trình tốt nhất
- ✅ **State Vector hiển thị**: 10 chiều DDPG được visualize bằng thanh mini real-time
- ✅ **Biểu đồ kép**: Best Distance + Reward trên cùng chart
