# SmartRoute — Hệ thống Điều phối Logistics Thích ứng

> **Dynamic TSP** giải quyết bằng **ACO (Ant Colony Optimization) + Q-Learning**

---

## Cách chạy

```bash
python server.py
# Trình duyệt tự mở tại http://localhost:8080
```

---

## Kiến trúc Module

```
Ant_Colony_v1/
│
├── index.html              # Entry point — Layout 3 khu vực
├── css/
│   └── style.css           # Design system (glassmorphism, dark theme)
├── js/
│   ├── graph.js            # [DATA] Graph: nodes, distances, pheromone matrix
│   ├── aco.js              # [ALGO] ACO Engine: đàn kiến, elitist strategy
│   ├── qlearning.js        # [AI]   Q-Learning Agent: state/action/reward
│   ├── renderer.js         # [VIEW] Canvas renderer: vẽ bản đồ, glow, particles
│   ├── chart_manager.js    # [VIEW] Chart.js: biểu đồ hội tụ real-time
│   └── app.js              # [CTRL] Orchestrator: vòng lặp, events, UI update
└── server.py               # HTTP server (Python stdlib)
```

---

## Thuật toán

### ACO Engine (`aco.js`)
- **30 kiến** mỗi generation, bắt đầu từ depot (node 0)
- Chọn node kế tiếp theo xác suất: `P(i→j) ∝ τ(i,j)^α × η(i,j)^β`
  - `τ`: pheromone, `η = 1/cost`: heuristic
- **Elitist strategy**: top 30% kiến + kiến tốt nhất được deposit mạnh hơn
- Pheromone bay hơi sau mỗi iteration: `τ *= (1 - ρ)`

### Q-Learning Agent (`qlearning.js`)

| State | Điều kiện |
|-------|-----------|
| S0: IMPROVING | Best distance đang giảm đều |
| S1: STUCK | ≥10 vòng không có kỷ lục mới |
| S2: DEGRADED | Tắc đường hoặc distance tệ hơn đột ngột |

| Action | Alpha | Rho | Chiến lược |
|--------|-------|-----|-----------|
| A0: Duy trì | 1.0 | 0.10 | Bình thường |
| A1: Khám phá | 0.5 | 0.80 | Xóa mùi cũ, tìm đường mới |
| A2: Khai thác | 2.0 | 0.05 | Bám chặt lộ trình tốt |

| Reward | Điều kiện |
|--------|-----------|
| +10 | Kỷ lục mới! |
| +1 | Duy trì ổn định |
| -5 | Vẫn kẹt |
| -10 | Đường đi tệ hơn |

---

## UI Layout

```
┌─────────────────── Header ──────────────────────┐
│ [Canvas 60%] │ [Chaos 15%] │ [Analytics 25%]  │
│              │ Start/Pause │ State Badge        │
│  Bản đồ      │ Tắc đường   │ ACO Params Live   │
│  mô phỏng    │ Reset       │ Gen / Best Dist   │
│              │ Legend      │ Line Chart         │
│              │             │ Q-Table            │
└──────────────────────────────────────────────────┘
```

---

## Tính năng nổi bật

- ✅ **Màu Pheromone động**: Cạnh đậm hơn = pheromone cao hơn
- ✅ **Particle kiến**: Hiệu ứng kiến chạy dọc lộ trình tốt nhất
- ✅ **Tắc đường real-time**: Click cạnh → đỏ + ×100 chi phí → AI phản ứng ngay
- ✅ **Q-Table hiển thị trực tiếp**: Trạng thái hiện tại được highlight
- ✅ **Biểu đồ kép**: Best Distance + Reward cùng một chart
- ✅ **Mũi tên hướng đi**: Hiển thị chiều di chuyển trên lộ trình tốt nhất
