import json
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu
with open('baseline_aco.json') as f:
    baseline = json.load(f)
with open('ql_aco.json') as f:
    ql = json.load(f)
with open('ddpg_aco.json') as f:
    ddpg = json.load(f)

# Lấy danh sách các thế hệ có tắc đường
traffic_gens = baseline.get('trafficGens', [200,400,600])

# 1. Biểu đồ Cumulative Reward
plt.figure(figsize=(12,5))
plt.plot(ql['generations'], ql['cumulativeRewards'], label='Q-Learning', color='#4a9eff')
plt.plot(ddpg['generations'], ddpg['cumulativeRewards'], label='DDPG', color='#a78bfa', linewidth=2)
plt.xlabel('Training Steps (Generations)')
plt.ylabel('Cumulative Reward')
plt.title('Learning Curve: Cumulative Reward theo thời gian')
plt.legend()
plt.grid(True, alpha=0.3)
for g in traffic_gens:
    plt.axvline(x=g, color='red', linestyle='--', alpha=0.5, label='Traffic' if g==traffic_gens[0] else '')
plt.savefig('fig1_learning_curve.png', dpi=150)
plt.show()

# 2. Biểu đồ Path Cost Over Time
plt.figure(figsize=(12,5))
plt.plot(baseline['generations'], baseline['bestCosts'], label='ACO thuần túy (α=1, ρ=0.1)', color='gray', linestyle='--')
plt.plot(ql['generations'], ql['bestCosts'], label='QL-ACO', color='#4a9eff')
plt.plot(ddpg['generations'], ddpg['bestCosts'], label='DDPG-ACO', color='#a78bfa', linewidth=2)
plt.xlabel('Generations')
plt.ylabel('Best Path Cost')
plt.title('So sánh chi phí lộ trình khi có tắc đường')
plt.legend()
plt.grid(True, alpha=0.3)
for g in traffic_gens:
    plt.axvline(x=g, color='red', linestyle='--', alpha=0.6)
    plt.text(g, plt.ylim()[1]*0.95, f'Traffic at gen {g}', rotation=90, fontsize=8)
plt.savefig('fig2_path_cost.png', dpi=150)
plt.show()

# 3. Biểu đồ Alpha & Rho Evolution (chỉ DDPG)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(ddpg['generations'], ddpg['alphaHistory'], color='#4a9eff', linewidth=1.5)
ax1.set_ylabel('Alpha (α)')
ax1.set_title('DDPG: Điều chỉnh tham số ACO theo thời gian')
ax1.grid(True, alpha=0.3)
for g in traffic_gens:
    ax1.axvline(x=g, color='red', linestyle='--', alpha=0.5)

ax2.plot(ddpg['generations'], ddpg['rhoHistory'], color='#a78bfa', linewidth=1.5)
ax2.set_xlabel('Generations')
ax2.set_ylabel('Rho (ρ)')
ax2.grid(True, alpha=0.3)
for g in traffic_gens:
    ax2.axvline(x=g, color='red', linestyle='--', alpha=0.5)
plt.savefig('fig3_alpha_rho_evolution.png', dpi=150)
plt.show()