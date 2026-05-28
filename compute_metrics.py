import json
import numpy as np
import sys


def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def compute_metrics(data, traffic_gens, window_stable=100):
    """
    Tính các chỉ số từ dữ liệu thí nghiệm.
    - data: dict chứa generations, bestCosts
    - traffic_gens: list các gen xảy ra tắc đường
    - window_stable: số gen cuối để tính chi phí ổn định (tĩnh)
    """
    costs = np.array(data['bestCosts'])
    gens = np.array(data['generations'])

    # Chi phí tĩnh: trung bình của 100 gen cuối cùng (giả định đã hội tụ)
    static_cost = np.mean(costs[-window_stable:]) if len(costs) >= window_stable else np.mean(costs)

    # Chi phí động: trung bình toàn bộ quá trình
    dynamic_cost = np.mean(costs)

    # Số thế hệ hội tụ: gen đầu tiên mà cost <= 1.05 * static_cost và giữ ổn định trong 20 gen tiếp
    threshold = static_cost * 1.05
    converged_gen = len(gens)  # default = max
    for i in range(len(costs) - 20):
        if costs[i] <= threshold and all(costs[i:i + 20] <= threshold * 1.02):
            converged_gen = gens[i]
            break

    # Tỷ lệ vượt qua tắc đường (recovery ratio) trung bình cho mỗi sự kiện
    recovery_ratios = []
    for tg in traffic_gens:
        # Tìm đỉnh cost sau traffic (trong vòng 30 gen sau)
        idx_traffic = np.where(gens == tg)[0]
        if len(idx_traffic) == 0:
            continue
        idx = idx_traffic[0]
        # cost ngay sau traffic (có thể gen chính xác chưa cập nhật, lấy gen tiếp theo)
        peak_idx = min(idx + 5, len(costs) - 1)  # tìm đỉnh trong 5 gen sau traffic
        peak_cost = max(costs[idx:peak_idx + 1])
        # cost ổn định sau 100 gen
        stable_idx = min(idx + 100, len(costs) - 1)
        stable_after = np.mean(costs[stable_idx:stable_idx + 50]) if stable_idx + 50 <= len(costs) else costs[
            stable_idx]
        if peak_cost > 0:
            recovery = (peak_cost - stable_after) / peak_cost
            recovery_ratios.append(max(0, recovery))

    avg_recovery = np.mean(recovery_ratios) if recovery_ratios else 0.0

    return {
        'static_cost': static_cost,
        'dynamic_cost': dynamic_cost,
        'converged_gen': converged_gen,
        'recovery_rate': avg_recovery,
    }


def main():
    # Đường dẫn file (sửa lại nếu cần)
    baseline_file = 'baseline_aco.json'
    ql_file = 'ql_aco.json'
    ddpg_file = 'ddpg_aco.json'

    try:
        baseline = load_data(baseline_file)
        ql = load_data(ql_file)
        ddpg = load_data(ddpg_file)
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file {e.filename}. Hãy chạy thí nghiệm trước.")
        sys.exit(1)

    # Lấy danh sách gen tắc đường từ bất kỳ file nào (giả sử giống nhau)
    traffic_gens = baseline.get('trafficGens', [200, 400, 600])

    metrics = {}
    for name, data in [('ACO Truyền thống', baseline), ('ACO + Q-Learning', ql), ('ACO + DDPG (Đề xuất)', ddpg)]:
        metrics[name] = compute_metrics(data, traffic_gens)

    # In bảng Markdown
    print("\n## Bảng 1: So sánh hiệu năng tổng quát (sau 1000 thế hệ)\n")
    print("| Tiêu chí | ACO Truyền thống | ACO + Q-Learning | ACO + DDPG (Đề xuất) |")
    print("|----------|------------------|------------------|----------------------|")
    print(
        f"| Chi phí trung bình (Tĩnh) | {metrics['ACO Truyền thống']['static_cost']:.1f} | {metrics['ACO + Q-Learning']['static_cost']:.1f} | {metrics['ACO + DDPG (Đề xuất)']['static_cost']:.1f} |")
    print(
        f"| Chi phí trung bình (Động) | {metrics['ACO Truyền thống']['dynamic_cost']:.1f} | {metrics['ACO + Q-Learning']['dynamic_cost']:.1f} | {metrics['ACO + DDPG (Đề xuất)']['dynamic_cost']:.1f} |")
    print(
        f"| Số thế hệ hội tụ | {metrics['ACO Truyền thống']['converged_gen']} | {metrics['ACO + Q-Learning']['converged_gen']} | {metrics['ACO + DDPG (Đề xuất)']['converged_gen']} |")

    # Xuất CSV
    import csv
    with open('comparison_table.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Tiêu chí', 'ACO Truyền thống', 'ACO + Q-Learning', 'ACO + DDPG (Đề xuất)'])
        writer.writerow(['Chi phí trung bình (Tĩnh)',
                         f"{metrics['ACO Truyền thống']['static_cost']:.1f}",
                         f"{metrics['ACO + Q-Learning']['static_cost']:.1f}",
                         f"{metrics['ACO + DDPG (Đề xuất)']['static_cost']:.1f}"])
        writer.writerow(['Chi phí trung bình (Động)',
                         f"{metrics['ACO Truyền thống']['dynamic_cost']:.1f}",
                         f"{metrics['ACO + Q-Learning']['dynamic_cost']:.1f}",
                         f"{metrics['ACO + DDPG (Đề xuất)']['dynamic_cost']:.1f}"])
        writer.writerow(['Số thế hệ hội tụ',
                         metrics['ACO Truyền thống']['converged_gen'],
                         metrics['ACO + Q-Learning']['converged_gen'],
                         metrics['ACO + DDPG (Đề xuất)']['converged_gen']])

    print("\n✅ Đã lưu bảng dạng CSV vào file 'comparison_table.csv'")


if __name__ == '__main__':
    main()