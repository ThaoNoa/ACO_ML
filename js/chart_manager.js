/**
 * chart_manager.js - Quản lý Line Chart thời gian thực
 * Sử dụng Chart.js để vẽ biểu đồ Best Distance theo Generation.
 * Hỗ trợ cập nhật động và hiển thị đẹp với dark theme.
 */

export class ChartManager {
  constructor(canvasId) {
    this.canvasId = canvasId;
    this.chart = null;
    this.maxDataPoints = 150; // Số điểm tối đa hiển thị (sliding window)
    this.generationData = [];
    this.distanceData = [];
    this.rewardData = [];
  }

  /**
   * Khởi tạo biểu đồ Chart.js
   */
  initialize() {
    if (this.chart) {
      this.chart.destroy();
    }

    const canvas = document.getElementById(this.canvasId);
    if (!canvas) {
      console.error(`Canvas #${this.canvasId} not found`);
      return;
    }

    const ctx = canvas.getContext('2d');

    // Gradient fill cho đường best distance
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.offsetHeight || 200);
    gradient.addColorStop(0, 'rgba(255, 215, 0, 0.3)');
    gradient.addColorStop(1, 'rgba(255, 215, 0, 0.0)');

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Best Distance',
            data: [],
            borderColor: '#ffd700',
            backgroundColor: gradient,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: '#ffd700',
            tension: 0.4,
            fill: true,
            yAxisID: 'yDist',
          },
          {
            label: 'Reward',
            data: [],
            borderColor: 'rgba(100,220,120,0.8)',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            fill: false,
            yAxisID: 'yReward',
            borderDash: [4, 3],
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,   // Tắt animation nội tại của Chart.js để nhanh hơn
        interaction: {
          intersect: false,
          mode: 'index',
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              color: 'rgba(200,210,240,0.8)',
              font: { size: 11, family: "'Inter', sans-serif" },
              boxWidth: 20,
              padding: 8,
            }
          },
          tooltip: {
            backgroundColor: 'rgba(10,15,30,0.9)',
            borderColor: 'rgba(74,158,255,0.4)',
            borderWidth: 1,
            titleColor: '#ffd700',
            bodyColor: 'rgba(200,210,240,0.9)',
            padding: 8,
            callbacks: {
              label: (ctx) => {
                if (ctx.dataset.label === 'Best Distance') {
                  return ` Distance: ${ctx.raw.toFixed(1)}`;
                }
                return ` Reward: ${ctx.raw}`;
              }
            }
          }
        },
        scales: {
          x: {
            ticks: {
              color: 'rgba(150,160,200,0.7)',
              font: { size: 10 },
              maxTicksLimit: 10,
              maxRotation: 0,
            },
            grid: {
              color: 'rgba(255,255,255,0.05)',
            },
          },
          yDist: {
            position: 'left',
            ticks: {
              color: 'rgba(255,215,0,0.7)',
              font: { size: 10 },
              maxTicksLimit: 6,
              callback: (val) => val.toFixed(0),
            },
            grid: {
              color: 'rgba(255,215,0,0.08)',
            },
          },
          yReward: {
            position: 'right',
            ticks: {
              color: 'rgba(100,220,120,0.7)',
              font: { size: 10 },
              maxTicksLimit: 5,
            },
            grid: { display: false },
          }
        }
      }
    });
  }

  /**
   * Thêm một điểm dữ liệu mới và cập nhật chart
   * @param {number} generation - Số thế hệ
   * @param {number} distance   - Khoảng cách tốt nhất
   * @param {number} reward     - Reward nhận được
   */
  addDataPoint(generation, distance, reward) {
    this.generationData.push(generation);
    this.distanceData.push(distance);
    this.rewardData.push(reward);

    // Sliding window
    if (this.generationData.length > this.maxDataPoints) {
      this.generationData.shift();
      this.distanceData.shift();
      this.rewardData.shift();
    }

    if (this.chart) {
      this.chart.data.labels = this.generationData.map(g => `G${g}`);
      this.chart.data.datasets[0].data = [...this.distanceData];
      this.chart.data.datasets[1].data = [...this.rewardData];
      this.chart.update('none'); // 'none' = no animation, fastest update
    }
  }

  /**
   * Reset chart về trạng thái ban đầu
   */
  reset() {
    this.generationData = [];
    this.distanceData = [];
    this.rewardData = [];
    if (this.chart) {
      this.chart.data.labels = [];
      this.chart.data.datasets[0].data = [];
      this.chart.data.datasets[1].data = [];
      this.chart.update('none');
    }
  }

  /**
   * Destroy chart (cleanup)
   */
  destroy() {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  }
}
