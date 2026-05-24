/**
 * renderer.js - Canvas Renderer
 * Vẽ toàn bộ bản đồ mô phỏng lên HTML Canvas:
 *   - Các cạnh (edges) với màu sắc thay đổi theo pheromone
 *   - Các node (cities/depot)
 *   - Lộ trình tốt nhất hiện tại (highlighted)
 *   - Các cạnh bị tắc đường (màu đỏ)
 *   - Particles/animation effects
 */

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // Màu sắc chủ đạo (theme)
    this.colors = {
      bg:            '#0a0e1a',
      bgGrid:        'rgba(255,255,255,0.03)',
      edgeBase:      'rgba(100,120,180,0.15)',
      edgePhero:     [0, 200, 120],      // Màu xanh ngọc cho pheromone cao
      edgeBlocked:   '#ff3b3b',          // Màu đỏ tắc đường
      bestPath:      '#ffd700',          // Vàng cho lộ trình tốt nhất
      nodeDefault:   '#4a9eff',          // Xanh dương cho node thường
      nodeDepot:     '#ff6b35',          // Cam cho depot
      nodeHover:     '#ffffff',
      nodeGlow:      'rgba(74,158,255,0.4)',
      antTrail:      'rgba(255,200,50,0.6)',
      text:          '#e0e8ff',
    };

    this.animFrame = 0;
    this.antParticles = []; // Hiệu ứng kiến di chuyển

    // Hover state
    this.hoveredEdge = null;
    this.hoveredNode = null;
  }

  /**
   * Vẽ lại toàn bộ frame
   * @param {import('./graph.js').Graph} graph
   * @param {number[]|null} bestPath - Lộ trình tốt nhất
   * @param {object} options
   */
  render(graph, bestPath, options = {}) {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;

    this.animFrame++;

    // === 1. Background ===
    this._drawBackground(W, H);

    // === 2. Vẽ các cạnh (pheromone-colored) ===
    this._drawEdges(graph, bestPath);

    // === 3. Highlight lộ trình tốt nhất ===
    if (bestPath && bestPath.length > 0) {
      this._drawBestPath(graph, bestPath);
    }

    // === 4. Vẽ các cạnh tắc đường ===
    this._drawBlockedEdges(graph);

    // === 5. Vẽ các node ===
    this._drawNodes(graph, options.highlightNode);

    // === 6. Hiệu ứng kiến di chuyển ===
    this._updateAndDrawParticles();

    // === 7. Thông tin lộ trình ===
    if (bestPath && options.bestCost !== undefined) {
      this._drawPathInfo(options.bestCost, options.generation);
    }
  }

  /**
   * Vẽ nền gradient + grid
   */
  _drawBackground(W, H) {
    const ctx = this.ctx;

    // Gradient nền
    const grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W,H));
    grad.addColorStop(0, '#0d1426');
    grad.addColorStop(1, '#060910');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Grid chấm
    ctx.fillStyle = this.colors.bgGrid;
    for (let x = 30; x < W; x += 40) {
      for (let y = 30; y < H; y += 40) {
        ctx.beginPath();
        ctx.arc(x, y, 0.8, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  /**
   * Vẽ tất cả cạnh với màu sắc theo pheromone
   */
  _drawEdges(graph, bestPath) {
    const ctx = this.ctx;
    const n = graph.size;

    // Tìm max pheromone để normalize
    let maxPheromone = 0.001;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (graph.pheromones[i][j] > maxPheromone) {
          maxPheromone = graph.pheromones[i][j];
        }
      }
    }

    // Tạo set cạnh của lộ trình tốt nhất để skip
    const bestEdgeSet = new Set();
    if (bestPath) {
      for (let k = 0; k < bestPath.length - 1; k++) {
        const a = bestPath[k], b = bestPath[k + 1];
        bestEdgeSet.add(`${Math.min(a,b)}-${Math.max(a,b)}`);
      }
    }

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const key = `${i}-${j}`;
        if (bestEdgeSet.has(key)) continue; // Vẽ sau
        if (graph.isEdgeBlocked(i, j)) continue; // Vẽ riêng

        const t = graph.pheromones[i][j] / maxPheromone; // 0..1
        const alpha = 0.08 + t * 0.5;
        const width  = 0.5 + t * 2.5;

        const [r, g, b] = this.colors.edgePhero;
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.lineWidth = width;

        ctx.beginPath();
        ctx.moveTo(graph.nodes[i].x, graph.nodes[i].y);
        ctx.lineTo(graph.nodes[j].x, graph.nodes[j].y);
        ctx.stroke();
      }
    }
  }

  /**
   * Vẽ lộ trình tốt nhất với hiệu ứng glow và animation
   */
  _drawBestPath(graph, path) {
    const ctx = this.ctx;
    const pulse = 0.7 + 0.3 * Math.sin(this.animFrame * 0.05);

    // Glow layer
    ctx.shadowBlur = 20;
    ctx.shadowColor = `rgba(255, 215, 0, ${0.5 * pulse})`;
    ctx.strokeStyle = `rgba(255,215,0,${0.4 * pulse})`;
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(graph.nodes[path[0]].x, graph.nodes[path[0]].y);
    for (let k = 1; k < path.length; k++) {
      ctx.lineTo(graph.nodes[path[k]].x, graph.nodes[path[k]].y);
    }
    ctx.stroke();

    // Core line
    ctx.shadowBlur = 0;
    ctx.strokeStyle = this.colors.bestPath;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([8, 4]);
    ctx.beginPath();
    ctx.moveTo(graph.nodes[path[0]].x, graph.nodes[path[0]].y);
    for (let k = 1; k < path.length; k++) {
      ctx.lineTo(graph.nodes[path[k]].x, graph.nodes[path[k]].y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';

    // Mũi tên hướng đi
    for (let k = 0; k < path.length - 1; k++) {
      const a = graph.nodes[path[k]];
      const b = graph.nodes[path[k+1]];
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      this._drawArrow(mx, my, angle, 10, 'rgba(255,215,0,0.8)');
    }
  }

  /**
   * Vẽ mũi tên nhỏ
   */
  _drawArrow(x, y, angle, size, color) {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(size, 0);
    ctx.lineTo(-size, -size * 0.5);
    ctx.lineTo(-size, size * 0.5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  /**
   * Vẽ các cạnh bị tắc đường
   */
  _drawBlockedEdges(graph) {
    const ctx = this.ctx;

    for (const key of graph.blockedEdges) {
      const [i, j] = key.split('-').map(Number);
      if (i >= graph.size || j >= graph.size) continue;

      const pulse = 0.6 + 0.4 * Math.sin(this.animFrame * 0.08);

      // Glow đỏ
      ctx.shadowBlur = 15;
      ctx.shadowColor = `rgba(255,50,50,${0.6 * pulse})`;
      ctx.strokeStyle = `rgba(255,60,60,${0.5 * pulse})`;
      ctx.lineWidth = 10;
      ctx.beginPath();
      ctx.moveTo(graph.nodes[i].x, graph.nodes[i].y);
      ctx.lineTo(graph.nodes[j].x, graph.nodes[j].y);
      ctx.stroke();

      // Core đỏ
      ctx.shadowBlur = 0;
      ctx.strokeStyle = this.colors.edgeBlocked;
      ctx.lineWidth = 3;
      ctx.setLineDash([6, 3]);
      ctx.beginPath();
      ctx.moveTo(graph.nodes[i].x, graph.nodes[i].y);
      ctx.lineTo(graph.nodes[j].x, graph.nodes[j].y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Icon cảnh báo ở giữa cạnh
      const mx = (graph.nodes[i].x + graph.nodes[j].x) / 2;
      const my = (graph.nodes[i].y + graph.nodes[j].y) / 2;
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('⚠️', mx, my);
    }
    ctx.shadowBlur = 0;
  }

  /**
   * Vẽ các node với glow effects
   */
  _drawNodes(graph, highlightNodeId = -1) {
    const ctx = this.ctx;

    for (const node of graph.nodes) {
      const isDepot = node.isDepot;
      const isHighlighted = node.id === highlightNodeId;

      const baseRadius = isDepot ? 18 : 12;
      const pulse = isDepot
        ? 0.8 + 0.2 * Math.sin(this.animFrame * 0.04)
        : 1;
      const radius = baseRadius * pulse;

      // Outer glow
      const glowRadius = radius + 8;
      const glowGrad = ctx.createRadialGradient(
        node.x, node.y, radius * 0.3,
        node.x, node.y, glowRadius
      );
      glowGrad.addColorStop(0, isDepot
        ? 'rgba(255,107,53,0.4)'
        : (isHighlighted ? 'rgba(255,255,255,0.4)' : 'rgba(74,158,255,0.3)')
      );
      glowGrad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = glowGrad;
      ctx.beginPath();
      ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
      ctx.fill();

      // Node body
      const bodyGrad = ctx.createRadialGradient(
        node.x - radius * 0.3, node.y - radius * 0.3, 0,
        node.x, node.y, radius
      );
      if (isDepot) {
        bodyGrad.addColorStop(0, '#ff9a5c');
        bodyGrad.addColorStop(1, '#cc4a1a');
      } else {
        bodyGrad.addColorStop(0, '#7ab8ff');
        bodyGrad.addColorStop(1, '#1a5bb5');
      }
      ctx.fillStyle = bodyGrad;
      ctx.strokeStyle = isDepot ? '#ffb88a' : (isHighlighted ? '#fff' : '#4a9eff');
      ctx.lineWidth = isDepot ? 2.5 : 1.5;
      ctx.shadowBlur = isDepot ? 20 : 8;
      ctx.shadowColor = isDepot ? 'rgba(255,107,53,0.8)' : 'rgba(74,158,255,0.6)';

      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Label / icon
      if (isDepot) {
        ctx.font = `${Math.round(radius * 1.1)}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('🏭', node.x, node.y);
      } else {
        ctx.font = `bold ${Math.round(radius * 0.85)}px 'Inter', sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(node.label, node.x, node.y);
      }
    }
  }

  /**
   * Sinh và cập nhật các particle kiến chạy theo best path
   */
  spawnAntParticles(graph, bestPath) {
    if (!bestPath || bestPath.length < 2) return;
    if (this.antParticles.length > 40) return;

    const segIndex = Math.floor(Math.random() * (bestPath.length - 1));
    const a = graph.nodes[bestPath[segIndex]];
    const b = graph.nodes[bestPath[segIndex + 1]];

    this.antParticles.push({
      x: a.x, y: a.y,
      tx: b.x, ty: b.y,
      progress: 0,
      speed: 0.01 + Math.random() * 0.02,
      size: 3 + Math.random() * 2,
      opacity: 1,
    });
  }

  _updateAndDrawParticles() {
    const ctx = this.ctx;
    this.antParticles = this.antParticles.filter(p => p.opacity > 0);

    for (const p of this.antParticles) {
      p.progress += p.speed;
      if (p.progress >= 1) {
        p.progress = 1;
        p.opacity -= 0.05;
      }
      const t = p.progress;
      const x = p.x + (p.tx - p.x) * t;
      const y = p.y + (p.ty - p.y) * t;

      ctx.globalAlpha = p.opacity * 0.8;
      ctx.fillStyle = '#ffd700';
      ctx.shadowBlur = 8;
      ctx.shadowColor = 'rgba(255,200,0,0.8)';
      ctx.beginPath();
      ctx.arc(x, y, p.size * (1 - t * 0.5), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
  }

  /**
   * Vẽ thông tin chi phí lên canvas
   */
  _drawPathInfo(cost, generation) {
    const ctx = this.ctx;
    ctx.font = 'bold 13px "Inter", monospace';
    ctx.fillStyle = 'rgba(255,215,0,0.9)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`Gen ${generation} | Best: ${cost.toFixed(1)}px`, 12, 12);
  }

  /**
   * Tìm cạnh gần nhất với điểm click (dùng cho chức năng tắc đường)
   * @returns {{ i, j } | null}
   */
  findNearestEdge(graph, mouseX, mouseY, threshold = 15) {
    let minDist = threshold;
    let nearest = null;
    const n = graph.size;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = this._pointToSegmentDist(
          mouseX, mouseY,
          graph.nodes[i].x, graph.nodes[i].y,
          graph.nodes[j].x, graph.nodes[j].y
        );
        if (d < minDist) {
          minDist = d;
          nearest = { i, j };
        }
      }
    }
    return nearest;
  }

  /**
   * Khoảng cách từ điểm (px,py) đến đoạn thẳng (ax,ay)-(bx,by)
   */
  _pointToSegmentDist(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - ax, py - ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  /**
   * Tìm node gần nhất với điểm click
   */
  findNearestNode(graph, mouseX, mouseY, threshold = 20) {
    let minDist = threshold;
    let nearest = null;

    for (const node of graph.nodes) {
      const d = Math.hypot(mouseX - node.x, mouseY - node.y);
      if (d < minDist) {
        minDist = d;
        nearest = node;
      }
    }
    return nearest;
  }
}
