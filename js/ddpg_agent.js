/**
 * ddpg_agent.js — Deep Deterministic Policy Gradient Agent
 *
 * Điều chỉnh liên tục 2 tham số ACO:
 *   alpha ∈ [0.3, 2.5]  (trọng số pheromone)
 *   rho   ∈ [0.02, 0.9] (tốc độ bay hơi)
 *
 * Kiến trúc:
 *   Actor:  state(10) → Dense(256) → Dense(256) → Dense(128) → tanh(2) → scale → {alpha, rho}
 *   Critic: [state(10) ∥ action(2)] → Dense(256) → Dense(256) → Dense(128) → Q(1)
 *   + Target networks (soft update Polyak, τ=0.005)
 *   + Experience Replay Buffer (5000 transitions)
 *   + Ornstein-Uhlenbeck Noise (exploration, σ decay 0.3→0.05)
 */

import * as tf from '@tensorflow/tfjs';
import { DDPG_CONSTANTS } from './environment.js';

// ─────────────────────────────────────────────
// HYPERPARAMETERS
// ─────────────────────────────────────────────
const HP = {
  STATE_DIM:      10,
  ACTION_DIM:     2,
  BUFFER_SIZE:    5000,
  BATCH_SIZE:     64,
  ACTOR_LR:       1e-4,
  CRITIC_LR:      1e-3,
  GAMMA:          0.95,    // discount factor
  TAU:            0.005,   // Polyak soft-update coefficient
  OU_THETA:       0.15,    // mean-reversion rate
  OU_SIGMA_INIT:  0.3,     // initial noise std
  OU_SIGMA_MIN:   0.05,    // minimum noise std
  OU_SIGMA_DECAY: 0.9995,  // per-step decay
  MIN_BUFFER:     200,     // min samples trước khi train
};

// ─────────────────────────────────────────────
// EXPERIENCE REPLAY BUFFER
// ─────────────────────────────────────────────
class ReplayBuffer {
  constructor(maxSize = HP.BUFFER_SIZE) {
    this.maxSize = maxSize;
    this.buffer  = [];
    this.ptr     = 0; // vòng tròn write pointer
  }

  /**
   * Lưu một transition
   * @param {number[]} state
   * @param {number[]} action
   * @param {number}   reward
   * @param {number[]} nextState
   */
  push(state, action, reward, nextState) {
    const entry = { state, action, reward, nextState };
    if (this.buffer.length < this.maxSize) {
      this.buffer.push(entry);
    } else {
      this.buffer[this.ptr] = entry;
    }
    this.ptr = (this.ptr + 1) % this.maxSize;
  }

  /**
   * Sample ngẫu nhiên một batch
   * @param {number} batchSize
   * @returns {Array<{state, action, reward, nextState}>}
   */
  sample(batchSize) {
    const n = this.buffer.length;
    return Array.from({ length: batchSize }, () =>
      this.buffer[Math.floor(Math.random() * n)]
    );
  }

  get size()  { return this.buffer.length; }
  get ready() { return this.buffer.length >= HP.MIN_BUFFER; }
}

// ─────────────────────────────────────────────
// ORNSTEIN-UHLENBECK NOISE
// ─────────────────────────────────────────────
class OUNoise {
  constructor() {
    this.theta    = HP.OU_THETA;
    this.sigma    = HP.OU_SIGMA_INIT;
    this.sigmaMin = HP.OU_SIGMA_MIN;
    this.decay    = HP.OU_SIGMA_DECAY;
    this.mu       = [0, 0];
    this.x        = [0, 0];
    this.dt       = 0.1;
  }

  /** Sinh noise vector trong tanh-normalized space */
  sample() {
    this.x = this.x.map((xi, i) => {
      const dx = this.theta * (this.mu[i] - xi) * this.dt
               + this.sigma * Math.sqrt(this.dt) * (Math.random() * 2 - 1);
      return xi + dx;
    });
    return [...this.x];
  }

  /** Giảm σ theo thời gian */
  decay_() {
    this.sigma = Math.max(this.sigmaMin, this.sigma * this.decay);
  }

  /** Reset trạng thái (đầu episode mới) */
  reset() {
    this.x     = [0, 0];
    this.sigma = HP.OU_SIGMA_INIT;
  }

  get sigmaValue() { return this.sigma; }
}

// ─────────────────────────────────────────────
// DDPG AGENT
// ─────────────────────────────────────────────
export class DDPGAgent {
  constructor() {
    this.stateDim  = HP.STATE_DIM;
    this.actionDim = HP.ACTION_DIM;

    this.actor        = null;
    this.actorTarget  = null;
    this.critic       = null;
    this.criticTarget = null;

    this.actorOpt  = tf.train.adam(HP.ACTOR_LR);
    this.criticOpt = tf.train.adam(HP.CRITIC_LR);

    this.buffer = new ReplayBuffer();
    this.noise  = new OUNoise();

    this.stepCount  = 0;
    this.trainCount = 0;

    this.actorLossHistory  = [];
    this.criticLossHistory = [];
  }

  // ─────────────────────────────────────────
  // NETWORK BUILDERS
  // ─────────────────────────────────────────

  /** Actor: state(10) → Dense×3 → tanh(2) */
  _buildActor() {
    const inp = tf.input({ shape: [this.stateDim] });
    let x = tf.layers.dense({ units: 256, activation: 'relu', kernelInitializer: 'glorotUniform' }).apply(inp);
    x     = tf.layers.dense({ units: 256, activation: 'relu', kernelInitializer: 'glorotUniform' }).apply(x);
    x     = tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'glorotUniform' }).apply(x);
    const out = tf.layers.dense({ units: this.actionDim, activation: 'tanh',
      kernelInitializer: 'glorotUniform' }).apply(x);
    return tf.model({ inputs: inp, outputs: out });
  }

  /** Critic: [state(10), action(2)] → Dense×3 → Q(1) */
  _buildCritic() {
    const sInp = tf.input({ shape: [this.stateDim] });
    const aInp = tf.input({ shape: [this.actionDim] });
    const cat  = tf.layers.concatenate().apply([sInp, aInp]);
    let x = tf.layers.dense({ units: 256, activation: 'relu', kernelInitializer: 'glorotUniform' }).apply(cat);
    x     = tf.layers.dense({ units: 256, activation: 'relu', kernelInitializer: 'glorotUniform' }).apply(x);
    x     = tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'glorotUniform' }).apply(x);
    const out = tf.layers.dense({ units: 1, activation: 'linear',
      kernelInitializer: 'glorotUniform' }).apply(x);
    return tf.model({ inputs: [sInp, aInp], outputs: out });
  }

  /** Hard copy weights src → dst */
  _copyWeights(src, dst) {
    dst.setWeights(src.getWeights());
  }

  /**
   * Polyak soft update: dst ← τ*src + (1-τ)*dst
   * Handles tensor disposal properly.
   */
  _softUpdate(src, dst) {
    const srcW = src.getWeights();
    const dstW = dst.getWeights();
    const newW = srcW.map((w, i) => {
      return tf.tidy(() => tf.add(tf.mul(w, HP.TAU), tf.mul(dstW[i], 1 - HP.TAU)));
    });
    dst.setWeights(newW);
    // Clean up only the newly created tensors
    tf.dispose(newW);
  }

  // ─────────────────────────────────────────
  // PUBLIC API: BUILD
  // ─────────────────────────────────────────

  /** Khởi tạo 4 networks — gọi một lần trước khi bắt đầu */
  async build() {
    this.actor        = this._buildActor();
    this.actorTarget  = this._buildActor();
    this.critic       = this._buildCritic();
    this.criticTarget = this._buildCritic();

    this._copyWeights(this.actor,  this.actorTarget);
    this._copyWeights(this.critic, this.criticTarget);

    console.log('[DDPG] Networks built ✓');
    console.log(`  Actor:  ${this.actor.countParams()} params`);
    console.log(`  Critic: ${this.critic.countParams()} params`);
  }

  // ─────────────────────────────────────────
  // PUBLIC API: ACTION SELECTION
  // ─────────────────────────────────────────

  /**
   * Chọn action từ Actor + OU noise
   * @param {Float32Array|number[]} stateArr — state vector 10D
   * @returns {{ rawAction: number[], alpha: number, rho: number }}
   */
  selectAction(stateArr) {
    const raw = tf.tidy(() => {
      const s = tf.tensor2d([Array.from(stateArr)], [1, this.stateDim]);
      return Array.from(this.actor.predict(s).dataSync());
    });

    const noise = this.noise.sample();
    const noisyAction = raw.map((v, i) =>
      Math.max(-1, Math.min(1, v + noise[i]))
    );

    return {
      rawAction: noisyAction,
      ...this.scaleAction(noisyAction),
    };
  }

  /**
   * Scale tanh-output [-1,1]² → {alpha, rho} thực tế
   * @param {number[]} raw
   * @returns {{ alpha: number, rho: number }}
   */
  scaleAction(raw) {
    const { ALPHA_MIN, ALPHA_MAX, RHO_MIN, RHO_MAX } = DDPG_CONSTANTS;
    return {
      alpha: Math.max(ALPHA_MIN, Math.min(ALPHA_MAX,
        ALPHA_MIN + (raw[0] + 1) / 2 * (ALPHA_MAX - ALPHA_MIN))),
      rho: Math.max(RHO_MIN, Math.min(RHO_MAX,
        RHO_MIN + (raw[1] + 1) / 2 * (RHO_MAX - RHO_MIN))),
    };
  }

  // ─────────────────────────────────────────
  // PUBLIC API: MEMORY
  // ─────────────────────────────────────────

  /**
   * Lưu transition vào replay buffer
   * @param {Float32Array} state
   * @param {number[]}     rawAction  ∈ [-1,1]²
   * @param {number}       reward
   * @param {Float32Array} nextState
   */
  remember(state, rawAction, reward, nextState) {
    this.buffer.push(
      Array.from(state),
      rawAction,
      reward,
      Array.from(nextState)
    );
  }

  // ─────────────────────────────────────────
  // PUBLIC API: TRAINING
  // ─────────────────────────────────────────

  /**
   * Sample một batch và update Actor + Critic theo DDPG rule.
   * @returns {{ actorLoss: number, criticLoss: number } | null}
   */
  async train() {
    if (!this.buffer.ready) return null;

    this.trainCount++;
    const batch      = this.buffer.sample(HP.BATCH_SIZE);
    const states     = batch.map(b => b.state);
    const actions    = batch.map(b => b.action);
    const rewards    = batch.map(b => b.reward);
    const nextStates = batch.map(b => b.nextState);

    // ── Critic update ─────────────────────────────────────────────────
    // Loss = MSE( r + γ*Q_target(s', μ_target(s')),  Q(s, a) )
    let criticLossVal = 0;
    {
      const { value: loss, grads } = this.criticOpt.computeGradients(() =>
        tf.tidy(() => {
          const sT    = tf.tensor2d(states,     [HP.BATCH_SIZE, this.stateDim]);
          const aT    = tf.tensor2d(actions,    [HP.BATCH_SIZE, this.actionDim]);
          const rT    = tf.tensor1d(rewards);
          const nsT   = tf.tensor2d(nextStates, [HP.BATCH_SIZE, this.stateDim]);

          const nextA  = this.actorTarget.predict(nsT);
          const nextQ  = this.criticTarget.predict([nsT, nextA]).squeeze();
          const target = tf.add(rT, tf.mul(HP.GAMMA, nextQ));
          const currQ  = this.critic.predict([sT, aT]).squeeze();
          return tf.losses.meanSquaredError(target, currQ);
        })
      );
      this.criticOpt.applyGradients(grads);
      criticLossVal = (await loss.data())[0];
      loss.dispose();
    }

    // ── Actor update ──────────────────────────────────────────────────
    // Loss = -E[ Q(s, μ(s)) ]  (maximize Q)
    let actorLossVal = 0;
    {
      const { value: loss, grads } = this.actorOpt.computeGradients(() =>
        tf.tidy(() => {
          const sT   = tf.tensor2d(states, [HP.BATCH_SIZE, this.stateDim]);
          const aP   = this.actor.predict(sT);
          const qVal = this.critic.predict([sT, aP]).squeeze();
          return tf.neg(tf.mean(qVal));
        })
      );
      this.actorOpt.applyGradients(grads);
      actorLossVal = (await loss.data())[0];
      loss.dispose();
    }

    // ── Soft update target networks ───────────────────────────────────
    this._softUpdate(this.actor,  this.actorTarget);
    this._softUpdate(this.critic, this.criticTarget);

    // ── Decay noise ───────────────────────────────────────────────────
    this.noise.decay_();
    this.stepCount++;

    // ── Lịch sử ──────────────────────────────────────────────────────
    this.actorLossHistory.push(actorLossVal);
    this.criticLossHistory.push(criticLossVal);
    if (this.actorLossHistory.length > 500) {
      this.actorLossHistory.shift();
      this.criticLossHistory.shift();
    }

    return { actorLoss: actorLossVal, criticLoss: criticLossVal };
  }

  // ─────────────────────────────────────────
  // PUBLIC API: EXPORT / IMPORT
  // ─────────────────────────────────────────

  /** Export model weights (JSON) */
  async exportWeights() {
    if (!this.actor) return;
    try {
      await this.actor.save('downloads://ddpg-actor-model');
      console.log('[DDPG] Đã export Actor model.');
    } catch (err) {
      console.error('[DDPG] Lỗi export:', err);
    }
  }

  /** Import model weights (JSON + BIN) từ file inputs */
  async importWeights(jsonFile, binFile) {
    try {
      const loadedModel = await tf.loadLayersModel(
        tf.io.browserFiles([jsonFile, binFile])
      );
      // Thay thế Actor
      if (this.actor) this.actor.dispose();
      this.actor = loadedModel;
      
      // Đồng bộ sang Actor Target
      this._copyWeights(this.actor, this.actorTarget);
      console.log('[DDPG] Đã import Actor model thành công.');
      return true;
    } catch (err) {
      console.error('[DDPG] Lỗi import:', err);
      return false;
    }
  }

  // ─────────────────────────────────────────
  // PUBLIC API: LIFECYCLE
  // ─────────────────────────────────────────

  /** Reset OU noise (khi bắt đầu episode mới hoặc switch mode) */
  resetNoise() { this.noise.reset(); }

  /** Giải phóng tất cả TF tensors khi không còn dùng */
  dispose() {
    [this.actor, this.actorTarget, this.critic, this.criticTarget]
      .forEach(m => { if (m) { m.dispose(); } });
    this.actor = this.actorTarget = this.critic = this.criticTarget = null;
    console.log('[DDPG] Networks disposed');
  }

  // ─────────────────────────────────────────
  // GETTERS
  // ─────────────────────────────────────────
  get bufferSize()    { return this.buffer.size; }
  get bufferReady()   { return this.buffer.ready; }
  get noiseLevel()    { return this.noise.sigmaValue; }
  get lastActorLoss() { return this.actorLossHistory.at(-1) ?? null; }
  get lastCriticLoss(){ return this.criticLossHistory.at(-1) ?? null; }
}
