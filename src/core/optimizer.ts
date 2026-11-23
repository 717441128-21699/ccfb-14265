/**
 * 通用 Levenberg–Marquardt 最小二乘优化器。
 * 残差与雅可比由调用方提供；雅可比可选数值差分，也支持解析/自动差分。
 */

export interface LMResult {
  params: number[];
  cost: number;
  iterations: number;
  converged: boolean;
}

export interface LMOptions {
  maxIterations?: number;
  /** 参数收敛阈值（相对） */
  paramTol?: number;
  /** 成本下降阈值 */
  costTol?: number;
  /** 初始阻尼 λ */
  lambdaInit?: number;
  /** 数值差分步长 */
  finiteDiffEps?: number;
  /** 逐参数下界（null/undefined 表示无约束） */
  lower?: (number | null)[];
  /** 逐参数上界 */
  upper?: (number | null)[];
  /** 每轮回调（用于 UI 进度） */
  onIteration?: (info: { iteration: number; cost: number; lambda: number }) => void;
}

function clampParams(p: number[], lower?: (number | null)[], upper?: (number | null)[]): number[] {
  if (!lower && !upper) return p;
  return p.map((x, j) => {
    if (lower && lower[j] != null) x = Math.max(x, lower[j]!);
    if (upper && upper[j] != null) x = Math.min(x, upper[j]!);
    return x;
  });
}

/**
 * @param residuals 给定参数返回残差数组
 * @param params0 初始参数
 * @param jacobian 可选解析雅可比（行=残差，列=参数，扁平 row-major）
 */
export function levenbergMarquardt(
  residuals: (p: number[]) => number[],
  params0: number[],
  opts: LMOptions = {},
  jacobian?: (p: number[]) => number[][],
): LMResult {
  const {
    maxIterations = 50,
    paramTol = 1e-8,
    costTol = 1e-10,
    lambdaInit = 1e-3,
    finiteDiffEps = 1e-6,
    lower,
    upper,
    onIteration,
  } = opts;

  let params = params0.slice();
  let r = residuals(params);
  let cost = 0.5 * r.reduce((s, x) => s + x * x, 0);
  let lambda = lambdaInit;
  let converged = false;
  let iter = 0;

  const numericJac = (p: number[]): number[][] => {
    const r0 = residuals(p);
    const J: number[][] = r0.map(() => new Array<number>(p.length).fill(0));
    for (let j = 0; j < p.length; j++) {
      const saved = p[j];
      // 相对步长：大参数（平移、焦距）按量级取 1e-5，小参数（角度）用绝对下限 1e-6
      const h = Math.max(Math.abs(saved) * (finiteDiffEps * 10), 1e-6);
      p[j] = saved + h;
      const rj = residuals(p);
      p[j] = saved;
      for (let i = 0; i < r0.length; i++) J[i][j] = (rj[i] - r0[i]) / h;
    }
    return J;
  };

  for (iter = 0; iter < maxIterations; iter++) {
    const J = jacobian ? jacobian(params) : numericJac(params);
    const m = params.length;
    // J^T J, J^T r
    const JTJ = new Float64Array(m * m);
    const JTr = new Float64Array(m);
    for (let i = 0; i < r.length; i++) {
      for (let a = 0; a < m; a++) {
        JTr[a] += J[i][a] * r[i];
        for (let b = 0; b < m; b++) JTJ[a * m + b] += J[i][a] * J[i][b];
      }
    }

    let accepted = false;
    let delta: number[] = [];
    for (let inner = 0; inner < 20 && !accepted; inner++) {
      const A = new Float64Array(JTJ);
      for (let a = 0; a < m; a++) A[a * m + a] += lambda * Math.max(JTJ[a * m + a], 1e-12);
      const sol = choleskySolve(A, JTr, m);
      if (!sol) {
        lambda *= 10;
        continue;
      }
      delta = sol;
      // 残差 r=观测−模型，J=∂r/∂p；GN 法方程 (JᵀJ)δ=Jᵀr，更新 p←p−δ
      // （等价于 r̃=模型−观测 时的标准 p←p+(J̃ᵀJ̃)⁻¹(−J̃ᵀr̃)）
      const trialRaw = params.map((x, j) => x - delta[j]);
      const trial = clampParams(trialRaw, lower, upper);
      const rTrial = residuals(trial);
      const costTrial = 0.5 * rTrial.reduce((s, x) => s + x * x, 0);
      if (costTrial < cost) {
        const paramChange = Math.max(...delta.map((d, j) => Math.abs(d) / (Math.abs(params[j]) + 1e-12)));
        params = trial;
        r = rTrial;
        const costDrop = cost - costTrial;
        cost = costTrial;
        lambda = Math.max(lambda / 3, 1e-12);
        accepted = true;
        onIteration?.({ iteration: iter, cost, lambda });
        if (costDrop < costTol * (cost + 1) || paramChange < paramTol) converged = true;
      } else {
        lambda *= 10;
      }
    }
    if (!accepted || converged) break;
  }

  return { params, cost, iterations: iter + 1, converged };
}

/** 正定对称矩阵 Cholesky 求解 A x = b，A 为 row-major n×n */
function choleskySolve(A: Float64Array, b: Float64Array, n: number): number[] | null {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 1e-14) return null;
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  // 前代 L y = b
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  // 回代 L^T x = y
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return Array.from(x);
}
