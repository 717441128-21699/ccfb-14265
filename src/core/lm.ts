/**
 * 通用 Levenberg–Marquardt 非线性最小二乘。
 * 残差函数返回所有观测残差（每个观测 2 个残差时长度为 2N）。
 * 雅可比用前向差分估计；参数典型尺度由调用方给出，差分步长按尺度取。
 */

export interface LmOptions {
  maxIterations?: number;
  tolerance?: number; // 参数变化与代价下降的收敛阈值
  lambda0?: number;
  lambdaUp?: number;
  lambdaDown?: number;
  /** 每个参数的典型尺度，用于差分步长与阻尼归一化 */
  scales?: number[];
  verbose?: boolean;
}

export interface LmResult {
  params: number[];
  cost: number;
  rms: number;
  iterations: number;
  converged: boolean;
}

export function levenbergMarquardt(
  initial: number[],
  residualsFn: (p: number[]) => number[],
  opts: LmOptions = {},
): LmResult {
  const maxIter = opts.maxIterations ?? 50;
  const tol = opts.tolerance ?? 1e-10;
  const scales =
    opts.scales ?? initial.map((v) => Math.max(Math.abs(v), 1e-3));
  let params = initial.slice();
  let res = residualsFn(params);
  let cost = 0.5 * res.reduce((s, r) => s + r * r, 0);
  let lambda = opts.lambda0 ?? 1e-3;
  const lambdaUp = opts.lambdaUp ?? 10;
  const lambdaDown = opts.lambdaDown ?? 0.3;
  const n = params.length;
  const m = res.length;

  let converged = false;
  let iter = 0;
  for (; iter < maxIter; iter++) {
    const J = new Float64Array(m * n);
    for (let j = 0; j < n; j++) {
      const h = Math.max(1e-8, 1e-6 * Math.abs(scales[j]));
      const p2 = params.slice();
      p2[j] += h;
      const r2 = residualsFn(p2);
      for (let i = 0; i < m; i++) {
        J[i * n + j] = (r2[i] - res[i]) / h;
      }
    }
    // J^T J 与 J^T r
    const JtJ = new Float64Array(n * n);
    const Jtr = new Float64Array(n);
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        let s = 0;
        for (let i = 0; i < m; i++) s += J[i * n + a] * J[i * n + b];
        JtJ[a * n + b] = s;
      }
      let s = 0;
      for (let i = 0; i < m; i++) s += J[i * n + a] * res[i];
      Jtr[a] = s;
    }

    let accepted = false;
    for (let attempt = 0; attempt < 12; attempt++) {
      const A = new Float64Array(JtJ);
      for (let a = 0; a < n; a++) {
        A[a * n + a] += lambda * (JtJ[a * n + a] + 1e-12);
      }
      const delta = solveSymmetric(A, Jtr, n);
      if (!delta) break;
      const candidate = params.map((v, j) => v - delta[j]);
      const rc = residualsFn(candidate);
      const costC = 0.5 * rc.reduce((s, r) => s + r * r, 0);
      if (Number.isFinite(costC) && costC < cost) {
        const denom = Math.max(
          1e-12,
          0.5 *
            delta.reduce(
              (s, d, j) => s + d * (lambda * (JtJ[j * n + j] + 1e-12) * d + Jtr[j]),
              0,
            ),
        );
        const rho = (cost - costC) / denom;
        params = candidate;
        res = rc;
        const stepNorm = Math.hypot(
          ...delta.map((d, j) => d / scales[j]),
        );
        const prevCost = cost;
        cost = costC;
        lambda = Math.max(1e-12, lambda * Math.max(lambdaDown, 1 - Math.pow(2 * rho - 1, 3)));
        lambda = Math.min(lambda, 1e12);
        accepted = true;
        if (stepNorm < tol || (prevCost - cost) / prevCost < tol) {
          converged = true;
        }
        break;
      }
      lambda *= lambdaUp;
    }
    if (!accepted || converged) break;
  }
  return {
    params,
    cost,
    rms: Math.sqrt((2 * cost) / Math.max(1, m)),
    iterations: iter + 1,
    converged,
  };
}

/** Cholesky 解对称正定方程组；奇异时返回 null */
function solveSymmetric(
  A: Float64Array,
  b: Float64Array,
  n: number,
): number[] | null {
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
  const y = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}
