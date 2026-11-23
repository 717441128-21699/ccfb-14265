// 水下折射标定: 棋盘格多视图 + Levenberg-Marquardt 光束法平差
//
// 待估参数:
//   全局 8 个: fx, fy, cx, cy, k1, k2, dAir, nWater
//   每视图 6 个: 旋转向量(3) + 平移(3) —— 棋盘格(世界系 z=0)到相机系的刚体变换
// 残差: 棋盘格三维角点经 刚体变换 -> 折射正向投影 与检出角点的像素差。
import {
  V3, M3, rodrigues, rotationToRvec, mat3MulVec, solveLinear, jacobiEigen, norm3, normalize3, cross3,
} from './linalg';
import { Housing, Intrinsics, projectPoint } from './refraction';

export interface BoardSpec {
  cols: number;    // 内角点列数
  rows: number;    // 内角点行数
  square: number;  // 格子边长 mm
}

export interface ViewObs {
  /** 检出的角点像素, 长度 = cols*rows, 行优先(与 OpenCV 输出一致) */
  pixels: [number, number][];
}

export interface Pose { rvec: V3; t: V3; }

export interface CalibResult {
  K: Intrinsics;
  dAir: number;
  nWater: number;
  poses: Pose[];
  rms: number;        // 重投影 RMS (像素)
  maxErr: number;     // 最大单点误差 (像素)
  perViewRms: number[];
  iterations: number;
  converged: boolean;
}

/** 棋盘格内角点三维坐标(板坐标系, z=0)。板的方格数为 (cols+1)x(rows+1)。 */
export function boardPoints(board: BoardSpec): V3[] {
  const pts: V3[] = [];
  for (let j = 0; j < board.rows; j++) {
    for (let i = 0; i < board.cols; i++) {
      pts.push([(i + 1) * board.square, (j + 1) * board.square, 0]);
    }
  }
  return pts;
}

function poseTransform(pose: Pose, p: V3): V3 {
  const R = rodrigues(pose.rvec);
  const rp = mat3MulVec(R, p);
  return [rp[0] + pose.t[0], rp[1] + pose.t[1], rp[2] + pose.t[2]];
}

/** 带位姿的正向投影 */
export function projectWithPose(h: Housing, K: Intrinsics, pose: Pose, pBoard: V3): [number, number] | null {
  return projectPoint(h, K, poseTransform(pose, pBoard));
}

// ---------- 单应矩阵初始化位姿 ----------

/** 归一化 DLT 求平面->图像的单应矩阵 */
function homography(board2d: [number, number][], img2d: [number, number][]): number[] {
  const n = board2d.length;
  const norm = (pts: [number, number][]) => {
    let mx = 0, my = 0;
    for (const p of pts) { mx += p[0]; my += p[1]; }
    mx /= n; my /= n;
    let md = 0;
    for (const p of pts) md += Math.hypot(p[0] - mx, p[1] - my);
    md /= n;
    const s = Math.SQRT2 / (md || 1);
    return { mx, my, s };
  };
  const nb = norm(board2d), ni = norm(img2d);
  const At: number[][] = Array.from({ length: 9 }, () => new Array<number>(9).fill(0));
  for (let k = 0; k < n; k++) {
    const X = (board2d[k][0] - nb.mx) * nb.s;
    const Y = (board2d[k][1] - nb.my) * nb.s;
    const x = (img2d[k][0] - ni.mx) * ni.s;
    const y = (img2d[k][1] - ni.my) * ni.s;
    const rows = [
      [X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x],
      [0, 0, 0, X, Y, 1, -y * X, -y * Y, -y],
    ];
    for (const row of rows) {
      for (let i = 0; i < 9; i++) for (let j = i; j < 9; j++) At[i][j] += row[i] * row[j];
    }
  }
  for (let i = 0; i < 9; i++) for (let j = 0; j < i; j++) At[i][j] = At[j][i];
  const { vectors } = jacobiEigen(At);
  const h = vectors[0]; // 最小特征值对应特征向量
  // 反归一化: H = T_img^-1 · Hn · T_board
  const mul3 = (A: number[], B: number[]): number[] => {
    const R = new Array<number>(9).fill(0);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      for (let k = 0; k < 3; k++) R[3 * r + c] += A[3 * r + k] * B[3 * k + c];
    }
    return R;
  };
  const Tb = [nb.s, 0, -nb.s * nb.mx, 0, nb.s, -nb.s * nb.my, 0, 0, 1];
  const TiInv = [1 / ni.s, 0, ni.mx, 0, 1 / ni.s, ni.my, 0, 0, 1];
  return mul3(TiInv, mul3(h, Tb));
}

/** 由单应矩阵分解初始位姿(板平面 z=0, 内参 K 已知) */
export function poseFromHomography(board: BoardSpec, pixels: [number, number][], K: Intrinsics): Pose {
  const pts = boardPoints(board);
  const b2d: [number, number][] = pts.map((p) => [p[0], p[1]]);
  const i2d: [number, number][] = pixels.map((p) => [(p[0] - K.cx) / K.fx, (p[1] - K.cy) / K.fy]);
  const H = homography(b2d, i2d);
  let h1: V3 = [H[0], H[3], H[6]];
  let h2: V3 = [H[1], H[4], H[7]];
  let h3: V3 = [H[2], H[5], H[8]];
  const lam = 2 / (norm3(h1) + norm3(h2));
  h1 = h1.map((v) => v * lam) as V3;
  h2 = h2.map((v) => v * lam) as V3;
  h3 = h3.map((v) => v * lam) as V3;
  let r1 = normalize3(h1);
  let r2 = normalize3([h2[0] - r1[0] * (r1[0] * h2[0] + r1[1] * h2[1] + r1[2] * h2[2]),
    h2[1] - r1[1] * (r1[0] * h2[0] + r1[1] * h2[1] + r1[2] * h2[2]),
    h2[2] - r1[2] * (r1[0] * h2[0] + r1[1] * h2[1] + r1[2] * h2[2])]);
  let r3 = cross3(r1, r2);
  let t = h3;
  if (t[2] < 0) { // 板必须在相机前方
    r1 = r1.map((v) => -v) as V3;
    r2 = r2.map((v) => -v) as V3;
    t = t.map((v) => -v) as V3;
    r3 = cross3(r1, r2);
  }
  const R: M3 = [r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]];
  return { rvec: rotationToRvec(R), t };
}

// ---------- LM 平差 ----------

const N_GLOBAL = 8;

function unpackGlobals(p: number[]): { K: Intrinsics; housing: Housing } {
  return {
    K: { fx: p[0], fy: p[1], cx: p[2], cy: p[3], k1: p[4], k2: p[5] },
    housing: { nGlass: 0, tGlass: 0, dAir: p[6], nWater: p[7] },
  };
}

export interface CalibOptions {
  nGlass: number;
  tGlass: number;
  imageWidth: number;
  imageHeight: number;
  initNWater?: number;   // 由水温/盐度计算, 默认 1.333
  initDAir?: number;     // 默认 25 mm
  fitNWater?: boolean;   // 是否拟合水折射率(默认 false: 固定为 initNWater, 避免与焦距的简并)
  nWaterSigma?: number;  // 拟合时的先验约束 σ, 默认 0.003
  dAirSigma?: number;    // dAir 先验约束 σ(mm), 0 = 不加约束, 默认 0
  maxIter?: number;
}

export function calibrate(board: BoardSpec, views: ViewObs[], opts: CalibOptions): CalibResult {
  const pts = boardPoints(board);
  const V = views.length;
  const N = pts.length;

  // 初始参数
  const K0: Intrinsics = {
    fx: 1.2 * Math.max(opts.imageWidth, opts.imageHeight),
    fy: 1.2 * Math.max(opts.imageWidth, opts.imageHeight),
    cx: opts.imageWidth / 2,
    cy: opts.imageHeight / 2,
    k1: 0, k2: 0,
  };
  const params: number[] = [
    K0.fx, K0.fy, K0.cx, K0.cy, 0, 0,
    opts.initDAir ?? 25, opts.initNWater ?? 1.333,
  ];
  const poses0: Pose[] = views.map((v) => poseFromHomography(board, v.pixels, K0));
  for (const pose of poses0) params.push(...pose.rvec, ...pose.t);

  const nP = params.length;
  const nObs = 2 * N * V;

  // 先验约束(正则化): 打破水下标定中 焦距↔折射率↔界面距离 的近简并
  interface Prior { paramIdx: number; mean: number; sigma: number; }
  const priors: Prior[] = [];
  if (opts.fitNWater) priors.push({ paramIdx: 7, mean: params[7], sigma: opts.nWaterSigma ?? 0.003 });
  if (opts.dAirSigma && opts.dAirSigma > 0) priors.push({ paramIdx: 6, mean: params[6], sigma: opts.dAirSigma });
  const m = nObs + priors.length;

  // 冻结的参数(列): 不拟合 nWater 时将其固定
  const frozen = new Set<number>();
  if (!opts.fitNWater) frozen.add(7);

  const housingBase = { nGlass: opts.nGlass, tGlass: opts.tGlass };

  // 计算单个视图的残差(写入 out 的 2*N*view 偏移处)
  const residualView = (p: number[], view: number, out: Float64Array) => {
    const { K, housing } = unpackGlobals(p);
    housing.nGlass = housingBase.nGlass;
    housing.tGlass = housingBase.tGlass;
    const off = N_GLOBAL + 6 * view;
    const pose: Pose = {
      rvec: [p[off], p[off + 1], p[off + 2]],
      t: [p[off + 3], p[off + 4], p[off + 5]],
    };
    const obs = views[view].pixels;
    const base = 2 * N * view;
    for (let j = 0; j < N; j++) {
      const proj = projectWithPose(housing, K, pose, pts[j]);
      if (proj) {
        out[base + 2 * j] = proj[0] - obs[j][0];
        out[base + 2 * j + 1] = proj[1] - obs[j][1];
      } else {
        out[base + 2 * j] = 1e3;
        out[base + 2 * j + 1] = 1e3;
      }
    }
  };
  const residualAll = (p: number[]): Float64Array => {
    const r = new Float64Array(m);
    for (let v = 0; v < V; v++) residualView(p, v, r);
    for (let k = 0; k < priors.length; k++) {
      r[nObs + k] = (p[priors[k].paramIdx] - priors[k].mean) / priors[k].sigma;
    }
    return r;
  };
  const cost = (r: Float64Array) => {
    let s = 0;
    for (let i = 0; i < r.length; i++) s += r[i] * r[i];
    return s;
  };

  let r = residualAll(params);
  let c = cost(r);
  let lambda = 1e-3;
  let converged = false;
  let iter = 0;
  const maxIter = opts.maxIter ?? 60;

  const J = new Float64Array(m * nP);

  for (; iter < maxIter; iter++) {
    // 数值雅可比(全局参数影响全部视图; 位姿参数只影响本视图; 冻结参数跳过)
    for (let j = 0; j < nP; j++) {
      if (frozen.has(j)) continue;
      const hStep = 1e-7 * Math.max(1, Math.abs(params[j]));
      const saved = params[j];
      params[j] = saved + hStep;
      if (j < N_GLOBAL) {
        const r2 = residualAll(params);
        for (let i = 0; i < m; i++) J[i * nP + j] = (r2[i] - r[i]) / hStep;
      } else {
        const view = Math.floor((j - N_GLOBAL) / 6);
        const r2 = new Float64Array(2 * N);
        const tmp = new Float64Array(m);
        residualView(params, view, tmp);
        for (let i = 0; i < 2 * N; i++) r2[i] = tmp[2 * N * view + i];
        for (let i = 0; i < 2 * N; i++) {
          J[(2 * N * view + i) * nP + j] = (r2[i] - r[2 * N * view + i]) / hStep;
        }
        // 其余视图对该参数的导数为 0
        for (let v = 0; v < V; v++) {
          if (v === view) continue;
          for (let i = 0; i < 2 * N; i++) J[(2 * N * v + i) * nP + j] = 0;
        }
      }
      params[j] = saved;
    }
    // 正规方程
    const A: number[][] = Array.from({ length: nP }, () => new Array<number>(nP).fill(0));
    const g = new Array<number>(nP).fill(0);
    for (let i = 0; i < m; i++) {
      const Ji = i * nP;
      for (let a = 0; a < nP; a++) {
        const ja = J[Ji + a];
        if (ja === 0) continue;
        g[a] += ja * r[i];
        for (let b = a; b < nP; b++) A[a][b] += ja * J[Ji + b];
      }
    }
    for (let a = 0; a < nP; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];
    // 冻结参数: 单位对角 + 零梯度 -> 步长为 0
    for (const j of frozen) {
      for (let k = 0; k < nP; k++) { A[j][k] = 0; A[k][j] = 0; }
      A[j][j] = 1;
      g[j] = 0;
    }

    let accepted = false;
    for (let inner = 0; inner < 40; inner++) {
      const Aλ = A.map((row, i) => row.map((v, j) => (i === j ? v + lambda * Math.max(A[i][i], 1e-12) : v)));
      const rhs = g.map((v) => -v);
      let delta: number[];
      try {
        delta = solveLinear(Aλ, rhs);
      } catch {
        lambda *= 10;
        continue;
      }
      const trial = params.map((v, i) => v + delta[i]);
      const rTrial = residualAll(trial);
      const cTrial = cost(rTrial);
      if (cTrial < c) {
        const rel = (c - cTrial) / Math.max(c, 1e-30);
        params.splice(0, nP, ...trial);
        r = rTrial;
        c = cTrial;
        lambda = Math.max(lambda / 10, 1e-12);
        accepted = true;
        if (rel < 1e-12) converged = true;
        break;
      }
      lambda *= 10;
      if (lambda > 1e12) break;
    }
    if (converged || !accepted) { converged = converged || accepted; break; }
  }

  // 汇总
  const { K, housing } = unpackGlobals(params);
  const poses: Pose[] = [];
  for (let v = 0; v < V; v++) {
    const off = N_GLOBAL + 6 * v;
    poses.push({
      rvec: [params[off], params[off + 1], params[off + 2]],
      t: [params[off + 3], params[off + 4], params[off + 5]],
    });
  }
  const perViewRms: number[] = [];
  let maxErr = 0;
  let sumObs = 0;
  for (let v = 0; v < V; v++) {
    let s = 0;
    for (let i = 0; i < 2 * N; i++) {
      const e = r[2 * N * v + i];
      s += e * e;
    }
    sumObs += s;
    for (let j = 0; j < N; j++) {
      const e = Math.hypot(r[2 * N * v + 2 * j], r[2 * N * v + 2 * j + 1]);
      if (e > maxErr) maxErr = e;
    }
    perViewRms.push(Math.sqrt(s / (2 * N)));
  }
  return {
    K,
    dAir: housing.dAir,
    nWater: housing.nWater,
    poses,
    rms: Math.sqrt(sumObs / nObs),
    maxErr,
    perViewRms,
    iterations: iter + 1,
    converged,
  };
}

/** 用已标定的参数, 对单张含棋盘格的照片解算位姿(用于"棋盘格定面"测量) */
export function solvePose(
  board: BoardSpec, pixels: [number, number][], K: Intrinsics, housing: Housing,
): Pose {
  const pts = boardPoints(board);
  const N = pts.length;
  const p0 = poseFromHomography(board, pixels, K);
  const params = [...p0.rvec, ...p0.t];
  const m = 2 * N;

  const residual = (p: number[]): Float64Array => {
    const pose: Pose = { rvec: [p[0], p[1], p[2]], t: [p[3], p[4], p[5]] };
    const out = new Float64Array(m);
    for (let j = 0; j < N; j++) {
      const proj = projectWithPose(housing, K, pose, pts[j]);
      out[2 * j] = proj ? proj[0] - pixels[j][0] : 1e3;
      out[2 * j + 1] = proj ? proj[1] - pixels[j][1] : 1e3;
    }
    return out;
  };

  let r = residual(params);
  let c = 0;
  for (let i = 0; i < m; i++) c += r[i] * r[i];
  let lambda = 1e-3;
  for (let iter = 0; iter < 60; iter++) {
    const J: number[][] = Array.from({ length: m }, () => new Array<number>(6).fill(0));
    for (let j = 0; j < 6; j++) {
      const hStep = 1e-7 * Math.max(1, Math.abs(params[j]));
      const saved = params[j];
      params[j] = saved + hStep;
      const r2 = residual(params);
      for (let i = 0; i < m; i++) J[i][j] = (r2[i] - r[i]) / hStep;
      params[j] = saved;
    }
    const A: number[][] = Array.from({ length: 6 }, () => new Array<number>(6).fill(0));
    const g = new Array<number>(6).fill(0);
    for (let i = 0; i < m; i++) {
      for (let a = 0; a < 6; a++) {
        g[a] += J[i][a] * r[i];
        for (let b = a; b < 6; b++) A[a][b] += J[i][a] * J[i][b];
      }
    }
    for (let a = 0; a < 6; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];
    let accepted = false;
    for (let inner = 0; inner < 40; inner++) {
      const Aλ = A.map((row, i) => row.map((v, j) => (i === j ? v + lambda * Math.max(A[i][i], 1e-12) : v)));
      let delta: number[];
      try {
        delta = solveLinear(Aλ, g.map((v) => -v));
      } catch {
        lambda *= 10;
        continue;
      }
      const trial = params.map((v, i) => v + delta[i]);
      const rT = residual(trial);
      let cT = 0;
      for (let i = 0; i < m; i++) cT += rT[i] * rT[i];
      if (cT < c) {
        params.splice(0, 6, ...trial);
        r = rT;
        const rel = (c - cT) / Math.max(c, 1e-30);
        c = cT;
        lambda = Math.max(lambda / 10, 1e-12);
        accepted = true;
        if (rel < 1e-12) return { rvec: [params[0], params[1], params[2]], t: [params[3], params[4], params[5]] };
        break;
      }
      lambda *= 10;
      if (lambda > 1e12) break;
    }
    if (!accepted) break;
  }
  return { rvec: [params[0], params[1], params[2]], t: [params[3], params[4], params[5]] };
}
