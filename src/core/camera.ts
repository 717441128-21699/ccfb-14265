import { projectPointNormalized } from './optics';
import type {
  BrownDistortion,
  Intrinsics,
  Pose,
  RefractionParams,
  Vec2,
  Vec3,
} from './types';
import { mv, add } from './linalg';

/** Brown-Conrady 畸变：归一化针孔坐标 -> 畸变后归一化坐标 */
export function distortNormalized(
  u: number,
  v: number,
  d: BrownDistortion,
): Vec2 {
  const r2 = u * u + v * v;
  const r4 = r2 * r2;
  const r6 = r4 * r2;
  const radial = 1 + d.k1 * r2 + d.k2 * r4 + d.k3 * r6;
  const uv2 = 2 * u * v;
  const xd =
    u * radial + d.p1 * uv2 + d.p2 * (r2 + 2 * u * u);
  const yd =
    v * radial + d.p1 * (r2 + 2 * v * v) + d.p2 * uv2;
  return { x: xd, y: yd };
}

/** 畸变逆映射：迭代求畸变前坐标，用于校正查表 */
export function undistortNormalized(
  ud: number,
  vd: number,
  d: BrownDistortion,
  iterations = 12,
): Vec2 {
  let u = ud;
  let v = vd;
  for (let i = 0; i < iterations; i++) {
    const p = distortNormalized(u, v, d);
    u += ud - p.x;
    v += vd - p.y;
  }
  return { x: u, y: v };
}

export function projectToPixel(
  u: number,
  v: number,
  K: Intrinsics,
  dist: BrownDistortion,
): Vec2 {
  const dp = distortNormalized(u, v, dist);
  return {
    x: K.fx * dp.x + K.cx,
    y: K.fy * dp.y + K.cy,
  };
}

/** 完整投影链：世界点 -> 像素（折射 + 镜头畸变 + 内参） */
export function projectWorldPoint(
  world: Vec3,
  pose: Pose,
  refr: RefractionParams,
  K: Intrinsics,
  dist: BrownDistortion,
): Vec2 | null {
  const Xc = add(mv(pose.R, world), pose.t);
  if (Xc.z <= 0) return null;
  const nrm = projectPointNormalized(Xc, refr);
  if (!nrm) return null;
  return projectToPixel(nrm.u, nrm.v, K, dist);
}

/** 生成棋盘格内角点的世界坐标（z=0 平面，单位 mm） */
export function checkerboardPoints(
  cols: number,
  rows: number,
  squareMm: number,
): Vec3[] {
  const pts: Vec3[] = [];
  const ox = -((cols - 1) * squareMm) / 2;
  const oy = -((rows - 1) * squareMm) / 2;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      pts.push({ x: ox + i * squareMm, y: oy + j * squareMm, z: 0 });
    }
  }
  return pts;
}

/**
 * 由单应矩阵分解位姿（平面 z=0 目标）。
 * H 把齐次板坐标 (x,y,1) 映射到像素，K^-1 H = [r1 r2 t] * lambda。
 */
export function poseFromHomography(
  H: number[],
  K: Intrinsics,
): Pose {
  const invK = invertIntrinsics(K);
  const M = matMul(invK, H);
  // 列向量
  const c1 = [M[0], M[3], M[6]];
  const c2 = [M[1], M[4], M[7]];
  const c3 = [M[2], M[5], M[8]];
  const lambda =
    1 / ((Math.hypot(...c1) + Math.hypot(...c2)) / 2);
  let r1 = c1.map((v) => v * lambda);
  let r2 = c2.map((v) => v * lambda);
  const t = c3.map((v) => v * lambda);
  // 正交化，保证 R 是旋转矩阵
  const n1 = Math.hypot(...r1) || 1;
  r1 = r1.map((v) => v / n1);
  const dot12 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
  r2 = r2.map((v, i) => v - dot12 * r1[i]);
  const n2 = Math.hypot(...r2) || 1;
  r2 = r2.map((v) => v / n2);
  const r3 = cross3(r1, r2);
  const R = [
    r1[0], r2[0], r3[0],
    r1[1], r2[1], r3[1],
    r1[2], r2[2], r3[2],
  ];
  return { R, t: { x: t[0], y: t[1], z: Math.max(t[2], 1) } };
}

export function invertIntrinsics(K: Intrinsics): number[] {
  const invFx = 1 / K.fx;
  const invFy = 1 / K.fy;
  return [
    invFx, 0, -K.cx * invFx,
    0, invFy, -K.cy * invFy,
    0, 0, 1,
  ];
}

function matMul(A: number[], B: number[]): number[] {
  const C = new Array<number>(9).fill(0);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
      }
    }
  }
  return C;
}

function cross3(a: number[], b: number[]): number[] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/** 由 4 对以上点对应求 3x3 单应（DLT + Hartley 归一化） */
export function findHomography(src: Vec2[], dst: Vec2[]): number[] | null {
  if (src.length < 4) return null;
  // Hartley 归一化：各点集平移到质心、平均距原点距离归一到 √2，
  // 改善 DLT 矩阵的条件数
  const Tsrc = normalizeTransform(src);
  const Tdst = normalizeTransform(dst);
  const sn = src.map((p) => applyAffine(Tsrc, p));
  const dn = dst.map((p) => applyAffine(Tdst, p));
  const A: number[][] = [];
  for (let k = 0; k < sn.length; k++) {
    const { x, y } = sn[k];
    const { x: u, y: v } = dn[k];
    // x'×(H x)=0 的两行：
    // 第 1 行（对应 v=y'）：[0 0 0 -x -y -1 v x v y v]
    // 第 2 行（对应 u=x'）：[x y 1 0 0 0 -u x -u y -u]
    A.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, -u]);
  }
  const h = smallestEigenVectorAtA(A);
  if (!h) return null;
  // Hn 作用于归一化坐标，反归一化：H = Tdst⁻¹ Hn Tsrc
  const Hn = [
    h[0], h[1], h[2],
    h[3], h[4], h[5],
    h[6], h[7], h[8],
  ];
  const H = matMul(matMul(invertAffine(Tdst), Hn), Tsrc);
  const denom = Math.abs(H[8]) > 1e-12 ? H[8] : 1;
  return H.map((v) => v / denom);
}

/** 点集归一化仿射变换（3x3），使质心在原点、平均距离 √2 */
function normalizeTransform(pts: Vec2[]): number[] {
  let cx = 0;
  let cy = 0;
  for (const p of pts) {
    cx += p.x;
    cy += p.y;
  }
  cx /= pts.length;
  cy /= pts.length;
  let meanDist = 0;
  for (const p of pts) meanDist += Math.hypot(p.x - cx, p.y - cy);
  meanDist /= pts.length;
  const s = meanDist > 1e-12 ? Math.SQRT2 / meanDist : 1;
  // [s 0 -s cx; 0 s -s cy; 0 0 1]
  return [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
}

function applyAffine(T: number[], p: Vec2): Vec2 {
  return {
    x: T[0] * p.x + T[1] * p.y + T[2],
    y: T[3] * p.x + T[4] * p.y + T[5],
  };
}

function invertAffine(T: number[]): number[] {
  const s = T[0];
  const tx = T[2];
  const ty = T[5];
  const inv = 1 / s;
  return [inv, 0, -tx * inv, 0, inv, -ty * inv, 0, 0, 1];
}

/**
 * 求 AᵀA 的最小特征值对应特征向量（DLT 单应的零空间）。
 * 逆幂迭代：对 (M + σI) 反复高斯消元求解，特征值被平移后
 * 最小者主导收敛，σ 取小正数保证正定可逆。对 9 维矩阵
 * 通常 20 次迭代内稳定到机器精度。
 */
export function smallestEigenVectorAtA(A: number[][]): number[] | null {
  const n = 9;
  const M = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (const row of A) {
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const v = row[i] * row[j];
        M[i][j] += v;
        if (i !== j) M[j][i] += v;
      }
    }
  }
  // 正则化对角线，确定位移 σ
  let trace = 0;
  for (let i = 0; i < n; i++) trace += M[i][i];
  const sigma = (trace / n) * 1e-10 + 1e-12;
  let x = new Array<number>(n).fill(1 / 3);
  const solve = (B: number[][], b: number[]): number[] | null => {
    const N = B.map((r, i) => [...r, b[i]]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let i = col + 1; i < n; i++) {
        if (Math.abs(N[i][col]) > Math.abs(N[piv][col])) piv = i;
      }
      if (Math.abs(N[piv][col]) < 1e-18) return null;
      [N[col], N[piv]] = [N[piv], N[col]];
      for (let i = 0; i < n; i++) {
        if (i === col) continue;
        const f = N[i][col] / N[col][col];
        for (let j = col; j <= n; j++) N[i][j] -= f * N[col][j];
      }
    }
    return N.map((row, i) => row[n] / row[i]);
  };
  for (let iter = 0; iter < 40; iter++) {
    const B = M.map((r, i) => r.map((v, j) => (i === j ? v + sigma : v)));
    const y = solve(B, x);
    if (!y) return null;
    const nn = Math.hypot(...y);
    const nx = y.map((v) => v / nn);
    if (Math.hypot(...nx.map((v, k) => v - x[k])) < 1e-12) {
      x = nx;
      break;
    }
    x = nx;
  }
  const nn = Math.hypot(...x);
  return nn > 0 ? x.map((v) => v / nn) : null;
}
