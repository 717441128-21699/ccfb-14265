/**
 * 最小化三维向量/矩阵工具（列主序 3x3 矩阵用 Float64Array(9)）。
 * 全部使用 number（float64），保证标定数值精度。
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export const v3 = (x = 0, y = 0, z = 0): Vec3 => ({ x, y, z });
export const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const scale = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
export const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
export const norm = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);
export const normalize = (a: Vec3): Vec3 => {
  const n = norm(a);
  return n > 0 ? scale(a, 1 / n) : v3(0, 0, 1);
};
export const neg = (a: Vec3): Vec3 => ({ x: -a.x, y: -a.y, z: -a.z });

/** 列主序矩阵乘向量 */
export function mv(m: Float64Array, p: Vec3): Vec3 {
  return {
    x: m[0] * p.x + m[3] * p.y + m[6] * p.z,
    y: m[1] * p.x + m[4] * p.y + m[7] * p.z,
    z: m[2] * p.x + m[5] * p.y + m[8] * p.z,
  };
}

export function mm(a: Float64Array, b: Float64Array): Float64Array {
  const r = new Float64Array(9);
  for (let c = 0; c < 3; c++) {
    for (let row = 0; row < 3; row++) {
      r[c * 3 + row] =
        a[row] * b[c * 3] + a[3 + row] * b[c * 3 + 1] + a[6 + row] * b[c * 3 + 2];
    }
  }
  return r;
}

export function transpose(m: Float64Array): Float64Array {
  return Float64Array.from([
    m[0], m[3], m[6],
    m[1], m[4], m[7],
    m[2], m[5], m[8],
  ]);
}

/** Rodrigues 旋转向量 -> 3x3 矩阵（列主序） */
export function rodrigues(rvec: Vec3): Float64Array {
  const theta = norm(rvec);
  const I = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  if (theta < 1e-12) return I;
  const k = scale(rvec, 1 / theta);
  const K = Float64Array.from([
    0, k.z, -k.y,
    -k.z, 0, k.x,
    k.y, -k.x, 0,
  ]);
  const s = Math.sin(theta);
  const c = Math.cos(theta);
  const r = new Float64Array(9);
  const K2 = mm(K, K);
  for (let i = 0; i < 9; i++) r[i] = I[i] + s * K[i] + (1 - c) * K2[i];
  return r;
}

/** 3x3 矩阵 -> Rodrigues 旋转向量 */
export function logRodrigues(m: Float64Array): Vec3 {
  const tr = m[0] + m[4] + m[8];
  const cosT = Math.min(1, Math.max(-1, (tr - 1) / 2));
  const theta = Math.acos(cosT);
  if (theta < 1e-10) {
    // 小角度：rvec ≈ 反对称部分的提取
    return v3((m[7] - m[5]) / 2, (m[2] - m[6]) / 2, (m[3] - m[1]) / 2);
  }
  const s = 2 * Math.sin(theta);
  return v3(
    ((m[7] - m[5]) / s) * theta,
    ((m[2] - m[6]) / s) * theta,
    ((m[3] - m[1]) / s) * theta,
  );
}

/** 解 3x3 线性方程 A x = b（高斯消元，带部分主元） */
export function solve3(A: Float64Array, b: Vec3): Vec3 | null {
  const a = new Float64Array(9);
  a.set(A);
  const y = [b.x, b.y, b.z];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(a[r * 3 + col]) > Math.abs(a[piv * 3 + col])) piv = r;
    }
    if (Math.abs(a[piv * 3 + col]) < 1e-14) return null;
    if (piv !== col) {
      for (let k = 0; k < 3; k++) {
        const tmp = a[col * 3 + k];
        a[col * 3 + k] = a[piv * 3 + k];
        a[piv * 3 + k] = tmp;
      }
      [y[col], y[piv]] = [y[piv], y[col]];
    }
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = a[r * 3 + col] / a[col * 3 + col];
      for (let k = col; k < 3; k++) a[r * 3 + k] -= f * a[col * 3 + k];
      y[r] -= f * y[col];
    }
  }
  return v3(y[0] / a[0], y[1] / (a[4] + 1e-300), y[2] / a[8]);
}
