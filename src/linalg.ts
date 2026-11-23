// 基础线性代数:3 维向量、3x3 矩阵、罗德里格斯公式、线性方程组、对称矩阵特征分解
export type V3 = [number, number, number];
export type M3 = [number, number, number, number, number, number, number, number, number];

export const dot3 = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm3 = (a: V3): number => Math.hypot(a[0], a[1], a[2]);
export const normalize3 = (a: V3): V3 => {
  const n = norm3(a);
  return [a[0] / n, a[1] / n, a[2] / n];
};
export const add3 = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub3 = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale3 = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];

export const mat3MulVec = (R: M3, v: V3): V3 => [
  R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
  R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
  R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
];

/** 旋转向量 -> 旋转矩阵 (罗德里格斯公式) */
export function rodrigues(rvec: V3): M3 {
  const theta = norm3(rvec);
  if (theta < 1e-14) {
    return [1, -rvec[2], rvec[1], rvec[2], 1, -rvec[0], -rvec[1], rvec[0], 1];
  }
  const [x, y, z] = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta];
  const c = Math.cos(theta), s = Math.sin(theta), C = 1 - c;
  return [
    c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C,
  ];
}

/** 旋转矩阵 -> 旋转向量 */
export function rotationToRvec(R: M3): V3 {
  const tr = R[0] + R[4] + R[8];
  const cosTheta = Math.min(1, Math.max(-1, (tr - 1) / 2));
  const theta = Math.acos(cosTheta);
  if (theta < 1e-12) return [0, 0, 0];
  const s = 2 * Math.sin(theta);
  const axis: V3 = [(R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s];
  return [axis[0] * theta, axis[1] * theta, axis[2] * theta];
}

/** 高斯消元(部分主元)解 n x n 线性方程组 A x = b,A 会被破坏 */
export function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    }
    if (piv !== col) {
      const tmp = A[piv]; A[piv] = A[col]; A[col] = tmp;
      const tb = b[piv]; b[piv] = b[col]; b[col] = tb;
    }
    const d = A[col][col];
    if (Math.abs(d) < 1e-300) throw new Error('奇异矩阵');
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / d;
      if (f === 0) continue;
      for (let c = col; c < n; c++) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= A[r][c] * x[c];
    x[r] = s / A[r][r];
  }
  return x;
}

/** 循环 Jacobi 法求对称矩阵特征分解(用于单应矩阵的零空间)。返回特征值升序及对应特征向量(列)。 */
export function jacobiEigen(Ain: number[][]): { values: number[]; vectors: number[][] } {
  const n = Ain.length;
  const A = Ain.map((row) => row.slice());
  let V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-24) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = A[p][p], aqq = A[q][q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi), s = Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[a][a] - A[b][b]);
  return {
    values: idx.map((i) => A[i][i]),
    vectors: idx.map((i) => V.map((row) => row[i])),
  };
}
