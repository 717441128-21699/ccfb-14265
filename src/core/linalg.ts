import type { Mat3, Pose, Vec3 } from './types';

export const v3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

export const add = (a: Vec3, b: Vec3): Vec3 => v3(a.x + b.x, a.y + b.y, a.z + b.z);
export const sub = (a: Vec3, b: Vec3): Vec3 => v3(a.x - b.x, a.y - b.y, a.z - b.z);
export const scale = (a: Vec3, s: number): Vec3 => v3(a.x * s, a.y * s, a.z * s);
export const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a: Vec3, b: Vec3): Vec3 =>
  v3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x,
  );
export const norm = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);
export const normalize = (a: Vec3): Vec3 => {
  const n = norm(a) || 1;
  return scale(a, 1 / n);
};

/** 行主序 3x3 乘列向量 */
export function mv(A: Mat3, x: Vec3): Vec3 {
  return v3(
    A[0] * x.x + A[1] * x.y + A[2] * x.z,
    A[3] * x.x + A[4] * x.y + A[5] * x.z,
    A[6] * x.x + A[7] * x.y + A[8] * x.z,
  );
}

export function mm(A: Mat3, B: Mat3): Mat3 {
  const C = new Array<number>(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      C[i * 3 + j] =
        A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    }
  }
  return C;
}

export function transpose(A: Mat3): Mat3 {
  return [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
}

/** 绕 x 轴（y->z）再绕 y 轴（z->x）的旋转矩阵 */
export function rotationFromTilt(tiltX: number, tiltY: number): Mat3 {
  const cx = Math.cos(tiltX),
    sx = Math.sin(tiltX);
  const cy = Math.cos(tiltY),
    sy = Math.sin(tiltY);
  const Rx: Mat3 = [1, 0, 0, 0, cx, -sx, 0, sx, cx];
  const Ry: Mat3 = [cy, 0, sy, 0, 1, 0, -sy, 0, cy];
  return mm(Ry, Rx);
}

/** Rodrigues 三参数 -> 旋转矩阵 */
export function rodrigues(rx: number, ry: number, rz: number): Mat3 {
  const r = v3(rx, ry, rz);
  const theta = norm(r);
  if (theta < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const k = scale(r, 1 / theta);
  const c = Math.cos(theta),
    c1 = 1 - c,
    s = Math.sin(theta);
  const K: Mat3 = [0, -k.z, k.y, k.z, 0, -k.x, -k.y, k.x, 0];
  const I: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const K2 = mm(K, K);
  return I.map((v, i) => v + s * K[i] + c1 * K2[i]);
}

export function transformPoint(pose: Pose, p: Vec3): Vec3 {
  return add(mv(pose.R, p), pose.t);
}

/** 点到平面（点 n0、法线 n）的有符号距离 */
export function planeDistance(p: Vec3, n0: Vec3, n: Vec3): number {
  return dot(sub(p, n0), n);
}

export const clamp = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));
