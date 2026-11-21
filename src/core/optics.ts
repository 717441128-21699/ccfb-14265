import {
  add,
  dot,
  mv,
  normalize,
  planeDistance,
  rotationFromTilt,
  scale,
  sub,
  v3,
} from './linalg';
import type { RefractionParams, Vec3 } from './types';

/**
 * 平面观察窗折射模型。
 *
 * 光路（从相机出发）：
 *   光心 O --空气射线 r_a--> 窗内表面 P1（法线 n）
 *     --Snell 折射--> 玻璃内射线 r_g
 *   平行玻璃片只产生横向平移，出射点 P2 = P1 + tg * r_g / (r_g·n)
 *     --再次 Snell 折射--> 水中射线 r_w
 *
 * 对于空气 n0=1 的入射，切向波矢守恒，水中方向满足
 *   r_w = r_a / nw + n * (c_a/nw - c_w)，其中 c_w = sqrt(1 - sin²a / nw²)，
 * 玻璃折射率只影响片内平移量。
 */

/** 矢量形式 Snell 折射，n 为界面法线（指向入射侧），返回透射方向；全反射时返回 null */
export function refract(
  incident: Vec3,
  n: Vec3,
  eta: number,
): Vec3 | null {
  const cosi = -dot(incident, n);
  const sin2t = eta * eta * (1 - cosi * cosi);
  if (sin2t > 1) return null;
  const cost = Math.sqrt(1 - sin2t);
  return normalize(add(scale(incident, eta), scale(n, eta * cosi - cost)));
}

export interface WaterRay {
  origin: Vec3; // 窗外表面出射点 P2
  direction: Vec3; // 水中单位方向 r_w
}

/** 由空气中单位射线（起点为相机光心）求进入水中后的射线 */
export function traceAirRay(
  rayAir: Vec3,
  p: RefractionParams,
): WaterRay | null {
  // 无折射：射线不偏折、不平移，水中射线即原空气射线
  if (p.ng === 1 && p.nw === 1) {
    return { origin: v3(0, 0, 0), direction: rayAir };
  }
  const Rn = rotationFromTilt(p.tiltX, p.tiltY);
  const n = mv(Rn, v3(0, 0, 1)); // 窗法线（指向相机一侧时取反使用）
  const P1center = mv(Rn, v3(0, 0, p.d));
  // 求 O + s*r_a 与窗内表面交点：(X - P1center)·n = 0
  const denom = dot(rayAir, n);
  if (denom <= 1e-9) return null;
  const s = dot(P1center, n) / denom;
  const P1 = scale(rayAir, s);

  const rGlass = refract(rayAir, scale(n, -1), 1 / p.ng);
  if (!rGlass) return null;
  const rgN = dot(rGlass, n);
  if (rgN <= 1e-9) return null;
  const P2 = add(P1, scale(rGlass, p.tg / rgN));

  const rWater = refract(rGlass, scale(n, -1), p.ng / p.nw);
  if (!rWater) return null;
  return { origin: P2, direction: rWater };
}

/** 射线与平面（点 q、单位法线 m）交点 */
export function rayPlane(
  o: Vec3,
  r: Vec3,
  q: Vec3,
  m: Vec3,
): Vec3 | null {
  const denom = dot(r, m);
  if (Math.abs(denom) < 1e-12) return null;
  const t = -planeDistance(o, q, m) / denom;
  return add(o, scale(r, t));
}

/** 水中射线与 z = depth 平面交点（共轴窗的快速路径用） */
export function rayZPlane(o: Vec3, r: Vec3, depth: number): Vec3 | null {
  if (Math.abs(r.z) < 1e-12) return null;
  const t = (depth - o.z) / r.z;
  return add(o, scale(r, t));
}

/**
 * 共轴窗（tiltX=tiltY=0）的快速投影：
 * 空气射线 r_a = (u,v,1)/sqrt，界面交点仅由 alpha 决定，
 * 问题降为关于 alpha = r_ax / r_az 的一维标量方程，
 * 用牛顿法（解析导数）求解，通常 4 次迭代内收敛。
 */
function projectCoaxial(
  X: Vec3,
  p: RefractionParams,
): { u: number; v: number } | null {
  const x = X.x;
  const y = X.y;
  const z = X.z;
  if (z <= p.d + p.tg) return null;
  const nw2 = p.nw * p.nw;
  const ng2 = p.ng * p.ng;

  // 选绝对值较大的轴作为主方向，保证比值 beta 有限（避免 x≈0 时除零）。
  // 主方向斜率 alpha 沿该轴求解，另一轴由 beta 比例恢复。
  const swap = Math.abs(y) > Math.abs(x);
  const main = swap ? y : x;
  const other = swap ? x : y;
  const beta = main !== 0 ? other / main : 0;
  let alpha = main / z; // 主方向初始针孔斜率
  for (let iter = 0; iter < 30; iter++) {
    const a2 = alpha * alpha;
    const qA = 1 + a2 * (1 + beta * beta);
    const sin2A = (a2 * (1 + beta * beta)) / qA;
    const sin2G = sin2A / ng2;
    const cosG = Math.sqrt(Math.max(0, 1 - sin2G));
    const sin2W = sin2A / nw2;
    const cosW = Math.sqrt(Math.max(0, 1 - sin2W));

    const P2main =
      p.d * alpha + (p.tg * (alpha / p.ng)) / cosG;
    const rMainOverRz = (alpha / p.nw) / cosW;
    const f = P2main + (z - (p.d + p.tg)) * rMainOverRz - main;

    const S = 1 + beta * beta;
    const dSin2A = (2 * alpha * S) / (qA * qA);
    const dCosG = -dSin2A / (2 * ng2 * cosG);
    const dCosW = -dSin2A / (2 * nw2 * cosW);
    const dP2 =
      p.d +
      (p.tg / p.ng) * ((cosG - alpha * dCosG) / (cosG * cosG));
    const dR =
      (1 / p.nw) * ((cosW - alpha * dCosW) / (cosW * cosW));
    const fp = dP2 + (z - p.d - p.tg) * dR;
    if (!Number.isFinite(fp) || Math.abs(fp) < 1e-12) break;
    const step = f / fp;
    alpha -= step;
    if (Math.abs(step) < 1e-11) break;
  }
  if (!Number.isFinite(alpha)) return null;
  const mainSlope = alpha;
  const otherSlope = alpha * beta;
  const u = swap ? otherSlope : mainSlope;
  const v = swap ? mainSlope : otherSlope;
  if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
  return { u, v };
}

/** 通用倾斜窗投影：对界面交点 (sx, sy) 做二维牛顿迭代 */
function projectTilted(
  X: Vec3,
  p: RefractionParams,
): { u: number; v: number } | null {
  const Rn = rotationFromTilt(p.tiltX, p.tiltY);
  const n = mv(Rn, v3(0, 0, 1));
  const P1center = mv(Rn, v3(0, 0, p.d));
  const c = dot(P1center, n);

  const buildRay = (sx: number, sy: number): WaterRay | null => {
    const rAir = normalize(v3(sx, sy, 1));
    const denom = dot(rAir, n);
    if (denom <= 1e-9) return null;
    const s = c / denom;
    const P1 = scale(rAir, s);
    const rGlass = refract(rAir, scale(n, -1), 1 / p.ng);
    if (!rGlass) return null;
    const rgN = dot(rGlass, n);
    if (rgN <= 1e-9) return null;
    const P2 = add(P1, scale(rGlass, p.tg / rgN));
    const rWater = refract(rGlass, scale(n, -1), p.ng / p.nw);
    if (!rWater) return null;
    return { origin: P2, direction: rWater };
  };
  const residual = (sx: number, sy: number): Vec3 | null => {
    const ray = buildRay(sx, sy);
    if (!ray) return null;
    const hit = rayPlane(ray.origin, ray.direction, X, v3(0, 0, 1));
    if (!hit) return null;
    return sub(hit, X);
  };

  // 求解变量是射线斜率 (sx, sy) = (rx/rz, ry/rz)，即归一化针孔坐标
  let sx = X.x / X.z;
  let sy = X.y / X.z;
  const eps = 1e-7;
  for (let iter = 0; iter < 40; iter++) {
    const r0 = residual(sx, sy);
    if (!r0) return null;
    const f = [r0.x, r0.y];
    const rp = residual(sx + eps, sy);
    const rm = residual(sx - eps, sy);
    const qp = residual(sx, sy + eps);
    const qm = residual(sx, sy - eps);
    if (!rp || !rm || !qp || !qm) return null;
    const J = [
      [(rp.x - rm.x) / (2 * eps), (qp.x - qm.x) / (2 * eps)],
      [(rp.y - rm.y) / (2 * eps), (qp.y - qm.y) / (2 * eps)],
    ];
    const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    if (Math.abs(det) < 1e-14) return null;
    const dx =
      (J[1][1] * f[0] - J[0][1] * f[1]) / det;
    const dy =
      (-J[1][0] * f[0] + J[0][0] * f[1]) / det;
    sx -= dx;
    sy -= dy;
    if (Math.abs(dx) + Math.abs(dy) < 1e-10) break;
  }
  if (!Number.isFinite(sx) || !Number.isFinite(sy)) return null;
  return { u: sx, v: sy };
}

/**
 * 把相机坐标系下的 3D 点投影为归一化针孔坐标 (u, v)（未加重镜头畸变）。
 * 共轴窗走一维快速路径，倾斜窗走二维牛顿。
 */
export function projectPointNormalized(
  X: Vec3,
  p: RefractionParams,
): { u: number; v: number } | null {
  // 无折射（玻璃与水折射率均为 1）：光线直线传播，精确针孔投影
  if (p.ng === 1 && p.nw === 1) {
    if (X.z <= 0) return null;
    return { u: X.x / X.z, v: X.y / X.z };
  }
  if (p.tiltX === 0 && p.tiltY === 0) {
    return projectCoaxial(X, p);
  }
  return projectTilted(X, p);
}

/** 由归一化针孔坐标构造空气中单位射线 */
export function airRayFromNormalized(u: number, v: number): Vec3 {
  return normalize(v3(u, v, 1));
}

/**
 * 校正（逆映射）核心：输出图是一台与真实相机共光心的虚拟针孔相机。
 * 给定输出像素的归一化坐标，其空气射线经窗折射后在参考平面 z=depth
 * 上击中物体点；由光路可逆，真实相机中抵达同一物体点的空气射线方向
 * 相同，故源图归一化针孔坐标就是 (uOut, vOut)。
 *
 * depth 的作用仅体现在折射参数随深度变化的场景（本模型中窗固定，
 * 折射方向只依赖入射角），这里显式追踪到 depth 是为了校验光线有效、
 * 命中点位于窗外表面之前的情况会被剔除，并保留深度相关扩展的接口。
 */
export function invertAtDepth(
  uOut: number,
  vOut: number,
  depth: number,
  p: RefractionParams,
): { u: number; v: number } | null {
  const rayOut = airRayFromNormalized(uOut, vOut);
  const water = traceAirRay(rayOut, p);
  if (!water) return null;
  const hit = rayZPlane(water.origin, water.direction, depth);
  if (!hit) return null;
  return { u: uOut, v: vOut };
}
