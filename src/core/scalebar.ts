/**
 * 比例尺（scale bar）处理，含半遮挡情形。
 *
 * 探方中比例尺常被泥沙盖住一部分。本模块支持：
 * 1. 已知剩余可见段真实长度 L_vis：用其两端点在平面上的反投影点核对几何长度；
 * 2. 只知道整根全长 L_full 与覆盖比例：L_vis = (1-buried)·L_full；
 * 3. 单根比例尺 + 比例尺平铺于探方平面的假设：求解平面使两端点反投影距离
 *    等于 L_vis。单线段存在法向锥面歧义，用朝向先验消歧。
 */
import { Vec3, v3, add, sub, scale, normalize, norm, cross } from './vec3';
import { Intrinsics, PortParams, traceBackward } from './model';
import { Plane3D, rayPlaneIntersect } from './rectify';

/** 两个源像素在给定平面上的反投影 3D 点 */
export function backprojectOnPlane(
  intr: Intrinsics,
  port: PortParams,
  plane: Plane3D,
  u: number,
  v: number,
): Vec3 | null {
  const ray = traceBackward(intr, port, u, v);
  return rayPlaneIntersect(ray.origin, ray.dir, plane);
}

export interface ScaleBarResult {
  mmPerPx: number;
  pA: Vec3;
  pB: Vec3;
  achievedLength: number;
}

/** 已知平面 + 可见段真实长度：反投影两端点并核对几何长度 */
export function calibrateFromVisibleSegment(
  intr: Intrinsics,
  port: PortParams,
  plane: Plane3D,
  a: { u: number; v: number },
  b: { u: number; v: number },
  visibleLengthMm: number,
): ScaleBarResult {
  const pA = backprojectOnPlane(intr, port, plane, a.u, a.v);
  const pB = backprojectOnPlane(intr, port, plane, b.u, b.v);
  if (!pA || !pB) throw new Error('比例尺端点反投影失败：光线与平面无交点');
  const achievedLength = norm(sub(pB, pA));
  if (achievedLength < 1e-9) throw new Error('比例尺两端点在平面上重合');
  return { mmPerPx: visibleLengthMm / achievedLength, pA, pB, achievedLength };
}


/**
 * 单比例尺平面求解。
 *
 * 直接在【先验法向的切平面】内对法向做二维参数化 n(α,β)，
 * 用 Levenberg 风格的二维最小化使两端点反投影距离 = L_vis，
 * 同时 n 始终接近先验（天然消歧，不需要从像素重建比例尺方向）。
 *
 * 约束「比例尺在平面内」通过：平面过锚点 q0，两端点各自与光线求交，
 * 交点连线即比例尺在该平面内的实现，长度闭合即可。
 */
export function solvePlaneFromSingleBar(
  intr: Intrinsics,
  port: PortParams,
  a: { u: number; v: number },
  b: { u: number; v: number },
  visibleLengthMm: number,
  q0: Vec3,
  normalPrior: Vec3 = v3(0, 0, 1),
): { plane: Plane3D; pA: Vec3; pB: Vec3; achievedLength: number; normalAmbiguity: boolean } {
  const rayA = traceBackward(intr, port, a.u, a.v);
  const rayB = traceBackward(intr, port, b.u, b.v);
  const n0 = normalize(normalPrior);

  // 先验法向的两个正交切向基
  const helper = Math.abs(n0.z) < 0.9 ? v3(0, 0, 1) : v3(0, 1, 0);
  const t1 = normalize(cross(n0, helper));
  const t2 = normalize(cross(n0, t1));

  const normalFromParams = (alpha: number, beta: number): Vec3 =>
    normalize(add(n0, add(scale(t1, alpha), scale(t2, beta))));

  const lengthAt = (alpha: number, beta: number): number => {
    const n = normalFromParams(alpha, beta);
    const plane: Plane3D = { q: q0, n };
    const pA = rayPlaneIntersect(rayA.origin, rayA.dir, plane);
    const pB = rayPlaneIntersect(rayB.origin, rayB.dir, plane);
    if (!pA || !pB) return NaN;
    return norm(sub(pB, pA));
  };

  // 二维高斯-牛顿：残差 r = length(α,β) - L_vis（单约束，沿梯度走，
  // 配合先验把解拉在 n0 附近——α,β 初值 0 即先验本身）
  let alpha = 0;
  let beta = 0;
  const h = 1e-5;
  let r = lengthAt(alpha, beta) - visibleLengthMm;
  for (let iter = 0; iter < 100; iter++) {
    if (Math.abs(r) < 1e-6) break;
    const la = lengthAt(alpha + h, beta);
    const lb = lengthAt(alpha, beta + h);
    const ga = (la - lengthAt(alpha, beta)) / h;
    const gb = (lb - lengthAt(alpha, beta)) / h;
    const g2 = ga * ga + gb * gb;
    if (g2 < 1e-18) break;
    // 高斯-牛顿步（单残差）：δ = r·g / g²，更新使 r→0
    const step = r / g2;
    alpha -= ga * step;
    beta -= gb * step;
    // 限制法向偏离先验不超过 ~30°，保证消歧与稳定
    const mag = Math.hypot(alpha, beta);
    if (mag > 0.55) {
      alpha *= 0.55 / mag;
      beta *= 0.55 / mag;
    }
    const rNew = lengthAt(alpha, beta) - visibleLengthMm;
    if (Math.abs(rNew) >= Math.abs(r)) break; // 无改善则停
    r = rNew;
  }

  const n = normalFromParams(alpha, beta);
  const plane: Plane3D = { q: q0, n };
  const pA = rayPlaneIntersect(rayA.origin, rayA.dir, plane)!;
  const pB = rayPlaneIntersect(rayB.origin, rayB.dir, plane)!;
  const achievedLength = norm(sub(pB, pA));

  // 歧义判定：单线段把法向约束到一条等值曲线（锥面），在先验附近的
  // 二维切平面邻域内，长度等值线是一维的，恒存在多个等长法向，
  // 因此只要残差能收敛到容差内，就标记存在法向歧义（由先验消歧）。
  const normalAmbiguity = Math.abs(r) < Math.max(1e-4, visibleLengthMm * 1e-6);

  return { plane, pA, pB, achievedLength, normalAmbiguity };
}

/** 由整根比例尺长度与覆盖比例计算可见段长度。buriedFraction∈[0,1] */
export function visibleLength(fullLengthMm: number, buriedFraction: number): number {
  return fullLengthMm * Math.max(0, 1 - buriedFraction);
}
