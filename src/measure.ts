// 测量: 校正后图像上的两点量测
//  模式一 比例尺残段: 用户在可见的那段比例尺上点两点并输入真实长度, 得 mm/px,
//         之后任意两点的像素距离乘以该比例。假设目标与比例尺同深度且近似正对。
//  模式二 棋盘格定面: 照片中含棋盘格时解出板平面(即探方面), 两条水中光线与平面求交,
//         得真实三维距离 —— 全画面精度一致, 不受透视缩短影响。
import { V3, rodrigues, mat3MulVec, dot3, norm3, sub3 } from './linalg';
import { Housing, Intrinsics, WaterRay, correctedPixelToRay } from './refraction';
import { Pose } from './calibration';

export interface Plane {
  n: V3;      // 单位法线(相机系)
  d: number;  // 平面方程 n·x + d = 0
}

/** 棋盘格位姿 -> 板平面(相机系)。板坐标系 z=0 即板面。 */
export function planeFromPose(pose: Pose): Plane {
  const R = rodrigues(pose.rvec);
  const n = mat3MulVec(R, [0, 0, 1]);
  return { n, d: -dot3(n, pose.t) };
}

/** 水中光线与平面求交, 返回相机系三维点(mm) */
export function rayPlaneIntersect(ray: WaterRay, plane: Plane): V3 | null {
  const denom = dot3(plane.n, ray.w);
  if (Math.abs(denom) < 1e-12) return null;
  const t = -(dot3(plane.n, ray.q) + plane.d) / denom;
  if (t <= 0) return null;
  return [ray.q[0] + ray.w[0] * t, ray.q[1] + ray.w[1] * t, ray.q[2] + ray.w[2] * t];
}

export function dist3(a: V3, b: V3): number {
  return norm3(sub3(a, b));
}

/** 比例尺模式: 由残段两点像素距离与真实长度求 mm/px */
export function scaleFromBar(p1: [number, number], p2: [number, number], trueLenMm: number): number {
  const px = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
  if (px < 1e-9) throw new Error('比例尺两点重合');
  return trueLenMm / px;
}

/** 比例尺模式量测(简单 mm/px, 仅适合与比例尺同深度的小范围) */
export function measureByScale(p1: [number, number], p2: [number, number], mmPerPx: number): number {
  return Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) * mmPerPx;
}

/**
 * 比例尺残段 -> 探方面(推荐): 假设相机近似垂直向下拍摄。
 * 由残段真实长度反解面深度 z0: 两个端点的水中光线已知(起点 q_i, 方向 w_i),
 * 求 z 使两交点距离等于真实长度 L —— 精确考虑光线起点在玻璃外表面的侧移,
 * 不能简单用 mm/px 外推(边缘处光线起点侧移可达数 mm, 会引入 4% 级误差)。
 * 在垂直拍摄的正对平面上该推断是精确的; 倾斜大时请改用棋盘格定面。
 */
export function planeFromScaleBar(
  h: Housing, Kv: Intrinsics,
  p1: [number, number], p2: [number, number], trueLenMm: number,
): Plane {
  const r1 = correctedPixelToRay(h, Kv, p1[0], p1[1]);
  const r2 = correctedPixelToRay(h, Kv, p2[0], p2[1]);
  if (!r1 || !r2) throw new Error('比例尺端点超出可校正范围');
  const qz = h.dAir + h.tGlass;
  // P_i(s) = q_i + w_i·s/wz_i, s = z - qz; D(s) = P2 - P1 = A + B·s, 解 |D| = L
  const A: V3 = [r2.q[0] - r1.q[0], r2.q[1] - r1.q[1], r2.q[2] - r1.q[2]];
  const B: V3 = [
    r2.w[0] / r2.w[2] - r1.w[0] / r1.w[2],
    r2.w[1] / r2.w[2] - r1.w[1] / r1.w[2],
    0,
  ];
  const L = trueLenMm;
  const bb = dot3(B, B);
  const ab = dot3(A, B);
  const cc = dot3(A, A) - L * L;
  if (bb < 1e-18) throw new Error('比例尺两点重合');
  const disc = ab * ab - bb * cc;
  if (disc < 0) throw new Error('比例尺长度与光线几何不一致');
  const sq = Math.sqrt(disc);
  // 两个根中取与朴素估计 (z-qz) ≈ fx·L/像素长 接近的
  const l = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
  const sNaive = (Kv.fx * L) / Math.max(l, 1e-9);
  const s1 = (-ab + sq) / bb;
  const s2 = (-ab - sq) / bb;
  const s = Math.abs(s1 - sNaive) <= Math.abs(s2 - sNaive) ? s1 : s2;
  if (s <= 0) throw new Error('解出的面深度在相机后方, 请检查比例尺输入');
  return { n: [0, 0, 1], d: -(qz + s) };
}

/** 定面模式量测: 校正图上两像素 -> 平面上的三维距离 mm */
export function measureOnPlane(
  h: Housing, Kv: Intrinsics, plane: Plane,
  p1: [number, number], p2: [number, number],
): number | null {
  const r1 = correctedPixelToRay(h, Kv, p1[0], p1[1]);
  const r2 = correctedPixelToRay(h, Kv, p2[0], p2[1]);
  if (!r1 || !r2) return null;
  const a = rayPlaneIntersect(r1, plane);
  const b = rayPlaneIntersect(r2, plane);
  if (!a || !b) return null;
  return dist3(a, b);
}
