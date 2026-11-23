/**
 * 平面玻璃窗口折射模型（flat-port refraction model）。
 *
 * 坐标系：相机坐标系，x 右、y 下、z 朝场景前方。
 * 窗口内表面（空气/玻璃）：n·X = d0
 * 窗口外表面（玻璃/水）：n·X = d0 + t
 * n 为单位法向，由相机指向水体，理想装配时 n=(0,0,1)，可带倾斜。
 *
 * 光线路径（物体 -> 相机）：
 *   水中物体 P
 *   -> 外表面折射进入玻璃（nWater -> nGlass）
 *   -> 内表面折射进入空气（nGlass -> nAir）
 *   -> 到达针孔 O，按内参成像
 *
 * 前向投影：从针孔沿像素光线出发，经两界面折射得到水中光线，
 * 用牛顿迭代调整像素，使水中光线通过目标点 P（不引入多项式近似）。
 */
import { Vec3, v3, add, sub, scale, dot, normalize, norm } from './vec3';

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export interface PortParams {
  /** 窗口法向（相机系，存储时不要求单位化，使用时归一化） */
  nx: number;
  ny: number;
  nz: number;
  /** 光心到内表面距离 n·X=d0 */
  d0: number;
  /** 玻璃厚度 */
  t: number;
  nAir: number;
  nGlass: number;
  nWater: number;
}

export function portNormal(p: PortParams): Vec3 {
  return normalize(v3(p.nx, p.ny, p.nz));
}

/**
 * 矢量形式 Snell 折射（Hecht 约定）。
 * @param I 入射方向单位向量（沿传播方向）
 * @param n 界面单位法向，指向【入射侧】（与 I 夹角 >90°，cosθi = −I·n > 0）
 * @param eta 入射侧折射率 / 折射侧折射率
 * @returns 折射方向单位向量；全反射时返回 null
 *
 * 公式：T = η I + (η cosθi − cosθt) n
 */
export function refract(I: Vec3, n: Vec3, eta: number): Vec3 | null {
  const cosi = -dot(I, n); // > 0
  const sin2t = eta * eta * (1 - cosi * cosi);
  if (sin2t >= 1) return null;
  const cost = Math.sqrt(1 - sin2t);
  return normalize(add(scale(I, eta), scale(n, eta * cosi - cost)));
}

/** 相机像素 -> 归一化相机光线方向 */
export function pixelRay(intr: Intrinsics, u: number, v: number): Vec3 {
  return normalize(v3((u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1));
}

export interface Ray {
  origin: Vec3;
  dir: Vec3;
}

/** 后向追踪：相机像素光线穿过两层界面，返回水中的光线（起点在外表面）。 */
export function traceBackward(intr: Intrinsics, port: PortParams, u: number, v: number): Ray {
  const n = portNormal(port);
  const nBack = scale(n, -1); // 指向入射侧（相机一侧）
  const dAir = pixelRay(intr, u, v);
  // 内表面 n·X = d0
  const s0 = port.d0 / dot(dAir, n);
  const x0 = scale(dAir, s0);
  // 空气 -> 玻璃
  const dGlass = refract(dAir, nBack, port.nAir / port.nGlass);
  if (!dGlass) throw new Error('内表面全反射（检查法向/入射角）');
  // 两平行界面间距 t：沿 dGlass 的行程
  const s1 = port.t / dot(dGlass, n);
  const x1 = add(x0, scale(dGlass, s1));
  // 玻璃 -> 水
  const dWater = refract(dGlass, nBack, port.nGlass / port.nWater);
  if (!dWater) throw new Error('外表面全反射（检查法向/入射角）');
  return { origin: x1, dir: dWater };
}

/** 水中光线上距目标点 P 最近的点 */
function closestOnWaterRay(
  intr: Intrinsics,
  port: PortParams,
  u: number,
  v: number,
  P: Vec3,
): Vec3 {
  const ray = traceBackward(intr, port, u, v);
  const lambda = dot(sub(P, ray.origin), ray.dir);
  return add(ray.origin, scale(ray.dir, lambda));
}

/** 3x2 最小二乘：J Δ = b，返回 Δu,Δv（J 列主序扁平，每列 3 行） */
function lstsq2(J: Float64Array, b: Vec3): { u: number; v: number } {
  const a00 = J[0] * J[0] + J[1] * J[1] + J[2] * J[2];
  const a01 = J[0] * J[3] + J[1] * J[4] + J[2] * J[5];
  const a11 = J[3] * J[3] + J[4] * J[4] + J[5] * J[5];
  const b0 = J[0] * b.x + J[1] * b.y + J[2] * b.z;
  const b1 = J[3] * b.x + J[4] * b.y + J[5] * b.z;
  const det = a00 * a11 - a01 * a01;
  if (Math.abs(det) < 1e-18) return { u: 0, v: 0 };
  return {
    u: (a11 * b0 - a01 * b1) / det,
    v: (a00 * b1 - a01 * b0) / det,
  };
}

/**
 * 前向投影：相机系中的 3D 点 P -> 像素。
 * 牛顿迭代调整像素坐标，使后向追踪得到的水中光线通过 P。
 * 收敛判据用像素步长（对外部有限差分更稳定）。
 */
export function projectPoint(
  intr: Intrinsics,
  port: PortParams,
  P: Vec3,
): { u: number; v: number } {
  // 初始猜测：忽略折射的针孔光线
  const guess = normalize(P);
  let u = intr.fx * (guess.x / guess.z) + intr.cx;
  let v = intr.fy * (guess.y / guess.z) + intr.cy;
  const eps = 0.05;

  for (let iter = 0; iter < 40; iter++) {
    const c0 = closestOnWaterRay(intr, port, u, v, P);
    const residual = sub(P, c0);
    if (norm(residual) < 1e-7) break;
    const cu = closestOnWaterRay(intr, port, u + eps, v, P);
    const cv = closestOnWaterRay(intr, port, u, v + eps, P);
    const J = Float64Array.from([
      (cu.x - c0.x) / eps,
      (cu.y - c0.y) / eps,
      (cu.z - c0.z) / eps,
      (cv.x - c0.x) / eps,
      (cv.y - c0.y) / eps,
      (cv.z - c0.z) / eps,
    ]);
    const delta = lstsq2(J, residual);
    u += delta.u;
    v += delta.v;
    if (Math.hypot(delta.u, delta.v) < 1e-8) break;
  }
  return { u, v };
}

/** 针孔投影（相机系点 -> 像素），供虚拟点投影/调试使用 */
export function pinholeProject(intr: Intrinsics, X: Vec3): { u: number; v: number } {
  return {
    u: intr.fx * (X.x / X.z) + intr.cx,
    v: intr.fy * (X.y / X.z) + intr.cy,
  };
}
