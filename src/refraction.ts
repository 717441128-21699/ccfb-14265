// 平面防水壳折射模型
//
// 几何: 相机投影中心在原点, 光轴为 +z。光线依次穿过
//   界面1 (z = dAir):       防水壳内空气 (n=1) -> 玻璃 (nGlass)
//   界面2 (z = dAir+tGlass): 玻璃 -> 水 (nWater)
// 两块界面均与光轴垂直(平面玻璃窗)。dAir 与 nWater 由标定拟合,
// 因此水深、水温、盐度引起的折射率变化会被 nWater 吸收, 每次下水重新标定即可。
import { V3, normalize3 } from './linalg';

export interface Housing {
  nGlass: number;  // 玻璃(或亚克力)折射率, 由壳体资料给定
  tGlass: number;  // 玻璃窗厚度 mm
  dAir: number;    // 相机投影中心到玻璃内表面的距离 mm —— 标定拟合
  nWater: number;  // 水的等效折射率 —— 标定拟合 (淡水≈1.333, 海水≈1.338-1.34)
}

export interface Intrinsics {
  fx: number; fy: number;   // 焦距 (像素)
  cx: number; cy: number;   // 主点 (像素)
  k1: number; k2: number;   // 镜头径向畸变 (在空气侧, 即防水壳内镜头本身的畸变)
}

export const N_AIR = 1.0;

/**
 * 由水温/盐度估算水的折射率 (589 nm 附近的线性经验式):
 *   n ≈ 1.33299 + 1.83e-4·S - 1.0e-4·(T-20)
 * S: 盐度 PSU (淡水 0, 海水约 35), T: 水温 °C。
 * 现场记录水温盐度即可; 若用折射仪实测, 可直接输入 n 值。
 */
export function waterIndexFromEnv(tempC: number, salinityPSU: number): number {
  return 1.33299 + 1.83e-4 * salinityPSU - 1.0e-4 * (tempC - 20);
}

/** 矢量形式 Snell 折射。I: 入射单位方向, N: 指向入射介质的界面法线, eta = n1/n2。全反射返回 null。 */
export function refractVec(I: V3, N: V3, eta: number): V3 | null {
  const cosi = -(I[0] * N[0] + I[1] * N[1] + I[2] * N[2]);
  const k = 1 - eta * eta * (1 - cosi * cosi);
  if (k < 0) return null;
  const f = eta * cosi - Math.sqrt(k);
  return [eta * I[0] + f * N[0], eta * I[1] + f * N[1], eta * I[2] + f * N[2]];
}

export interface WaterRay {
  q: V3;  // 光线离开玻璃外表面(进入水中)的点
  w: V3;  // 水中单位方向
}

/**
 * 正向追踪: 相机中心发出、方向正比于 (a, b, 1) 的空气光线, 经两层界面折射进入水中。
 * (a, b) 即无畸变归一化图像坐标。
 */
export function traceForward(h: Housing, a: number, b: number): WaterRay | null {
  const z1 = h.dAir;
  const z2 = h.dAir + h.tGlass;
  const N: V3 = [0, 0, -1];
  let d: V3 = normalize3([a, b, 1]);
  // 界面1: 空气 -> 玻璃 (向光密介质, 不会全反射)
  let t = z1 / d[2];
  const o1: V3 = [d[0] * t, d[1] * t, z1];
  d = refractVec(d, N, N_AIR / h.nGlass)!;
  // 界面2: 玻璃 -> 水
  t = (z2 - o1[2]) / d[2];
  const o2: V3 = [o1[0] + d[0] * t, o1[1] + d[1] * t, z2];
  const w = refractVec(d, N, h.nGlass / h.nWater);
  if (!w) return null;
  return { q: o2, w };
}

/**
 * 反折射: 水中单位方向 w -> 空气光线斜率 (a, b)。
 * 平面平行玻璃只平移光线、不改变方向关系, 因此方向上映射只含 nWater:
 * 空气侧单位方向 = (nW·wx, nW·wy, sqrt(1-nW²·(wx²+wy²)))。
 * 超出水-空气全反射临界角(约 48.6°)时返回 null。
 */
export function unrefract(h: Housing, w: V3): [number, number] | null {
  if (w[2] <= 0) return null;
  const s = h.nWater / N_AIR;
  const t2 = s * s * (w[0] * w[0] + w[1] * w[1]);
  if (t2 >= 1) return null;
  const az = Math.sqrt(1 - t2);
  return [(s * w[0]) / az, (s * w[1]) / az];
}

/** 镜头径向畸变(正向: 无畸变归一化坐标 -> 像素) */
export function distortPixel(K: Intrinsics, a: number, b: number): [number, number] {
  const r2 = a * a + b * b;
  const s = 1 + K.k1 * r2 + K.k2 * r2 * r2;
  return [K.fx * a * s + K.cx, K.fy * b * s + K.cy];
}

/** 去畸变(迭代): 像素 -> 无畸变归一化坐标 */
export function undistortPixel(K: Intrinsics, u: number, v: number): [number, number] {
  const xd = (u - K.cx) / K.fx;
  const yd = (v - K.cy) / K.fy;
  let x = xd, y = yd;
  for (let i = 0; i < 8; i++) {
    const r2 = x * x + y * y;
    const s = 1 + K.k1 * r2 + K.k2 * r2 * r2;
    x = xd / s;
    y = yd / s;
  }
  return [x, y];
}

/**
 * 正向投影: 水中三维点 P(相机坐标系, mm) -> 像素。
 * 需求解光线经折射后恰好通过 P 的出射方向, 用二维牛顿法迭代(对斜率 a, b)。
 */
export function projectPoint(h: Housing, K: Intrinsics, P: V3): [number, number] | null {
  const r = projectPointEx(h, K, P);
  return r ? [r.u, r.v] : null;
}

/** 正向投影并返回对应的水中光线(供三维测量/测试) */
export function projectPointEx(h: Housing, K: Intrinsics, P: V3): { u: number; v: number; ray: WaterRay } | null {
  if (P[2] <= h.dAir + h.tGlass + 1e-6) return null;
  let a = P[0] / P[2];
  let b = P[1] / P[2];
  for (let it = 0; it < 16; it++) {
    const ray = traceForward(h, a, b);
    if (!ray) return null;
    const t = (P[2] - ray.q[2]) / ray.w[2];
    const rx = ray.q[0] + ray.w[0] * t;
    const ry = ray.q[1] + ray.w[1] * t;
    const ex = rx - P[0];
    const ey = ry - P[1];
    if (Math.abs(ex) < 1e-10 && Math.abs(ey) < 1e-10) break;
    const eps = 1e-7;
    const rA = traceForward(h, a + eps, b);
    const rB = traceForward(h, a, b + eps);
    if (!rA || !rB) return null;
    const tA = (P[2] - rA.q[2]) / rA.w[2];
    const tB = (P[2] - rB.q[2]) / rB.w[2];
    const jxa = (rA.q[0] + rA.w[0] * tA - rx) / eps;
    const jya = (rA.q[1] + rA.w[1] * tA - ry) / eps;
    const jxb = (rB.q[0] + rB.w[0] * tB - rx) / eps;
    const jyb = (rB.q[1] + rB.w[1] * tB - ry) / eps;
    const det = jxa * jyb - jxb * jya;
    if (Math.abs(det) < 1e-300) return null;
    const da = (-ex * jyb + ey * jxb) / det;
    const db = (-jxa * ey + jya * ex) / det;
    a += da;
    b += db;
    if (Math.abs(da) < 1e-13 && Math.abs(db) < 1e-13) break;
  }
  const finalRay = traceForward(h, a, b);
  if (!finalRay) return null;
  const [u, v] = distortPixel(K, a, b);
  return { u, v, ray: finalRay };
}

/**
 * 图像校正映射(与 WebGL2 shader 中的公式严格一致):
 * 校正后图像(虚拟针孔相机, 内参 Kv)的像素 (u, v) -> 原始水下照片上的采样位置。
 * 只涉及方向, 与 dAir / tGlass 无关。
 */
export function correctPixel(
  h: Housing, K: Intrinsics, Kv: Intrinsics, u: number, v: number,
): [number, number] | null {
  const wx = (u - Kv.cx) / Kv.fx;
  const wy = (v - Kv.cy) / Kv.fy;
  const wn = 1 / Math.hypot(wx, wy, 1);
  const w: V3 = [wx * wn, wy * wn, wn];
  const ab = unrefract(h, w);
  if (!ab) return null;
  return distortPixel(K, ab[0], ab[1]);
}

/** 校正后像素 -> 水中光线(起点在玻璃外表面, 用于三维测量) */
export function correctedPixelToRay(h: Housing, Kv: Intrinsics, u: number, v: number): WaterRay | null {
  const wx = (u - Kv.cx) / Kv.fx;
  const wy = (v - Kv.cy) / Kv.fy;
  const wn = 1 / Math.hypot(wx, wy, 1);
  const w: V3 = [wx * wn, wy * wn, wn];
  const ab = unrefract(h, w);
  if (!ab) return null;
  const ray = traceForward(h, ab[0], ab[1]);
  if (!ray) return null;
  return { q: ray.q, w };
}
