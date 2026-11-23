/**
 * 校正（rectification）与测量几何。
 *
 * 思路：把每张照片「重投影」到探方所在的物理平面上，生成正射影像，
 * 影像像素与毫米之间为线性比例，两点像素距离乘 mm/px 即真实长度。
 *
 * - 平面在相机系中由一点 Q 与单位法向 n 定义：n·(X-Q)=0
 * - 对每个输出像素：输出像素 -> 平面 3D 点 X -> projectPoint（含折射）-> 源像素采样
 * - 倾斜平面用 2D 牛顿迭代求交，平行平面直接解析求交
 */
import { Vec3, v3, add, sub, scale, dot, normalize, norm, cross } from './vec3';
import { Intrinsics, PortParams, traceBackward, projectPoint } from './model';

export interface Plane3D {
  /** 平面上一点（相机系） */
  q: Vec3;
  /** 单位法向 */
  n: Vec3;
}

/** 光线与平面交点；平行/向后时返回 null */
export function rayPlaneIntersect(origin: Vec3, dir: Vec3, plane: Plane3D): Vec3 | null {
  const denom = dot(dir, plane.n);
  if (Math.abs(denom) < 1e-12) return null;
  const s = dot(sub(plane.q, origin), plane.n) / denom;
  if (s <= 0) return null;
  return add(origin, scale(dir, s));
}

/**
 * 通用平面求源像素：对输出像素的后向水中光线与平面求交，
 * 再用前向折射投影回源像素。倾斜平面用牛顿迭代处理非线性。
 */
export function rectifyMapPixel(
  intr: Intrinsics,
  port: PortParams,
  plane: Plane3D,
  u: number,
  v: number,
): { su: number; sv: number; valid: boolean } {
  // 输出像素本身就是要在原图上定位的查询点：直接从该像素发出光线
  const ray = traceBackward(intr, port, u, v);
  const hit = rayPlaneIntersect(ray.origin, ray.dir, plane);
  if (!hit) return { su: 0, sv: 0, valid: false };
  const src = projectPoint(intr, port, hit);
  return { su: src.u, sv: src.v, valid: true };
}

export interface OrthoFrame {
  /** 平面原点（相机系） */
  origin: Vec3;
  /** 输出影像 x/y 方向的单位基向量（位于平面内） */
  ex: Vec3;
  ey: Vec3;
  /** 每像素对应的毫米数 */
  mmPerPx: number;
}

/** 输出影像像素坐标 -> 平面 3D 点 */
export function orthoPixelToWorld(frame: OrthoFrame, px: number, py: number): Vec3 {
  return add(frame.origin, add(scale(frame.ex, px * frame.mmPerPx), scale(frame.ey, py * frame.mmPerPx)));
}

/** 平面 3D 点 -> 输出影像像素坐标 */
export function worldToOrthoPixel(frame: OrthoFrame, X: Vec3): { x: number; y: number } {
  const d = sub(X, frame.origin);
  return {
    x: dot(d, frame.ex) / frame.mmPerPx,
    y: dot(d, frame.ey) / frame.mmPerPx,
  };
}

/**
 * 正射影像输出像素 -> 源图采样坐标。
 * 输出像素先映射为平面 3D 点，再做含折射的前向投影。
 */
export function orthoMapPixel(
  intr: Intrinsics,
  port: PortParams,
  frame: OrthoFrame,
  outX: number,
  outY: number,
): { su: number; sv: number; valid: boolean } {
  const X = orthoPixelToWorld(frame, outX, outY);
  const src = projectPoint(intr, port, X);
  return { su: src.u, sv: src.v, valid: true };
}

/** 正射影像上两点距离（毫米） */
export function measureOrtho(frame: OrthoFrame, p1: { x: number; y: number }, p2: { x: number; y: number }): number {
  return Math.hypot(p2.x - p1.x, p2.y - p1.y) * frame.mmPerPx;
}

/**
 * 构造正射帧：以 anchor（平面上一点）为影像中心，
 * ex 取 viewUp 在平面内的投影方向，ey 由法向叉乘得到。
 */
export function buildOrthoFrame(plane: Plane3D, anchor: Vec3, mmPerPx: number, viewUp: Vec3 = v3(0, -1, 0)): OrthoFrame {
  const ex = normalize(sub(viewUp, scale(plane.n, dot(viewUp, plane.n))));
  const ey = cross(plane.n, ex);
  return { origin: anchor, ex, ey, mmPerPx };
}

/** 自动确定正射影像的覆盖范围（像素宽高 + 使 anchor 居中的原点偏移） */
export function fitOrthoBounds(
  intr: Intrinsics,
  port: PortParams,
  plane: Plane3D,
  frame: OrthoFrame,
  srcWidth: number,
  srcHeight: number,
): { width: number; height: number; origin: Vec3 } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const samples = 12;
  for (let iy = 0; iy <= samples; iy++) {
    for (let ix = 0; ix <= samples; ix++) {
      const su = (ix / samples) * (srcWidth - 1);
      const sv = (iy / samples) * (srcHeight - 1);
      const ray = traceBackward(intr, port, su, sv);
      const hit = rayPlaneIntersect(ray.origin, ray.dir, plane);
      if (!hit) continue;
      const d = sub(hit, frame.origin);
      const x = dot(d, frame.ex) / frame.mmPerPx;
      const y = dot(d, frame.ey) / frame.mmPerPx;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  }
  if (!isFinite(minX)) {
    return { width: srcWidth, height: srcHeight, origin: frame.origin };
  }
  const pad = 4;
  minX -= pad; minY -= pad; maxX += pad; maxY += pad;
  // 调整原点，使 (minX,minY) 对应输出 (0,0)
  const origin = add(frame.origin, add(scale(frame.ex, minX * frame.mmPerPx), scale(frame.ey, minY * frame.mmPerPx)));
  return {
    width: Math.ceil(maxX - minX),
    height: Math.ceil(maxY - minY),
    origin,
  };
}

/** 平面内两点物理距离（毫米），用于比例尺标定残差 */
export function worldDistance(a: Vec3, b: Vec3): number {
  return norm(sub(a, b));
}
