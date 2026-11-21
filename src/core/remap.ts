import {
  airRayFromNormalized,
  projectPointNormalized,
  rayZPlane,
  traceAirRay,
} from './optics';
import { distortNormalized } from './camera';
import type {
  BrownDistortion,
  Intrinsics,
  Measurement,
  RefractionParams,
  ScaleBar,
  Vec2,
} from './types';

/**
 * 校正图 = 与真实相机共光心的虚拟针孔相机（保留原内参）。
 * 对输出像素 (u,v) 的逆映射：
 *   1. 虚拟针孔射线（理想、无畸变）经窗两层折射，击中参考平面 z=depth
 *      得到水中物体点 X；
 *   2. 对 X 做真实相机的完整正向投影（折射几何 + Brown 畸变 + 内参），
 *      得到该物体点实际落在源图上的像素；
 *   3. GPU 按该坐标双三次采样。
 * 物理折射的逐像素光线追踪只在粗网格上做，再双线性展开，
 * 共轴/倾斜窗共用同一条路径。
 */

export interface RemapTable {
  width: number;
  height: number;
  /** 与输出图同尺寸的源像素坐标 (x,y) 交错存储 */
  mapX: Float32Array;
  mapY: Float32Array;
  /** 无有效光线的像素掩码 */
  valid: Uint8Array;
  referenceDepthMm: number;
}

export interface RemapOptions {
  /** 粗网格间距，像素；默认 32（24MP 约 7.5 万个网格点） */
  gridStep?: number;
}

export function buildRemap(
  width: number,
  height: number,
  K: Intrinsics,
  dist: BrownDistortion,
  refr: RefractionParams,
  referenceDepthMm: number,
  opts: RemapOptions = {},
): RemapTable {
  // 折射映射非线性强，网格步长取 16 保证双三次展开的亚像素精度；
  // 24MP 图约 9.4 万网格点，CPU 建表仍在秒级。
  const step = opts.gridStep ?? 16;
  const mapX = new Float32Array(width * height);
  const mapY = new Float32Array(width * height);
  const valid = new Uint8Array(width * height);

  const gx = Math.ceil(width / step) + 1;
  const gy = Math.ceil(height / step) + 1;
  const cx = new Float32Array(gx * gy);
  const cy = new Float32Array(gx * gy);
  const cv = new Uint8Array(gx * gy);

  for (let j = 0; j < gy; j++) {
    const py = Math.min(height - 1, j * step);
    for (let i = 0; i < gx; i++) {
      const px = Math.min(width - 1, i * step);
      const u = (px - K.cx) / K.fx;
      const v = (py - K.cy) / K.fy;
      // 1) 虚拟针孔射线经折射击中参考平面
      const ray = airRayFromNormalized(u, v);
      const water = traceAirRay(ray, refr);
      const idx = j * gx + i;
      if (!water) {
        cv[idx] = 0;
        cx[idx] = px;
        cy[idx] = py;
        continue;
      }
      const hit = rayZPlane(water.origin, water.direction, referenceDepthMm);
      if (!hit || hit.z <= 0) {
        cv[idx] = 0;
        cx[idx] = px;
        cy[idx] = py;
        continue;
      }
      // 2) 该物体点在真实相机中的归一化投影（折射几何）
      const proj = projectPointNormalized(hit, refr);
      if (!proj) {
        cv[idx] = 0;
        cx[idx] = px;
        cy[idx] = py;
        continue;
      }
      // 3) 施加镜头畸变与内参，得到源像素坐标
      const dp = distortNormalized(proj.u, proj.v, dist);
      cx[idx] = K.fx * dp.x + K.cx;
      cy[idx] = K.fy * dp.y + K.cy;
      cv[idx] = 1;
    }
  }

  // Catmull-Rom bicubic expansion. The refraction mapping is nonlinear,
  // so bilinear interpolation on a coarse grid leaves multi-pixel error;
  // the 4x4 cubic kernel reconstructs it to sub-pixel accuracy (the same
  // kernel used in the WebGL resampler). Validity still uses the central
  // 2x2 bilinear mask so border/occluded pixels are flagged correctly.
  const catmull = (t: number): number => {
    const at = Math.abs(t);
    if (at < 1) return 1.5 * at * at * at - 2.5 * at * at + 1;
    if (at < 2) return -0.5 * at * at * at + 2.5 * at * at - 4 * at + 2;
    return 0;
  };
  for (let py = 0; py < height; py++) {
    const gj = py / step;
    const j0 = Math.floor(gj);
    const ty = gj - j0;
    for (let px = 0; px < width; px++) {
      const gi = px / step;
      const i0 = Math.floor(gi);
      const tx = gi - i0;
      const i1 = Math.min(gx - 1, i0 + 1);
      const j1 = Math.min(gy - 1, j0 + 1);
      const out = py * width + px;

      // validity from central 2x2
      const a00 = j0 * gx + i0;
      const a10 = j0 * gx + i1;
      const a01 = j1 * gx + i0;
      const a11 = j1 * gx + i1;
      const wsum =
        (1 - tx) * (1 - ty) * (cv[a00] ? 1 : 0) +
        tx * (1 - ty) * (cv[a10] ? 1 : 0) +
        (1 - tx) * ty * (cv[a01] ? 1 : 0) +
        tx * ty * (cv[a11] ? 1 : 0);
      valid[out] = wsum > 0.5 ? 1 : 0;

      let mx = 0;
      let my = 0;
      for (let mj = -1; mj <= 2; mj++) {
        const wy = catmull(ty - mj);
        const gjc = Math.max(0, Math.min(gy - 1, j0 + mj));
        for (let mi = -1; mi <= 2; mi++) {
          const wx = catmull(tx - mi);
          const gic = Math.max(0, Math.min(gx - 1, i0 + mi));
          const idx = gjc * gx + gic;
          const w = wx * wy * (cv[idx] ? 1 : 0);
          mx += w * cx[idx];
          my += w * cy[idx];
        }
      }
      mapX[out] = mx;
      mapY[out] = my;
    }
  }
  return { width, height, mapX, mapY, valid, referenceDepthMm };
}

/** 校正后参考平面上的像素当量（mm/像素），由虚拟针孔模型直接给出 */
export function mmPerPixelAtDepth(
  K: Intrinsics,
  depthMm: number,
): number {
  return depthMm / K.fx;
}

/**
 * 两点测量。优先使用比例尺残段标定像素当量（可被泥沙盖住一部分），
 * 否则回退到参考深度的针孔当量。
 * 同时给出深度失配造成的相对误差界：|Δz|/z。
 */
export function measure(
  p1: Vec2,
  p2: Vec2,
  K: Intrinsics,
  referenceDepthMm: number,
  bar?: ScaleBar | null,
): Measurement {
  const pixels = Math.hypot(p2.x - p1.x, p2.y - p1.y);
  let mmPerPixel: number;
  if (bar) {
    const barPx = Math.hypot(bar.p2.x - bar.p1.x, bar.p2.y - bar.p1.y);
    mmPerPixel = bar.knownLengthMm / Math.max(1e-6, barPx);
  } else {
    mmPerPixel = mmPerPixelAtDepth(K, referenceDepthMm);
  }
  return {
    p1,
    p2,
    referenceDepthMm,
    pixels,
    mmPerPixel,
    lengthMm: pixels * mmPerPixel,
  };
}

/** 估计被测物实际深度与参考深度不一致时的相对误差（百分比） */
export function depthMismatchError(
  actualDepthMm: number,
  referenceDepthMm: number,
): number {
  return Math.abs(actualDepthMm - referenceDepthMm) / referenceDepthMm * 100;
}
