/**
 * 按真实折射几何合成水下照片（CPU 光线追踪参考实现）。
 *
 * 对每个源像素：traceBackward 得到水中光线，与场景平面求交，
 * 在物理坐标上采样棋盘格/比例尺/器物纹理，颜色经简单衰减。
 * 角点检测则用「已知角点 3D 位置 -> projectPoint」生成亚像素观测，
 * 与渲染共用同一套折射模型，保证测试图物理自洽。
 */
import { Vec3, v3, add, sub, scale, dot, norm } from './vec3';
import { Intrinsics, PortParams, traceBackward, projectPoint } from './model';
import { Plane3D, rayPlaneIntersect } from './rectify';
import { boardPoint } from './calibration';

export interface RGB {
  r: number;
  g: number;
  b: number;
}

export interface SyntheticScene {
  width: number;
  height: number;
  intrinsics: Intrinsics;
  port: PortParams;
  /** 探方平面（相机系） */
  plane: Plane3D;
  board: { rows: number; cols: number; squareMm: number; origin: Vec3; ex: Vec3; ey: Vec3 };
  /** 比例尺（平面内），start/end 为相机系 3D 端点，buriedStart 段视为被泥沙覆盖 */
  scaleBars: {
    start: Vec3;
    end: Vec3;
    widthMm: number;
    fullLengthMm: number;
    buriedFraction: number;
  }[];
  /** 已知尺寸的器物：圆盘（中心、平面内半径 mm） */
  artifacts: { center: Vec3; radiusMm: number; shade: RGB }[];
}

export interface SyntheticImage {
  width: number;
  height: number;
  /** RGBA，每通道 0~255 */
  data: Uint8ClampedArray;
}

/** 场景平面上的 3D 点 -> 板系局部坐标 */
function toBoardLocal(scene: SyntheticScene, X: Vec3): { x: number; y: number } {
  const d = sub(X, scene.board.origin);
  return { x: dot(d, scene.board.ex), y: dot(d, scene.board.ey) };
}

function shade(base: RGB, attenuation: number, tint: number): RGB {
  return {
    r: Math.min(255, base.r * attenuation + tint),
    g: Math.min(255, base.g * attenuation + tint),
    b: Math.min(255, base.b * attenuation + tint),
  };
}

function sampleScene(scene: SyntheticScene, X: Vec3): RGB {
  const local = toBoardLocal(scene, X);
  const sq = scene.board.squareMm;
  const ix = Math.floor(local.x / sq);
  const iy = Math.floor(local.y / sq);
  const inBoard =
    local.x >= 0 &&
    local.y >= 0 &&
    local.x < scene.board.cols * sq &&
    local.y < scene.board.rows * sq;

  let color: RGB = inBoard
    ? (ix + iy) % 2 === 0
      ? { r: 235, g: 232, b: 220 }
      : { r: 35, g: 38, b: 42 }
    : { r: 96, g: 108, b: 92 }; // 泥沙底色

  // 器物（圆盘）
  for (const art of scene.artifacts) {
    if (norm(sub(X, art.center)) <= art.radiusMm) {
      color = shade(art.shade, 0.95, 0);
    }
  }

  // 比例尺：黑底白刻度，被覆盖段画成泥沙色
  for (const bar of scene.scaleBars) {
    const axis = sub(bar.end, bar.start);
    const L = norm(axis);
    const dir = scale(axis, 1 / L);
    const w = sub(X, bar.start);
    const along = dot(w, dir);
    const perp = norm(sub(w, scale(dir, along)));
    if (along >= 0 && along <= L && perp <= bar.widthMm / 2) {
      if (along / L < bar.buriedFraction) {
        color = { r: 110, g: 100, b: 80 }; // 泥沙覆盖
      } else {
        color = { r: 245, g: 245, b: 240 };
        // 每 10mm 一条粗刻度，5mm 细刻度
        const mod10 = along % 10;
        const mod5 = along % 5;
        const tickLen = mod10 < 0.8 ? bar.widthMm * 0.9 : mod5 < 0.5 ? bar.widthMm * 0.55 : 0;
        if (perp <= tickLen / 2) color = { r: 20, g: 20, b: 20 };
      }
    }
  }

  // 水体衰减（距离越远越蓝绿）
  const dist = norm(X);
  const atten = Math.max(0.55, 1 - dist / 20000);
  return shade(color, atten, 6);
}

/** 渲染合成照片；ss 为每边超采样数（1=不超采样） */
export function renderSynthetic(scene: SyntheticScene, ss = 2): SyntheticImage {
  const { width, height, intrinsics, port, plane } = scene;
  const out = new Uint8ClampedArray(width * height * 4);
  const invSs2 = 1 / (ss * ss);

  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      let r = 0, g = 0, b = 0;
      for (let sy = 0; sy < ss; sy++) {
        for (let sx = 0; sx < ss; sx++) {
          const u = px + (sx + 0.5) / ss;
          const v = py + (sy + 0.5) / ss;
          const ray = traceBackward(intrinsics, port, u, v);
          const hit = rayPlaneIntersect(ray.origin, ray.dir, plane);
          const c = hit ? sampleScene(scene, hit) : { r: 8, g: 24, b: 34 };
          r += c.r; g += c.g; b += c.b;
        }
      }
      const o = (py * width + px) * 4;
      out[o] = r * invSs2;
      out[o + 1] = g * invSs2;
      out[o + 2] = b * invSs2;
      out[o + 3] = 255;
    }
  }
  return { width, height, data: out };
}

export interface DetectedCorner {
  r: number;
  c: number;
  u: number;
  v: number;
}

/**
 * 生成棋盘格角点的亚像素观测（模拟角点检测）。
 * 角点 3D 位置由板位姿严格定义，经含折射的前向投影得到像素。
 */
export function detectCheckerboardCorners(scene: SyntheticScene): DetectedCorner[] {
  const out: DetectedCorner[] = [];
  const { rows, cols, squareMm, origin, ex, ey } = scene.board;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const local = boardPoint(rows, cols, squareMm, r, c);
      const X = add(origin, add(scale(ex, local.x), scale(ey, local.y)));
      const p = projectPoint(scene.intrinsics, scene.port, X);
      if (p.u >= 0 && p.u < scene.width && p.v >= 0 && p.v < scene.height) {
        out.push({ r, c, u: p.u, v: p.v });
      }
    }
  }
  return out;
}

/** 比例尺两端点（可见段）的像素位置，供测量流程测试 */
export function visibleBarEndpointsPx(scene: SyntheticScene, barIndex = 0) {
  const bar = scene.scaleBars[barIndex];
  const axis = sub(bar.end, bar.start);
  const L = norm(axis);
  const dir = scale(axis, 1 / L);
  const visStart = add(bar.start, scale(dir, L * bar.buriedFraction));
  const pA = projectPoint(scene.intrinsics, scene.port, visStart);
  const pB = projectPoint(scene.intrinsics, scene.port, bar.end);
  return { a: pA, b: pB, visibleLengthMm: L * (1 - bar.buriedFraction) };
}

/** 器物圆盘边缘两点（给定直径方向）的像素位置 */
export function artifactDiameterPx(scene: SyntheticScene, artIndex: number, angleRad: number) {
  const art = scene.artifacts[artIndex];
  const d = v3(Math.cos(angleRad), Math.sin(angleRad), 0);
  const p1 = projectPoint(scene.intrinsics, scene.port, add(art.center, scale(d, art.radiusMm)));
  const p2 = projectPoint(scene.intrinsics, scene.port, sub(art.center, scale(d, art.radiusMm)));
  return { p1, p2, diameterMm: art.radiusMm * 2 };
}
