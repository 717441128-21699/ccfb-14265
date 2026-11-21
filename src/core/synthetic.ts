import { projectWorldPoint, checkerboardPoints } from './camera';
import { invertIntrinsics } from './camera';
import { airRayFromNormalized, rayPlane, traceAirRay } from './optics';
import { rodrigues, v3 } from './linalg';
import type {
  BrownDistortion,
  Intrinsics,
  Pose,
  RefractionParams,
  Vec3,
} from './types';
import { undistortNormalized } from './camera';
import type { GrayImage, RgbaImage } from './image';
import { sampleBilinearGray } from './image';

export interface SynthesisOptions {
  width: number;
  height: number;
  intrinsics: Intrinsics;
  distortion: BrownDistortion;
  refraction: RefractionParams;
  pose: Pose;
  cols: number;
  rows: number;
  squareMm: number;
  noiseSigma?: number;
  /** 泥沙覆盖：0..1，按比例随机生成斑块遮挡 */
  sedimentFraction?: number;
  seed?: number;
  scaleBar?: {
    /** 板坐标系下的条带中心点与长度（mm），条带为黑白相间 10 mm 格 */
    x: number;
    y: number;
    lengthMm: number;
    widthMm: number;
    /** 遮挡掉的比例 0..1（模拟被泥沙盖住一半） */
    occludeFraction?: number;
  };
}

/** 确定性 PRNG（mulberry32），保证测试可复现 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 逆映射合成：对每个像素反推它在棋盘格平面上的落点，
 * 完整经过 Brown 逆畸变 -> 空气射线 -> 两层 Snell 折射 -> 平面求交，
 * 因此输出图像包含与真实水下成像一致的折射几何。
 */
export function synthesizeCheckerboard(
  opts: SynthesisOptions,
): RgbaImage {
  const {
    width,
    height,
    intrinsics: K,
    distortion,
    refraction,
    pose,
    cols,
    rows,
    squareMm,
  } = opts;
  const rng = mulberry32(opts.seed ?? 1);
  const noiseSigma = opts.noiseSigma ?? 3;
  const data = new Uint8ClampedArray(width * height * 4);
  const invK = invertIntrinsics(K);

  // 板平面在相机系下的参数
  const normalCam = (() => {
    const R = pose.R;
    // 板法线 (0,0,1) 变换到相机系
    return v3(R[2], R[5], R[8]);
  })();
  const originCam = pose.t;

  const boardW = (cols - 1) * squareMm;
  const boardH = (rows - 1) * squareMm;

  const sedimentMap =
    opts.sedimentFraction && opts.sedimentFraction > 0
      ? makeSediment(width, height, opts.sedimentFraction, rng)
      : null;

  // 物理 PSF 模拟。关键：核必须关于像素中心严格对称且移不变，
  // 否则离散采样会使视觉鞍点随边缘相位漂移。采用 5x5 对称高斯核
  // （偏移 ±0.5、±1.5，正负权重相等），对连续棋盘着色做卷积，
  // 合成约 1.5–2px 的边缘过渡带，与真实相机 MTF 一致且保持临界点。
  const offsets = [-1.5, -0.5, 0.5, 1.5];
  const psfSigma = 1.0;
  const kernel1d = offsets.map(
    (d) => Math.exp(-(d * d) / (2 * psfSigma * psfSigma)),
  );
  const ksum1 = kernel1d.reduce((a, b) => a + b, 0);
  const gaussW = (idx: number) => kernel1d[idx] / ksum1;

  // 连续坐标（亚像素位置）下的棋盘着色，返回 0..1 的归一化亮度
  const shadeAtContinuous = (px: number, py: number): number => {
    const xd = invK[0] * px + invK[2];
    const yd = invK[4] * py + invK[5];
    const uvn = undistortNormalized(xd, yd, distortion);
    const rayAir = airRayFromNormalized(uvn.x, uvn.y);
    const water = traceAirRay(rayAir, refraction);
    if (!water) return 18 / 255;
    const hit = rayPlane(water.origin, water.direction, originCam, normalCam);
    if (!hit) return 18 / 255;
    const Rt = transpose3(pose.R);
    const local = applyR(
      Rt,
      v3(hit.x - pose.t.x, hit.y - pose.t.y, hit.z - pose.t.z),
    );
    const gx = local.x + boardW / 2;
    const gy = local.y + boardH / 2;
    if (
      gx < -squareMm * 0.5 ||
      gy < -squareMm * 0.5 ||
      gx > boardW + squareMm * 0.5 ||
      gy > boardH + squareMm * 0.5
    ) {
      return 18 / 255;
    }
    const cellIndex = (val: number) =>
      val >= 0
        ? Math.floor(val / squareMm)
        : Math.ceil(val / squareMm) - 1;
    const ix = cellIndex(gx);
    const iy = cellIndex(gy);
    const dark = (ix + iy) % 2 !== 0;
    let shade = dark ? 36 : 232;
    if (opts.scaleBar) {
      const bar = scaleBarShade(local, opts.scaleBar);
      if (bar !== null) shade = bar;
    }
    return shade / 255;
  };

  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      let shade = 0;
      for (let jy = 0; jy < offsets.length; jy++) {
        for (let ix = 0; ix < offsets.length; ix++) {
          const w = gaussW(ix) * gaussW(jy);
          shade +=
            shadeAtContinuous(px + offsets[ix], py + offsets[jy]) * w;
        }
      }
      shade = shade * 255;
      let g = shade + (rng() - 0.5) * 2 * noiseSigma;
      if (sedimentMap && sedimentMap[py * width + px]) {
        g = 96 + (rng() - 0.5) * 28; // 泥沙棕灰色
      }
      const idx = (py * width + px) * 4;
      // 水下轻微蓝绿色偏
      data[idx] = clamp8(g * 0.82);
      data[idx + 1] = clamp8(g * 0.95);
      data[idx + 2] = clamp8(g * 1.02);
      data[idx + 3] = 255;
    }
  }
  return { width, height, data };
}

/** 比例尺条带着色：返回灰度值；不在条带内返回 null */
function scaleBarShade(
  local: Vec3,
  bar: NonNullable<SynthesisOptions['scaleBar']>,
): number | null {
  const dx = local.x - bar.x;
  const dy = local.y - bar.y;
  const halfL = bar.lengthMm / 2;
  const halfW = bar.widthMm / 2;
  if (Math.abs(dy) > halfW || Math.abs(dx) > halfL) return null;
  const occ = bar.occludeFraction ?? 0;
  if (dx < -halfL + bar.lengthMm * occ) return 100; // 被泥沙覆盖段
  const cell = 10;
  const seg = Math.floor((dx + halfL) / cell);
  return seg % 2 === 0 ? 250 : 12;
}

function makeSediment(
  w: number,
  h: number,
  fraction: number,
  rng: () => number,
): Uint8Array {
  const mask = new Uint8Array(w * h);
  const blobs = Math.floor((w * h * fraction) / 900);
  for (let b = 0; b < blobs; b++) {
    const cx = rng() * w;
    const cy = rng() * h;
    const r = 8 + rng() * 34;
    const x0 = Math.max(0, Math.floor(cx - r));
    const x1 = Math.min(w - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(h - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.hypot(x - cx, y - cy);
        if (d < r * (0.7 + rng() * 0.4)) mask[y * w + x] = 1;
      }
    }
  }
  return mask;
}

function transpose3(A: number[]): number[] {
  return [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
}
function applyR(A: number[], x: Vec3): Vec3 {
  return v3(
    A[0] * x.x + A[1] * x.y + A[2] * x.z,
    A[3] * x.x + A[4] * x.y + A[5] * x.z,
    A[6] * x.x + A[7] * x.y + A[8] * x.z,
  );
}
function clamp8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/** 生成一组合成标定姿态（不同深度、倾角、朝向） */
export function makePoses(
  count: number,
  rng: () => number,
  baseDepth = 900,
): Pose[] {
  const poses: Pose[] = [];
  for (let i = 0; i < count; i++) {
    const rx = (rng() - 0.5) * 0.35;
    const ry = (rng() - 0.5) * 0.4;
    const rz = (rng() - 0.5) * 0.2;
    const R = rodrigues(rx, ry, rz);
    const depth = baseDepth + (rng() - 0.5) * 200;
    const t = v3((rng() - 0.5) * 120, (rng() - 0.5) * 100, depth);
    poses.push({ R, t });
  }
  return poses;
}

/** 正视姿态：板平面与像面平行、位于指定深度，用于校正查表的严格验证 */
export function frontoParallelPose(depthMm: number): Pose {
  return { R: rodrigues(0, 0, 0), t: v3(0, 0, depthMm) };
}

/** 生成已知角点的投影真值（连续几何位置，用于折射/投影链路核验） */
export function groundTruthCorners(
  opts: SynthesisOptions,
): { x: number; y: number }[] {
  const board = checkerboardPoints(opts.cols, opts.rows, opts.squareMm);
  return board
    .map((p) =>
      projectWorldPoint(
        p,
        opts.pose,
        opts.refraction,
        opts.intrinsics,
        opts.distortion,
      ),
    )
    .filter((p): p is { x: number; y: number } => p !== null);
}

/**
 * 可观测角点真值：在渲染图上把连续投影点精化到实际的梯度零点
 * （像素积分 + PSF 会使可观测鞍点相对连续几何位置偏移亚像素级，
 * 这是真实相机能测量到的位置，检测器应以此为精度基准）。
 */
export function observableCorners(
  opts: SynthesisOptions,
  gray: GrayImage,
): { x: number; y: number }[] {
  const projected = groundTruthCorners(opts);
  return projected.map((p) => refineToSaddle(gray, p));
}

function refineToSaddle(
  img: GrayImage,
  p: { x: number; y: number },
): { x: number; y: number } {
  const grad = (x: number, y: number): [number, number] => [
    sampleBilinearGray(img, x + 0.5, y) -
      sampleBilinearGray(img, x - 0.5, y),
    sampleBilinearGray(img, x, y + 0.5) -
      sampleBilinearGray(img, x, y - 0.5),
  ];
  // Saddle-constrained score. A checkerboard corner is a hyperbolic
  // critical point (Hessian det < 0); uniform regions and edges also
  // have near-zero gradient but positive/zero det, so they must carry a
  // dominant penalty. This locates the observable saddle rather than the
  // nearest arbitrary gradient zero.
  const score = (x: number, y: number): number => {
    const [gx, gy] = grad(x, y);
    const eps = 0.5;
    const hxx = (grad(x + eps, y)[0] - grad(x - eps, y)[0]) / (2 * eps);
    const hyy = (grad(x, y + eps)[1] - grad(x, y - eps)[1]) / (2 * eps);
    const hxy = (grad(x + eps, y)[1] - grad(x - eps, y)[1]) / (2 * eps);
    const det = hxx * hyy - hxy * hxy;
    const g2 = gx * gx + gy * gy;
    return det < 0 ? g2 : g2 + 1e6;
  };
  const search = (
    center: { x: number; y: number },
    radius: number,
    step: number,
  ) => {
    let best = { x: center.x, y: center.y, v: Infinity };
    for (let dy = -radius; dy <= radius + 1e-9; dy += step) {
      for (let dx = -radius; dx <= radius + 1e-9; dx += step) {
        const x = center.x + dx;
        const y = center.y + dy;
        const v = score(x, y);
        if (v < best.v) best = { x, y, v };
      }
    }
    return best;
  };
  // Small radius: the observable saddle stays within ~1px of the geometric
  // corner under a shift-invariant PSF; larger radii only invite non-saddle
  // gradient zeros. Coarse then fine grid for sub-0.1px localization.
  const coarse = search(p, 1.5, 0.2);
  const fine = search(coarse, 0.25, 0.05);
  if (Math.hypot(fine.x - p.x, fine.y - p.y) > 1.5) return p;
  return { x: fine.x, y: fine.y };
}

export function toGray(img: RgbaImage): GrayImage {
  const out = new Uint8ClampedArray(img.width * img.height);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.round(
      0.299 * img.data[i * 4] +
        0.587 * img.data[i * 4 + 1] +
        0.114 * img.data[i * 4 + 2],
    );
  }
  return { width: img.width, height: img.height, data: out };
}
