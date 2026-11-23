// 合成测试图: 按真实折射几何(两层平面界面光线追踪)正向渲染
// 每个像素: 去畸变 -> 空气斜率 -> traceForward 折射 -> 与棋盘格平面求交 -> 图案着色
// 与标定/校正使用完全相同的物理模型, 因此可用于闭环精度自检。
import { V3, rodrigues, mat3MulVec } from './linalg';
import { Housing, Intrinsics, traceForward, undistortPixel, projectPoint } from './refraction';
import { BoardSpec, Pose } from './calibration';

export interface SynthOptions {
  width: number;
  height: number;
  K: Intrinsics;
  housing: Housing;
  pose: Pose;            // 棋盘格(或测试图案平面)相对相机的位姿
  board: BoardSpec;      // 图案几何(棋盘格时为内角点规格)
  pattern: 'checker' | 'grid';
  noiseSigma?: number;   // 高斯噪声 (0-255 灰度级), 默认 0
  background?: number;   // 板外背景灰度 0-255, 默认 90
}

function patternValue(
  pattern: 'checker' | 'grid', board: BoardSpec, X: number, Y: number,
): number {
  const s = board.square;
  const W = (board.cols + 1) * s;
  const H = (board.rows + 1) * s;
  if (X < 0 || Y < 0 || X >= W || Y >= H) return -1; // 板外
  if (pattern === 'checker') {
    const ix = Math.floor(X / s);
    const iy = Math.floor(Y / s);
    return (ix + iy) % 2 === 0 ? 235 : 20;
  }
  // grid: 每格一条线, 线宽 1.5mm, 用于目视检验校正后直线是否恢复笔直
  const lw = 1.5;
  const fx = X % s;
  const fy = Y % s;
  const onLine = fx < lw || fy < lw || s - fx < lw || s - fy < lw;
  return onLine ? 20 : 235;
}

/** 渲染一帧合成图, 返回 RGBA 像素 */
export function renderSynth(opts: SynthOptions): ImageData {
  const { width, height, K, housing, pose, board, pattern } = opts;
  const bg = opts.background ?? 90;
  const sigma = opts.noiseSigma ?? 0;
  const R = rodrigues(pose.rvec);
  const t = pose.t;
  // 板平面(相机系): n·x + d = 0
  const n = mat3MulVec(R, [0, 0, 1]);
  const d = -(n[0] * t[0] + n[1] * t[1] + n[2] * t[2]);
  // 相机系点 -> 板系: pB = R^T (pC - t)
  const RT: number[] = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];

  const img = new ImageData(width, height);
  const data = img.data;
  // Box-Muller
  let spare: number | null = null;
  const gauss = (): number => {
    if (sigma === 0) return 0;
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    v = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u)) * sigma;
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [a, b] = undistortPixel(K, x + 0.5, y + 0.5);
      const ray = traceForward(housing, a, b);
      let gray = bg;
      if (ray) {
        const denom = n[0] * ray.w[0] + n[1] * ray.w[1] + n[2] * ray.w[2];
        if (Math.abs(denom) > 1e-12) {
          const tt = -(n[0] * ray.q[0] + n[1] * ray.q[1] + n[2] * ray.q[2] + d) / denom;
          if (tt > 0) {
            const pC: V3 = [ray.q[0] + ray.w[0] * tt, ray.q[1] + ray.w[1] * tt, ray.q[2] + ray.w[2] * tt];
            const rel: V3 = [pC[0] - t[0], pC[1] - t[1], pC[2] - t[2]];
            const X = RT[0] * rel[0] + RT[1] * rel[1] + RT[2] * rel[2];
            const Y = RT[3] * rel[0] + RT[4] * rel[1] + RT[5] * rel[2];
            const val = patternValue(pattern, board, X, Y);
            gray = val >= 0 ? val : bg;
          }
        }
      }
      const g = Math.max(0, Math.min(255, gray + gauss()));
      const idx = 4 * (y * width + x);
      data[idx] = g;
      data[idx + 1] = g;
      data[idx + 2] = g;
      data[idx + 3] = 255;
    }
  }
  return img;
}

/** 直接计算合成视图的角点像素真值(不经图像渲染, 供数值标定测试) */
export function synthCornerPixels(
  K: Intrinsics, housing: Housing, pose: Pose, boardPts: V3[],
): [number, number][] {
  const R = rodrigues(pose.rvec);
  const out: [number, number][] = [];
  for (const p of boardPts) {
    const pc = mat3MulVec(R, p);
    const P: V3 = [pc[0] + pose.t[0], pc[1] + pose.t[1], pc[2] + pose.t[2]];
    const uv = projectPoint(housing, K, P);
    if (!uv) throw new Error('合成角点投影失败(点在视野外?)');
    out.push(uv);
  }
  return out;
}

// ---------- 自检用: 合成位姿与带噪声视图 ----------

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussNoise(rnd: () => number, sigma: number): number {
  const u = Math.max(rnd(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()) * sigma;
}

/** 生成一组保证全部角点落在画面内的标定位姿(围绕板中心、不同距离与倾角) */
export function synthPoses(
  board: BoardSpec, K: Intrinsics, housing: Housing,
  imgW: number, imgH: number, count: number, seed = 42,
): Pose[] {
  const rnd = mulberry32(seed);
  const pts: V3[] = [];
  for (let j = 0; j < board.rows; j++) {
    for (let i = 0; i < board.cols; i++) pts.push([(i + 1) * board.square, (j + 1) * board.square, 0]);
  }
  const cxB = ((board.cols + 1) * board.square) / 2;
  const cyB = ((board.rows + 1) * board.square) / 2;
  const poses: Pose[] = [];
  let guard = 0;
  while (poses.length < count && guard++ < 8000) {
    const pose: Pose = {
      rvec: [(rnd() - 0.5) * 0.9, (rnd() - 0.5) * 0.9, (rnd() - 0.5) * 0.6],
      t: [
        -cxB + (rnd() - 0.5) * cxB * 1.6,
        -cyB + (rnd() - 0.5) * cyB * 1.6,
        700 + rnd() * 400,
      ],
    };
    try {
      const px = synthCornerPixels(K, housing, pose, pts);
      if (px.every(([u, v]) => u > 40 && u < imgW - 40 && v > 40 && v < imgH - 40)) poses.push(pose);
    } catch { /* 视野外, 重采样 */ }
  }
  if (poses.length < count) throw new Error('无法生成足够的有效位姿, 请调整参数');
  return poses;
}
