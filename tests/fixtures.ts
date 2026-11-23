// 测试公共装置: 真值参数、合成位姿与视图
import { Housing, Intrinsics } from '../src/refraction';
import { BoardSpec, ViewObs, boardPoints, Pose } from '../src/calibration';
import { synthCornerPixels } from '../src/synth';
import { V3 } from '../src/linalg';

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const TRUE_K: Intrinsics = { fx: 2400, fy: 2410, cx: 1500.4, cy: 1000.2, k1: -0.08, k2: 0.012 };
export const TRUE_HOUSING: Housing = { nGlass: 1.52, tGlass: 4, dAir: 28, nWater: 1.339 };
export const BOARD: BoardSpec = { cols: 9, rows: 6, square: 30 };
export const IMG_W = 3000;
export const IMG_H = 2000;

/** 生成一组保证全部角点落在画面内的位姿 */
export function makePoses(count: number, seed = 42): Pose[] {
  const rnd = mulberry32(seed);
  const pts = boardPoints(BOARD);
  const poses: Pose[] = [];
  let guard = 0;
  while (poses.length < count && guard++ < 5000) {
    const rx = (rnd() - 0.5) * 0.9;   // 约 ±26°
    const ry = (rnd() - 0.5) * 0.9;
    const rz = (rnd() - 0.5) * 0.6;
    const t: V3 = [
      -150 + (rnd() - 0.5) * 260,   // 板中心 x≈150
      -105 + (rnd() - 0.5) * 200,   // 板中心 y≈105
      700 + rnd() * 400,
    ];
    const pose: Pose = { rvec: [rx, ry, rz], t };
    try {
      const px = synthCornerPixels(TRUE_K, TRUE_HOUSING, pose, pts);
      const ok = px.every(([u, v]) => u > 40 && u < IMG_W - 40 && v > 40 && v < IMG_H - 40);
      if (ok) poses.push(pose);
    } catch {
      // 有点在视野外, 重新采样
    }
  }
  if (poses.length < count) throw new Error('无法生成足够的有效位姿');
  return poses;
}

export function gauss(rnd: () => number, sigma: number): number {
  const u = Math.max(rnd(), 1e-12);
  const v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * sigma;
}

export function makeViews(poses: Pose[], noiseSigma: number, seed = 7): ViewObs[] {
  const rnd = mulberry32(seed);
  const pts = boardPoints(BOARD);
  return poses.map((pose) => {
    const px = synthCornerPixels(TRUE_K, TRUE_HOUSING, pose, pts);
    return {
      pixels: px.map(([u, v]) => [u + gauss(rnd, noiseSigma), v + gauss(rnd, noiseSigma)] as [number, number]),
    };
  });
}
