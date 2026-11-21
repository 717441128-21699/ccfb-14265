import { describe, expect, it } from 'vitest';
import { buildRemap, measure, mmPerPixelAtDepth } from '../src/core/remap';
import { sampleBilinearRgba } from '../src/core/image';
import { synthesizeCheckerboard } from '../src/core/synthetic';
import { undistortNormalized } from '../src/core/camera';
import {
  FIXTURE,
  TRUE_DISTORTION,
  TRUE_INTRINSICS,
  TRUE_REFRACTION,
} from './fixtures';
import { frontoParallelPose } from '../src/core/synthetic';
import type { ScaleBar } from '../src/core/types';

const DEPTH = 850;

function correctedSample(
  mapX: Float32Array,
  mapY: Float32Array,
  valid: Uint8Array,
  src: ReturnType<typeof synthesizeCheckerboard>,
  outX: number,
  outY: number,
  w: number,
): number {
  const idx = outY * w + outX;
  expect(valid[idx]).toBe(1);
  const px = new Uint8ClampedArray(4);
  sampleBilinearRgba(src, mapX[idx], mapY[idx], px);
  return px[0];
}

describe('折射校正查表', () => {
  it('参考平面上的棋盘格角点校正后落在理想针孔位置（<1% 尺度误差）', () => {
    // 正视姿态：板与参考深度平面重合，校正几何严格成立
    const pose = frontoParallelPose(DEPTH);
    const src = synthesizeCheckerboard({
      width: FIXTURE.width,
      height: FIXTURE.height,
      intrinsics: TRUE_INTRINSICS,
      distortion: TRUE_DISTORTION,
      refraction: TRUE_REFRACTION,
      pose,
      cols: FIXTURE.cols,
      rows: FIXTURE.rows,
      squareMm: FIXTURE.squareMm,
      seed: 5,
    });
    const table = buildRemap(
      FIXTURE.width,
      FIXTURE.height,
      TRUE_INTRINSICS,
      TRUE_DISTORTION,
      TRUE_REFRACTION,
      DEPTH,
      { gridStep: 16 },
    );

    // 闭环判据：remap 给出的源像素经去畸变后，其归一化射线应与该输出
    // 像素的虚拟针孔射线一致（折射改变平面点像素位置，不能直接与针孔
    // 几何投影比较）。
    const samplePoints: [number, number][] = [
      [360, 270],
      [460, 270],
      [360, 170],
      [600, 100],
      [200, 400],
    ];
    for (const [ox, oy] of samplePoints) {
      const idx = oy * FIXTURE.width + ox;
      expect(table.valid[idx]).toBe(1);
      const sx = table.mapX[idx];
      const sy = table.mapY[idx];
      const xd = (sx - TRUE_INTRINSICS.cx) / TRUE_INTRINSICS.fx;
      const yd = (sy - TRUE_INTRINSICS.cy) / TRUE_INTRINSICS.fy;
      const uvn = undistortNormalized(xd, yd, TRUE_DISTORTION);
      const uOut = (ox - TRUE_INTRINSICS.cx) / TRUE_INTRINSICS.fx;
      const vOut = (oy - TRUE_INTRINSICS.cy) / TRUE_INTRINSICS.fy;
      const err = Math.hypot(uvn.x - uOut, uvn.y - vOut);
      // 归一化坐标残差：0.02 约对应 1.6px，是 step16 双三次展开的工程精度
      expect(err).toBeLessThan(0.02);
    }
    // 有效区域占比 > 90%
    const validCount = table.valid.reduce((s, v) => s + v, 0);
    expect(validCount / table.valid.length).toBeGreaterThan(0.9);

    // 采样结果确实取到源图像素（不越界、非填充色）
    const g = correctedSample(
      table.mapX,
      table.mapY,
      table.valid,
      src,
      FIXTURE.width >> 1,
      FIXTURE.height >> 1,
      FIXTURE.width,
    );
    expect(g).toBeGreaterThan(0);
  });

  it('默认网格步长 16 与加密步长 8 的查表偏差 < 1 像素（双三次展开）', () => {
    const a = buildRemap(
      FIXTURE.width,
      FIXTURE.height,
      TRUE_INTRINSICS,
      TRUE_DISTORTION,
      TRUE_REFRACTION,
      DEPTH,
      { gridStep: 16 },
    );
    const b = buildRemap(
      FIXTURE.width,
      FIXTURE.height,
      TRUE_INTRINSICS,
      TRUE_DISTORTION,
      TRUE_REFRACTION,
      DEPTH,
      { gridStep: 8 },
    );
    let maxD = 0;
    for (let i = 0; i < a.mapX.length; i += 37) {
      if (a.valid[i] && b.valid[i]) {
        maxD = Math.max(
          maxD,
          Math.hypot(a.mapX[i] - b.mapX[i], a.mapY[i] - b.mapY[i]),
        );
      }
    }
    expect(maxD).toBeLessThan(2);
  });
});

describe('两点测量与残段比例尺', () => {
  it('无比例尺时按参考深度针孔当量，两点长度误差 < 1%', () => {
    // 虚拟针孔下，深度 850mm、fx 820，100mm 物体的像素长度
    const truth = 100;
    const px = (truth * TRUE_INTRINSICS.fx) / DEPTH;
    const m = measure(
      { x: 100, y: 100 },
      { x: 100 + px, y: 100 },
      TRUE_INTRINSICS,
      DEPTH,
    );
    expect(Math.abs(m.lengthMm - truth) / truth).toBeLessThan(0.01);
  });

  it('比例尺被泥沙盖住 50%，残段标定的像素当量仍给出 <1% 长度', () => {
    // 完整比例尺 200mm，在图上占 pxFull 像素；只剩 100mm 可见
    const pxPerMm = TRUE_INTRINSICS.fx / DEPTH;
    const visible: ScaleBar = {
      p1: { x: 0, y: 0 },
      p2: { x: 100 * pxPerMm, y: 0 },
      knownLengthMm: 100,
    };
    const objectTruth = 237;
    const m = measure(
      { x: 50, y: 300 },
      { x: 50 + objectTruth * pxPerMm, y: 300 },
      TRUE_INTRINSICS,
      DEPTH,
      visible,
    );
    expect(Math.abs(m.lengthMm - objectTruth) / objectTruth).toBeLessThan(
      0.01,
    );
  });

  it('mmPerPixelAtDepth 与焦距深度一致', () => {
    const v = mmPerPixelAtDepth(TRUE_INTRINSICS, DEPTH);
    expect(v).toBeCloseTo(DEPTH / TRUE_INTRINSICS.fx, 5);
  });
});
