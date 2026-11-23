import { describe, it, expect } from 'vitest';
import { calibrate, CalibInput, CalibView } from '../src/core/calibration';
import { detectCheckerboardCorners, renderSynthetic } from '../src/core/synthetic';
import { makeScene, makeGroundTruth } from './fixtures';

function buildInput(viewCount: number): CalibInput {
  const views: CalibView[] = [];
  const scene0 = makeScene(0);
  for (let i = 0; i < viewCount; i++) {
    const scene = makeScene(i);
    const corners = detectCheckerboardCorners(scene);
    views.push({ name: `view-${i}`, width: scene.width, height: scene.height, corners });
  }
  return {
    rows: scene0.board.rows,
    cols: scene0.board.cols,
    squareMm: scene0.board.squareMm,
    views,
    water: { temperatureC: 18, salinityPSU: 32, wavelengthNm: 589 },
    // 装配图给定的机械参数（真值）+ 空气中标定的焦距先验
    knownPort: { d0: 28, t: 6, nGlass: 1.517 },
    intrinsicsPrior: { fx: 1200, fy: 1200, lock: true },
  };
}

describe('多视角折射标定', () => {
  it('合成数据重投影 RMS < 0.5 px 且最大误差 < 0.5 px', () => {
    const input = buildInput(6);
    const result = calibrate(input, { maxIterations: 40 });
    expect(result.rms).toBeLessThan(0.5);
    expect(result.maxError).toBeLessThan(0.5);
    expect(result.converged).toBe(true);
  });

  it('恢复的内参与玻璃参数接近真值', () => {
    const gt = makeGroundTruth();
    const result = calibrate(buildInput(6), { maxIterations: 40 });
    const relErr = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);
    expect(relErr(result.intrinsics.fx, gt.intrinsics.fx)).toBeLessThan(0.01);
    expect(relErr(result.intrinsics.fy, gt.intrinsics.fy)).toBeLessThan(0.01);
    expect(relErr(result.intrinsics.cx, gt.intrinsics.cx)).toBeLessThan(0.01);
    expect(relErr(result.intrinsics.cy, gt.intrinsics.cy)).toBeLessThan(0.01);
    expect(relErr(result.port.d0, gt.port.d0)).toBeLessThan(0.05);
    expect(relErr(result.port.t, gt.port.t)).toBeLessThan(0.1);
    expect(relErr(result.port.nGlass, gt.port.nGlass)).toBeLessThan(0.02);
    // 法向倾斜角误差 < 0.3°
    const dot =
      result.port.nx * gt.port.nx + result.port.ny * gt.port.ny + result.port.nz * gt.port.nz;
    const angleDeg = (Math.acos(Math.min(1, dot)) * 180) / Math.PI;
    expect(angleDeg).toBeLessThan(0.3);
  });

  it('水温盐度变化时折射率随之变化（每次下水独立标定）', () => {
    const cold = calibrate(buildInput(6), { maxIterations: 20 });
    const inputWarm = buildInput(6);
    inputWarm.water = { temperatureC: 28, salinityPSU: 35, wavelengthNm: 589 };
    const warm = calibrate(inputWarm, { maxIterations: 20 });
    expect(warm.port.nWater).not.toBeCloseTo(cold.port.nWater, 6);
    expect(warm.port.nWater).toBeLessThan(cold.port.nWater); // 暖水折射率更低
  });

  it('视角不足 3 张时报错', () => {
    expect(() => calibrate(buildInput(2))).toThrow();
  });

  it('合成图可渲染且角点全部落在画面内', () => {
    const scene = makeScene(0);
    const img = renderSynthetic(scene, 1);
    expect(img.data.length).toBe(scene.width * scene.height * 4);
    const corners = detectCheckerboardCorners(scene);
    expect(corners.length).toBe(scene.board.rows * scene.board.cols);
    for (const k of corners) {
      expect(k.u).toBeGreaterThanOrEqual(0);
      expect(k.u).toBeLessThan(scene.width);
    }
  });
});
