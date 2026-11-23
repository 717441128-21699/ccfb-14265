import { describe, it, expect } from 'vitest';
import { calibrate, CalibInput, CalibView, CalibrationResult } from '../src/core/calibration';
import { detectCheckerboardCorners } from '../src/core/synthetic';
import { makeScene } from '../src/core/demoScene';
import { projectPoint } from '../src/core/model';

function buildInput(): CalibInput {
  const views: CalibView[] = [];
  const scene0 = makeScene(0);
  for (let i = 0; i < 6; i++) {
    const scene = makeScene(i);
    views.push({
      name: `v${i}`,
      width: scene.width,
      height: scene.height,
      corners: detectCheckerboardCorners(scene),
    });
  }
  return {
    rows: scene0.board.rows,
    cols: scene0.board.cols,
    squareMm: scene0.board.squareMm,
    views,
    water: { temperatureC: 18, salinityPSU: 32 },
    knownPort: { d0: 28, t: 6, nGlass: 1.517 },
    intrinsicsPrior: { fx: 1200, fy: 1200, lock: true },
  };
}

describe('标定结果序列化（目录持久化）', () => {
  it('JSON 往返后投影结果与原对象一致', () => {
    const result = calibrate(buildInput(), { maxIterations: 40 });
    const json = JSON.stringify(result);
    const restored = JSON.parse(json) as CalibrationResult;

    expect(restored.intrinsics.fx).toBeCloseTo(result.intrinsics.fx, 8);
    expect(restored.port.nWater).toBeCloseTo(result.port.nWater, 8);
    expect(restored.views.length).toBe(result.views.length);

    // 用恢复后的参数投影一个相机系点，结果一致
    const P = { x: 120, y: -80, z: 1500 };
    const p1 = projectPoint(result.intrinsics, result.port, P);
    const p2 = projectPoint(restored.intrinsics, restored.port, P);
    expect(p2.u).toBeCloseTo(p1.u, 6);
    expect(p2.v).toBeCloseTo(p1.v, 6);
  });

  it('每次下水（不同温盐）生成不同 nWater 且各自 RMS 达标', () => {
    const a = calibrate(buildInput(), { maxIterations: 30 });
    const inputB = buildInput();
    inputB.water = { temperatureC: 27, salinityPSU: 35 };
    const b = calibrate(inputB, { maxIterations: 30 });
    expect(a.port.nWater).not.toBeCloseTo(b.port.nWater, 6);
    expect(a.rms).toBeLessThan(0.5);
    expect(b.rms).toBeLessThan(0.5);
  });
});
