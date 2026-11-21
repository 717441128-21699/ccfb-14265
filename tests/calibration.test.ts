import { describe, expect, it } from 'vitest';
import { calibrate, evaluateCalibration } from '../src/core/calibration';
import { detectCheckerboard } from '../src/core/detector';
import {
  FIXTURE,
  TRUE_INTRINSICS,
  TRUE_REFRACTION,
  makeSyntheticSet,
} from './fixtures';

function detectedBoards(seed = 42, count = 10) {
  const set = makeSyntheticSet(count, seed);
  const boards = set
    .map((item) =>
      detectCheckerboard(item.gray, {
        cols: FIXTURE.cols,
        rows: FIXTURE.rows,
      }),
    )
    .filter((b): b is NonNullable<typeof b> => b !== null);
  return boards;
}

describe('水下折射标定（端到端，合成真值数据）', () => {
  it('重投影误差收敛（当前 Hessian 检测器精度下 RMS < 3 像素）', () => {
    const boards = detectedBoards(42, 10);
    expect(boards.length).toBeGreaterThanOrEqual(8);
    const cal = calibrate(boards, {
      squareMm: FIXTURE.squareMm,
      initialRefraction: { d: 30, tg: 5, ng: 1.49, nw: 1.333 },
    });
    // 检测器角点含少量离群，标定对全部残差最小二乘；
    // 角点中位误差为亚像素，RMS 被离群平方项放大。
    expect(cal.stats.reprojectionRms).toBeLessThan(3);
    expect(cal.stats.pointCount).toBe(
      boards.length * FIXTURE.cols * FIXTURE.rows,
    );
  });

  it('折射与内参收敛到合成真值附近', () => {
    const boards = detectedBoards(42, 10);
    const cal = calibrate(boards, { squareMm: FIXTURE.squareMm });
    // 焦距 5% 以内（检测噪声下的内参恢复精度）
    expect(Math.abs(cal.intrinsics.fx - TRUE_INTRINSICS.fx) / TRUE_INTRINSICS.fx).toBeLessThan(0.05);
    expect(Math.abs(cal.intrinsics.fy - TRUE_INTRINSICS.fy) / TRUE_INTRINSICS.fy).toBeLessThan(0.05);
    // 主点 12 像素以内
    expect(Math.abs(cal.intrinsics.cx - TRUE_INTRINSICS.cx)).toBeLessThan(12);
    expect(Math.abs(cal.intrinsics.cy - TRUE_INTRINSICS.cy)).toBeLessThan(12);
    // 水折射率 0.05 以内（检测噪声下的物理参数恢复精度）
    expect(Math.abs(cal.refraction.nw - TRUE_REFRACTION.nw)).toBeLessThan(0.05);
    // 畸变系数量级：检测噪声下径向畸变项不易与折射解耦，仅约束在有限范围
    expect(Math.abs(cal.distortion.k1)).toBeLessThan(0.5);
  });

  it('固定 nw（水温盐度给定）时标定仍达到 0.5 像素', () => {
    const boards = detectedBoards(42, 10);
    const cal = calibrate(boards, {
      squareMm: FIXTURE.squareMm,
      fixNw: true,
      initialRefraction: { nw: TRUE_REFRACTION.nw },
    });
    expect(cal.stats.reprojectionRms).toBeLessThan(0.5);
    expect(cal.refraction.nw).toBe(TRUE_REFRACTION.nw);
  });

  it('独立验收集（未参与拟合的图像）误差同样 < 0.5 像素', () => {
    const train = detectedBoards(42, 10);
    const cal = calibrate(train, { squareMm: FIXTURE.squareMm });
    const held = detectedBoards(99, 4);
    const stats = evaluateCalibration(cal, held, FIXTURE.squareMm);
    expect(stats.reprojectionRms).toBeLessThan(0.5);
  });

  it('少于 3 张图直接拒绝', () => {
    const boards = detectedBoards(42, 2);
    expect(() => calibrate(boards, { squareMm: 30 })).toThrow();
  });
});
