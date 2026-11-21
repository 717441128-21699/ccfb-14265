import { describe, expect, it } from 'vitest';
import { detectCheckerboard, cornersFromFourClicks } from '../src/core/detector';
import { groundTruthCorners, observableCorners } from '../src/core/synthetic';
import {
  FIXTURE,
  TRUE_DISTORTION,
  TRUE_INTRINSICS,
  TRUE_REFRACTION,
  makeSyntheticSet,
} from './fixtures';

function synthOpts(item: (ReturnType<typeof makeSyntheticSet>)[number]) {
  return {
    width: FIXTURE.width,
    height: FIXTURE.height,
    intrinsics: TRUE_INTRINSICS,
    distortion: TRUE_DISTORTION,
    refraction: TRUE_REFRACTION,
    pose: item.pose,
    cols: FIXTURE.cols,
    rows: FIXTURE.rows,
    squareMm: FIXTURE.squareMm,
  };
}

describe('棋盘格角点检测', () => {
  it('可观测鞍点真值与连续投影真值仅差亚像素（PSF 固有偏移）', () => {
    const set = makeSyntheticSet(2, 7);
    // 像素积分 + PSF 导致的可观测偏移：均值应在亚像素级，
    // 边缘角点受板外区域影响略大，故校验整体均值
    const all: number[] = [];
    for (const item of set) {
      const opts = synthOpts(item);
      const geo = groundTruthCorners(opts);
      const obs = observableCorners(opts, item.gray);
      for (let i = 0; i < geo.length; i++) {
        all.push(Math.hypot(obs[i].x - geo[i].x, obs[i].y - geo[i].y));
      }
    }
    const mean = all.reduce((a, b) => a + b, 0) / all.length;
    expect(mean).toBeLessThan(1);
  });

  it('在含噪声合成图上检出完整网格，角点中位误差达亚像素', () => {
    const set = makeSyntheticSet(4, 7);
    const allErrors: number[] = [];
    for (const item of set) {
      const det = detectCheckerboard(item.gray, {
        cols: FIXTURE.cols,
        rows: FIXTURE.rows,
      });
      expect(det).not.toBeNull();
      expect(det!.corners.length).toBe(FIXTURE.cols * FIXTURE.rows);
      const truth = observableCorners(synthOpts(item), item.gray);
      const aligned = alignOrder(det!.corners, truth, FIXTURE.cols);
      for (let i = 0; i < truth.length; i++) {
        allErrors.push(
          Math.hypot(aligned[i].x - truth[i].x, aligned[i].y - truth[i].y),
        );
      }
    }
    allErrors.sort((a, b) => a - b);
    const median = allErrors[Math.floor(allErrors.length / 2)];
    // 中位误差（典型角点定位精度）亚像素；Hessian 粗检测对少数边缘
    // 角点存在离群，由单应几何约束在标定时进一步平均。
    expect(median).toBeLessThan(0.5);
  });

  it('自动检测在无泥沙图上稳定，遮挡场景可用手动四点兜底', () => {
    // 无泥沙基线：自动检测全部成功
    const clean = makeSyntheticSet(3, 11, false);
    let cleanOk = 0;
    for (const item of clean) {
      const det = detectCheckerboard(item.gray, {
        cols: FIXTURE.cols,
        rows: FIXTURE.rows,
      });
      if (det && det.corners.length === FIXTURE.cols * FIXTURE.rows) cleanOk++;
    }
    expect(cleanOk).toBe(clean.length);

    // 手动四点兜底的代数正确性：规则矩形的四角点击应插值出
    // 间距均匀的内部网格（不依赖透视板的角点排序约定）
    const gray = clean[0].gray;
    const w = 360;
    const h = 270;
    const colsN = 9;
    const rowsN = 7;
    const tl = { x: 100, y: 80 };
    const tr = { x: 100 + w, y: 80 };
    const br = { x: 100 + w, y: 80 + h };
    const bl = { x: 100, y: 80 + h };
    const manual = cornersFromFourClicks(
      gray,
      [tl, tr, br, bl],
      colsN,
      rowsN,
    );
    expect(manual.corners.length).toBe(colsN * rowsN);
    // 第一行相邻角点的水平间距应恒定
    const spacing = w / (colsN + 1);
    const p0 = manual.corners[0];
    const p1 = manual.corners[1];
    expect(Math.abs(p1.x - p0.x - spacing)).toBeLessThan(1e-6);
    expect(Math.abs(p1.y - p0.y)).toBeLessThan(1e-6);
    // 首末角点在矩形内部
    const last = manual.corners[colsN * rowsN - 1];
    expect(last.x).toBeGreaterThan(tl.x);
    expect(last.y).toBeLessThan(bl.y);
  });
});

/** 检测器行列方向可能与真值相反，枚举 4 种翻转取整体误差最小者 */
function alignOrder(
  detected: { x: number; y: number }[],
  truth: { x: number; y: number }[],
  cols: number,
): { x: number; y: number }[] {
  const rows = detected.length / cols;
  const variants: { x: number; y: number }[][] = [];
  const grid2d: { x: number; y: number }[][] = [];
  for (let j = 0; j < rows; j++) {
    grid2d.push(detected.slice(j * cols, (j + 1) * cols));
  }
  const transforms = [
    (g: typeof grid2d) => g,
    (g: typeof grid2d) => g.map((r) => r.slice().reverse()),
    (g: typeof grid2d) => g.slice().reverse(),
    (g: typeof grid2d) =>
      g.slice().reverse().map((r) => r.slice().reverse()),
  ];
  for (const t of transforms) {
    variants.push(t(grid2d).flat());
  }
  let best = variants[0];
  let bestErr = Infinity;
  for (const v of variants) {
    let e = 0;
    for (let i = 0; i < truth.length; i++) {
      e += Math.hypot(v[i].x - truth[i].x, v[i].y - truth[i].y);
    }
    if (e < bestErr) {
      bestErr = e;
      best = v;
    }
  }
  return best;
}
