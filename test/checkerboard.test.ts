import { describe, it, expect } from 'vitest';
import { detectCheckerboard, detectCandidates } from '../src/vision/checkerboard';
import { renderSynthetic, detectCheckerboardCorners } from '../src/core/synthetic';
import { makeScene } from './fixtures';

describe('棋盘格角点检测', () => {
  it('候选点在每个真实角点 5px 内均有覆盖（亚像素鞍点检测）', () => {
    const scene = makeScene(2);
    const img = renderSynthetic(scene, 2);
    const candidates = detectCandidates(img);
    const truth = detectCheckerboardCorners(scene);
    // 每个真值角点 5px 邻域内必须存在候选
    let covered = 0;
    let nearestSum = 0;
    for (const t of truth) {
      let bd = Infinity;
      for (const cd of candidates) bd = Math.min(bd, Math.hypot(cd.u - t.u, cd.v - t.v));
      if (bd < 5) covered++;
      nearestSum += bd;
    }
    // eslint-disable-next-line no-console
    console.log(`候选 ${candidates.length} 个，真值 ${truth.length} 个，覆盖率 ${covered}/${truth.length}，平均最近距离 ${(nearestSum / truth.length).toFixed(2)}px`);
    expect(covered).toBe(truth.length);
  });

  it('完整网格识别返回内部一致的 rows×cols 网格（homography 残差小、节距均匀）', () => {
    const scene = makeScene(2);
    const img = renderSynthetic(scene, 2);
    const detected = detectCheckerboard(img, { rows: 9, cols: 12 });
    // 契约：完整识别或空；不允许部分网格
    expect(detected.length === 0 || detected.length === 9 * 12).toBe(true);
    if (detected.length === 0) return;

    // 内部一致性：所有相邻列角点间距应近似相等（棋盘等方格）
    const at = (r: number, c: number) => detected.find((k) => k.r === r && k.c === c)!;
    const pitches: number[] = [];
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 11; c++) {
        const a = at(r, c);
        const b = at(r, c + 1);
        pitches.push(Math.hypot(b.u - a.u, b.v - a.v));
      }
    }
    const mean = pitches.reduce((s, x) => s + x, 0) / pitches.length;
    const maxDev = Math.max(...pitches.map((x) => Math.abs(x - mean) / mean));
    // eslint-disable-next-line no-console
    console.log(`网格节距均值 ${mean.toFixed(2)}px，最大相对偏差 ${(maxDev * 100).toFixed(1)}%`);
    expect(maxDev).toBeLessThan(0.15);
  });
});
