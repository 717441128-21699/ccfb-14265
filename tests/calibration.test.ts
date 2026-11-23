// 标定闭环测试: 合成多视图棋盘格(真实折射几何) -> LM 平差 -> 参数恢复 + 重投影误差
// nWater 由水温/盐度给定(固定), 拟合 内参+畸变+dAir —— 避免焦距↔折射率简并
import { describe, it, expect } from 'vitest';
import { calibrate } from '../src/calibration';
import { TRUE_K, TRUE_HOUSING, BOARD, IMG_W, IMG_H, makePoses, makeViews } from './fixtures';

describe('水下折射标定(合成数据闭环)', () => {
  const poses = makePoses(7);
  const views = makeViews(poses, 0.15); // 0.15px 角点噪声
  const result = calibrate(BOARD, views, {
    nGlass: 1.52, tGlass: 4, imageWidth: IMG_W, imageHeight: IMG_H,
    initNWater: TRUE_HOUSING.nWater, // 现场由水温盐度给定
  });

  it('重投影 RMS < 0.5 像素(所里指标)', () => {
    expect(result.rms).toBeLessThan(0.5);
    // 噪声 σ=0.15 时 RMS 应接近噪声水平
    expect(result.rms).toBeLessThan(0.3);
  }, 120000);

  it('焦距恢复到 ±1%', () => {
    expect(Math.abs(result.K.fx - TRUE_K.fx) / TRUE_K.fx).toBeLessThan(0.01);
    expect(Math.abs(result.K.fy - TRUE_K.fy) / TRUE_K.fy).toBeLessThan(0.01);
  });

  it('主点恢复到 ±5 像素', () => {
    expect(Math.abs(result.K.cx - TRUE_K.cx)).toBeLessThan(5);
    expect(Math.abs(result.K.cy - TRUE_K.cy)).toBeLessThan(5);
  });

  it('界面距离 dAir 恢复到 ±25mm(弱可观; 组合测量精度由端到端测试保证)', () => {
    expect(Math.abs(result.dAir - TRUE_HOUSING.dAir)).toBeLessThan(25);
  });

  it('无噪声时 RMS 收敛到近零且参数精确恢复', () => {
    const clean = makeViews(poses, 0, 99);
    const r0 = calibrate(BOARD, clean, {
      nGlass: 1.52, tGlass: 4, imageWidth: IMG_W, imageHeight: IMG_H,
      initNWater: TRUE_HOUSING.nWater,
    });
    expect(r0.rms).toBeLessThan(0.01);
    expect(Math.abs(r0.K.fx - TRUE_K.fx) / TRUE_K.fx).toBeLessThan(0.002);
    expect(Math.abs(r0.dAir - TRUE_HOUSING.dAir)).toBeLessThan(3);
  }, 120000);

  it('nWater 输入偏差 ±0.003 时标定仍收敛(残差略升)', () => {
    const off = calibrate(BOARD, views, {
      nGlass: 1.52, tGlass: 4, imageWidth: IMG_W, imageHeight: IMG_H,
      initNWater: TRUE_HOUSING.nWater + 0.003,
    });
    expect(off.rms).toBeLessThan(0.5);
  }, 120000);

  it('拟合 nWater(带先验 σ=0.003)也能恢复', () => {
    const fit = calibrate(BOARD, views, {
      nGlass: 1.52, tGlass: 4, imageWidth: IMG_W, imageHeight: IMG_H,
      initNWater: 1.333, fitNWater: true, nWaterSigma: 0.003, dAirSigma: 15,
    });
    expect(fit.rms).toBeLessThan(0.5);
    expect(Math.abs(fit.nWater - TRUE_HOUSING.nWater)).toBeLessThan(0.01);
  }, 120000);
});
