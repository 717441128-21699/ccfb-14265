import { describe, it, expect } from 'vitest';
import { levenbergMarquardt } from '../src/core/optimizer';

describe('LM 优化器单元验证', () => {
  it('线性最小二乘 y = a x + b 恢复真值', () => {
    const aTrue = 2.5;
    const bTrue = -1.3;
    const data = [0, 1, 2, 3, 4, 5].map((x) => ({ x, y: aTrue * x + bTrue }));
    // 残差 = 观测 - 模型
    const res = (p: number[]) => data.map((d) => d.y - (p[0] * d.x + p[1]));
    const out = levenbergMarquardt(res, [0, 0], { maxIterations: 50 });
    expect(out.params[0]).toBeCloseTo(aTrue, 6);
    expect(out.params[1]).toBeCloseTo(bTrue, 6);
    expect(out.converged).toBe(true);
  });

  it('非线性 Rosenbrock 收敛到 (1,1)', () => {
    const res = (p: number[]) => [1 - p[0], 10 * (p[1] - p[0] * p[0])];
    const out = levenbergMarquardt(res, [-1.2, 1], { maxIterations: 200 });
    expect(out.params[0]).toBeCloseTo(1, 4);
    expect(out.params[1]).toBeCloseTo(1, 4);
  });

  it('大尺度参数（平移上千）仍能收敛', () => {
    // 模型 y = t（单参数），两个观测都在 1500 附近
    const res = (p: number[]) => [1500 - p[0], 1502 - p[0]];
    const out = levenbergMarquardt(res, [1100], { maxIterations: 50 });
    expect(out.params[0]).toBeCloseTo(1501, 3);
  });
});
