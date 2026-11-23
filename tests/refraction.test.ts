// 折射模型单元测试: Snell 定律、可逆性、投影/校正映射闭环
import { describe, it, expect } from 'vitest';
import {
  Housing, Intrinsics, traceForward, unrefract, projectPoint, projectPointEx,
  correctPixel, correctedPixelToRay,
} from '../src/refraction';
import { V3, norm3 } from '../src/linalg';

const housing: Housing = { nGlass: 1.52, tGlass: 4, dAir: 28, nWater: 1.339 };
const K: Intrinsics = { fx: 2400, fy: 2400, cx: 1500.4, cy: 1000.2, k1: -0.08, k2: 0.012 };
const Kv: Intrinsics = { fx: 2400, fy: 2400, cx: 1500, cy: 1000, k1: 0, k2: 0 };

describe('折射几何', () => {
  it('满足 Snell 定律 n_air·sinθ_air = n_water·sinθ_water', () => {
    for (const [a, b] of [[0.1, 0.05], [0.3, -0.2], [-0.25, 0.35], [0.45, 0.4]] as const) {
      const ray = traceForward(housing, a, b)!;
      expect(ray).not.toBeNull();
      const sinAir = Math.hypot(a, b) / Math.hypot(a, b, 1);
      const sinWater = Math.hypot(ray.w[0], ray.w[1]);
      expect(sinWater * housing.nWater).toBeCloseTo(sinAir, 10);
    }
  });

  it('玻璃只平移光线: 出射点随玻璃厚度平移, 方向不变', () => {
    const thin = traceForward({ ...housing, tGlass: 0.001 }, 0.3, 0.1)!;
    const thick = traceForward({ ...housing, tGlass: 12 }, 0.3, 0.1)!;
    for (let i = 0; i < 3; i++) expect(thick.w[i]).toBeCloseTo(thin.w[i], 10);
    expect(Math.abs(thick.q[0] - thin.q[0])).toBeGreaterThan(0.1); // 横向平移
    expect(thick.q[2]).toBeCloseTo(housing.dAir + 12, 10);
  });

  it('反折射是正向追踪的逆(方向)', () => {
    for (const [a, b] of [[0.05, 0.1], [0.4, -0.3], [-0.35, -0.25]] as const) {
      const ray = traceForward(housing, a, b)!;
      const ab = unrefract(housing, ray.w)!;
      expect(ab[0]).toBeCloseTo(a, 10);
      expect(ab[1]).toBeCloseTo(b, 10);
    }
  });

  it('超过全反射临界角的水中方向返回 null', () => {
    // 临界角 arcsin(1/1.339) ≈ 48.35°; 取 60° 方向
    const w: V3 = [Math.sin(Math.PI / 3), 0, Math.cos(Math.PI / 3)];
    expect(unrefract(housing, w)).toBeNull();
  });

  it('正向投影: 投影点落在其光线上, 且像素与光线方向一致', () => {
    const P: V3 = [120, -80, 900];
    const r = projectPointEx(housing, K, P)!;
    // 光线 q + t·w 应穿过 P
    const t = (P[2] - r.ray.q[2]) / r.ray.w[2];
    const hit: V3 = [r.ray.q[0] + r.ray.w[0] * t, r.ray.q[1] + r.ray.w[1] * t, P[2]];
    expect(norm3([hit[0] - P[0], hit[1] - P[1], 0])).toBeLessThan(1e-6);
    // 与 projectPoint 一致
    const uv = projectPoint(housing, K, P)!;
    expect(uv[0]).toBeCloseTo(r.u, 12);
    expect(uv[1]).toBeCloseTo(r.v, 12);
  });

  it('校正映射闭环: 校正像素 -> 光线 -> 取点 -> 正向投影 = 原图采样位置', () => {
    for (const [u, v] of [[200, 300], [1500, 1000], [2800, 1800], [500, 1700]] as const) {
      const src = correctPixel(housing, K, Kv, u, v);
      expect(src).not.toBeNull();
      const ray = correctedPixelToRay(housing, Kv, u, v)!;
      const t = (900 - ray.q[2]) / ray.w[2];
      const P: V3 = [ray.q[0] + ray.w[0] * t, ray.q[1] + ray.w[1] * t, 900];
      const uv = projectPoint(housing, K, P)!;
      expect(uv[0]).toBeCloseTo(src![0], 6);
      expect(uv[1]).toBeCloseTo(src![1], 6);
    }
  });
});
