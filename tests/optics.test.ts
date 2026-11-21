import { describe, expect, it } from 'vitest';
import { waterRefractiveIndex, WATER_PRESETS } from '../src/core/water';
import { traceAirRay, refract, projectPointNormalized } from '../src/core/optics';
import { v3, normalize } from '../src/core/linalg';
import type { RefractionParams } from '../src/core/types';

const baseRefr: RefractionParams = {
  d: 30,
  tg: 5,
  ng: 1.49,
  nw: 1.333,
  tiltX: 0,
  tiltY: 0,
};

describe('水折射率经验模型', () => {
  it('纯水 20°C 钠 D 线接近公认值 1.333', () => {
    const n = waterRefractiveIndex({
      temperatureC: 20,
      salinity: 0,
      wavelengthNm: 589,
    });
    expect(n).toBeCloseTo(1.333, 3);
  });

  it('蓝光（500nm）折射率高于红光（650nm），体现色散', () => {
    const blue = waterRefractiveIndex({
      temperatureC: 20,
      salinity: 0,
      wavelengthNm: 500,
    });
    const red = waterRefractiveIndex({
      temperatureC: 20,
      salinity: 0,
      wavelengthNm: 650,
    });
    expect(blue).toBeGreaterThan(red);
  });

  it('海水盐度升高折射率增大', () => {
    const fresh = waterRefractiveIndex({
      temperatureC: 20,
      salinity: 0,
      wavelengthNm: 589,
    });
    const sea = waterRefractiveIndex({
      temperatureC: 20,
      salinity: 35,
      wavelengthNm: 589,
    });
    expect(sea).toBeGreaterThan(fresh);
    // 35 PSU 盐度贡献标准值约 0.0097
    expect(sea - fresh).toBeGreaterThan(0.008);
    expect(sea - fresh).toBeLessThan(0.011);
  });

  it('温度升高折射率下降', () => {
    const cold = waterRefractiveIndex({
      temperatureC: 5,
      salinity: 35,
      wavelengthNm: 589,
    });
    const warm = waterRefractiveIndex({
      temperatureC: 28,
      salinity: 35,
      wavelengthNm: 589,
    });
    expect(warm).toBeLessThan(cold);
  });

  it('预设工况全部落在物理合理区间', () => {
    for (const p of WATER_PRESETS) {
      const n = waterRefractiveIndex({
        ...p.conditions,
        wavelengthNm: 589,
      });
      expect(n).toBeGreaterThan(1.3);
      expect(n).toBeLessThan(1.36);
    }
  });
});

describe('平面观察窗折射', () => {
  it('近轴光线方向几乎不变，仅产生平移', () => {
    const ray = traceAirRay(normalize(v3(0.001, 0, 1)), baseRefr);
    expect(ray).not.toBeNull();
    expect(ray!.direction.x).toBeCloseTo(0.001 / 1.333, 4);
    expect(ray!.direction.z).toBeGreaterThan(0.999);
  });

  it('出射水中射线满足 Snell 定律（切向波矢守恒）', () => {
    const incident = normalize(v3(0.4, 0.2, 1));
    const ray = traceAirRay(incident, baseRefr)!;
    const sinA = Math.hypot(incident.x, incident.y);
    const sinW = Math.hypot(ray.direction.x, ray.direction.y);
    expect(sinW * baseRefr.nw).toBeCloseTo(sinA, 5);
  });

  it('玻璃片只平移不改变出射方向：改厚度方向不变', () => {
    const r1 = traceAirRay(normalize(v3(0.3, 0.1, 1)), baseRefr)!;
    const r2 = traceAirRay(
      normalize(v3(0.3, 0.1, 1)),
      { ...baseRefr, tg: 20 },
    )!;
    expect(r2.direction.x).toBeCloseTo(r1.direction.x, 6);
    expect(r2.direction.y).toBeCloseTo(r1.direction.y, 6);
    expect(r2.origin.x).not.toBeCloseTo(r1.origin.x, 4);
  });

  it('nw 增大折射变强（同入射角水中方向更靠轴）', () => {
    const incident = normalize(v3(0.5, 0, 1));
    const rFresh = traceAirRay(incident, { ...baseRefr, nw: 1.33 })!;
    const rSalt = traceAirRay(incident, { ...baseRefr, nw: 1.345 })!;
    expect(Math.abs(rSalt.direction.x)).toBeLessThan(
      Math.abs(rFresh.direction.x),
    );
  });

  it('全反射时返回 null', () => {
    // 玻璃->水临界角 arcsin(nw/ng) ~ 63°，构造玻璃内掠射
    const t = refract(
      normalize(v3(0.95, 0, 0.31)),
      v3(0, 0, -1),
      1.49 / 1.333,
    );
    expect(t).toBeNull();
  });

  it('倾斜窗与共轴窗在倾角趋于 0 时结果一致', () => {
    const X = v3(60, -40, 900);
    const a = projectPointNormalized(X, baseRefr);
    const b = projectPointNormalized(X, {
      ...baseRefr,
      tiltX: 1e-9,
      tiltY: 1e-9,
    });
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    // 倾斜路径用有限差分（eps=1e-7），一致性到 1e-3 即收敛
    expect(Math.abs(b!.u - a!.u)).toBeLessThan(2e-3);
    expect(Math.abs(b!.v - a!.v)).toBeLessThan(2e-3);
  });
});
