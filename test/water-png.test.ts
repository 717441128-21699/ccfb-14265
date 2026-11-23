import { describe, it, expect } from 'vitest';
import { waterRefractiveIndex, refractiveIndices } from '../src/core/water';
import { encodePng } from '../src/core/png';

describe('Quan–Fry 海水折射率', () => {
  it('20°C 淡水 589nm 接近公认值 1.333', () => {
    const n = waterRefractiveIndex(20, 0, 589);
    expect(n).toBeGreaterThan(1.332);
    expect(n).toBeLessThan(1.334);
  });

  it('盐度升高折射率增大，温度升高折射率减小', () => {
    const base = waterRefractiveIndex(15, 30);
    expect(waterRefractiveIndex(15, 35)).toBeGreaterThan(base);
    expect(waterRefractiveIndex(25, 30)).toBeLessThan(base);
  });

  it('折射率组结构完整', () => {
    const r = refractiveIndices({ temperatureC: 10, salinityPSU: 32 });
    expect(r.nAir).toBeCloseTo(1.000293, 6);
    expect(r.nGlass).toBeGreaterThan(1.5);
    expect(r.nWater).toBeGreaterThan(1.33);
  });
});

describe('PNG 编码器', () => {
  it('输出合法 PNG 签名与 IHDR/IEND 结构', () => {
    const data = new Uint8ClampedArray(4 * 4 * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = (i * 7) & 0xff;
      data[i + 1] = (i * 13) & 0xff;
      data[i + 2] = (i * 29) & 0xff;
      data[i + 3] = 255;
    }
    const png = encodePng({ width: 4, height: 4, data });
    expect(png.length).toBeGreaterThan(100);
    expect(Array.from(png.subarray(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    // IEND 结尾
    expect(Array.from(png.subarray(png.length - 8, png.length - 4))).toEqual([73, 69, 78, 68]);
  });

  it('1000x750 合成图编码不抛错且尺寸合理', () => {
    const data = new Uint8ClampedArray(1000 * 750 * 4);
    const png = encodePng({ width: 1000, height: 750, data });
    // stored block 近似原始大小 + 头
    expect(png.length).toBeGreaterThan(1000 * 750 * 4 * 0.9);
  });
});
