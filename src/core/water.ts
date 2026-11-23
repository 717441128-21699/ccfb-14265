/**
 * 海水/淡水折射率：Quan & Fry (1995) 经验公式（波长以 nm 代入的系数形式）。
 * T 单位 °C，S 单位 PSU(‰)，λ 单位 nm。
 * 适用 T∈[0,30]、S∈[0,35]、λ∈[400,700]，精度 ~1e-5。
 *
 * 参考量级：20°C 淡水 589nm ≈ 1.3330；15°C S=30 海水 ≈ 1.3389。
 */

/** 折射率，λ 默认 589nm（钠 D 线） */
export function waterRefractiveIndex(
  temperatureC: number,
  salinityPSU: number,
  wavelengthNm = 589,
): number {
  const T = temperatureC;
  const S = salinityPSU;
  const L = wavelengthNm;
  const base = 1.31405 - 2.02e-6 * T * T;
  const salinityTerm = (1.779e-4 - 1.05e-6 * T) * S;
  const wavelengthTerm =
    (15.868 - 0.00423 * T + 0.01155 * S) / L -
    4382 / (L * L) +
    1.1455e6 / (L * L * L);
  return base + salinityTerm + wavelengthTerm;
}

/** 常用光学玻璃（BK7 类）在参考波长下的折射率；也允许用户直接填实测值 */
export const DEFAULT_GLASS_N = 1.517;

export interface WaterConditions {
  temperatureC: number;
  salinityPSU: number;
  wavelengthNm?: number;
}

/** 由潜水环境生成完整折射率组 */
export function refractiveIndices(cond: WaterConditions): { nAir: number; nGlass: number; nWater: number } {
  return {
    nAir: 1.000293,
    nGlass: DEFAULT_GLASS_N,
    nWater: waterRefractiveIndex(
      cond.temperatureC,
      cond.salinityPSU,
      cond.wavelengthNm ?? 589,
    ),
  };
}
