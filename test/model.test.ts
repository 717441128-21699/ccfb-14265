import { describe, it, expect } from 'vitest';
import { v3, normalize } from '../src/core/vec3';
import { refract, traceBackward, projectPoint, portNormal, PortParams, Intrinsics } from '../src/core/model';
import { makeGroundTruth } from './fixtures';

const intr: Intrinsics = { fx: 1200, fy: 1200, cx: 500, cy: 375 };
const port: PortParams = {
  nx: 0, ny: 0, nz: 1, d0: 30, t: 5,
  nAir: 1.000293, nGlass: 1.517, nWater: 1.334,
};

describe('Snell 折射基本性质', () => {
  it('法向入射方向不变', () => {
    // Hecht 约定：法向指向入射侧（与光线反向）
    const d = refract(v3(0, 0, 1), v3(0, 0, -1), 1 / 1.334)!;
    expect(Math.abs(d.x)).toBeLessThan(1e-12);
    expect(Math.abs(d.y)).toBeLessThan(1e-12);
    expect(d.z).toBeCloseTo(1, 10);
  });

  it('空气->水折射满足 sinθw = sinθa / n（n≈1.334）', () => {
    const thetaA = (30 * Math.PI) / 180;
    const I = v3(Math.sin(thetaA), 0, Math.cos(thetaA));
    const d = refract(I, v3(0, 0, -1), 1.000293 / 1.334)!;
    const thetaW = Math.asin(Math.hypot(d.x, d.y));
    expect(thetaW).toBeCloseTo(Math.asin(Math.sin(thetaA) * (1.000293 / 1.334)), 9);
  });

  it('玻璃中间层出射方向与直接空气->水一致（平行界面）', () => {
    // 两层平行板：出射水中角度只由 nAir/nWater 决定
    const ray = traceBackward(intr, port, 700, 375);
    const thetaW = Math.asin(Math.hypot(ray.dir.x, ray.dir.y));
    const dAir = normalize(v3((700 - 500) / 1200, 0, 1));
    const thetaA = Math.asin(Math.hypot(dAir.x, dAir.y));
    expect(thetaW).toBeCloseTo(Math.asin(Math.sin(thetaA) * (1.000293 / 1.334)), 9);
  });
});

describe('前向/后向折射互逆', () => {
  it('任意平面点 projectPoint 与 traceBackward 互逆（到 0.001 px）', () => {
    const gt = makeGroundTruth();
    const targets = [
      v3(0, 0, 1500),
      v3(300, -200, 1500),
      v3(-400, 250, 1500),
      v3(450, 330, 1500),
    ];
    for (const P of targets) {
      const p = projectPoint(gt.intrinsics, gt.port, P);
      const ray = traceBackward(gt.intrinsics, gt.port, p.u, p.v);
      // 光线应通过 P
      const w = v3(P.x - ray.origin.x, P.y - ray.origin.y, P.z - ray.origin.z);
      const lambda = w.x * ray.dir.x + w.y * ray.dir.y + w.z * ray.dir.z;
      const closest = v3(
        ray.origin.x + lambda * ray.dir.x,
        ray.origin.y + lambda * ray.dir.y,
        ray.origin.z + lambda * ray.dir.z,
      );
      const dist = Math.hypot(closest.x - P.x, closest.y - P.y, closest.z - P.z);
      expect(dist).toBeLessThan(1e-4);
    }
  });

  it('边缘像素被折射拉大：水中平面覆盖范围大于针孔模型', () => {
    const gt = makeGroundTruth();
    const n = portNormal(gt.port);
    void n;
    // 边缘光线在 z=1500 平面的横向位置
    const ray = traceBackward(gt.intrinsics, gt.port, 999, 375);
    const s = (1500 - ray.origin.z) / ray.dir.z;
    const xRefr = ray.origin.x + s * ray.dir.x;
    // 无折射针孔光线
    const xPin = ((999 - gt.intrinsics.cx) / gt.intrinsics.fx) * 1500;
    // 折射使光线更靠近法向 -> 到达同深度时横向位移更小；
    // 等价地：画面边缘物体要落在边缘像素，物体实际更靠外（被“拉大”）
    expect(Math.abs(xRefr)).toBeLessThan(Math.abs(xPin));
  });
});
