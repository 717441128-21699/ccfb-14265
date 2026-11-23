import { describe, it, expect } from 'vitest';
import { makeScene, makeGroundTruth } from '../src/core/demoScene';
import { renderSynthetic, detectCheckerboardCorners, visibleBarEndpointsPx, artifactDiameterPx } from '../src/core/synthetic';
import { buildOrthoFrame, fitOrthoBounds, rayPlaneIntersect, worldToOrthoPixel, measureOrtho, Plane3D } from '../src/core/rectify';
import { traceBackward } from '../src/core/model';
import { rectifyCPU } from '../src/core/cpuRectify';
import { calibrate } from '../src/core/calibration';

describe('UI 工作流冒烟（无 DOM 核心路径）', () => {
  it('演示场景渲染 + 完整标定 + 校正 + 测量闭环', () => {
    // 1. 标定
    const gt = makeGroundTruth();
    const scene0 = makeScene(0);
    const views = Array.from({ length: 6 }, (_, i) => {
      const scene = makeScene(i);
      return {
        name: `v${i}`,
        width: scene.width,
        height: scene.height,
        corners: detectCheckerboardCorners(scene),
      };
    });
    const cal = calibrate(
      {
        rows: scene0.board.rows,
        cols: scene0.board.cols,
        squareMm: scene0.board.squareMm,
        views,
        water: { temperatureC: 18, salinityPSU: 32 },
        knownPort: { d0: 28, t: 6, nGlass: 1.517 },
        intrinsicsPrior: { fx: 1200, fy: 1200, lock: true },
      },
      { maxIterations: 40 },
    );
    expect(cal.rms).toBeLessThan(0.5);

    // 2. 渲染一张待校正照片并 CPU 正射校正
    const scene = makeScene(0);
    const img = renderSynthetic(scene, 1);
    const plane: Plane3D = { q: gt.plane.q, n: gt.plane.n };
    const frame = buildOrthoFrame(plane, plane.q, 0.5);
    const bounds = fitOrthoBounds(cal.intrinsics, cal.port, plane, frame, img.width, img.height);
    frame.origin = bounds.origin;
    const out = rectifyCPU({
      intr: cal.intrinsics,
      port: cal.port,
      frame,
      src: { width: img.width, height: img.height, data: img.data },
      outWidth: bounds.width,
      outHeight: bounds.height,
    });
    expect(out.length).toBe(bounds.width * bounds.height * 4);

    // 3. 两点测量圆盘直径
    const dia = artifactDiameterPx(scene, 0, 0.4);
    const toOrtho = (su: number, sv: number) => {
      const ray = traceBackward(cal.intrinsics, cal.port, su, sv);
      const X = rayPlaneIntersect(ray.origin, ray.dir, plane)!;
      return worldToOrthoPixel(frame, X);
    };
    const o1 = toOrtho(dia.p1.u, dia.p1.v);
    const o2 = toOrtho(dia.p2.u, dia.p2.v);
    const measured = measureOrtho(frame, o1, o2);
    expect(Math.abs(measured - dia.diameterMm) / dia.diameterMm).toBeLessThan(0.01);

    // 4. 半埋比例尺可见段（180mm）
    const bar = visibleBarEndpointsPx(scene, 0);
    expect(bar.visibleLengthMm).toBeCloseTo(180, 6);
  });
});
