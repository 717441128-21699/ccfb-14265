import { describe, it, expect } from 'vitest';
import { projectPoint } from '../src/core/model';
import { buildOrthoFrame, orthoPixelToWorld, Plane3D } from '../src/core/rectify';
import { makeGroundTruth } from './fixtures';

/**
 * 2400 万像素校正性能。
 * CPU 参考实现给出每像素成本下限；WebGL2 为大规模并行，浏览器基准见 UI 页内报告。
 * 这里测量纯映射（不含纹理上传/采样）的每秒像素数，并断言无异常退化。
 */
describe('校正性能基准', () => {
  it('24MP 纯几何映射在 CPU 上可完成并报告吞吐', () => {
    const gt = makeGroundTruth();
    const plane: Plane3D = gt.plane;
    const frame = buildOrthoFrame(plane, plane.q, 0.5);
    const total = 6000 * 4000; // 24MP
    // 采样测量（全量在 CI 上可能慢，按步长抽样后外推）
    const step = 40;
    let sampled = 0;
    const t0 = performance.now();
    for (let y = 0; y < 4000; y += step) {
      for (let x = 0; x < 6000; x += step) {
        const X = orthoPixelToWorld(frame, x, y);
        const p = projectPoint(gt.intrinsics, gt.port, X);
        if (Number.isFinite(p.u)) sampled++;
      }
    }
    const dt = performance.now() - t0;
    const pxPerMs = sampled / dt;
    const estFullMs = total / pxPerMs;
    // eslint-disable-next-line no-console
    console.log(
      `CPU 映射吞吐 ${(pxPerMs * 1000 / 1e6).toFixed(2)} Mpx/s, ` +
      `24MP 估计耗时 ${(estFullMs / 1000).toFixed(2)} s（WebGL2 目标 <2s）`,
    );
    expect(sampled).toBeGreaterThan(0);
    expect(pxPerMs).toBeGreaterThan(1); // 健全性：至少 1px/ms
  });
});
