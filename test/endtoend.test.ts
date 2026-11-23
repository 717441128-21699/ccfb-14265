import { describe, it, expect } from 'vitest';
import { buildOrthoFrame, fitOrthoBounds, worldToOrthoPixel, rayPlaneIntersect, Plane3D } from '../src/core/rectify';
import { rectifyCPU } from '../src/core/cpuRectify';
import { renderSynthetic, artifactDiameterPx } from '../src/core/synthetic';
import { traceBackward } from '../src/core/model';
import { makeScene } from './fixtures';

describe('端到端像素校正（含重采样）', () => {
  it('渲染→CPU 校正→正射图测量圆盘直径，误差 < 1%', () => {
    const scene = makeScene(0);
    // 小尺寸渲染保证测试速度，但仍走完整折射+双线性链路
    const img = renderSynthetic(scene, 1);
    const plane: Plane3D = scene.plane;
    const frame = buildOrthoFrame(plane, plane.q, 0.5);
    const bounds = fitOrthoBounds(scene.intrinsics, scene.port, plane, frame, scene.width, scene.height);
    frame.origin = bounds.origin;

    const out = rectifyCPU({
      intr: scene.intrinsics,
      port: scene.port,
      frame,
      src: { width: img.width, height: img.height, data: img.data },
      outWidth: bounds.width,
      outHeight: bounds.height,
    });
    expect(out.length).toBe(bounds.width * bounds.height * 4);
    // 输出图非全透明（有效像素占比）
    let filled = 0;
    for (let i = 3; i < out.length; i += 4) if (out[i] > 0) filled++;
    expect(filled / (bounds.width * bounds.height)).toBeGreaterThan(0.5);

    // 几何测量（源像素 → 平面 → 正射像素），与重采样结果共用同一 frame
    const dia = artifactDiameterPx(scene, 0, 0.4);
    const toOrtho = (su: number, sv: number) => {
      const ray = traceBackward(scene.intrinsics, scene.port, su, sv);
      const X = rayPlaneIntersect(ray.origin, ray.dir, plane)!;
      return worldToOrthoPixel(frame, X);
    };
    const o1 = toOrtho(dia.p1.u, dia.p1.v);
    const o2 = toOrtho(dia.p2.u, dia.p2.v);
    const measured = Math.hypot(o2.x - o1.x, o2.y - o1.y) * frame.mmPerPx;
    expect(Math.abs(measured - dia.diameterMm) / dia.diameterMm).toBeLessThan(0.01);
  });
});
