import { describe, it, expect } from 'vitest';
import { v3, add, sub, scale, norm } from '../src/core/vec3';
import {
  buildOrthoFrame,
  fitOrthoBounds,
  measureOrtho,
  worldToOrthoPixel,
  Plane3D,
} from '../src/core/rectify';
import {
  calibrateFromVisibleSegment,
  solvePlaneFromSingleBar,
  visibleLength,
} from '../src/core/scalebar';
import {
  renderSynthetic,
  visibleBarEndpointsPx,
  artifactDiameterPx,
} from '../src/core/synthetic';
import { makeScene, makeTiltedPlaneScene, makeGroundTruth } from './fixtures';
import { projectPoint, traceBackward } from '../src/core/model';
import { rayPlaneIntersect } from '../src/core/rectify';

describe('半遮挡比例尺', () => {
  it('可见段长度按覆盖比例正确计算', () => {
    expect(visibleLength(300, 0.4)).toBeCloseTo(180, 6);
    expect(visibleLength(300, 0)).toBeCloseTo(300, 6);
  });

  it('已知平面上可见段反投影长度等于真实 180mm（误差 < 0.1%）', () => {
    const scene = makeScene(0);
    const eps = visibleBarEndpointsPx(scene, 0);
    const res = calibrateFromVisibleSegment(
      scene.intrinsics,
      scene.port,
      scene.plane,
      eps.a,
      eps.b,
      eps.visibleLengthMm,
    );
    expect(res.achievedLength).toBeGreaterThan(0);
    // achievedLength 是纯几何反投影，本身就是毫米；与真值对比
    expect(Math.abs(res.achievedLength - 180) / 180).toBeLessThan(0.001);
  });

  it('单比例尺在倾斜平面上求根恢复的长度误差 < 1%，且借助朝向先验使法向接近真值', () => {
    const scene = makeTiltedPlaneScene();
    const eps = visibleBarEndpointsPx(scene, 0);
    const solved = solvePlaneFromSingleBar(
      scene.intrinsics,
      scene.port,
      eps.a,
      eps.b,
      eps.visibleLengthMm,
      scene.plane.q,
      scene.plane.n, // 朝向先验（探方近似正对相机时可用光轴代替）
    );
    expect(Math.abs(solved.achievedLength - eps.visibleLengthMm) / eps.visibleLengthMm).toBeLessThan(0.01);
    expect(solved.normalAmbiguity).toBe(true); // 单线段确实存在歧义
    const dot =
      solved.plane.n.x * scene.plane.n.x +
      solved.plane.n.y * scene.plane.n.y +
      solved.plane.n.z * scene.plane.n.z;
    // 法向夹角 < 1°
    expect((Math.acos(Math.min(1, Math.abs(dot))) * 180) / Math.PI).toBeLessThan(1);
  });
});

describe('正射校正与两点测量', () => {
  it('已知尺寸圆盘（直径 110mm）测量误差 < 1%', () => {
    const scene = makeScene(0);
    const plane: Plane3D = scene.plane;
    const frame = buildOrthoFrame(plane, plane.q, 0.5);
    const bounds = fitOrthoBounds(scene.intrinsics, scene.port, plane, frame, scene.width, scene.height);
    frame.origin = bounds.origin;

    const dia = artifactDiameterPx(scene, 0, 0.4);
    // 源像素 -> 平面 3D -> 正射像素
    const toOrtho = (su: number, sv: number) => {
      const ray = traceBackward(scene.intrinsics, scene.port, su, sv);
      const X = rayPlaneIntersect(ray.origin, ray.dir, plane)!;
      return worldToOrthoPixel(frame, X);
    };
    const o1 = toOrtho(dia.p1.u, dia.p1.v);
    const o2 = toOrtho(dia.p2.u, dia.p2.v);
    const measured = measureOrtho(frame, o1, o2);
    expect(Math.abs(measured - dia.diameterMm) / dia.diameterMm).toBeLessThan(0.01);
  });

  it('校正映射自洽：正射像素映射回源像素再反投影应回到同一平面点', () => {
    const gt = makeGroundTruth();
    const plane: Plane3D = gt.plane;
    // 取一个平面点，投影到源图，再用后向光线回平面，应闭合
    const X = add(plane.q, v3(180, -120, 0));
    const src = projectPoint(gt.intrinsics, gt.port, X);
    const ray = traceBackward(gt.intrinsics, gt.port, src.u, src.v);
    const back = rayPlaneIntersect(ray.origin, ray.dir, plane)!;
    expect(norm(sub(back, X))).toBeLessThan(1e-3);
  });

  it('CPU 校正整图（1000x750）可在合理时间完成', () => {
    const scene = makeScene(0);
    const img = renderSynthetic(scene, 1);
    const t0 = performance.now();
    const plane: Plane3D = scene.plane;
    const frame = buildOrthoFrame(plane, plane.q, 0.5);
    const bounds = fitOrthoBounds(scene.intrinsics, scene.port, plane, frame, scene.width, scene.height);
    frame.origin = bounds.origin;
    // 抽样执行映射（全量在 CPU 上仅作冒烟，WebGL2 性能由专门基准测试覆盖）
    let valid = 0;
    for (let y = 0; y < bounds.height; y += 7) {
      for (let x = 0; x < bounds.width; x += 7) {
        const X = add(
          frame.origin,
          add(scale(frame.ex, x * frame.mmPerPx), scale(frame.ey, y * frame.mmPerPx)),
        );
        const p = projectPoint(scene.intrinsics, scene.port, X);
        if (p.u >= 0 && p.u < img.width && p.v >= 0 && p.v < img.height) valid++;
      }
    }
    const dt = performance.now() - t0;
    expect(valid).toBeGreaterThan(1000);
    expect(dt).toBeLessThan(5000);
  });
});
