/**
 * 测量工作流：加载标定 → 构造探方平面/正射帧 → 校正 → 两点量长度。
 */
import { CalibrationResult } from '../core/calibration';
import { Intrinsics, PortParams, traceBackward } from '../core/model';
import { v3 } from '../core/vec3';
import {
  Plane3D,
  OrthoFrame,
  buildOrthoFrame,
  fitOrthoBounds,
  measureOrtho,
  rayPlaneIntersect,
  worldToOrthoPixel,
} from '../core/rectify';
import { rectifyCPU } from '../core/cpuRectify';

export interface MeasureSetup {
  intr: Intrinsics;
  port: PortParams;
  plane: Plane3D;
  frame: OrthoFrame;
  outWidth: number;
  outHeight: number;
}

/**
 * 由标定结果和平面深度构造测量环境。
 * 平面默认正对相机（n=光轴），深度 depthMm 由比例尺或先验给出。
 */
export function setupMeasure(
  cal: CalibrationResult,
  depthMm: number,
  mmPerPx: number,
  normal: { x: number; y: number; z: number } = { x: 0, y: 0, z: 1 },
): MeasureSetup {
  const intr = cal.intrinsics;
  const port = cal.port;
  const plane: Plane3D = { q: v3(0, 0, depthMm), n: v3(normal.x, normal.y, normal.z) };
  const frame = buildOrthoFrame(plane, plane.q, mmPerPx);
  return { intr, port, plane, frame, outWidth: 0, outHeight: 0 };
}

/** 根据源图尺寸计算正射影像覆盖范围 */
export function computeBounds(setup: MeasureSetup, srcWidth: number, srcHeight: number): MeasureSetup {
  const bounds = fitOrthoBounds(
    setup.intr,
    setup.port,
    setup.plane,
    setup.frame,
    srcWidth,
    srcHeight,
  );
  setup.frame.origin = bounds.origin;
  setup.outWidth = bounds.width;
  setup.outHeight = bounds.height;
  return setup;
}

/** CPU 校正（无 WebGL2 或小图时） */
export function rectifyImage(
  setup: MeasureSetup,
  src: { width: number; height: number; data: Uint8ClampedArray },
): Uint8ClampedArray {
  return rectifyCPU({
    intr: setup.intr,
    port: setup.port,
    frame: setup.frame,
    src,
    outWidth: setup.outWidth,
    outHeight: setup.outHeight,
  });
}

/** 源图像素 → 正射影像像素（用于把点击坐标映射到校正图） */
export function sourceToOrtho(setup: MeasureSetup, su: number, sv: number) {
  const ray = traceBackward(setup.intr, setup.port, su, sv);
  const X = rayPlaneIntersect(ray.origin, ray.dir, setup.plane);
  if (!X) return null;
  return worldToOrthoPixel(setup.frame, X);
}

/** 正射图上两点测量（毫米） */
export function measureTwoPoints(
  setup: MeasureSetup,
  p1: { x: number; y: number },
  p2: { x: number; y: number },
): number {
  return measureOrtho(setup.frame, p1, p2);
}
