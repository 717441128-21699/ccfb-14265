// 测量精度测试: 定面模式与比例尺残段模式, 验证 1% 指标
import { describe, it, expect } from 'vitest';
import {
  Intrinsics, projectPointEx, correctedPixelToRay, traceForward, undistortPixel,
} from '../src/refraction';
import { planeFromPose, planeFromScaleBar, measureOnPlane } from '../src/measure';
import { calibrate } from '../src/calibration';
import { rodrigues, mat3MulVec, V3 } from '../src/linalg';
import { TRUE_K, TRUE_HOUSING, BOARD, IMG_W, IMG_H, makePoses, makeViews } from './fixtures';

const Kv: Intrinsics = { fx: TRUE_K.fx, fy: TRUE_K.fy, cx: 1500, cy: 1000, k1: 0, k2: 0 };

/** 板坐标 -> 相机系三维点 */
function boardToCam(pose: { rvec: V3; t: V3 }, pB: V3): V3 {
  const R = rodrigues(pose.rvec);
  const pc = mat3MulVec(R, pB);
  return [pc[0] + pose.t[0], pc[1] + pose.t[1], pc[2] + pose.t[2]];
}

/** 板坐标 -> 校正后图像像素(经真实折射正向投影, 再映射到虚拟针孔) */
function boardToCorrectedPixel(pose: { rvec: V3; t: V3 }, pB: V3): [number, number] {
  const r = projectPointEx(TRUE_HOUSING, TRUE_K, boardToCam(pose, pB))!;
  const w = r.ray.w;
  return [Kv.fx * (w[0] / w[2]) + Kv.cx, Kv.fy * (w[1] / w[2]) + Kv.cy];
}

describe('长度测量(误差 < 1%)', () => {
  const poses = makePoses(7, 42);

  it('定面模式: 多个位姿下多种距离恢复', () => {
    for (const pose of poses) {
      const plane = planeFromPose(pose);
      for (const [len, p1B, p2B] of [
        [100, [20, 20, 0], [120, 20, 0]],
        [250, [10, 10, 0], [260, 10, 0]],
        [180, [30, 30, 0], [30, 210, 0]],
      ] as const) {
        const px1 = boardToCorrectedPixel(pose, p1B as unknown as V3);
        const px2 = boardToCorrectedPixel(pose, p2B as unknown as V3);
        const m = measureOnPlane(TRUE_HOUSING, Kv, plane, px1, px2)!;
        expect(m).not.toBeNull();
        expect(Math.abs(m - len) / len).toBeLessThan(0.01);
      }
    }
  });

  it('定面模式: 画面边缘同样达标(折射畸变最大的区域)', () => {
    const pose = poses[0];
    const plane = planeFromPose(pose);
    const px1 = boardToCorrectedPixel(pose, [10, 10, 0]);
    const px2 = boardToCorrectedPixel(pose, [10, 160, 0]);
    const m = measureOnPlane(TRUE_HOUSING, Kv, plane, px1, px2)!;
    expect(Math.abs(m - 150) / 150).toBeLessThan(0.01);
  });

  it('比例尺残段模式: 垂直拍摄时由残段推面深度, 全画面达标', () => {
    // 近似垂直向下的拍摄位姿(探方照相架的典型情况)
    const pose = { rvec: [0.03, -0.02, 0.01] as V3, t: [-140, -100, 850] as V3 };
    // 比例尺真实 100mm, 泥沙盖住一半, 只露出 50mm
    const barA = boardToCorrectedPixel(pose, [50, 100, 0]);
    const barB = boardToCorrectedPixel(pose, [100, 100, 0]);
    const plane = planeFromScaleBar(TRUE_HOUSING, Kv, barA, barB, 50);
    // 在画面不同位置量测不同长度
    for (const [len, p1B, p2B] of [
      [130, [60, 150, 0], [190, 150, 0]],
      [200, [20, 20, 0], [220, 20, 0]],
      [90, [250, 180, 0], [250, 90, 0]],
    ] as const) {
      const px1 = boardToCorrectedPixel(pose, p1B as unknown as V3);
      const px2 = boardToCorrectedPixel(pose, p2B as unknown as V3);
      const m = measureOnPlane(TRUE_HOUSING, Kv, plane, px1, px2)!;
      expect(Math.abs(m - len) / len).toBeLessThan(0.01);
    }
  });

  it('端到端: 带噪声标定 -> 用标定结果测量, 误差 < 1%', () => {
    const views = makeViews(poses, 0.15);
    const calib = calibrate(BOARD, views, {
      nGlass: 1.52, tGlass: 4, imageWidth: IMG_W, imageHeight: IMG_H,
      initNWater: TRUE_HOUSING.nWater,
    });
    expect(calib.rms).toBeLessThan(0.5);
    const housing = { nGlass: 1.52, tGlass: 4, dAir: calib.dAir, nWater: calib.nWater };
    const KvEst: Intrinsics = { fx: calib.K.fx, fy: calib.K.fy, cx: 1500, cy: 1000, k1: 0, k2: 0 };
    // 真值模型生成原图像素, 估计模型做校正映射, 模拟真实使用流程
    const pose = poses[3];
    const plane = planeFromPose(calib.poses[3]);
    const toCorrected = (pB: V3): [number, number] => {
      const r = projectPointEx(TRUE_HOUSING, TRUE_K, boardToCam(pose, pB))!;
      const [a, b] = undistortPixel(TRUE_K, r.u, r.v);       // 原图像素 -> 空气斜率
      const tr = traceForward(housing, a, b)!;                // 估计模型折射入水
      return [KvEst.fx * (tr.w[0] / tr.w[2]) + KvEst.cx, KvEst.fy * (tr.w[1] / tr.w[2]) + KvEst.cy];
    };
    const px1 = toCorrected([30, 40, 0]);
    const px2 = toCorrected([180, 40, 0]);
    const m = measureOnPlane(housing, KvEst, plane, px1, px2)!;
    expect(Math.abs(m - 150) / 150).toBeLessThan(0.01);
  }, 120000);

  it('校正像素 -> 光线 -> 平面交点自洽', () => {
    const pose = poses[2];
    const plane = planeFromPose(pose);
    const pB: V3 = [100, 80, 0];
    const px = boardToCorrectedPixel(pose, pB);
    const ray = correctedPixelToRay(TRUE_HOUSING, Kv, px[0], px[1])!;
    const denom = plane.n[0] * ray.w[0] + plane.n[1] * ray.w[1] + plane.n[2] * ray.w[2];
    const t = -(plane.n[0] * ray.q[0] + plane.n[1] * ray.q[1] + plane.n[2] * ray.q[2] + plane.d) / denom;
    const hit: V3 = [ray.q[0] + ray.w[0] * t, ray.q[1] + ray.w[1] * t, ray.q[2] + ray.w[2] * t];
    const P = boardToCam(pose, pB);
    expect(Math.hypot(hit[0] - P[0], hit[1] - P[1], hit[2] - P[2])).toBeLessThan(0.01);
  });
});
