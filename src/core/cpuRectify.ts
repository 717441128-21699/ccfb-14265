/**
 * CPU 正射校正（回退实现 + GPU 结果的对照基准）。
 * 数学与 WebGL 着色器完全一致：OrthoFrame 平面点 → projectPoint → 双线性采样。
 */
import { Intrinsics, PortParams, projectPoint } from '../core/model';
import { OrthoFrame, orthoPixelToWorld } from '../core/rectify';

export interface CPURectifyInput {
  intr: Intrinsics;
  port: PortParams;
  frame: OrthoFrame;
  src: { width: number; height: number; data: Uint8Array | Uint8ClampedArray };
  outWidth: number;
  outHeight: number;
}

function sampleBilinear(
  data: Uint8Array | Uint8ClampedArray,
  w: number,
  h: number,
  u: number,
  v: number,
): [number, number, number, number] {
  const x0 = Math.floor(u);
  const y0 = Math.floor(v);
  const tx = u - x0;
  const ty = v - y0;
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  const clampX0 = Math.max(0, x0);
  const clampY0 = Math.max(0, y0);
  const idx = (x: number, y: number) => (y * w + x) * 4;
  const i00 = idx(clampX0, clampY0);
  const i10 = idx(x1, clampY0);
  const i01 = idx(clampX0, y1);
  const i11 = idx(x1, y1);
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let c = 0; c < 4; c++) {
    const top = data[i00 + c] * (1 - tx) + data[i10 + c] * tx;
    const bot = data[i01 + c] * (1 - tx) + data[i11 + c] * tx;
    out[c] = top * (1 - ty) + bot * ty;
  }
  return out;
}

export function rectifyCPU(input: CPURectifyInput): Uint8ClampedArray {
  const { intr, port, frame, src, outWidth, outHeight } = input;
  const out = new Uint8ClampedArray(outWidth * outHeight * 4);
  for (let y = 0; y < outHeight; y++) {
    for (let x = 0; x < outWidth; x++) {
      const X = orthoPixelToWorld(frame, x, y);
      const s = projectPoint(intr, port, X);
      const o = (y * outWidth + x) * 4;
      if (s.u < 0 || s.u >= src.width || s.v < 0 || s.v >= src.height) continue;
      const px = sampleBilinear(src.data, src.width, src.height, s.u, s.v);
      out[o] = px[0];
      out[o + 1] = px[1];
      out[o + 2] = px[2];
      out[o + 3] = px[3];
    }
  }
  return out;
}
