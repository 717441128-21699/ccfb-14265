import type { Calibration, CheckerboardImage, RefractionParams, ScaleBar, Measurement, Vec2 } from '../core/types';
import { calibrate, evaluateCalibration } from '../core/calibration';
import { detectCheckerboard } from '../core/detector';
import { toGray, type GrayImage, type RgbaImage } from '../core/image';
import { waterRefractiveIndex } from '../core/water';
import { buildRemap } from '../core/remap';
import { measure, mmPerPixelAtDepth } from '../core/remap';
import { GlResampler, isWebGL2Available } from '../webgl/resampler';
import { db, type StoredCalibration, type ImageRecord } from '../core/db';

export interface AppState {
  calibration: Calibration | null;
  calibrationId: string;
  checkerImages: { name: string; rgba: RgbaImage; gray: GrayImage; board: CheckerboardImage | null }[];
  currentImage: RgbaImage | null;
  currentFileName: string;
  remap: ReturnType<typeof buildRemap> | null;
  correctedCanvas: HTMLCanvasElement | null;
  measurePoints: Vec2[];
  scaleBar: ScaleBar | null;
  measurements: Measurement[];
  env: { tempC: number; salinity: number; depthM: number };
}

export const state: AppState = {
  calibration: null,
  calibrationId: '',
  checkerImages: [],
  currentImage: null,
  currentFileName: '',
  remap: null,
  correctedCanvas: null,
  measurePoints: [],
  scaleBar: null,
  measurements: [],
  env: { tempC: 20, salinity: 35, depthM: 2 },
};

export function currentNw(): number {
  return waterRefractiveIndex({
    temperatureC: state.env.tempC,
    salinity: state.env.salinity,
    depthM: state.env.depthM,
  });
}

export function loadImageFromFile(file: File): Promise<RgbaImage> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve({
        width: canvas.width,
        height: canvas.height,
        data: imageData.data,
      });
    };
    img.onerror = reject;
    img.src = url;
  });
}

export function rgbaToCanvas(rgba: RgbaImage): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = rgba.width;
  canvas.height = rgba.height;
  const ctx = canvas.getContext('2d')!;
  // 显式构造 Uint8ClampedArray 以满足 ImageData 的 ArrayBuffer 类型约束
  const data = new Uint8ClampedArray(rgba.data);
  ctx.putImageData(new ImageData(data, rgba.width, rgba.height), 0, 0);
  return canvas;
}

export function addCheckerImage(name: string, rgba: RgbaImage, cols: number, rows: number) {
  const gray = toGray(rgba);
  const board = detectCheckerboard(gray, { cols, rows });
  state.checkerImages.push({ name, rgba, gray, board });
  return board;
}

export function runCalibrationFlow(
  squareMm: number,
  fixNw: boolean,
): Calibration {
  const boards = state.checkerImages
    .map((c) => c.board)
    .filter((b): b is CheckerboardImage => b !== null);
  if (boards.length < 3) throw new Error('至少需要 3 张成功检测的棋盘格图像');
  const initialRefraction: RefractionParams = {
    d: 30,
    tg: 5,
    ng: 1.49,
    nw: currentNw(),
    tiltX: 0,
    tiltY: 0,
  };
  const cal = calibrate(boards, {
    squareMm,
    initialRefraction,
    fixNw,
  });
  state.calibration = cal;
  state.calibrationId = `cal_${Date.now().toString(36)}`;
  // 持久化标定
  const stored: StoredCalibration = {
    id: state.calibrationId,
    name: `标定 ${new Date().toLocaleString('zh-CN')}`,
    createdAt: new Date().toISOString(),
    imageWidth: boards[0].width,
    imageHeight: boards[0].height,
    intrinsics: cal.intrinsics,
    distortion: cal.distortion,
    refraction: cal.refraction,
    stats: cal.stats,
    site: '',
    diveDate: '',
  };
  db.putCalibration(stored).catch((e) => console.error('标定持久化失败', e));
  return cal;
}

export function runCorrectionFlow(
  rgba: RgbaImage,
  referenceDepthMm: number,
): { canvas: HTMLCanvasElement; perfMs: number } {
  if (!state.calibration) throw new Error('请先完成折射标定');
  const { intrinsics, distortion, refraction } = state.calibration;
  const t0 = performance.now();
  const remap = buildRemap(
    rgba.width,
    rgba.height,
    intrinsics,
    distortion,
    refraction,
    referenceDepthMm,
  );
  state.remap = remap;
  const canvas = document.createElement('canvas');
  canvas.width = rgba.width;
  canvas.height = rgba.height;
  let perfMs: number;
  if (isWebGL2Available()) {
    const resampler = new GlResampler(canvas);
    resampler.uploadRgba(rgba);
    resampler.uploadMap(remap);
    perfMs = resampler.render(rgba.width, rgba.height);
    resampler.dispose();
  } else {
    const out = cpuFallback(rgba, remap.mapX, remap.mapY, remap.valid);
    const ctx = canvas.getContext('2d')!;
    const data = new Uint8ClampedArray(out);
    ctx.putImageData(new ImageData(data, rgba.width, rgba.height), 0, 0);
    perfMs = performance.now() - t0;
  }
  state.correctedCanvas = canvas;
  return { canvas, perfMs };
}

function cpuFallback(
  rgba: RgbaImage,
  mapX: Float32Array,
  mapY: Float32Array,
  valid: Uint8Array,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgba.width * rgba.height * 4);
  for (let y = 0; y < rgba.height; y++) {
    for (let x = 0; x < rgba.width; x++) {
      const i = y * rgba.width + x;
      const o = i * 4;
      if (!valid[i]) {
        out[o] = out[o + 1] = out[o + 2] = 12;
        out[o + 3] = 255;
        continue;
      }
      const px = mapX[i];
      const py = mapY[i];
      const x0 = Math.floor(px);
      const y0 = Math.floor(py);
      const tx = px - x0;
      const ty = py - y0;
      const w = rgba.width;
      const h = rgba.height;
      const clampX = (v: number) => Math.max(0, Math.min(w - 1, v));
      const clampY = (v: number) => Math.max(0, Math.min(h - 1, v));
      const i00 = clampY(y0) * w + clampX(x0);
      const i10 = clampY(y0) * w + clampX(x0 + 1);
      const i01 = clampY(y0 + 1) * w + clampX(x0);
      const i11 = clampY(y0 + 1) * w + clampX(x0 + 1);
      for (let c = 0; c < 4; c++) {
        const v00 = rgba.data[i00 * 4 + c];
        const v10 = rgba.data[i10 * 4 + c];
        const v01 = rgba.data[i01 * 4 + c];
        const v11 = rgba.data[i11 * 4 + c];
        out[o + c] =
          (1 - tx) * (1 - ty) * v00 +
          tx * (1 - ty) * v10 +
          (1 - tx) * ty * v01 +
          tx * ty * v11;
      }
    }
  }
  return out;
}

export function addMeasurePoint(p: Vec2, referenceDepthMm: number) {
  state.measurePoints.push(p);
  recomputeMeasurements(referenceDepthMm);
}

export function setScaleBarFlow(
  p1: Vec2,
  p2: Vec2,
  knownLengthMm: number,
  referenceDepthMm: number,
) {
  state.scaleBar = { p1, p2, knownLengthMm };
  recomputeMeasurements(referenceDepthMm);
}

function recomputeMeasurements(referenceDepthMm: number) {
  if (!state.calibration || state.measurePoints.length < 2) {
    state.measurements = [];
    return;
  }
  const K = state.calibration.intrinsics;
  const out: Measurement[] = [];
  for (let k = 1; k < state.measurePoints.length; k++) {
    const m = measure(
      state.measurePoints[k - 1],
      state.measurePoints[k],
      K,
      referenceDepthMm,
      state.scaleBar ?? undefined,
    );
    out.push(m);
  }
  state.measurements = out;
}

export function mmPerPixel(referenceDepthMm: number): number {
  return state.calibration
    ? mmPerPixelAtDepth(state.calibration.intrinsics, referenceDepthMm)
    : 0;
}

export async function saveImageRecord(
  site: string,
  diveDate: string,
  referenceDepthMm: number,
): Promise<void> {
  if (!state.calibration || !state.currentImage) return;
  const blob = await new Promise<Blob | null>((resolve) =>
    state.correctedCanvas?.toBlob(resolve, 'image/jpeg', 0.92) ?? resolve(null),
  );
  const rec: ImageRecord = {
    id: `img_${Date.now().toString(36)}`,
    fileName: state.currentFileName,
    width: state.currentImage.width,
    height: state.currentImage.height,
    calibrationId: state.calibrationId,
    referenceDepthMm,
    scaleBar: state.scaleBar,
    measurements: state.measurements,
    createdAt: new Date().toISOString(),
    site,
    diveDate,
    correctedBlob: blob ?? undefined,
  };
  await db.putImage(rec);
}

export { evaluateCalibration };
