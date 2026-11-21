/** 纯数据图像容器，Node 与浏览器通用，不依赖任何 DOM API */

export interface GrayImage {
  width: number;
  height: number;
  /** 行主序，0..255 */
  data: Uint8ClampedArray;
}

export interface RgbaImage {
  width: number;
  height: number;
  /** RGBA 行主序 */
  data: Uint8ClampedArray;
}

export function toGray(img: RgbaImage): GrayImage {
  const { width, height, data } = img;
  const out = new Uint8ClampedArray(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    out[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return { width, height, data: out };
}

export function sampleBilinearGray(
  img: GrayImage,
  x: number,
  y: number,
): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  if (x0 < 0 || y0 < 0 || x0 >= img.width - 1 || y0 >= img.height - 1) {
    const ix = Math.max(0, Math.min(img.width - 1, Math.round(x)));
    const iy = Math.max(0, Math.min(img.height - 1, Math.round(y)));
    return img.data[iy * img.width + ix];
  }
  const i00 = img.data[y0 * img.width + x0];
  const i10 = img.data[y0 * img.width + x0 + 1];
  const i01 = img.data[(y0 + 1) * img.width + x0];
  const i11 = img.data[(y0 + 1) * img.width + x0 + 1];
  const top = i00 * (1 - tx) + i10 * tx;
  const bot = i01 * (1 - tx) + i11 * tx;
  return top * (1 - ty) + bot * ty;
}

export function sampleBilinearRgba(
  img: RgbaImage,
  x: number,
  y: number,
  out: Uint8ClampedArray,
  outOffset = 0,
): void {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const w = img.width;
  const clampX = Math.max(0, Math.min(img.width - 1, x0));
  const clampY = Math.max(0, Math.min(img.height - 1, y0));
  const clampX1 = Math.max(0, Math.min(img.width - 1, x0 + 1));
  const clampY1 = Math.max(0, Math.min(img.height - 1, y0 + 1));
  const i00 = (clampY * w + clampX) * 4;
  const i10 = (clampY * w + clampX1) * 4;
  const i01 = (clampY1 * w + clampX) * 4;
  const i11 = (clampY1 * w + clampX1) * 4;
  for (let c = 0; c < 4; c++) {
    const top = img.data[i00 + c] * (1 - tx) + img.data[i10 + c] * tx;
    const bot = img.data[i01 + c] * (1 - tx) + img.data[i11 + c] * tx;
    out[outOffset + c] = top * (1 - ty) + bot * ty;
  }
}
