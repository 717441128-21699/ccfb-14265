// OpenCV.js 棋盘格角点检测封装 (findChessboardCornersSB + cornerSubPix)
// opencv.js 本地打包 (node_modules), 离线可用 —— 船上/现场无外网也能跑。
// @ts-ignore
import opencvUrl from '@techstark/opencv-js/dist/opencv.js?url';

let cvPromise: Promise<any> | null = null;

export function loadOpenCV(onProgress?: (msg: string) => void): Promise<any> {
  if (cvPromise) return cvPromise;
  cvPromise = new Promise((resolve, reject) => {
    const w = window as any;
    if (w.cv && w.cv.Mat) {
      resolve(w.cv);
      return;
    }
    onProgress?.('正在加载 OpenCV.js (约 8MB, 首次较慢)…');
    const s = document.createElement('script');
    s.src = opencvUrl;
    s.async = true;
    s.onload = async () => {
      try {
        let cv = w.cv;
        if (cv instanceof Promise) cv = await cv;
        if (cv && typeof cv.then === 'function') cv = await cv;
        // 某些构建需要等待运行时初始化
        if (cv && !cv.Mat && cv.onRuntimeInitialized !== undefined) {
          await new Promise<void>((res) => { cv.onRuntimeInitialized = () => res(); });
        }
        if (!cv || !cv.Mat) throw new Error('OpenCV 初始化失败');
        resolve(cv);
      } catch (e) {
        reject(e);
      }
    };
    s.onerror = () => reject(new Error('opencv.js 加载失败'));
    document.head.appendChild(s);
  });
  return cvPromise;
}

/**
 * 检测棋盘格内角点。
 * @param source 图像 (canvas / ImageData / HTMLImageElement)
 * @param cols 内角点列数
 * @param rows 内角点行数
 * @returns 行优先角点像素坐标, 失败返回 null
 */
export async function detectChessboard(
  source: HTMLCanvasElement | HTMLImageElement | ImageData,
  cols: number,
  rows: number,
): Promise<[number, number][] | null> {
  const cv = await loadOpenCV();
  let src: any = null;
  const gray = new cv.Mat();
  const corners = new cv.Mat();
  try {
    if (source instanceof ImageData) {
      src = cv.matFromImageData(source);
    } else {
      src = cv.imread(source);
    }
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    const flags =
      (cv.CALIB_CB_NORMALIZE_IMAGE ?? 2) |
      (cv.CALIB_CB_EXHAUSTIVE ?? 16) |
      (cv.CALIB_CB_ACCURACY ?? 32);
    let ok = false;
    try {
      ok = cv.findChessboardCornersSB(gray, new cv.Size(cols, rows), corners, flags);
    } catch {
      ok = false;
    }
    if (!ok) {
      // 回退到经典检测器
      ok = cv.findChessboardCorners(gray, new cv.Size(cols, rows), corners,
        (cv.CALIB_CB_ADAPTIVE_THRESH ?? 1) | (cv.CALIB_CB_NORMALIZE_IMAGE ?? 2) | (cv.CALIB_CB_FAST_CHECK ?? 8));
      if (ok) {
        const term = new cv.TermCriteria(
          (cv.TermCriteria_EPS ?? 2) | (cv.TermCriteria_MAX_ITER ?? 1), 40, 0.001);
        cv.cornerSubPix(gray, corners, new cv.Size(5, 5), new cv.Size(-1, -1), term);
      }
    }
    if (!ok) return null;
    const data: Float32Array = corners.data32F;
    const pts: [number, number][] = [];
    for (let i = 0; i < cols * rows; i++) pts.push([data[2 * i], data[2 * i + 1]]);
    return pts;
  } finally {
    src?.delete?.();
    gray.delete();
    corners.delete();
  }
}
