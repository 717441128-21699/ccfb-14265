/** 共享数学/标定类型 */

export interface Vec2 {
  x: number;
  y: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** 针孔内参（像素单位） */
export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

/** Brown-Conrady 径向 + 切向畸变系数（归一化针孔坐标上） */
export interface BrownDistortion {
  k1: number;
  k2: number;
  p1: number;
  p2: number;
  k3: number;
}

/**
 * 平面观察窗折射参数。
 * 坐标系：相机光心为原点，+z 指向被摄物，像面 x 右、y 下。
 * d    : 光心到玻璃窗内表面（空气-玻璃界面）的轴向距离，mm
 * tg   : 玻璃厚度，mm
 * ng   : 玻璃折射率
 * nw   : 水的折射率（随水温/盐度/波长变化，每次下水重标）
 * tiltX: 窗法线绕相机 x 轴倾角，rad
 * tiltY: 窗法线绕相机 y 轴倾角，rad
 */
export interface RefractionParams {
  d: number;
  tg: number;
  ng: number;
  nw: number;
  tiltX: number;
  tiltY: number;
}

/** 世界->相机刚体变换：Pc = R * Pw + t */
export interface Pose {
  R: Mat3;
  t: Vec3;
}

export type Mat3 = number[]; // 行主序 9 个元素

export interface CheckerboardImage {
  /** 角点像素坐标，按行（board 坐标 j 固定、i 递增）排列 */
  corners: Vec2[];
  cols: number; // 内角点列数
  rows: number; // 内角点行数
  width: number;
  height: number;
}

export interface CalibrationStats {
  reprojectionRms: number;
  maxError: number;
  imageCount: number;
  pointCount: number;
  perImageRms: number[];
}

export interface Calibration {
  id: string;
  name: string;
  createdAt: string;
  imageWidth: number;
  imageHeight: number;
  intrinsics: Intrinsics;
  distortion: BrownDistortion;
  refraction: RefractionParams;
  stats: CalibrationStats;
}

export interface ScaleBar {
  /** 比例尺可见段两端像素坐标（校正后图像上） */
  p1: Vec2;
  p2: Vec2;
  /** 该段真实长度，mm */
  knownLengthMm: number;
}

export interface Measurement {
  p1: Vec2;
  p2: Vec2;
  /** 参考平面（标定棋盘格平面）深度，mm */
  referenceDepthMm: number;
  pixels: number;
  mmPerPixel: number;
  lengthMm: number;
}
