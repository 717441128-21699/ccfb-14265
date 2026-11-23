/**
 * 多视角棋盘格联合标定：
 * 共享参数（相机内参 4 + 窗口参数 5）+ 每视角位姿（rvec/tvec 6）。
 * 残差为棋盘格角点的重投影误差（像素），数值雅可比 + LM 求解。
 */
import { Vec3, v3, normalize, cross, rodrigues, mv, add, scale, logRodrigues } from './vec3';
import { Intrinsics, PortParams, projectPoint } from './model';
import { waterRefractiveIndex, DEFAULT_GLASS_N } from './water';
import { levenbergMarquardt } from './optimizer';

export interface CornerObs {
  r: number;
  c: number;
  u: number;
  v: number;
}

export interface CalibView {
  name: string;
  width: number;
  height: number;
  corners: CornerObs[];
}

export interface WaterConditionsInput {
  temperatureC: number;
  salinityPSU: number;
  wavelengthNm?: number;
}

export interface KnownPort {
  /** 光心到窗口内表面的机械距离（壳体装配图/实测，毫米） */
  d0: number;
  /** 玻璃厚度（毫米），未知时给标称值，标定会微调 */
  t: number;
  /** 玻璃折射率（材料属性，默认 BK7=1.517） */
  nGlass?: number;
}

export interface IntrinsicsPrior {
  /** 空气中标定/EXIF 得到的焦距（像素）。单平面水下标定存在焦距-深度歧义，
   *  提供先验可将其锁定，强烈建议。 */
  fx: number;
  fy?: number;
  /** true=锁定焦距不优化；false(默认)=作为强先验，允许小范围调整 */
  lock?: boolean;
}

export interface CalibInput {
  /** 棋盘格内角点行列数 */
  rows: number;
  cols: number;
  squareMm: number;
  views: CalibView[];
  water: WaterConditionsInput;
  /** 已知的窗口机械参数；d0 通常由装配图给定，不作为自由标定参数 */
  knownPort?: KnownPort;
  /** 相机焦距先验（空气中标定或 EXIF），用于消除单平面焦距-深度歧义 */
  intrinsicsPrior?: IntrinsicsPrior;
}

export interface ViewPose {
  rvec: Vec3;
  tvec: Vec3;
  rms: number;
  cornerCount: number;
}

export interface CalibrationResult {
  intrinsics: Intrinsics;
  port: PortParams;
  water: Required<WaterConditionsInput>;
  board: { rows: number; cols: number; squareMm: number };
  views: ViewPose[];
  rms: number;
  maxError: number;
  iterations: number;
  converged: boolean;
  createdAtIso: string;
}

/** 棋盘格角点在标定板坐标系中的 3D 位置（z=0 平面） */
export function boardPoint(rows: number, cols: number, squareMm: number, r: number, c: number): Vec3 {
  void rows;
  void cols;
  return v3(c * squareMm, r * squareMm, 0);
}

// ---- 参数打包 -----------------------------------------------------------
// [fx, fy, cx, cy, tiltX, tiltY, d0, t, nGlass, 每视角(rvec3,tvec3)...]
const SHARED_COUNT = 9;
const VIEW_STRIDE = 6;

function unpack(p: number[], viewCount: number, nWater: number, nAir: number) {
  const intrinsics: Intrinsics = { fx: p[0], fy: p[1], cx: p[2], cy: p[3] };
  const n = normalize(v3(p[4], p[5], 1));
  const port: PortParams = {
    nx: n.x,
    ny: n.y,
    nz: n.z,
    d0: p[6],
    t: p[7],
    nAir,
    nGlass: p[8],
    nWater,
  };
  const poses: { rvec: Vec3; tvec: Vec3 }[] = [];
  for (let i = 0; i < viewCount; i++) {
    const o = SHARED_COUNT + i * VIEW_STRIDE;
    poses.push({
      rvec: v3(p[o], p[o + 1], p[o + 2]),
      tvec: v3(p[o + 3], p[o + 4], p[o + 5]),
    });
  }
  return { intrinsics, port, poses };
}

/** 单视角位姿粗初始化（成像面轴方向正交化 + 面积法估深） */
function initPose(view: CalibView, input: CalibInput, fx0: number, fy0: number, cx: number, cy: number) {
  const { rows, cols, squareMm } = input;
  const at = (r: number, c: number) => view.corners.find((k) => k.r === r && k.c === c);

  // 棋盘 x 轴（列方向）、y 轴（行方向）在图像上的平均像素方向
  let dxU = 0, dxV = 0, nX = 0;
  let dyU = 0, dyV = 0, nY = 0;
  let pitch = 0, nP = 0;
  for (const k of view.corners) {
    const right = at(k.r, k.c + 1);
    if (right) {
      dxU += right.u - k.u;
      dxV += right.v - k.v;
      nX++;
      pitch += Math.hypot(right.u - k.u, right.v - k.v);
      nP++;
    }
    const down = at(k.r + 1, k.c);
    if (down) {
      dyU += down.u - k.u;
      dyV += down.v - k.v;
      nY++;
    }
  }
  dxU /= Math.max(1, nX); dxV /= Math.max(1, nX);
  dyU /= Math.max(1, nY); dyV /= Math.max(1, nY);
  pitch /= Math.max(1, nP);

  // 转到归一化相机光线方向（z=1），再归一化——这才是 3D 方向的正确近似
  const r1raw = normalize(v3(dxU / fx0, dxV / fy0, 0));
  const r2raw = normalize(v3(dyU / fx0, dyV / fy0, 0));
  // r3：两轴叉乘，保证朝前
  let r3 = normalize(cross(r1raw, r2raw));
  if (r3.z < 0) r3 = scale(r3, -1);
  // 以 r3 为准重建正交基：r1 ⟂ r3 且保留 r1raw 方向，r2 = r3 × r1
  const r1 = normalize(cross(r2raw, r3));
  const r2 = cross(r3, r1);
  // 列主序：列依次为 r1, r2, r3（与 mv/rodrigues 约定一致）
  const R = Float64Array.from([
    r1.x, r2.x, r3.x,
    r1.y, r2.y, r3.y,
    r1.z, r2.z, r3.z,
  ]);

  // 估深：针孔关系 pitch(px) ≈ fx * squareMm / Z（平面近正对，r3.z≈1）
  const Z = (fx0 * squareMm) / Math.max(pitch, 1e-6);
  const uMean = view.corners.reduce((s, k) => s + k.u, 0) / view.corners.length;
  const vMean = view.corners.reduce((s, k) => s + k.v, 0) / view.corners.length;
  // 板中心的相机系位置：用平均归一化光线方向，按 z 分量反推到深度 Z
  const centerRay = normalize(v3((uMean - cx) / fx0, (vMean - cy) / fy0, 1));
  const centerCam = scale(centerRay, Z / centerRay.z);
  // 板中心在板系中的坐标
  const boardCenter = v3(((cols - 1) * squareMm) / 2, ((rows - 1) * squareMm) / 2, 0);
  const Rb = mv(R, boardCenter);
  const tvec = v3(centerCam.x - Rb.x, centerCam.y - Rb.y, centerCam.z - Rb.z);

  return { rvec: logRodrigues(R), tvec };
}

/** 测试钩子：暴露单视角初始化 */
export function initPoseForTest(
  view: CalibView,
  input: CalibInput,
  fx0: number,
  fy0: number,
  cx: number,
  cy: number,
) {
  return initPose(view, input, fx0, fy0, cx, cy);
}


export interface CalibOptions {
  maxIterations?: number;
  onProgress?: (info: { iteration: number; cost: number }) => void;
  /** 是否优化主点（默认 false：钉在画面中心，平面端口推荐） */
  refinePrincipalPoint?: boolean;
  /** 各阶段诊断回调 */
  onStageInfo?: (info: { stage: number; fx: number; fy: number; rms: number }) => void;
}

export function calibrate(input: CalibInput, opts: CalibOptions = {}): CalibrationResult {
  const viewCount = input.views.length;
  if (viewCount < 3) throw new Error('至少需要 3 张不同位姿的棋盘格图像');
  const w0 = input.views[0].width;
  const h0 = input.views[0].height;

  const water: Required<WaterConditionsInput> = {
    temperatureC: input.water.temperatureC,
    salinityPSU: input.water.salinityPSU,
    wavelengthNm: input.water.wavelengthNm ?? 589,
  };
  const nWater = waterRefractiveIndex(water.temperatureC, water.salinityPSU, water.wavelengthNm);
  const nAir = 1.000293;

  // 窗口机械/材料参数：d0 与 nGlass 由装配图和玻璃牌号给定（不可由中心视场标定），
  // t 与倾斜作为自由参数。
  const knownD0 = input.knownPort?.d0 ?? 30;
  const knownT = input.knownPort?.t ?? 5;
  const knownNGlass = input.knownPort?.nGlass ?? DEFAULT_GLASS_N;

  // 初始内参：优先使用焦距先验；否则取 1.2 倍图宽的经验值。主点取画面中心。
  const prior = input.intrinsicsPrior;
  const fx0 = prior?.fx ?? w0 * 1.2;
  const fy0 = prior?.fy ?? fx0;
  const params0: number[] = [
    fx0, fy0, w0 / 2, h0 / 2,           // 内参
    0, 0, knownD0, knownT, knownNGlass, // 窗口：无倾斜 + 已知机械参数
  ];
  for (const view of input.views) {
    const pose = initPose(view, input, fx0, fx0, w0 / 2, h0 / 2);
    params0.push(pose.rvec.x, pose.rvec.y, pose.rvec.z);
    params0.push(pose.tvec.x, pose.tvec.y, pose.tvec.z);
  }

  const residuals = (p: number[]): number[] => {
    const { intrinsics, port, poses } = unpack(p, viewCount, nWater, nAir);
    const out: number[] = [];
    for (let i = 0; i < viewCount; i++) {
      const view = input.views[i];
      const R = rodrigues(poses[i].rvec);
      for (const k of view.corners) {
        const Xb = boardPoint(input.rows, input.cols, input.squareMm, k.r, k.c);
        const Xc = add(mv(R, Xb), poses[i].tvec);
        const proj = projectPoint(intrinsics, port, Xc);
        out.push(k.u - proj.u, k.v - proj.v);
      }
    }
    return out;
  };

  // 物理边界：fx/fy 为正且在合理焦距区间。
  // 主点默认钉在画面中心（平面端口装配误差小；若放开，主点会吸收焦距-深度
  // 耦合误差导致 fx 偏差）。refinePrincipalPoint=true 时才在 ±5% 内优化。
  // 倾斜 ∈[-0.15,0.15]（约 ±8.5°）；t∈[0.5,50]。d0(6)、nGlass(8) 锁定。
  const refinePP = opts.refinePrincipalPoint === true;
  // 焦距边界：有先验且 lock=true 时钉死；lock=false 时限制在先验 ±2%；
  // 无先验时给宽物理区间（此时单平面结果存在歧义，需谨慎解读）。
  const fxBound = (f: number): [number, number] => {
    if (!prior) return [w0 * 0.3, w0 * 5];
    if (prior.lock === true) return [f, f];
    return [f * 0.98, f * 1.02];
  };
  const [fxLo, fxHi] = fxBound(fx0);
  const [fyLo, fyHi] = fxBound(fy0);
  const lower: (number | null)[] = [
    fxLo, fyLo,
    refinePP ? w0 * 0.45 : w0 / 2,
    refinePP ? h0 * 0.45 : h0 / 2,
    -0.15, -0.15, null, 0.5, null,
  ];
  const upper: (number | null)[] = [
    fxHi, fyHi,
    refinePP ? w0 * 0.55 : w0 / 2,
    refinePP ? h0 * 0.55 : h0 / 2,
    0.15, 0.15, null, 50, null,
  ];
  for (let i = 0; i < viewCount; i++) lower.push(null, null, null, null, null, null);
  for (let i = 0; i < viewCount; i++) upper.push(null, null, null, null, null, null);

  // 锁定的窗口参数初值（d0 与 nGlass 全程固定）
  const lockedPort = params0.slice(4, SHARED_COUNT);

  // 阶段 1：固定全部窗口参数，只优化内参 + 每视角位姿
  const stage1Residuals = (p: number[]): number[] => {
    const full = p.slice();
    for (let j = 4; j < SHARED_COUNT; j++) full[j] = lockedPort[j - 4];
    return residuals(full);
  };
  const stage1 = levenbergMarquardt(stage1Residuals, params0, {
    maxIterations: 80,
    lower,
    upper,
    onIteration: (info) => opts.onProgress?.({ iteration: info.iteration, cost: info.cost }),
  });
  const stage1Params = stage1.params.slice();
  for (let j = 4; j < SHARED_COUNT; j++) stage1Params[j] = lockedPort[j - 4];
  opts.onStageInfo?.({ stage: 1, fx: stage1Params[0], fy: stage1Params[1], rms: stage1.cost });

  // 阶段 2：固定内参(0..3)，只放开倾斜(4,5) 与 t(7)，同时调位姿。
  // 避免内参与端口参数在数值差分噪声下互相带偏。
  const stage2Residuals = (p: number[]): number[] => {
    const full = p.slice();
    for (let j = 0; j < 4; j++) full[j] = stage1Params[j];
    full[6] = lockedPort[6 - 4]; // d0
    full[8] = lockedPort[8 - 4]; // nGlass
    return residuals(full);
  };
  const stage2 = levenbergMarquardt(stage2Residuals, stage1Params, {
    maxIterations: 60,
    lambdaInit: 1e-2,
    lower,
    upper,
    onIteration: (info) => opts.onProgress?.({ iteration: info.iteration + 40, cost: info.cost }),
  });
  const stage2Params = stage2.params.slice();
  for (let j = 0; j < 4; j++) stage2Params[j] = stage1Params[j];
  stage2Params[6] = lockedPort[6 - 4];
  stage2Params[8] = lockedPort[8 - 4];
  opts.onStageInfo?.({ stage: 2, fx: stage2Params[0], fy: stage2Params[1], rms: stage2.cost });

  // 以阶段 2 收尾。不再做内参+端口全自由联合：平板端口下 fx 与倾斜/t、
  // 各视角深度存在退化相关，全自由会以更低重投影误差为代价把 fx 拉偏，
  // 损害后续测量精度。分阶段优化在物理上更稳健。
  const result = stage2;
  result.params = stage2Params;
  result.params[6] = lockedPort[6 - 4];
  result.params[8] = lockedPort[8 - 4];

  const { intrinsics, port, poses } = unpack(result.params, viewCount, nWater, nAir);

  // 汇总误差
  const viewStats: ViewPose[] = [];
  let sumSq = 0;
  let count = 0;
  let maxError = 0;
  for (let i = 0; i < viewCount; i++) {
    const view = input.views[i];
    const R = rodrigues(poses[i].rvec);
    let vSq = 0;
    for (const k of view.corners) {
      const Xb = boardPoint(input.rows, input.cols, input.squareMm, k.r, k.c);
      const Xc = add(mv(R, Xb), poses[i].tvec);
      const proj = projectPoint(intrinsics, port, Xc);
      const e = Math.hypot(k.u - proj.u, k.v - proj.v);
      vSq += e * e;
      sumSq += e * e;
      maxError = Math.max(maxError, e);
      count++;
    }
    viewStats.push({
      rvec: poses[i].rvec,
      tvec: poses[i].tvec,
      rms: Math.sqrt(vSq / view.corners.length),
      cornerCount: view.corners.length,
    });
  }

  const finalRms = Math.sqrt(sumSq / count);

  return {
    intrinsics,
    port,
    water,
    board: { rows: input.rows, cols: input.cols, squareMm: input.squareMm },
    views: viewStats,
    rms: finalRms,
    maxError,
    iterations: result.iterations,
    // 优化器内部收敛，或最终重投影 RMS 已达亚像素验收线（0.5px），均视为成功
    converged: result.converged || finalRms < 0.5,
    createdAtIso: new Date().toISOString(),
  };
}
