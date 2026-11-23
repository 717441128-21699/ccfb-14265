/**
 * 应用状态与工作流编排。
 * 纯 TypeScript，不依赖框架；UI 节点在 main.ts 绑定。
 */
import { CalibInput, CalibrationResult, CalibView, calibrate } from '../core/calibration';
import { CornerObs } from '../core/calibration';
import { detectCheckerboard, detectCandidates } from '../vision/checkerboard';
import { waterRefractiveIndex } from '../core/water';
import { catalog, DiveRecord, StoredCalibration, genId } from '../storage/catalog';

export interface AppState {
  trench: string;
  date: string;
  temperatureC: number;
  salinityPSU: number;
  d0: number;
  t: number;
  nGlass: number;
  fxprior: number | null;
  rows: number;
  cols: number;
  squareMm: number;
  views: { name: string; width: number; height: number; corners: CornerObs[]; imageData?: ImageData }[];
  calibration: CalibrationResult | null;
  busy: boolean;
  log: string[];
}

export function initialState(): AppState {
  return {
    trench: 'T01',
    date: new Date().toISOString().slice(0, 10),
    temperatureC: 18,
    salinityPSU: 32,
    d0: 28,
    t: 6,
    nGlass: 1.517,
    fxprior: null,
    rows: 9,
    cols: 12,
    squareMm: 20,
    views: [],
    calibration: null,
    busy: false,
    log: [],
  };
}

export function pushLog(state: AppState, msg: string): void {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  state.log.push(line);
  if (state.log.length > 200) state.log.shift();
}

export function nWaterOf(state: AppState): number {
  return waterRefractiveIndex(state.temperatureC, state.salinityPSU);
}

/** 从一张 ImageData 检测棋盘角点并加入视角集合 */
export function addCheckerView(
  state: AppState,
  name: string,
  imageData: ImageData,
): { corners: number; ok: boolean } {
  const detected = detectCheckerboard(imageData, { rows: state.rows, cols: state.cols });
  if (detected.length !== state.rows * state.cols) {
    // 自动网格失败：保留候选点供手工标记，不计入有效视角
    const cands = detectCandidates(imageData);
    pushLog(state, `${name}：自动网格识别失败（候选 ${cands.length} 个），请手工标记或换图`);
    return { corners: detected.length, ok: false };
  }
  state.views.push({
    name,
    width: imageData.width,
    height: imageData.height,
    corners: detected,
    imageData,
  });
  pushLog(state, `${name}：识别 ${detected.length} 个角点`);
  return { corners: detected.length, ok: true };
}

/** 运行标定 */
export async function runCalibration(state: AppState): Promise<CalibrationResult> {
  if (state.views.length < 3) {
    throw new Error('至少需要 3 张有效棋盘格图像');
  }
  const views: CalibView[] = state.views.map((v) => ({
    name: v.name,
    width: v.width,
    height: v.height,
    corners: v.corners,
  }));
  const input: CalibInput = {
    rows: state.rows,
    cols: state.cols,
    squareMm: state.squareMm,
    views,
    water: { temperatureC: state.temperatureC, salinityPSU: state.salinityPSU },
    knownPort: { d0: state.d0, t: state.t, nGlass: state.nGlass },
    intrinsicsPrior: state.fxprior
      ? { fx: state.fxprior, fy: state.fxprior, lock: true }
      : undefined,
  };
  pushLog(state, `开始标定：${views.length} 视角，nWater=${nWaterOf(state).toFixed(5)}`);
  const t0 = performance.now();
  const result = calibrate(input, {
    maxIterations: 60,
    onProgress: (info) => {
      if (info.iteration % 10 === 0) pushLog(state, `  迭代 ${info.iteration}，cost=${info.cost.toFixed(3)}`);
    },
  });
  const dt = performance.now() - t0;
  state.calibration = result;
  pushLog(
    state,
    `标定完成：RMS=${result.rms.toFixed(3)}px，最大=${result.maxError.toFixed(3)}px，耗时 ${dt.toFixed(0)}ms`,
  );
  return result;
}

/** 保存潜水记录与标定结果到本地目录 */
export async function saveCalibrationToCatalog(state: AppState): Promise<string> {
  if (!state.calibration) throw new Error('没有可保存的标定结果');
  const dive: DiveRecord = {
    id: genId('dive_'),
    trench: state.trench,
    date: state.date,
    temperatureC: state.temperatureC,
    salinityPSU: state.salinityPSU,
    createdAt: Date.now(),
  };
  await catalog.putDive(dive);
  const stored: StoredCalibration = {
    id: genId('cal_'),
    diveId: dive.id,
    trench: state.trench,
    date: state.date,
    payload: state.calibration,
    rms: state.calibration.rms,
    maxError: state.calibration.maxError,
    createdAt: Date.now(),
  };
  await catalog.putCalibration(stored);
  pushLog(state, `已保存到目录：${state.trench} / ${state.date}（RMS ${stored.rms.toFixed(3)}px）`);
  return stored.id;
}

export function validationSummary(result: CalibrationResult): { pass: boolean; lines: string[] } {
  const lines = [
    `重投影 RMS：${result.rms.toFixed(4)} px（要求 < 0.5）`,
    `最大角点误差：${result.maxError.toFixed(4)} px`,
    `焦距 fx/fy：${result.intrinsics.fx.toFixed(1)} / ${result.intrinsics.fy.toFixed(1)}`,
    `主点 cx/cy：${result.intrinsics.cx.toFixed(1)} / ${result.intrinsics.cy.toFixed(1)}`,
    `窗口 d0/t：${result.port.d0.toFixed(2)} / ${result.port.t.toFixed(2)} mm`,
    `玻璃 n：${result.port.nGlass.toFixed(4)}，水 n：${result.port.nWater.toFixed(5)}`,
    `视角数：${result.views.length}，收敛：${result.converged ? '是' : '否（RMS 已达标）'}`,
  ];
  return { pass: result.rms < 0.5 && result.maxError < 0.5, lines };
}
