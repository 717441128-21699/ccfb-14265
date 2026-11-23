// 主界面: 标定 / 校正 / 量测 / 合成自检 / 档案
import {
  Housing, Intrinsics, waterIndexFromEnv, projectPointEx, undistortPixel, traceForward,
} from './refraction';
import { rodrigues, mat3MulVec, V3 } from './linalg';
import {
  BoardSpec, ViewObs, CalibResult, calibrate, solvePose,
} from './calibration';
import { detectChessboard, loadOpenCV } from './chessboard';
import { Corrector, makeVirtualIntrinsics } from './gl';
import { Plane, planeFromScaleBar, planeFromPose, measureOnPlane } from './measure';
import {
  renderSynth, synthPoses, synthCornerPixels, mulberry32, gaussNoise,
} from './synth';
import * as storage from './storage';

// ---------------- 全局状态 ----------------

interface CalibData {
  K: Intrinsics;
  housing: Housing;
  board: BoardSpec;
  rms: number;
  maxErr: number;
  imageWidth: number;
  imageHeight: number;
  savedAt?: string;
}

interface MeasureRecord {
  photo: string;
  mode: 'bar' | 'board';
  plane: Plane;
  results: { p1: [number, number]; p2: [number, number]; mm: number }[];
}

const state = {
  calib: null as CalibData | null,
  root: null as FileSystemDirectoryHandle | null,
  corrector: null as Corrector | null,
  calibImgs: [] as { name: string; canvas: HTMLCanvasElement; corners: [number, number][] | null }[],
  measure: {
    img: null as HTMLCanvasElement | null,   // 校正后图像
    origCanvas: null as HTMLCanvasElement | null, // 原图(供棋盘格检测)
    photoName: '',
    plane: null as Plane | null,
    barPoints: [] as [number, number][],
    pending: [] as [number, number][],
    results: [] as { p1: [number, number]; p2: [number, number]; mm: number }[],
    view: { s: 1, ox: 0, oy: 0 },
    mode: 'bar' as 'bar' | 'board',
  },
};

// ---------------- 工具 ----------------

const $ = <T extends HTMLElement = HTMLElement>(id: string): T =>
  document.getElementById(id) as T;
const inp = (id: string): HTMLInputElement => $(id);

function getCorrector(): Corrector {
  if (!state.corrector) state.corrector = new Corrector();
  return state.corrector;
}

function currentBoard(): BoardSpec {
  return { cols: +inp('boardCols').value, rows: +inp('boardRows').value, square: +inp('boardSq').value };
}

async function fileToCanvas(file: File | Blob): Promise<HTMLCanvasElement> {
  const bmp = await createImageBitmap(file);
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  c.getContext('2d')!.drawImage(bmp, 0, 0);
  return c;
}

function canvasToBlob(c: HTMLCanvasElement, type: string): Promise<Blob> {
  return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('编码失败'))), type, 0.95));
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function archivePathText(): string {
  const t = inp('tanfang').value.trim() || '未命名探方';
  const d = inp('diveDate').value || today();
  return `档案路径: <根目录>/${t}/${d}/`;
}

async function saveToArchive(subdir: string, name: string, blob: Blob): Promise<string> {
  const t = inp('tanfang').value.trim() || '未命名探方';
  const d = inp('diveDate').value || today();
  if (state.root) {
    const dir = await storage.diveFolder(state.root, t, d);
    const target = subdir ? await dir.getDirectoryHandle(subdir, { create: true }) : dir;
    await storage.saveBlob(target, name, blob);
    return `${t}/${d}/${subdir ? subdir + '/' : ''}${name}`;
  }
  storage.downloadBlob(`${t}_${d}_${name}`, blob);
  return `(下载) ${name}`;
}

// ---------------- 标签页切换 ----------------

document.querySelectorAll<HTMLButtonElement>('nav button').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    btn.classList.add('active');
    $(`tab-${btn.dataset.tab}`).classList.add('active');
  });
});

// ---------------- ① 标定 ----------------

function updateNWater(): void {
  inp('nWater').value = waterIndexFromEnv(+inp('waterT').value, +inp('waterS').value).toFixed(5);
}
inp('waterT').addEventListener('input', updateNWater);
inp('waterS').addEventListener('input', updateNWater);
updateNWater();

inp('calibFiles').addEventListener('change', () => {
  state.calibImgs = [];
  renderCalibFileList();
  $('btnDetect').textContent = '检测角点';
});

function renderCalibFileList(): void {
  const el = $('calibFileList');
  el.innerHTML = state.calibImgs.length
    ? state.calibImgs.map((f) =>
      `<div><span>${f.name}</span><span>${
        f.corners === null ? '<span class="bad">未检出</span>'
          : f.corners.length ? `<span class="ok">✓ ${f.corners.length} 角点</span>` : '待检测'
      }</span></div>`).join('')
    : '';
}

inp('btnDetect').addEventListener('click', async () => {
  const files = Array.from(inp('calibFiles').files ?? []);
  if (!files.length) return;
  const btn = inp('btnDetect') as HTMLButtonElement;
  btn.disabled = true;
  try {
    await loadOpenCV((m) => { btn.textContent = m; });
    const board = currentBoard();
    state.calibImgs = [];
    for (const f of files) {
      const canvas = await fileToCanvas(f);
      btn.textContent = `检测中 ${f.name}…`;
      const corners = await detectChessboard(canvas, board.cols, board.rows);
      state.calibImgs.push({ name: f.name, canvas, corners: corners ?? null });
      renderCalibFileList();
    }
    const ok = state.calibImgs.filter((x) => x.corners).length;
    btn.textContent = `检测角点(成功 ${ok}/${files.length})`;
    ($('btnCalibrate') as HTMLButtonElement).disabled = ok < 4;
  } catch (e) {
    btn.textContent = '检测失败: ' + (e as Error).message;
  } finally {
    btn.disabled = false;
  }
});

let lastCalibResult: CalibResult | null = null;

inp('btnCalibrate').addEventListener('click', () => {
  const views: ViewObs[] = state.calibImgs
    .filter((f) => f.corners)
    .map((f) => ({ pixels: f.corners! }));
  const board = currentBoard();
  const imgW = state.calibImgs[0].canvas.width;
  const imgH = state.calibImgs[0].canvas.height;
  const fitN = inp('fitNWater').checked;
  $('calibResult').textContent = '标定计算中(LM 平差, 约需几秒)…';
  setTimeout(() => {
    try {
      const res = calibrate(board, views, {
        nGlass: +inp('nGlass').value,
        tGlass: +inp('tGlass').value,
        imageWidth: imgW,
        imageHeight: imgH,
        initNWater: +inp('nWater').value,
        initDAir: +inp('dAirInit').value,
        fitNWater: fitN,
        dAirSigma: fitN ? 15 : 0,
      });
      lastCalibResult = res;
      state.calib = {
        K: res.K,
        housing: { nGlass: +inp('nGlass').value, tGlass: +inp('tGlass').value, dAir: res.dAir, nWater: res.nWater },
        board,
        rms: res.rms,
        maxErr: res.maxErr,
        imageWidth: imgW,
        imageHeight: imgH,
      };
      const pass = res.rms < 0.5;
      $('calibResult').innerHTML = [
        `<span class="${pass ? 'ok' : 'bad'}">重投影 RMS = ${res.rms.toFixed(3)} px ${pass ? '✓ 达标(<0.5)' : '✗ 超差(≥0.5)'}</span>`,
        `最大单点误差 = ${res.maxErr.toFixed(3)} px · 迭代 ${res.iterations} 次 · 视图 ${views.length} 张`,
        `fx=${res.K.fx.toFixed(1)} fy=${res.K.fy.toFixed(1)} cx=${res.K.cx.toFixed(1)} cy=${res.K.cy.toFixed(1)}`,
        `k1=${res.K.k1.toFixed(5)} k2=${res.K.k2.toFixed(5)}`,
        `dAir=${res.dAir.toFixed(1)} mm · n水=${res.nWater.toFixed(5)}`,
        `各视图 RMS: ${res.perViewRms.map((r) => r.toFixed(2)).join('  ')}`,
      ].join('<br>');
      ($('btnSaveCalib') as HTMLButtonElement).disabled = false;
      updateCorrectCalibInfo();
    } catch (e) {
      $('calibResult').innerHTML = `<span class="bad">标定失败: ${(e as Error).message}</span>`;
    }
  }, 30);
});

function calibJSON(): string {
  const c = state.calib!;
  return JSON.stringify({
    app: 'uw-calib', version: 1,
    savedAt: new Date().toISOString(),
    tanfang: inp('tanfang').value.trim(),
    date: inp('diveDate').value || today(),
    board: c.board,
    housing: c.housing,
    K: c.K,
    imageWidth: c.imageWidth,
    imageHeight: c.imageHeight,
    rms: c.rms,
    maxErr: c.maxErr,
    water: { tempC: +inp('waterT').value, salinityPSU: +inp('waterS').value },
  }, null, 2);
}

inp('btnSaveCalib').addEventListener('click', async () => {
  if (!state.calib) return;
  const path = await saveToArchive('', 'calibration.json', new Blob([calibJSON()], { type: 'application/json' }));
  $('calibResult').innerHTML += `<br><span class="ok">已保存: ${path}</span>`;
});

inp('btnLoadCalib').addEventListener('click', () => inp('loadCalibFile').click());
inp('loadCalibFile').addEventListener('change', async () => {
  const f = inp('loadCalibFile').files?.[0];
  if (!f) return;
  loadCalibJSON(JSON.parse(await f.text()));
});

function loadCalibJSON(j: any): void {
  if (!j || !j.K || !j.housing || !j.board) {
    $('calibResult').innerHTML = '<span class="bad">标定文件格式不正确</span>';
    return;
  }
  state.calib = {
    K: j.K, housing: j.housing, board: j.board,
    rms: j.rms ?? NaN, maxErr: j.maxErr ?? NaN,
    imageWidth: j.imageWidth ?? 0, imageHeight: j.imageHeight ?? 0,
  };
  $('calibResult').innerHTML =
    `已载入标定 (${j.savedAt ?? '未知时间'}${j.tanfang ? ' · 探方 ' + j.tanfang : ''})<br>` +
    `RMS=${Number(j.rms).toFixed(3)} px · fx=${j.K.fx.toFixed(1)} · dAir=${j.housing.dAir.toFixed(1)}mm · n水=${j.housing.nWater.toFixed(5)}`;
  updateCorrectCalibInfo();
}

// ---------------- ② 校正 ----------------

function updateCorrectCalibInfo(): void {
  const el = $('correctCalibInfo');
  if (state.calib) {
    const c = state.calib;
    el.innerHTML = `当前标定: fx=${c.K.fx.toFixed(1)} · n水=${c.housing.nWater.toFixed(5)} · dAir=${c.housing.dAir.toFixed(1)}mm · RMS=${c.rms.toFixed(3)}px`;
    ($('btnCorrect') as HTMLButtonElement).disabled = false;
  } else {
    el.textContent = '未载入标定, 请先在「下水标定」页完成或载入标定。';
  }
}

inp('btnCorrect').addEventListener('click', async () => {
  if (!state.calib) return;
  const files = Array.from(inp('correctFiles').files ?? []);
  if (!files.length) return;
  const btn = $('btnCorrect') as HTMLButtonElement;
  btn.disabled = true;
  const list = $('correctList');
  list.innerHTML = '';
  const fmt = inp('outFormat').value;
  const mime = fmt === 'png' ? 'image/png' : 'image/jpeg';
  const c = state.calib;
  try {
    const corr = getCorrector();
    for (const f of files) {
      const row = document.createElement('div');
      row.innerHTML = `<span>${f.name}</span><span>处理中…</span>`;
      list.appendChild(row);
      try {
        const bmp = await createImageBitmap(f);
        corr.setImage(bmp, bmp.width, bmp.height);
        const Kv = makeVirtualIntrinsics(c.K, bmp.width, bmp.height);
        const t0 = performance.now();
        const out = corr.correct(bmp.width, bmp.height, { K: c.K, Kv, nWater: c.housing.nWater });
        const dt = performance.now() - t0;
        const blob = await canvasToBlob(out, mime);
        const outName = f.name.replace(/\.[^.]+$/, '') + '_corr.' + fmt;
        const path = await saveToArchive('corrected', outName, blob);
        row.innerHTML = `<span>${f.name} → ${outName}</span><span class="ok">✓ ${dt.toFixed(0)} ms · ${(blob.size / 1048576).toFixed(1)}MB · ${path}</span>`;
        // 预览
        const pv = $('correctPreview') as HTMLCanvasElement;
        const scale = Math.min(1, 1100 / out.width);
        pv.width = Math.round(out.width * scale);
        pv.height = Math.round(out.height * scale);
        pv.getContext('2d')!.drawImage(out, 0, 0, pv.width, pv.height);
      } catch (e) {
        row.innerHTML = `<span>${f.name}</span><span class="bad">失败: ${(e as Error).message}</span>`;
      }
    }
  } catch (e) {
    list.innerHTML = `<div><span class="bad">${(e as Error).message}</span></div>`;
  } finally {
    btn.disabled = false;
  }
});

// ---------------- ③ 量测 ----------------

const mCanvas = $('measureCanvas') as HTMLCanvasElement;

function measureKv(): Intrinsics {
  const c = state.calib!;
  const img = state.measure.img!;
  return makeVirtualIntrinsics(c.K, img.width, img.height);
}

function fitMeasureView(): void {
  const img = state.measure.img;
  if (!img) return;
  const wrap = $('measureCanvasWrap');
  const w = wrap.clientWidth || 1000;
  const h = Math.round(Math.min(window.innerHeight * 0.7, (img.height / img.width) * w));
  mCanvas.width = w;
  mCanvas.height = Math.max(300, h);
  const s = Math.min(w / img.width, mCanvas.height / img.height);
  state.measure.view = {
    s,
    ox: (w - img.width * s) / 2,
    oy: (mCanvas.height - img.height * s) / 2,
  };
}

function redrawMeasure(): void {
  const m = state.measure;
  const ctx = mCanvas.getContext('2d')!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, mCanvas.width, mCanvas.height);
  if (!m.img) return;
  ctx.setTransform(m.view.s, 0, 0, m.view.s, m.view.ox, m.view.oy);
  ctx.drawImage(m.img, 0, 0);
  const drawPt = (p: [number, number], color: string) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p[0], p[1], 5 / m.view.s, 0, Math.PI * 2);
    ctx.fill();
  };
  const drawSeg = (a: [number, number], b: [number, number], color: string, label?: string) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 / m.view.s;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    if (label) {
      ctx.fillStyle = color;
      ctx.font = `${16 / m.view.s}px sans-serif`;
      ctx.fillText(label, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 8 / m.view.s);
    }
  };
  // 比例尺点
  m.barPoints.forEach((p) => drawPt(p, '#00c853'));
  if (m.barPoints.length === 2) drawSeg(m.barPoints[0], m.barPoints[1], '#00c853', '比例尺');
  // 已完成的量测
  for (const r of m.results) {
    drawPt(r.p1, '#ff5252');
    drawPt(r.p2, '#ff5252');
    drawSeg(r.p1, r.p2, '#ff5252', `${r.mm.toFixed(1)}mm`);
  }
  // 待定的点
  m.pending.forEach((p) => drawPt(p, '#ffd740'));
}

function screenToImg(e: MouseEvent): [number, number] {
  const rect = mCanvas.getBoundingClientRect();
  const m = state.measure.view;
  return [
    (e.clientX - rect.left - m.ox) / m.s,
    (e.clientY - rect.top - m.oy) / m.s,
  ];
}

let dragInfo: { x: number; y: number; moved: boolean } | null = null;

mCanvas.addEventListener('mousedown', (e) => {
  dragInfo = { x: e.clientX, y: e.clientY, moved: false };
});
mCanvas.addEventListener('mousemove', (e) => {
  if (!dragInfo || !state.measure.img) return;
  const dx = e.clientX - dragInfo.x;
  const dy = e.clientY - dragInfo.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) dragInfo.moved = true;
  if (dragInfo.moved) {
    state.measure.view.ox += dx;
    state.measure.view.oy += dy;
    dragInfo.x = e.clientX;
    dragInfo.y = e.clientY;
    redrawMeasure();
  }
});
mCanvas.addEventListener('mouseup', (e) => {
  const wasDrag = dragInfo?.moved;
  dragInfo = null;
  if (wasDrag || !state.measure.img || !state.calib) return;
  const p = screenToImg(e);
  const m = state.measure;
  if (m.mode === 'bar' && !m.plane) {
    if (m.barPoints.length >= 2) m.barPoints = [];
    m.barPoints.push(p);
    ($('btnBarSet') as HTMLButtonElement).disabled = m.barPoints.length !== 2;
    $('measureStatus').textContent = m.barPoints.length === 2
      ? '已选比例尺两端点, 确认残段真实长度后点「设定比例」。'
      : '请在比例尺残段上点两个端点。';
  } else {
    m.pending.push(p);
    if (m.pending.length === 2) {
      const c = state.calib;
      const mm = measureOnPlane(c.housing, measureKv(), m.plane!, m.pending[0], m.pending[1]);
      if (mm !== null && isFinite(mm)) {
        m.results.push({ p1: m.pending[0], p2: m.pending[1], mm });
      }
      m.pending = [];
      renderMeasureResults();
    }
  }
  redrawMeasure();
});
mCanvas.addEventListener('dblclick', () => {
  const m = state.measure;
  if (m.pending.length) m.pending.pop();
  else if (m.results.length) m.results.pop();
  else if (m.barPoints.length) { m.barPoints.pop(); m.plane = null; }
  renderMeasureResults();
  redrawMeasure();
});
mCanvas.addEventListener('wheel', (e) => {
  if (!state.measure.img) return;
  e.preventDefault();
  const rect = mCanvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const v = state.measure.view;
  const ns = Math.min(50, Math.max(0.05, v.s * Math.exp(-e.deltaY * 0.0012)));
  v.ox = cx - ((cx - v.ox) / v.s) * ns;
  v.oy = cy - ((cy - v.oy) / v.s) * ns;
  v.s = ns;
  redrawMeasure();
}, { passive: false });

function renderMeasureResults(): void {
  const m = state.measure;
  $('measureResults').innerHTML = m.results.length
    ? `<table><tr><th>#</th><th>长度 mm</th><th>长度 cm</th><th>端点1</th><th>端点2</th></tr>${
      m.results.map((r, i) =>
        `<tr><td>${i + 1}</td><td class="mono"><b>${r.mm.toFixed(1)}</b></td><td>${(r.mm / 10).toFixed(2)}</td>` +
        `<td class="mono">${r.p1.map((v) => v.toFixed(0)).join(',')}</td>` +
        `<td class="mono">${r.p2.map((v) => v.toFixed(0)).join(',')}</td></tr>`).join('')
    }</table>`
    : '<span class="hint">尚无量测结果。</span>';
  ($('btnSaveMeasure') as HTMLButtonElement).disabled = !m.results.length;
}

inp('measureMode').addEventListener('change', () => {
  state.measure.mode = inp('measureMode').value as 'bar' | 'board';
  state.measure.plane = null;
  state.measure.barPoints = [];
  state.measure.pending = [];
  $('barRow').style.display = state.measure.mode === 'bar' ? 'flex' : 'none';
  $('measureStatus').textContent = state.measure.mode === 'bar'
    ? '比例尺模式: 先在残段上点两个端点。'
    : '棋盘格定面: 载入含标定板的照片后自动解算平面。';
  redrawMeasure();
});

inp('measureFile').addEventListener('change', async () => {
  const f = inp('measureFile').files?.[0];
  if (!f || !state.calib) {
    if (!state.calib) $('measureStatus').textContent = '请先完成或载入标定。';
    return;
  }
  const m = state.measure;
  m.photoName = f.name;
  m.plane = null;
  m.barPoints = [];
  m.pending = [];
  m.results = [];
  $('measureStatus').textContent = '校正照片中…';
  try {
    const bmp = await createImageBitmap(f);
    const orig = document.createElement('canvas');
    orig.width = bmp.width;
    orig.height = bmp.height;
    orig.getContext('2d')!.drawImage(bmp, 0, 0);
    m.origCanvas = orig;
    const corr = getCorrector();
    corr.setImage(bmp, bmp.width, bmp.height);
    const c = state.calib;
    const Kv = makeVirtualIntrinsics(c.K, bmp.width, bmp.height);
    m.img = corr.correct(bmp.width, bmp.height, { K: c.K, Kv, nWater: c.housing.nWater });
    fitMeasureView();
    if (m.mode === 'board') {
      $('measureStatus').textContent = '检测棋盘格并解算平面…';
      const corners = await detectChessboard(orig, c.board.cols, c.board.rows);
      if (!corners) {
        $('measureStatus').innerHTML = '<span class="bad">未检出棋盘格, 无法定面。可改用比例尺模式。</span>';
      } else {
        const pose = solvePose(c.board, corners, c.K, c.housing);
        m.plane = planeFromPose(pose);
        $('measureStatus').innerHTML =
          `<span class="ok">✓ 平面已解算(板距 ${pose.t[2].toFixed(0)}mm)。点两点即量长度。</span>`;
      }
    } else {
      $('barRow').style.display = 'flex';
      $('measureStatus').textContent = '校正完成。请在比例尺残段上点两个端点。';
    }
    renderMeasureResults();
    redrawMeasure();
  } catch (e) {
    $('measureStatus').innerHTML = `<span class="bad">失败: ${(e as Error).message}</span>`;
  }
});

inp('btnBarSet').addEventListener('click', () => {
  const m = state.measure;
  if (m.barPoints.length !== 2 || !state.calib) return;
  try {
    m.plane = planeFromScaleBar(state.calib.housing, measureKv(), m.barPoints[0], m.barPoints[1], +inp('barLen').value);
    const z0 = -m.plane.d;
    $('measureStatus').innerHTML =
      `<span class="ok">✓ 比例已设定(推得探方面深度 ≈${z0.toFixed(0)}mm)。点两点即量长度。</span>`;
    redrawMeasure();
  } catch (e) {
    $('measureStatus').innerHTML = `<span class="bad">${(e as Error).message}</span>`;
  }
});

inp('btnSaveMeasure').addEventListener('click', async () => {
  const m = state.measure;
  if (!m.results.length || !m.plane) return;
  const rec: MeasureRecord = {
    photo: m.photoName,
    mode: m.mode,
    plane: m.plane,
    results: m.results,
  };
  // 追加到既有记录
  let all: MeasureRecord[] = [];
  if (state.root) {
    const dir = await storage.diveFolder(state.root, inp('tanfang').value.trim() || '未命名探方', inp('diveDate').value || today());
    all = (await storage.loadJSON<MeasureRecord[]>(dir, 'measurements.json')) ?? [];
    all.push(rec);
    await storage.saveJSON(dir, 'measurements.json', all);
    $('measureStatus').innerHTML += `<br><span class="ok">已追加保存到 measurements.json</span>`;
  } else {
    storage.downloadBlob(`measurements_${m.photoName}.json`, new Blob([JSON.stringify([rec], null, 2)]));
  }
});

// ---------------- ④ 合成自检 ----------------

inp('btnSelfTest').addEventListener('click', () => {
  const out = $('selftestResult');
  out.textContent = '自检运行中…';
  setTimeout(async () => {
    try {
      const W = +inp('stW').value;
      const H = +inp('stH').value;
      const nViews = +inp('stViews').value;
      const sigma = +inp('stNoise').value;
      // 合成真值
      const TRUE_K: Intrinsics = { fx: 1.2 * W, fy: 1.2 * W, cx: W / 2 + 0.4, cy: H / 2 + 0.2, k1: -0.08, k2: 0.012 };
      const TRUE_H: Housing = { nGlass: 1.52, tGlass: 4, dAir: 28, nWater: 1.339 };
      const board: BoardSpec = { cols: 9, rows: 6, square: 30 };
      const poses = synthPoses(board, TRUE_K, TRUE_H, W, H, nViews, 42);
      const rnd = mulberry32(7);
      const boardPts: V3[] = [];
      for (let j = 0; j < board.rows; j++) {
        for (let i = 0; i < board.cols; i++) {
          boardPts.push([(i + 1) * board.square, (j + 1) * board.square, 0]);
        }
      }
      const pts: [number, number][][] = poses.map((pose) =>
        synthCornerPixels(TRUE_K, TRUE_H, pose, boardPts)
          .map(([u, v]) => [u + gaussNoise(rnd, sigma), v + gaussNoise(rnd, sigma)] as [number, number]));
      const res = calibrate(board, pts.map((pixels) => ({ pixels })), {
        nGlass: TRUE_H.nGlass, tGlass: TRUE_H.tGlass,
        imageWidth: W, imageHeight: H, initNWater: TRUE_H.nWater,
      });
      const estH: Housing = { nGlass: TRUE_H.nGlass, tGlass: TRUE_H.tGlass, dAir: res.dAir, nWater: res.nWater };
      const KvEst = makeVirtualIntrinsics(res.K, W, H);
      // 量测自检: 真值模型成像 -> 估计模型校正与量测(模拟真实使用流程)
      const pose0 = poses[0];
      const planeEst = planeFromPose(res.poses[0]);
      const boardToCorr = (pB: V3): [number, number] => {
        const Rm = rodrigues(pose0.rvec);
        const pc = mat3MulVec(Rm, pB);
        const P: V3 = [pc[0] + pose0.t[0], pc[1] + pose0.t[1], pc[2] + pose0.t[2]];
        const r = projectPointEx(TRUE_H, TRUE_K, P)!;
        const [a, b] = undistortPixel(TRUE_K, r.u, r.v);
        const tr = traceForward(estH, a, b)!;
        return [KvEst.fx * (tr.w[0] / tr.w[2]) + KvEst.cx, KvEst.fy * (tr.w[1] / tr.w[2]) + KvEst.cy];
      };
      const mPx1 = boardToCorr([30, 40, 0]);
      const mPx2 = boardToCorr([180, 40, 0]);
      const mLen = measureOnPlane(estH, KvEst, planeEst, mPx1, mPx2)!;
      const mErr = Math.abs(mLen - 150) / 150 * 100;
      const fxErr = Math.abs(res.K.fx - TRUE_K.fx) / TRUE_K.fx * 100;
      const passRms = res.rms < 0.5;
      const passM = mErr < 1;
      out.innerHTML = [
        `<span class="${passRms ? 'ok' : 'bad'}">重投影 RMS = ${res.rms.toFixed(3)} px ${passRms ? '✓ <0.5' : '✗ ≥0.5'}</span>`,
        `焦距恢复误差 ${fxErr.toFixed(2)}% · dAir 估计 ${res.dAir.toFixed(1)}mm(真值 28) · n水 ${res.nWater.toFixed(5)}(真值 1.339)`,
        `<span class="${passM ? 'ok' : 'bad'}">量测自检: 真值 150.0mm → 测得 ${mLen.toFixed(2)}mm, 误差 ${mErr.toFixed(3)}% ${passM ? '✓ <1%' : '✗ ≥1%'}</span>`,
      ].join('<br>');
      // 渲染网格场景并校正(目视检验直线恢复)
      const gridImg = renderSynth({
        width: W, height: H, K: TRUE_K, housing: TRUE_H, pose: pose0,
        board, pattern: 'grid', noiseSigma: 2,
      });
      const rawC = $('stCanvasRaw') as HTMLCanvasElement;
      rawC.width = W;
      rawC.height = H;
      rawC.getContext('2d')!.putImageData(gridImg, 0, 0);
      const corr = getCorrector();
      corr.setImage(rawC, W, H);
      const corrected = corr.correct(W, H, { K: res.K, Kv: KvEst, nWater: estH.nWater });
      const corrC = $('stCanvasCorr') as HTMLCanvasElement;
      corrC.width = W;
      corrC.height = H;
      corrC.getContext('2d')!.drawImage(corrected, 0, 0);
      // WebGL2 shader 与 CPU 折射模型逐点一致性抽查
      const { correctPixel } = await import('./refraction');
      const corrData = corrC.getContext('2d')!.getImageData(0, 0, W, H).data;
      const rawData = gridImg.data;
      const bilinear = (x: number, y: number): number => {
        const x0 = Math.floor(x), y0 = Math.floor(y);
        const fx2 = x - x0, fy2 = y - y0;
        const px = (xx: number, yy: number) =>
          rawData[4 * (Math.min(H - 1, Math.max(0, yy)) * W + Math.min(W - 1, Math.max(0, xx)))];
        return px(x0, y0) * (1 - fx2) * (1 - fy2) + px(x0 + 1, y0) * fx2 * (1 - fy2)
          + px(x0, y0 + 1) * (1 - fx2) * fy2 + px(x0 + 1, y0 + 1) * fx2 * fy2;
      };
      let maxDiff = 0;
      let samples = 0;
      for (let i = 0; i < 40; i++) {
        const u = Math.round((0.1 + 0.8 * ((i * 37) % 40) / 40) * W);
        const v = Math.round((0.1 + 0.8 * ((i * 23) % 40) / 40) * H);
        const src = correctPixel(estH, res.K, KvEst, u, v);
        if (!src) continue;
        const cpu = bilinear(src[0], src[1]);
        const gpu = corrData[4 * (v * W + u)];
        maxDiff = Math.max(maxDiff, Math.abs(cpu - gpu));
        samples++;
      }
      const passGl = maxDiff <= 8;
      out.innerHTML += `<br><span class="${passGl ? 'ok' : 'bad'}">WebGL2↔CPU 映射抽查 ${samples} 点, 最大灰度差 ${maxDiff.toFixed(1)} ${passGl ? '✓' : '✗'}</span>`;
    } catch (e) {
      out.innerHTML = `<span class="bad">自检失败: ${(e as Error).message}</span>`;
    }
  }, 30);
});

// ---------------- ⑤ 档案 ----------------

inp('diveDate').value = today();
inp('tanfang').addEventListener('input', () => { $('archivePath').textContent = archivePathText(); });
inp('diveDate').addEventListener('input', () => { $('archivePath').textContent = archivePathText(); });
$('archivePath').textContent = archivePathText();

inp('btnPickRoot').addEventListener('click', async () => {
  if (!storage.fsAccessSupported()) {
    $('rootInfo').textContent = '当前浏览器不支持目录访问(请用 Chrome/Edge), 保存将退化为下载。';
    return;
  }
  try {
    state.root = await storage.pickRoot();
    $('rootInfo').textContent = `已选择: ${state.root.name}`;
    refreshTree();
  } catch { /* 用户取消 */ }
});

async function refreshTree(): Promise<void> {
  if (!state.root) return;
  const dirs = await storage.listDirs(state.root);
  const lines: string[] = [];
  for (const t of dirs) {
    const th = await state.root.getDirectoryHandle(t);
    const dates = await storage.listDirs(th);
    lines.push(`${t}/  ${dates.map((d) => `[${d}]`).join(' ')}`);
  }
  $('archiveTree').innerHTML = lines.length ? lines.join('<br>') : '<span class="hint">(空)</span>';
}

inp('btnSaveAll').addEventListener('click', async () => {
  if (!state.calib) {
    $('archiveTree').innerHTML = '<span class="bad">没有可保存的标定。</span>';
    return;
  }
  const path = await saveToArchive('', 'calibration.json', new Blob([calibJSON()], { type: 'application/json' }));
  $('archiveTree').innerHTML = `<span class="ok">已保存 ${path}</span>`;
  refreshTree();
});

inp('btnLoadFromArchive').addEventListener('click', async () => {
  if (!state.root) {
    $('archiveTree').innerHTML = '<span class="bad">请先选择根目录。</span>';
    return;
  }
  const dir = await storage.diveFolder(state.root, inp('tanfang').value.trim() || '未命名探方', inp('diveDate').value || today());
  const j = await storage.loadJSON<any>(dir, 'calibration.json');
  if (j) {
    loadCalibJSON(j);
    $('archiveTree').innerHTML = '<span class="ok">已载入该探方/日期的标定。</span>';
  } else {
    $('archiveTree').innerHTML = '<span class="bad">该探方/日期下没有 calibration.json。</span>';
  }
});

// 初始状态
updateCorrectCalibInfo();
renderMeasureResults();
