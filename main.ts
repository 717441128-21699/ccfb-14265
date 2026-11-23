import {
  AppState,
  initialState,
  pushLog,
  nWaterOf,
  addCheckerView,
  runCalibration,
  saveCalibrationToCatalog,
  validationSummary,
} from './src/ui/appState';
import { buildDemoBundle, buildDemoMeasurementScene } from './src/ui/demo';
import { setupMeasure, computeBounds, rectifyImage, measureTwoPoints, MeasureSetup } from './src/ui/measure';
import { isWebGL2Supported, WebGLRectifier } from './src/webgl/rectifier';
import { catalog, StoredCalibration } from './src/storage/catalog';
import { encodePng } from './src/core/png';

const state = initialState();
const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

// ---- 标签页切换 ----
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $('#tab-' + (tab as HTMLElement).dataset.tab).classList.add('active');
    if ((tab as HTMLElement).dataset.tab === 'catalog') void renderCatalog();
    if ((tab as HTMLElement).dataset.tab === 'measure') void refreshCalibSelect();
  });
});

// ---- 环境参数双向绑定 ----
const bindInputs = () => {
  const map: [string, keyof AppState][] = [
    ['trench', 'trench'],
    ['date', 'date'],
    ['temperatureC', 'temperatureC'],
    ['salinityPSU', 'salinityPSU'],
    ['d0', 'd0'],
    ['t', 't'],
    ['nGlass', 'nGlass'],
    ['rows', 'rows'],
    ['cols', 'cols'],
    ['squareMm', 'squareMm'],
  ];
  for (const [id, key] of map) {
    const el = $(id) as HTMLInputElement;
    el.addEventListener('input', () => {
      (state[key] as unknown as string | number) =
        el.type === 'number' ? parseFloat(el.value) : el.value;
      updateNWater();
    });
  }
  $('#fxprior').addEventListener('input', (e) => {
    const v = parseFloat((e.target as HTMLInputElement).value);
    state.fxprior = Number.isFinite(v) && v > 0 ? v : null;
  });
};

const updateNWater = () => {
  ($('#nWater') as HTMLInputElement).value = nWaterOf(state).toFixed(6);
};

const renderLog = () => {
  $('#log').textContent = state.log.slice(-40).join('\n');
  $('#log').scrollTop = $('#log').scrollHeight;
};

const renderViewList = () => {
  const ul = $('#viewList');
  ul.innerHTML = '';
  state.views.forEach((v, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${i + 1}. ${v.name}</span><span>${v.corners.length} 角点 · ${v.width}×${v.height}</span>`;
    ul.appendChild(li);
  });
};

// ---- 图像加载为 ImageData ----
const loadImageData = (file: File): Promise<ImageData> =>
  new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      resolve(ctx.getImageData(0, 0, c.width, c.height));
    };
    img.onerror = reject;
    img.src = url;
  });

$('#btnLoadFiles').addEventListener('click', () => $('#fileInput').click());
$('#fileInput').addEventListener('change', async (e) => {
  const files = Array.from((e.target as HTMLInputElement).files ?? []);
  for (const f of files) {
    try {
      const imgData = await loadImageData(f);
      addCheckerView(state, f.name, imgData);
    } catch (err) {
      pushLog(state, `加载 ${f.name} 失败：${(err as Error).message}`);
    }
  }
  renderViewList();
  renderLog();
});

$('#btnDemoViews').addEventListener('click', () => {
  const bundle = buildDemoBundle(6);
  state.views = bundle.views.map((bv) => ({
    name: bv.name,
    width: bv.imageData.width,
    height: bv.imageData.height,
    corners: bv.view.corners,
    imageData: bv.imageData,
  }));
  pushLog(state, `已载入 ${bundle.views.length} 张合成演示图（真实折射几何）`);
  renderViewList();
  renderLog();
});

$('#btnClearViews').addEventListener('click', () => {
  state.views = [];
  state.calibration = null;
  renderViewList();
  $('#btnSaveCalib').disabled = true;
  $('#calibReport').innerHTML = '';
  pushLog(state, '已清空视角');
  renderLog();
});

// ---- 标定 ----
$('#btnCalibrate').addEventListener('click', async () => {
  try {
    const result = await runCalibration(state);
    const summary = validationSummary(result);
    $('#calibReport').innerHTML =
      `<p class="${summary.pass ? 'ok' : 'bad'}">` +
      (summary.pass ? '✓ 达标（RMS < 0.5 px）' : '✗ 未达标') +
      '</p>' +
      summary.lines.map((l) => `<div class="metric"><span>${l.split('：')[0]}</span><span>${l.split('：')[1]}</span></div>`).join('');
    $('#btnSaveCalib').disabled = false;
  } catch (err) {
    pushLog(state, '标定失败：' + (err as Error).message);
  }
  renderLog();
});

$('#btnSaveCalib').addEventListener('click', async () => {
  try {
    await saveCalibrationToCatalog(state);
  } catch (err) {
    pushLog(state, '保存失败：' + (err as Error).message);
  }
  renderLog();
});

// ---- 目录 ----
async function renderCatalog() {
  const trenches = await catalog.listTrenches();
  const tree = $('#catalogTree');
  tree.innerHTML = '';
  if (trenches.length === 0) {
    tree.innerHTML = '<p class="hint">暂无保存的标定记录。</p>';
    return;
  }
  for (const trench of trenches) {
    const dives = await catalog.listDives(trench);
    const wrap = document.createElement('div');
    wrap.className = 'catalog-trench';
    wrap.innerHTML = `<div class="trench-head">探方 ${trench}</div>`;
    for (const dive of dives) {
      const cals = await catalog.listCalibrations(trench, dive.date);
      for (const cal of cals) {
        const item = document.createElement('div');
        item.className = 'catalog-item';
        item.innerHTML =
          `<div><strong>${dive.date}</strong> · RMS ${cal.rms.toFixed(3)}px · 最大 ${cal.maxError.toFixed(3)}px</div>` +
          `<div class="meta">水温 ${dive.temperatureC}°C · 盐度 ${dive.salinityPSU} PSU · ${new Date(cal.createdAt).toLocaleString()}</div>`;
        wrap.appendChild(item);
      }
    }
    tree.appendChild(wrap);
  }
}

$('#btnRefreshCatalog').addEventListener('click', () => void renderCatalog());

// ---- 测量页 ----
let savedCalibs: StoredCalibration[] = [];
async function refreshCalibSelect() {
  savedCalibs = await catalog.listCalibrations(state.trench, state.date);
  const sel = $('#calibSelect') as HTMLSelectElement;
  sel.innerHTML = '';
  const all: StoredCalibration[] = [];
  for (const t of await catalog.listTrenches()) {
    for (const d of await catalog.listDives(t)) {
      all.push(...(await catalog.listCalibrations(t, d.date)));
    }
  }
  if (state.calibration) {
    const opt = document.createElement('option');
    opt.textContent = '当前会话标定（未保存）';
    opt.value = 'session';
    sel.appendChild(opt);
  }
  for (const c of all) {
    const opt = document.createElement('option');
    opt.textContent = `${c.trench} / ${c.date} RMS${c.rms.toFixed(3)}`;
    opt.value = c.id;
    sel.appendChild(opt);
  }
  savedCalibs = all;
}

let measureImage: ImageData | null = null;
let measureSetup: MeasureSetup | null = null;
let rectifiedImage: ImageData | null = null;
let clickPoints: { x: number; y: number }[] = [];

$('#btnLoadMeasure').addEventListener('click', () => $('#measureFile').click());
$('#measureFile').addEventListener('change', async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  measureImage = await loadImageData(f);
  drawSourceImage();
  $('#btnRectify').disabled = false;
});

$('#btnDemoMeasure').addEventListener('click', () => {
  const bundle = buildDemoMeasurementScene();
  measureImage = bundle.imageData;
  drawSourceImage();
  $('#btnRectify').disabled = false;
  pushLog(state, '已载入演示测量照片（含半埋比例尺与 110mm 圆盘）');
  renderLog();
});

function drawSourceImage() {
  if (!measureImage) return;
  rectifiedImage = null;
  clickPoints = [];
  const canvas = $('#measureCanvas') as HTMLCanvasElement;
  canvas.width = measureImage.width;
  canvas.height = measureImage.height;
  canvas.getContext('2d')!.putImageData(measureImage, 0, 0);
}

function activeCalibration() {
  const sel = $('#calibSelect') as HTMLSelectElement;
  if (sel.value === 'session' || !sel.value) return state.calibration;
  const found = savedCalibs.find((c) => c.id === sel.value);
  return found ? (found.payload as never as typeof state.calibration) : state.calibration;
}

$('#btnRectify').addEventListener('click', () => {
  const cal = activeCalibration();
  if (!cal || !measureImage) {
    alert('请先选择标定并载入照片');
    return;
  }
  const depth = parseFloat(($('#planeDepth') as HTMLInputElement).value);
  const mmPerPx = parseFloat(($('#mmPerPx') as HTMLInputElement).value);
  let setup = setupMeasure(cal, depth, mmPerPx);
  setup = computeBounds(setup, measureImage.width, measureImage.height);
  measureSetup = setup;

  const t0 = performance.now();
  let out: Uint8ClampedArray;
  if (isWebGL2Supported()) {
    try {
      const glr = new WebGLRectifier();
      out = glr.rectify(measureImage, {
        intr: setup.intr,
        port: setup.port,
        frame: setup.frame,
        srcWidth: measureImage.width,
        srcHeight: measureImage.height,
        outWidth: setup.outWidth,
        outHeight: setup.outHeight,
      });
      glr.dispose();
    } catch (err) {
      pushLog(state, 'WebGL2 校正失败，回退 CPU：' + (err as Error).message);
      out = rectifyImage(setup, measureImage);
    }
  } else {
    out = rectifyImage(setup, measureImage);
  }
  const dt = performance.now() - t0;
  const rectified = new ImageData(out, setup.outWidth, setup.outHeight);
  rectifiedImage = rectified;
  const canvas = $('#measureCanvas') as HTMLCanvasElement;
  canvas.width = setup.outWidth;
  canvas.height = setup.outHeight;
  canvas.getContext('2d')!.putImageData(rectified, 0, 0);
  clickPoints = [];
  $('#btnClearPoints').disabled = false;
  $('#measureResult').innerHTML =
    `<div class="metric"><span>正射图尺寸</span><span>${setup.outWidth}×${setup.outHeight}px</span></div>` +
    `<div class="metric"><span>校正耗时</span><span>${dt.toFixed(0)} ms</span></div>` +
    `<div class="metric"><span>分辨率</span><span>${mmPerPx} mm/px</span></div>`;
  pushLog(state, `正射校正完成，耗时 ${dt.toFixed(0)}ms`);
  renderLog();
});

$('#measureCanvas').addEventListener('click', (e) => {
  if (!measureSetup) return;
  const canvas = e.target as HTMLCanvasElement;
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
  clickPoints.push({ x, y });
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#ff4d4d';
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fill();
  if (clickPoints.length === 2) {
    const len = measureTwoPoints(measureSetup, clickPoints[0], clickPoints[1]);
    ctx.strokeStyle = '#ffd24d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(clickPoints[0].x, clickPoints[0].y);
    ctx.lineTo(clickPoints[1].x, clickPoints[1].y);
    ctx.stroke();
    $('#measureResult').innerHTML +=
      `<div class="metric"><span>两点距离</span><span class="ok">${len.toFixed(2)} mm</span></div>`;
    clickPoints = [];
  }
});

$('#btnClearPoints').addEventListener('click', () => {
  clickPoints = [];
  if (rectifiedImage) {
    const canvas = $('#measureCanvas') as HTMLCanvasElement;
    canvas.width = rectifiedImage.width;
    canvas.height = rectifiedImage.height;
    canvas.getContext('2d')!.putImageData(rectifiedImage, 0, 0);
  }
});

// ---- 演示：导出合成 PNG（验证 PNG 编码链路）----
window.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.shiftKey && e.key === 'E') {
    const bundle = buildDemoMeasurementScene();
    const png = encodePng({ width: bundle.imageData.width, height: bundle.imageData.height, data: bundle.imageData.data });
    const blob = new Blob([png], { type: 'image/png' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'synthetic-refraction.png';
    a.click();
  }
});

// 初始化
($('#date') as HTMLInputElement).value = state.date;
bindInputs();
updateNWater();
renderLog();
