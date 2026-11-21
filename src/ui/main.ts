import {
  state,
  currentNw,
  loadImageFromFile,
  rgbaToCanvas,
  addCheckerImage,
  runCalibrationFlow,
  runCorrectionFlow,
  addMeasurePoint,
  setScaleBarFlow,
  mmPerPixel,
  saveImageRecord,
} from './app';
import { db } from '../core/db';

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

// ---- 标签页切换 ----
document.querySelectorAll<HTMLButtonElement>('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $(`tab-${tab.dataset.tab}`).classList.add('active');
    if (tab.dataset.tab === 'catalog') void renderCatalog();
  });
});

// ---- 工况应用（温盐深 -> 水折射率）----
$('applyEnv').addEventListener('click', () => {
  state.env.tempC = parseFloat($<HTMLInputElement>('tempC').value) || 20;
  state.env.salinity = parseFloat($<HTMLInputElement>('salinity').value) || 0;
  state.env.depthM = parseFloat($<HTMLInputElement>('diveDepth').value) || 0;
  const nw = currentNw();
  alert(
    `工况已应用：水温 ${state.env.tempC}°C，盐度 ${state.env.salinity} PSU\n` +
      `当前水折射率 nw = ${nw.toFixed(4)}\n` +
      `下次下水若工况变化，请重新标定或勾选固定 nw。`,
  );
});

// ---- 标定页 ----
const calFiles = $<HTMLInputElement>('calFiles');
$('calAddImages').addEventListener('click', () => calFiles.click());
calFiles.addEventListener('change', async () => {
  const files = Array.from(calFiles.files ?? []);
  const cols = parseInt($<HTMLInputElement>('boardCols').value, 10);
  const rows = parseInt($<HTMLInputElement>('boardRows').value, 10);
  const strip = $('calThumbs');
  for (const file of files) {
    const rgba = await loadImageFromFile(file);
    const board = addCheckerImage(file.name, rgba, cols, rows);
    const canvas = rgbaToCanvas(rgba);
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    const badge = document.createElement('div');
    badge.className = `badge ${board ? 'ok' : 'bad'}`;
    badge.textContent = board ? `检测到 ${board.corners.length} 角点` : '检测失败，可手动标';
    thumb.appendChild(badge);
    thumb.appendChild(canvas);
    strip.appendChild(thumb);
  }
  const ok = state.checkerImages.filter((c) => c.board).length;
  $('calStatus').textContent =
    `已导入 ${state.checkerImages.length} 张，其中 ${ok} 张成功检测角点。` +
    (ok < 3 ? '至少需要 3 张成功图像才能标定。' : '可以运行标定。');
});

$('runCalibration').addEventListener('click', () => {
  try {
    const squareMm = parseFloat($<HTMLInputElement>('squareMm').value);
    const fixNw = $<HTMLInputElement>('fixNw').checked;
    const cal = runCalibrationFlow(squareMm, fixNw);
    $('calStatus').textContent =
      `标定完成：重投影 RMS ${cal.stats.reprojectionRms.toFixed(3)} px，` +
      `最大误差 ${cal.stats.maxError.toFixed(3)} px，使用 ${cal.stats.imageCount} 张图像、` +
      `${cal.stats.pointCount} 个角点。`;
    renderCalParams();
  } catch (e) {
    alert((e as Error).message);
  }
});

function renderCalParams() {
  const cal = state.calibration;
  const box = $('calParams');
  if (!cal) {
    box.innerHTML = '';
    return;
  }
  const cards: [string, string][] = [
    ['焦距 fx', cal.intrinsics.fx.toFixed(2)],
    ['焦距 fy', cal.intrinsics.fy.toFixed(2)],
    ['主点 cx', cal.intrinsics.cx.toFixed(2)],
    ['主点 cy', cal.intrinsics.cy.toFixed(2)],
    ['水折射率 nw', cal.refraction.nw.toFixed(4)],
    ['玻璃折射率 ng', cal.refraction.ng.toFixed(3)],
    ['窗内表面距离 d (mm)', String(cal.refraction.d)],
    ['玻璃厚度 tg (mm)', String(cal.refraction.tg)],
    ['径向畸变 k1', cal.distortion.k1.toFixed(4)],
    ['重投影 RMS (px)', cal.stats.reprojectionRms.toFixed(3)],
  ];
  box.innerHTML = cards
    .map(([k, v]) => `<div class="param-card"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join('');
}

// ---- 校正页 ----
const imageFile = $<HTMLInputElement>('imageFile');
$('loadImage').addEventListener('click', () => imageFile.click());
imageFile.addEventListener('change', async () => {
  const file = imageFile.files?.[0];
  if (!file) return;
  const rgba = await loadImageFromFile(file);
  state.currentImage = rgba;
  state.currentFileName = file.name;
  const srcCanvas = rgbaToCanvas(rgba);
  const host = $<HTMLDivElement>('srcCanvas').parentElement!;
  $('srcCanvas').replaceWith(srcCanvas);
  srcCanvas.id = 'srcCanvas';
  const out = document.createElement('canvas');
  out.id = 'outCanvas';
  const existing = $('outCanvas');
  if (existing) existing.replaceWith(out);
  else host.appendChild(out);
});

$('runCorrection').addEventListener('click', () => {
  if (!state.currentImage) {
    alert('请先导入照片');
    return;
  }
  try {
    const depth = parseFloat($<HTMLInputElement>('refDepth').value);
    const { canvas, perfMs } = runCorrectionFlow(state.currentImage, depth);
    canvas.id = 'outCanvas';
    $('outCanvas').replaceWith(canvas);
    $<HTMLSpanElement>('corrPerf').textContent =
      `校正耗时 ${perfMs.toFixed(0)} ms（${state.currentImage.width}×${state.currentImage.height}）`;
  } catch (e) {
    alert((e as Error).message);
  }
});

$('downloadCorrected').addEventListener('click', () => {
  if (!state.correctedCanvas) {
    alert('请先执行折射校正');
    return;
  }
  state.correctedCanvas.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = state.currentFileName.replace(/\.[^.]+$/, '') + '_corrected.jpg';
    a.click();
  }, 'image/jpeg', 0.92);
});

// ---- 测量页 ----
let scaleClicks: { x: number; y: number }[] = [];
let settingScale = false;
const measureCanvas = $<HTMLCanvasElement>('measureCanvas');

function ensureMeasureImage() {
  if (!state.correctedCanvas) {
    alert('请先在校正页完成折射校正');
    return false;
  }
  if (measureCanvas.width !== state.correctedCanvas.width) {
    measureCanvas.width = state.correctedCanvas.width;
    measureCanvas.height = state.correctedCanvas.height;
  }
  const ctx = measureCanvas.getContext('2d')!;
  ctx.drawImage(state.correctedCanvas, 0, 0);
  return true;
}

$('setScaleBar').addEventListener('click', () => {
  if (!ensureMeasureImage()) return;
  settingScale = true;
  scaleClicks = [];
  $('measureInfo').textContent = '请依次点击比例尺残段的两个端点（泥沙盖住一半也可用可见段）';
});

$('clearMeasure').addEventListener('click', () => {
  state.measurePoints = [];
  state.scaleBar = null;
  state.measurements = [];
  scaleClicks = [];
  redrawMeasure();
  $('measureInfo').textContent = '';
  $('measureList').innerHTML = '';
});

measureCanvas.addEventListener('click', (ev) => {
  if (!ensureMeasureImage()) return;
  const rect = measureCanvas.getBoundingClientRect();
  const sx = measureCanvas.width / rect.width;
  const sy = measureCanvas.height / rect.height;
  const p = {
    x: (ev.clientX - rect.left) * sx,
    y: (ev.clientY - rect.top) * sy,
  };
  const depth = parseFloat($<HTMLInputElement>('refDepth').value);
  if (settingScale) {
    scaleClicks.push(p);
    if (scaleClicks.length === 2) {
      const known = parseFloat(
        prompt('该比例尺残段的已知长度（mm）：') ?? '0',
      );
      if (known > 0) {
        setScaleBarFlow(scaleClicks[0], scaleClicks[1], known, depth);
        $('measureInfo').textContent = `比例尺已设置：${known} mm`;
      }
      settingScale = false;
    }
  } else {
    addMeasurePoint(p, depth);
  }
  redrawMeasure();
  renderMeasurements();
});

function redrawMeasure() {
  if (!state.correctedCanvas) return;
  const ctx = measureCanvas.getContext('2d')!;
  ctx.drawImage(state.correctedCanvas, 0, 0);
  const drawDot = (p: { x: number; y: number }, color: string) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
  };
  // 比例尺
  if (state.scaleBar) {
    ctx.strokeStyle = '#ffb454';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(state.scaleBar.p1.x, state.scaleBar.p1.y);
    ctx.lineTo(state.scaleBar.p2.x, state.scaleBar.p2.y);
    ctx.stroke();
    drawDot(state.scaleBar.p1, '#ffb454');
    drawDot(state.scaleBar.p2, '#ffb454');
  }
  // 测量点与连线
  state.measurePoints.forEach((p, i) => {
    drawDot(p, '#3b9eff');
    ctx.fillStyle = '#fff';
    ctx.fillText(String(i + 1), p.x + 8, p.y - 8);
    if (i > 0) {
      ctx.strokeStyle = '#2bd0a8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(state.measurePoints[i - 1].x, state.measurePoints[i - 1].y);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
  });
}

function renderMeasurements() {
  const depth = parseFloat($<HTMLInputElement>('refDepth').value);
  const list = $('measureList');
  const scale = state.scaleBar ? '比例尺标定' : `针孔当量 ${mmPerPixel(depth).toFixed(4)} mm/px`;
  const header = `<div class="measure-row"><strong>测量结果（${scale}）</strong></div>`;
  const rows = state.measurements
    .map(
      (m, i) =>
        `<div class="measure-row"><span>段 ${i + 1}：像素 ${m.pixels.toFixed(1)}</span><span>${m.lengthMm.toFixed(2)} mm</span></div>`,
    )
    .join('');
  list.innerHTML = header + rows;
}

// ---- 目录页 ----
async function renderCatalog() {
  const sites = await db.listSites();
  const box = $('catalogList');
  if (sites.length === 0) {
    box.innerHTML = '<div class="catalog-card">暂无保存的记录。</div>';
    return;
  }
  box.innerHTML = '';
  for (const s of sites) {
    for (const diveDate of s.dates) {
      const dates = await db.listImages(s.site, diveDate);
      const card = document.createElement('div');
      card.className = 'catalog-card';
      card.innerHTML = `
        <h3>${s.site} · ${diveDate}</h3>
        <div class="meta"><span>图像 ${dates.length} 张</span></div>`;
      for (const rec of dates) {
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.style.marginTop = '8px';
        const lines = rec.measurements
          .map((m, i) => `段${i + 1}: ${m.lengthMm.toFixed(1)}mm`)
          .join(' · ');
        meta.textContent = `${rec.fileName} · 深度 ${rec.referenceDepthMm}mm · ${lines || '无测量'}`;
        card.appendChild(meta);
      }
      box.appendChild(card);
    }
  }
}

$('saveRecord').addEventListener('click', async () => {
  const site = $<HTMLInputElement>('siteId').value.trim();
  const diveDate = $<HTMLInputElement>('diveDate').value;
  if (!site || !diveDate) {
    alert('请填写探方编号和日期');
    return;
  }
  const depth = parseFloat($<HTMLInputElement>('refDepth').value);
  await saveImageRecord(site, diveDate, depth);
  alert('记录已保存到本地目录');
  await renderCatalog();
});

void renderCatalog();
