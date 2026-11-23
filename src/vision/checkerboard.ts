/**
 * 轻量棋盘格内角点检测（无外部依赖）。
 *
 * 两阶段：
 * 1) 候选检测：灰度 → 四象限明暗翻转打分（鞍点）→ NMS → 亚像素拟合 → 去重；
 * 2) 网格排序：最近邻图上做 BFS 网格坐标传播，再用 homography 最小二乘精化。
 *
 * 适用于对比度良好、平面平整的水下棋盘照片；网格断裂（遮挡/出界）时返回空，
 * 由调用方换图或手工标记。
 */
import { CornerObs } from '../core/calibration';

export interface DetectOptions {
  rows: number;
  cols: number;
  threshold?: number;
  nmsRadius?: number;
}

function toGray(data: Uint8ClampedArray): Float32Array {
  const g = new Float32Array((data.length / 4) | 0);
  for (let i = 0; i < g.length; i++) {
    const o = i * 4;
    g[i] = 0.299 * data[o] + 0.587 * data[o + 1] + 0.114 * data[o + 2];
  }
  return g;
}

/** 四象限明暗翻转响应：对角同色、相邻异色 */
function checkerResponse(g: Float32Array, w: number, x: number, y: number, r: number): number {
  let a = 0, b = 0, c = 0, d = 0;
  let na = 0, nb = 0, nc = 0, nd = 0;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const val = g[(y + dy) * w + (x + dx)];
      if (dx <= 0 && dy <= 0) { a += val; na++; }
      else if (dx > 0 && dy <= 0) { b += val; nb++; }
      else if (dx <= 0 && dy > 0) { c += val; nc++; }
      else { d += val; nd++; }
    }
  }
  a /= na; b /= nb; c /= nc; d /= nd;
  return Math.max(Math.abs((a + d) - (b + c)), Math.abs((a + b) - (c + d)));
}

function subPixelOffset(g: Float32Array, w: number, x: number, y: number): { dx: number; dy: number } {
  const at = (xx: number, yy: number) => g[yy * w + xx];
  const gx = (at(x + 1, y) - at(x - 1, y)) / 2;
  const gy = (at(x, y + 1) - at(x, y - 1)) / 2;
  const gxx = at(x + 1, y) + at(x - 1, y) - 2 * at(x, y);
  const gyy = at(x, y + 1) + at(x, y - 1) - 2 * at(x, y);
  const gxy = (at(x + 1, y + 1) - at(x + 1, y - 1) - at(x - 1, y + 1) + at(x - 1, y - 1)) / 4;
  const det = gxx * gyy - gxy * gxy;
  if (Math.abs(det) < 1e-9) return { dx: 0, dy: 0 };
  const dx = (gyy * gx - gxy * gy) / -det;
  const dy = (gxx * gy - gxy * gx) / -det;
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || Math.abs(dx) > 1 || Math.abs(dy) > 1) {
    return { dx: 0, dy: 0 };
  }
  return { dx, dy };
}

export interface Candidate {
  u: number;
  v: number;
  score: number;
}

/** NMS + 亚像素 + 去重，返回无序候选点 */
export function detectCandidates(
  img: { width: number; height: number; data: Uint8ClampedArray },
  opts: { threshold?: number; nmsRadius?: number } = {},
): Candidate[] {
  const threshold = opts.threshold ?? 0.15;
  const nmsRadius = opts.nmsRadius ?? Math.max(4, Math.round(Math.min(img.width, img.height) / 130));
  const w = img.width;
  const h = img.height;
  const g = toGray(img.data);
  const r = Math.max(4, Math.round(Math.min(w, h) / 40));
  const resp = new Float32Array(w * h);
  let maxResp = 0;
  for (let y = r + 2; y < h - r - 2; y++) {
    for (let x = r + 2; x < w - r - 2; x++) {
      const s = checkerResponse(g, w, x, y, r);
      resp[y * w + x] = s;
      if (s > maxResp) maxResp = s;
    }
  }
  const cut = maxResp * threshold;
  const raw: Candidate[] = [];
  for (let y = r + 2; y < h - r - 2; y++) {
    for (let x = r + 2; x < w - r - 2; x++) {
      const s = resp[y * w + x];
      if (s < cut) continue;
      let isMax = true;
      for (let dy = -nmsRadius; dy <= nmsRadius && isMax; dy++) {
        for (let dx = -nmsRadius; dx <= nmsRadius; dx++) {
          if (resp[(y + dy) * w + (x + dx)] > s) {
            isMax = false;
            break;
          }
        }
      }
      if (isMax) {
        const { dx, dy } = subPixelOffset(g, w, x, y);
        raw.push({ u: x + dx, v: y + dy, score: s });
      }
    }
  }
  raw.sort((a, b) => b.score - a.score);
  const mergeDist = nmsRadius * 0.7;
  const merged: Candidate[] = [];
  for (const cand of raw) {
    if (!merged.some((m) => Math.hypot(m.u - cand.u, m.v - cand.v) < mergeDist)) {
      merged.push(cand);
    }
  }
  return merged;
}

/** 通用 NxN 高斯消元（部分主元），A row-major */
function solveNxN(Ain: Float64Array, bin: Float64Array, n: number): number[] | null {
  const A = new Float64Array(Ain);
  const b = Array.from(bin);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let rr = col + 1; rr < n; rr++) {
      if (Math.abs(A[rr * n + col]) > Math.abs(A[piv * n + col])) piv = rr;
    }
    if (Math.abs(A[piv * n + col]) < 1e-12) return null;
    if (piv !== col) {
      for (let k = 0; k < n; k++) [A[col * n + k], A[piv * n + k]] = [A[piv * n + k], A[col * n + k]];
      [b[col], b[piv]] = [b[piv], b[col]];
    }
    for (let rr = 0; rr < n; rr++) {
      if (rr === col) continue;
      const f = A[rr * n + col] / A[col * n + col];
      for (let k = col; k < n; k++) A[rr * n + k] -= f * A[col * n + k];
      b[rr] -= f * b[col];
    }
  }
  return b.map((bi, i) => bi / A[i * n + i]);
}

/**
 * 网格排序：BFS 传播 + homography 精化。
 *
 * 1) 估计节距 gap，构建每个候选在 ~1.4gap 内的最近邻表；
 * 2) 对多个候选种子，尝试其邻居对作为两个网格轴，BFS 给每个可达点
 *    分配 (r,c)：右邻居 c+1、下邻居 r+1；
 * 3) 用分配结果拟合 homography，按模型重吸附并校验 rows×cols 完整命中。
 */
function orderGrid(points: Candidate[], rows: number, cols: number): CornerObs[] {
  const need = rows * cols;
  if (points.length < need) return [];

  // 节距：最近邻距离的下四分位
  const nn: number[] = [];
  for (let i = 0; i < points.length; i++) {
    let bd = Infinity;
    for (let j = 0; j < points.length; j++) {
      if (i !== j) bd = Math.min(bd, Math.hypot(points[i].u - points[j].u, points[i].v - points[j].v));
    }
    nn.push(bd);
  }
  nn.sort((a, b) => a - b);
  const gap = nn[Math.floor(nn.length * 0.25)];
  if (!Number.isFinite(gap) || gap < 2) return [];
  const linkDist = gap * 1.4;

  // 邻居表
  const neighbors: number[][] = points.map(() => []);
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      if (Math.hypot(points[i].u - points[j].u, points[i].v - points[j].v) <= linkDist) {
        neighbors[i].push(j);
        neighbors[j].push(i);
      }
    }
  }

  const fitHomography = (corr: { r: number; c: number; u: number; v: number }[]) => {
    const M = new Float64Array(64);
    const b = new Float64Array(8);
    for (const k of corr) {
      const x = [k.r, k.c, 1, 0, 0, 0, -k.u * k.r, -k.u * k.c];
      const y = [0, 0, 0, k.r, k.c, 1, -k.v * k.r, -k.v * k.c];
      for (let i = 0; i < 8; i++) {
        b[i] += x[i] * k.u + y[i] * k.v;
        for (let j = 0; j < 8; j++) M[i * 8 + j] += x[i] * x[j] + y[i] * y[j];
      }
    }
    const h = solveNxN(M, b, 8);
    if (!h) return null;
    return (r: number, c: number) => {
      const ww = h[6] * r + h[7] * c + 1;
      return { u: (h[0] * r + h[1] * c + h[2]) / ww, v: (h[3] * r + h[4] * c + h[5]) / ww };
    };
  };

  // 以每个候选为种子，取其两个邻居作为轴，BFS 传播网格坐标
  const seedOrder = points.map((_, i) => i).slice(0, Math.min(30, points.length));
  for (const seedIdx of seedOrder) {
    const nbrs = neighbors[seedIdx];
    if (nbrs.length < 2) continue;
    for (let ai = 0; ai < nbrs.length; ai++) {
      for (let bi = ai + 1; bi < nbrs.length; bi++) {
        const colIdx = nbrs[ai];
        const rowIdx = nbrs[bi];
        // BFS 传播：label[i] = (r,c)
        const label = new Map<number, { r: number; c: number }>();
        label.set(seedIdx, { r: 0, c: 0 });
        label.set(colIdx, { r: 0, c: 1 });
        label.set(rowIdx, { r: 1, c: 0 });
        const queue = [seedIdx, colIdx, rowIdx];
        while (queue.length) {
          const cur = queue.shift()!;
          const lab = label.get(cur)!;
          // 找当前点的右(c+1)、下(r+1)：用已标定邻居的方向向量预测
          for (const nb of neighbors[cur]) {
            if (label.has(nb)) continue;
            // 预测：从 col 轴/row 轴基向量
            const baseCol = { u: points[colIdx].u - points[seedIdx].u, v: points[colIdx].v - points[seedIdx].v };
            const baseRow = { u: points[rowIdx].u - points[seedIdx].u, v: points[rowIdx].v - points[seedIdx].v };
            const predC = {
              u: points[cur].u + baseCol.u,
              v: points[cur].v + baseCol.v,
            };
            const predR = { u: points[cur].u + baseRow.u, v: points[cur].v + baseRow.v };
            const dC = Math.hypot(points[nb].u - predC.u, points[nb].v - predC.v);
            const dR = Math.hypot(points[nb].u - predR.u, points[nb].v - predR.v);
            if (dC < gap * 0.7) {
              label.set(nb, { r: lab.r, c: lab.c + 1 });
              queue.push(nb);
            } else if (dR < gap * 0.7) {
              label.set(nb, { r: lab.r + 1, c: lab.c });
              queue.push(nb);
            }
          }
        }

        // 坐标平移使最小 r,c 为 0
        let minR = Infinity, minC = Infinity;
        for (const lab of label.values()) {
          minR = Math.min(minR, lab.r);
          minC = Math.min(minC, lab.c);
        }
        // BFS 仅用于确认存在一个 rows×cols 的连通网格点集；
        // 精确 (r,c) 标签交给 homography 软分配迭代，避免传播累积误差。
        const gridPoints: number[] = [];
        for (const [idx, lab] of label) {
          const r = lab.r - minR;
          const c = lab.c - minC;
          if (r >= 0 && r < rows && c >= 0 && c < cols) gridPoints.push(idx);
        }
        if (gridPoints.length < need) continue;

        // 用 BFS 的粗略标签做初值拟合 homography
        const initCorr = Array.from(label.entries())
          .map(([idx, lab]) => ({ r: lab.r - minR, c: lab.c - minC, u: points[idx].u, v: points[idx].v }))
          .filter((k) => k.r >= 0 && k.r < rows && k.c >= 0 && k.c < cols);
        let model = fitHomography(initCorr);
        if (!model) continue;

        // 软分配迭代：模型预测 → 一对一吸附 → 重拟合
        let result: CornerObs[] | null = null;
        for (let iter = 0; iter < 15; iter++) {
          result = resnap(model, points, rows, cols, gap * 0.85);
          if (!result) break;
          const refit = fitHomography(result.map((k) => ({ r: k.r, c: k.c, u: k.u, v: k.v })));
          if (!refit) break;
          let move = 0;
          for (const k of result) {
            const pred = refit(k.r, k.c);
            move += Math.hypot(pred.u - k.u, pred.v - k.v);
          }
          model = refit;
          if (move / result.length < 0.05) break;
        }
        // 残差校验：正确网格的 homography 拟合残差应远小于节距；
        // 错位/错误定向的网格残差大，判为失败（返回空，交给手工标记）。
        if (result && result.length === need) {
          const verify = fitHomography(result.map((k) => ({ r: k.r, c: k.c, u: k.u, v: k.v })));
          if (verify) {
            let resid = 0;
            for (const k of result) {
              const pred = verify(k.r, k.c);
              resid += Math.hypot(pred.u - k.u, pred.v - k.v);
            }
            // 仅当 homography 拟合残差很小时接受；定向错误的网格虽自洽，
            // 但 BFS 传播通常伴随较大局部残差，阈值取 0.25 节距。
            if (resid / need < gap * 0.25) {
              result.sort((a, b) => a.r - b.r || a.c - b.c);
              return result;
            }
          }
        }
      }
    }
  }
  return [];
}

/** 按 homography 模型把候选一对一重吸附到 rows×cols 节点，全部命中才返回 */
function resnap(
  model: (r: number, c: number) => { u: number; v: number },
  points: Candidate[],
  rows: number,
  cols: number,
  tol: number,
): CornerObs[] | null {
  const pairs: { r: number; c: number; idx: number; d: number }[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const pred = model(r, c);
      for (let i = 0; i < points.length; i++) {
        const d = Math.hypot(points[i].u - pred.u, points[i].v - pred.v);
        if (d <= tol) pairs.push({ r, c, idx: i, d });
      }
    }
  }
  pairs.sort((a, b) => a.d - b.d);
  const nodeTaken = new Set<string>();
  const candTaken = new Set<number>();
  const out: CornerObs[] = [];
  for (const p of pairs) {
    const key = p.r + ':' + p.c;
    if (nodeTaken.has(key) || candTaken.has(p.idx)) continue;
    nodeTaken.add(key);
    candTaken.add(p.idx);
    out.push({ r: p.r, c: p.c, u: points[p.idx].u, v: points[p.idx].v });
  }
  return out.length === rows * cols ? out : null;
}

export function detectCheckerboard(
  img: { width: number; height: number; data: Uint8ClampedArray },
  opts: DetectOptions,
): CornerObs[] {
  const candidates = detectCandidates(img, { threshold: opts.threshold, nmsRadius: opts.nmsRadius });
  return orderGrid(candidates, opts.rows, opts.cols);
}
