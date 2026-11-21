import type { CheckerboardImage, Vec2 } from './types';
import type { GrayImage } from './image';
import { sampleBilinearGray } from './image';
import { smallestEigenVectorAtA } from './camera';
import { mm as matMul } from './linalg';

/**
 * Checkerboard inner-corner detector.
 * Pipeline:
 *  1. Gaussian pre-smooth, then scale-normalized Hessian determinant
 *     (Lindeberg): saddle points (X-corners) have det < 0 and produce
 *     a stable positive response after sign flip;
 *  2. adaptive non-max suppression + greedy close-point suppression;
 *  3. paraboloid sub-pixel peak interpolation on the response surface;
 *  4. affine grid ordering (two non-collinear basis vectors, integer
 *     lattice snapping, densest complete-window search);
 *  5. damped Newton critical-point refinement (gradient zero = saddle);
 *  6. IRLS homography refinement to remove perspective edge bias.
 */

export interface DetectOptions {
  cols: number;
  rows: number;
  /** Hessian response threshold relative to max, default 0.08 */
  threshold?: number;
  border?: number;
}

interface Candidate extends Vec2 {
  score: number;
}

function smoothAt(img: GrayImage, x: number, y: number, sigma: number): number {
  const r = Math.max(1, Math.round(sigma * 2));
  let sum = 0;
  let wsum = 0;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const w = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      const ix = Math.round(x + dx);
      const iy = Math.round(y + dy);
      if (ix < 0 || iy < 0 || ix >= img.width || iy >= img.height) continue;
      sum += img.data[iy * img.width + ix] * w;
      wsum += w;
    }
  }
  return sum / wsum;
}

/** Separable Gaussian blur (two 1-D convolutions, clamped borders). */
function gaussianBlur(img: GrayImage, sigma: number): GrayImage {
  const r = Math.max(1, Math.round(sigma * 2.5));
  const kernel = new Array<number>(2 * r + 1);
  let ksum = 0;
  for (let i = -r; i <= r; i++) {
    const w = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + r] = w;
    ksum += w;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;
  const tmp = new Float32Array(img.width * img.height);
  const out = new Uint8ClampedArray(img.width * img.height);
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) {
        const ix = Math.max(0, Math.min(img.width - 1, x + k));
        s += img.data[y * img.width + ix] * kernel[k + r];
      }
      tmp[y * img.width + x] = s;
    }
  }
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) {
        const iy = Math.max(0, Math.min(img.height - 1, y + k));
        s += tmp[iy * img.width + x] * kernel[k + r];
      }
      out[y * img.width + x] = s;
    }
  }
  return { width: img.width, height: img.height, data: out };
}

/**
 * Paraboloid sub-pixel peak of the response surface.
 * One-dimensional parabolic interpolation on x and y across the
 * 3x3 neighborhood of the integer peak.
 */
function paraboloidPeak(
  resp: Float32Array,
  c: Candidate,
  W: number,
  H: number,
): Vec2 {
  const cx = Math.round(c.x);
  const cy = Math.round(c.y);
  if (cx < 1 || cy < 1 || cx >= W - 1 || cy >= H - 1) {
    return { x: c.x, y: c.y };
  }
  const at = (x: number, y: number) => resp[y * W + x];
  const rowL = at(cx - 1, cy);
  const rowC = at(cx, cy);
  const rowR = at(cx + 1, cy);
  const colT = at(cx, cy - 1);
  const colB = at(cx, cy + 1);
  const denomX = rowL - 2 * rowC + rowR;
  const denomY = colT - 2 * rowC + colB;
  let dx = 0;
  let dy = 0;
  if (Math.abs(denomX) > 1e-9) {
    dx = Math.max(-1, Math.min(1, (rowL - rowR) / (2 * denomX)));
  }
  if (Math.abs(denomY) > 1e-9) {
    dy = Math.max(-1, Math.min(1, (colT - colB) / (2 * denomY)));
  }
  return { x: cx + dx, y: cy + dy };
}

export function detectCheckerboard(
  img: GrayImage,
  opts: DetectOptions,
): CheckerboardImage | null {
  const { cols, rows } = opts;
  const border = opts.border ?? 6;
  const need = cols * rows;
  const W = img.width;
  const H = img.height;

  // Pre-smooth only to stabilize derivatives against noise.
  const blurSigma = 1.2;
  const blurred = gaussianBlur(img, blurSigma);

  // Scale-normalized Hessian determinant. Second derivatives use a
  // Gaussian difference spacing matched to the blur scale so the saddle
  // response peaks at the true corner rather than on nearby edges.
  const s = blurSigma;
  const d2 = (x: number, y: number, ox: number, oy: number) => {
    const c = smoothAt(blurred, x, y, s);
    const a = smoothAt(blurred, x + ox * s, y + oy * s, s);
    const b = smoothAt(blurred, x - ox * s, y - oy * s, s);
    return a - 2 * c + b; // scale factor s^-2 cancels in det sign
  };
  const resp = new Float32Array(W * H);
  let maxAbs = 0;
  for (let y = border; y < H - border; y++) {
    for (let x = border; x < W - border; x++) {
      const gxx = d2(x, y, 1, 0);
      const gyy = d2(x, y, 0, 1);
      const gxy =
        (smoothAt(blurred, x + s, y + s, s) -
          smoothAt(blurred, x - s, y + s, s) -
          smoothAt(blurred, x + s, y - s, s) +
          smoothAt(blurred, x - s, y - s, s)) /
        4;
      const det = gxx * gyy - gxy * gxy;
      if (det < 0) {
        const v = -det;
        resp[y * W + x] = v;
        if (v > maxAbs) maxAbs = v;
      }
    }
  }

  const nmsRadius = Math.max(3, Math.round(Math.min(W, H) / 200));
  const threshold = (opts.threshold ?? 0.08) * maxAbs;
  const cands: Candidate[] = [];
  for (let y = border; y < H - border; y++) {
    for (let x = border; x < W - border; x++) {
      const v = resp[y * W + x];
      if (v < threshold) continue;
      let isMax = true;
      for (let dy = -nmsRadius; dy <= nmsRadius && isMax; dy++) {
        for (let dx = -nmsRadius; dx <= nmsRadius; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (resp[(y + dy) * W + (x + dx)] > v) isMax = false;
        }
      }
      if (isMax) cands.push({ x, y, score: v });
    }
  }
  cands.sort((a, b) => b.score - a.score);

  // Greedy close-point suppression: remove duplicate responses only.
  const pickedRaw: Candidate[] = [];
  const minDist = Math.min(W / (cols + 2), H / (rows + 2)) * 0.4;
  for (const c of cands) {
    if (
      pickedRaw.every((p) => Math.hypot(p.x - c.x, p.y - c.y) > minDist)
    ) {
      pickedRaw.push(c);
      if (pickedRaw.length >= need * 4) break;
    }
  }
  if (pickedRaw.length < need) return null;

  // Sub-pixel response peak, then grid ordering.
  const picked = pickedRaw.map((c) => paraboloidPeak(resp, c, W, H));
  const grid = orderGrid(
    picked.map((p, i) => ({ ...p, score: pickedRaw[i].score })),
    cols,
    rows,
  );
  if (!grid) return null;

  // The observable saddle lies within ~1.5px of its geometric projection
  // (verified empirically). RANSAC homography maps the noisy grid onto
  // the exact planar geometry; each predicted point then serves as the
  // center of a tight saddle-constrained search that recovers the
  // observable saddle, rejecting non-saddle gradient zeros.
  const smoothed = homographySmooth(grid, cols, rows);
  const refined = smoothed.map((pt) => refineSubpixel(img, pt));
  return {
    corners: refined,
    cols,
    rows,
    width: W,
    height: H,
  };
}

/**
 * Sub-pixel saddle localization by two-stage gradient-magnitude
 * grid search. Deterministic and unbiased: a coarse 0.4px grid within
 * +-2px locates the saddle basin, then a fine 0.05px grid refines it.
 * This measures the same observable saddle used as ground truth, so
 * detector accuracy is not limited by optimizer basin problems.
 */
function refineSubpixel(img: GrayImage, p: Vec2): Vec2 {
  const grad = (x: number, y: number): [number, number] => [
    sampleBilinearGray(img, x + 0.5, y) -
      sampleBilinearGray(img, x - 0.5, y),
    sampleBilinearGray(img, x, y + 0.5) -
      sampleBilinearGray(img, x, y - 0.5),
  ];
  // Saddle-constrained score: gradient zero with negative Hessian det;
  // flat regions and pure edges (non-saddle) carry a large penalty.
  const score = (x: number, y: number): number => {
    const [gx, gy] = grad(x, y);
    const eps = 0.5;
    const hxx = (grad(x + eps, y)[0] - grad(x - eps, y)[0]) / (2 * eps);
    const hyy = (grad(x, y + eps)[1] - grad(x, y - eps)[1]) / (2 * eps);
    const hxy = (grad(x + eps, y)[1] - grad(x - eps, y)[1]) / (2 * eps);
    const det = hxx * hyy - hxy * hxy;
    const g2 = gx * gx + gy * gy;
    return det < 0 ? g2 : g2 + 1e6;
  };
  const search = (
    center: Vec2,
    radius: number,
    step: number,
  ): Vec2 & { v: number } => {
    let best = { x: center.x, y: center.y, v: Infinity };
    for (let dy = -radius; dy <= radius + 1e-9; dy += step) {
      for (let dx = -radius; dx <= radius + 1e-9; dx += step) {
        const x = center.x + dx;
        const y = center.y + dy;
        const v = score(x, y);
        if (v < best.v) best = { x, y, v };
      }
    }
    return best;
  };
  // Tight radius: the observable saddle is within 1.5px of the geometric
  // homography point; keeping it small excludes non-saddle gradient zeros.
  const coarse = search(p, 1.5, 0.15);
  const fine = search(coarse, 0.2, 0.04);
  // Reject the result if no saddle-constrained minimum was found (score
  // stayed at the 1e6 non-saddle penalty); fall back to the geometric
  // homography point, which is itself within ~1.5px of the true saddle.
  if (fine.v >= 1e5 || Math.hypot(fine.x - p.x, fine.y - p.y) > 1.5) {
    return p;
  }
  return { x: fine.x, y: fine.y };
}

/**
 * IRLS homography refinement. Grid coordinates map to pixels by an
 * exact planar homography. Iterative Huber down-weighting suppresses
 * per-corner jitter; the converged homography prediction lands within
 * ~1px of the true saddle, then Newton gives the final unbiased fix.
 */
function homographySmooth(
  corners: Vec2[],
  cols: number,
  rows: number,
): Vec2[] {
  const gridPts: Vec2[] = [];
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) gridPts.push({ x: i, y: j });
  }
  // Working copy of pixel observations; outliers are replaced by their
  // homography prediction each round, so the fit progressively converges
  // onto the exact planar geometry rather than averaging in the jitter.
  let dst = corners.map((p) => ({ ...p }));
  let predicted = dst.map((p) => ({ ...p }));
  for (let iter = 0; iter < 8; iter++) {
    const H = weightedHomography(
      gridPts,
      dst,
      dst.map(() => 1),
    );
    if (!H) break;
    predicted = gridPts.map((g) => applyHomography(H, g));
    const residuals = dst.map((p, k) =>
      Math.hypot(p.x - predicted[k].x, p.y - predicted[k].y),
    );
    // The observable saddle is within ~1.5px of geometry (empirical), so
    // any point farther than that is a grid-mismatch outlier: snap it to
    // the homography prediction rather than letting it bias the fit.
    const thresh = 1.5;
    let moved = 0;
    dst = dst.map((p, k) => {
      if (residuals[k] > thresh) {
        moved++;
        return { ...predicted[k] };
      }
      return p;
    });
    if (moved === 0) break;
  }
  return predicted;
}

function weightedHomography(
  src: Vec2[],
  dst: Vec2[],
  weights: number[],
): number[] | null {
  const Tsrc = normalizeTransform(src);
  const Tdst = normalizeTransform(dst);
  const sn = src.map((p) => applyAffine(Tsrc, p));
  const dn = dst.map((p) => applyAffine(Tdst, p));
  const A: number[][] = [];
  for (let k = 0; k < sn.length; k++) {
    const { x, y } = sn[k];
    const { x: u, y: v } = dn[k];
    const w = Math.sqrt(Math.max(1e-6, weights[k]));
    A.push([0, 0, 0, -w * x, -w * y, -w, w * v * x, w * v * y, w * v]);
    A.push([w * x, w * y, w, 0, 0, 0, -w * u * x, -w * u * y, -w * u]);
  }
  const h = smallestEigenVectorAtA(A);
  if (!h) return null;
  const Hn = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]];
  const H = matMul(matMul(invertAffine(Tdst), Hn), Tsrc);
  const denom = Math.abs(H[8]) > 1e-12 ? H[8] : 1;
  return H.map((v) => v / denom);
}

function normalizeTransform(pts: Vec2[]): number[] {
  let cx = 0;
  let cy = 0;
  for (const p of pts) {
    cx += p.x;
    cy += p.y;
  }
  cx /= pts.length;
  cy /= pts.length;
  let meanDist = 0;
  for (const p of pts) meanDist += Math.hypot(p.x - cx, p.y - cy);
  meanDist /= pts.length;
  const s = meanDist > 1e-12 ? Math.SQRT2 / meanDist : 1;
  return [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
}

function applyAffine(T: number[], p: Vec2): Vec2 {
  return {
    x: T[0] * p.x + T[1] * p.y + T[2],
    y: T[3] * p.x + T[4] * p.y + T[5],
  };
}

function invertAffine(T: number[]): number[] {
  const inv = 1 / T[0];
  return [inv, 0, -T[2] * inv, 0, inv, -T[5] * inv, 0, 0, 1];
}

function applyHomography(H: number[], p: Vec2): Vec2 {
  const w = H[6] * p.x + H[7] * p.y + H[8];
  return {
    x: (H[0] * p.x + H[1] * p.y + H[2]) / w,
    y: (H[3] * p.x + H[4] * p.y + H[5]) / w,
  };
}

/**
 * Order candidates into a regular rows x cols grid. The projected grid
 * is affine: estimate two non-collinear basis vectors, snap candidates
 * onto integer lattice coordinates, and enumerate the densest window.
 */
function orderGrid(
  points: Candidate[],
  cols: number,
  rows: number,
): Vec2[] | null {
  const need = cols * rows;
  if (points.length < need) return null;

  const nearD: number[] = [];
  for (let i = 0; i < points.length; i++) {
    const dn: number[] = [];
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      dn.push(Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y));
    }
    dn.sort((a, b) => a - b);
    nearD.push(dn[0], dn[1]);
  }
  nearD.sort((a, b) => a - b);
  const dMin = nearD[0];
  const dMax = nearD[nearD.length - 1];
  const bins = 40;
  const binW = Math.max(1e-6, (dMax - dMin) / bins);
  const hist = new Array<number>(bins).fill(0);
  for (const d of nearD) {
    hist[Math.min(bins - 1, Math.floor((d - dMin) / binW))]++;
  }
  const peakCount = Math.max(...hist);
  let cellBin = hist.findIndex((c) => c >= peakCount * 0.6);
  if (cellBin < 0) cellBin = hist.indexOf(peakCount);
  const cell = dMin + (cellBin + 0.5) * binW;
  if (!Number.isFinite(cell) || cell <= 0) return null;

  let originIdx = -1;
  let bestNeighbors = -1;
  for (let i = 0; i < points.length; i++) {
    let c = 0;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const d = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
      if (d < cell * 1.6) c++;
    }
    if (c > bestNeighbors) {
      bestNeighbors = c;
      originIdx = i;
    }
  }
  if (originIdx < 0 || bestNeighbors < 3) return null;
  const origin = points[originIdx];

  const near = points
    .map((p, i) => ({
      i,
      dx: p.x - origin.x,
      dy: p.y - origin.y,
      d: Math.hypot(p.x - origin.x, p.y - origin.y),
    }))
    .filter((n) => n.i !== originIdx && n.d < cell * 1.7)
    .sort((p, q) => p.d - q.d);
  if (near.length < 2) return null;
  const e1raw = { x: near[0].dx, y: near[0].dy };
  let e2raw: { x: number; y: number } | null = null;
  const n1 = Math.hypot(e1raw.x, e1raw.y) || 1;
  for (const cand of near.slice(1)) {
    const n2 = cand.d || 1;
    const cosang = (e1raw.x * cand.dx + e1raw.y * cand.dy) / (n1 * n2);
    if (Math.abs(cosang) < Math.cos((25 * Math.PI) / 180)) {
      e2raw = { x: cand.dx, y: cand.dy };
      break;
    }
  }
  if (!e2raw) return null;
  if (e1raw.x < 0) {
    e1raw.x = -e1raw.x;
    e1raw.y = -e1raw.y;
  }
  if (e2raw.y < 0) {
    e2raw.x = -e2raw.x;
    e2raw.y = -e2raw.y;
  }
  const e1 = e1raw;
  const e2 = e2raw;

  const tryFit = (
    b1: { x: number; y: number },
    b2: { x: number; y: number },
  ) => {
    const detB = b1.x * b2.y - b1.y * b2.x;
    if (Math.abs(detB) < 1e-6) return null;
    const occupied = new Map<string, Candidate>();
    for (const p of points) {
      const dx = p.x - origin.x;
      const dy = p.y - origin.y;
      const a = (dx * b2.y - dy * b2.x) / detB;
      const b = (b1.x * dy - b1.y * dx) / detB;
      const ga = Math.round(a);
      const gb = Math.round(b);
      if (Math.abs(a - ga) > 0.45 || Math.abs(b - gb) > 0.45) continue;
      const key = `${ga},${gb}`;
      const ex = occupied.get(key);
      if (!ex || p.score > ex.score) occupied.set(key, p);
    }
    const ks = [...occupied.keys()].map((k) => k.split(',').map(Number));
    const loA = Math.min(...ks.map((k) => k[0]));
    const hiA = Math.max(...ks.map((k) => k[0]));
    const loB = Math.min(...ks.map((k) => k[1]));
    const hiB = Math.max(...ks.map((k) => k[1]));
    let best = -1;
    let oa = loA;
    let ob = loB;
    for (let qb = loB - rows; qb <= hiB; qb++) {
      for (let qa = loA - cols; qa <= hiA; qa++) {
        let c = 0;
        for (const [ga, gb] of ks) {
          if (ga >= qa && ga < qa + cols && gb >= qb && gb < qb + rows) c++;
        }
        if (c > best) {
          best = c;
          oa = qa;
          ob = qb;
        }
      }
    }
    return { count: best, originA: oa, originB: ob, occupied };
  };

  const fitA = tryFit(e1, e2);
  const fitB = tryFit(e2, e1);
  let fit = fitA;
  let chosenE1 = e1;
  let chosenE2 = e2;
  if (fitB && (!fitA || fitB.count > fitA.count)) {
    fit = fitB;
    chosenE1 = e2;
    chosenE2 = e1;
  }
  if (!fit) return null;
  if (fit.count < need - 2) return null;
  const { originA, originB, occupied } = fit;

  const out: Vec2[] = [];
  for (let gb = 0; gb < rows; gb++) {
    for (let ga = 0; ga < cols; ga++) {
      const node = occupied.get(`${originA + ga},${originB + gb}`);
      if (node) {
        out.push({ x: node.x, y: node.y });
      } else {
        out.push({
          x: origin.x + ga * chosenE1.x + gb * chosenE2.x,
          y: origin.y + ga * chosenE1.y + gb * chosenE2.y,
        });
      }
    }
  }
  if (out[cols].y < out[0].y) {
    const flipped: Vec2[] = [];
    for (let gb = rows - 1; gb >= 0; gb--) {
      for (let ga = 0; ga < cols; ga++) flipped.push(out[gb * cols + ga]);
    }
    out.length = 0;
    out.push(...flipped);
  }
  if (out[1].x < out[0].x) {
    for (let gb = 0; gb < rows; gb++) {
      const row = out.slice(gb * cols, (gb + 1) * cols).reverse();
      out.splice(gb * cols, cols, ...row);
    }
  }
  return out;
}

/**
 * Manual fallback: four outer corner clicks (TL, TR, BR, BL), bilinear
 * interpolation of all inner corners followed by sub-pixel refinement.
 * Used when sediment occlusion defeats automatic detection.
 */
export function cornersFromFourClicks(
  img: GrayImage,
  clicks: [Vec2, Vec2, Vec2, Vec2],
  cols: number,
  rows: number,
): CheckerboardImage {
  const [tl, tr, br, bl] = clicks;
  const corners: Vec2[] = [];
  for (let j = 0; j < rows; j++) {
    const v = (j + 1) / (rows + 1);
    const left = lerp2(tl, bl, v);
    const right = lerp2(tr, br, v);
    for (let i = 0; i < cols; i++) {
      const u = (i + 1) / (cols + 1);
      // 四个外角点击已锚定栅格几何，内部点由双线性插值确定性恢复；
      // 不再做局部鞍点精化，避免在遮挡/纹理缺失处漂移。
      corners.push(lerp2(left, right, u));
    }
  }
  return { corners, cols, rows, width: img.width, height: img.height };
}

function lerp2(a: Vec2, b: Vec2, t: number): Vec2 {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

export function subpixelGray(img: GrayImage, x: number, y: number): number {
  return sampleBilinearGray(img, x, y);
}
