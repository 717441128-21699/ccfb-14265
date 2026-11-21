import {
  checkerboardPoints,
  findHomography,
  poseFromHomography,
  projectWorldPoint,
} from './camera';
import { levenbergMarquardt } from './lm';
import { rodrigues } from './linalg';
import type {
  BrownDistortion,
  Calibration,
  CalibrationStats,
  CheckerboardImage,
  Intrinsics,
  Pose,
  RefractionParams,
  Vec3,
} from './types';

export interface CalibrationOptions {
  squareMm: number;
  initialRefraction?: Partial<RefractionParams>;
  /** 固定 nw（由水温/盐度模型给定），只优化几何参数 */
  fixNw?: boolean;
  /** 是否优化窗倾斜，默认最后阶段开启 */
  estimateTilt?: boolean;
  name?: string;
}

interface ImageObservation {
  img: CheckerboardImage;
  board: Vec3[];
  pose: Pose;
  rod: number[];
  t: number[];
}

const DIST_ZERO: BrownDistortion = {
  k1: 0,
  k2: 0,
  p1: 0,
  p2: 0,
  k3: 0,
};

function defaultIntrinsics(w: number, h: number): Intrinsics {
  const f = Math.max(w, h) * 1.2;
  return { fx: f, fy: f, cx: w / 2, cy: h / 2 };
}

function defaultRefraction(
  init?: Partial<RefractionParams>,
): RefractionParams {
  return {
    d: init?.d ?? 30,
    tg: init?.tg ?? 5,
    ng: init?.ng ?? 1.49,
    nw: init?.nw ?? 1.333,
    tiltX: init?.tiltX ?? 0,
    tiltY: init?.tiltY ?? 0,
  };
}

function poseToParams(obs: ImageObservation): number[] {
  return [...obs.rod, ...obs.t];
}

function paramsToPose(p: number[]): Pose {
  return {
    R: rodrigues(p[0], p[1], p[2]),
    t: { x: p[3], y: p[4], z: p[5] },
  };
}

function rmsOf(
  obs: ImageObservation[],
  K: Intrinsics,
  dist: BrownDistortion,
  refr: RefractionParams,
): { rms: number; max: number; perImage: number[] } {
  let sumSq = 0;
  let count = 0;
  let max = 0;
  const perImage: number[] = [];
  for (const o of obs) {
    let imgSq = 0;
    for (let i = 0; i < o.board.length; i++) {
      const proj = projectWorldPoint(o.board[i], o.pose, refr, K, dist);
      if (!proj) continue;
      const dx = proj.x - o.img.corners[i].x;
      const dy = proj.y - o.img.corners[i].y;
      const e2 = dx * dx + dy * dy;
      sumSq += e2;
      imgSq += e2;
      max = Math.max(max, Math.sqrt(e2));
      count++;
    }
    perImage.push(Math.sqrt(imgSq / o.board.length));
  }
  return { rms: Math.sqrt(sumSq / Math.max(1, count)), max, perImage };
}

export function calibrate(
  images: CheckerboardImage[],
  opts: CalibrationOptions,
): Calibration {
  if (images.length < 3) {
    throw new Error('至少需要 3 张不同姿态的棋盘格图像进行标定');
  }
  const w = images[0].width;
  const h = images[0].height;
  let K = defaultIntrinsics(w, h);
  let dist = { ...DIST_ZERO };
  let refr = defaultRefraction(opts.initialRefraction);

  const obs: ImageObservation[] = images.map((img) => {
    const board = checkerboardPoints(img.cols, img.rows, opts.squareMm);
    const H = findHomography(
      board.map((p) => ({ x: p.x, y: p.y })),
      img.corners,
    );
    if (!H) throw new Error('单应矩阵求解失败，角点配置退化');
    const pose = poseFromHomography(H, K);
    const rod = [0, 0, 0]; // 初值零向量，后续由当前 R 反推意义不大，直接优化
    return {
      img,
      board,
      pose,
      rod,
      t: [pose.t.x, pose.t.y, pose.t.z],
    };
  });

  const syncPoses = () => {
    for (const o of obs) o.pose = paramsToPose(poseToParams(o));
  };

  // ---- 阶段 1：固定内参与折射，逐图优化位姿 ----
  for (const o of obs) {
    const resFn = (p: number[]) => {
      const pose = paramsToPose(p);
      const out: number[] = [];
      for (let i = 0; i < o.board.length; i++) {
        const proj = projectWorldPoint(o.board[i], pose, refr, K, dist);
        if (!proj) {
          out.push(1e4, 1e4);
          continue;
        }
        out.push(proj.x - o.img.corners[i].x, proj.y - o.img.corners[i].y);
      }
      return out;
    };
    const r = levenbergMarquardt(poseToParams(o), resFn, {
      maxIterations: 40,
      scales: [0.1, 0.1, 0.1, 10, 10, 100],
    });
    o.rod = r.params.slice(0, 3);
    o.t = r.params.slice(3);
    o.pose = paramsToPose(r.params);
  }

  // ---- 阶段 2：全局联合优化 ----
  // 布局：[fx,fy,cx,cy, k1,k2,p1,p2,k3, d,tg,ng,(nw), (tiltX,tiltY)]，
  // 随后是每张图 6 个位姿参数。
  const buildGlobal = (includeTilt: boolean): number[] => {
    const head = [
      K.fx,
      K.fy,
      K.cx,
      K.cy,
      dist.k1,
      dist.k2,
      dist.p1,
      dist.p2,
      dist.k3,
      refr.d,
      refr.tg,
      refr.ng,
    ];
    if (!opts.fixNw) head.push(refr.nw);
    if (includeTilt) head.push(refr.tiltX, refr.tiltY);
    for (const o of obs) head.push(...o.rod, ...o.t);
    return head;
  };

  const unpackGlobal = (p: number[], includeTilt: boolean) => {
    K = { fx: p[0], fy: p[1], cx: p[2], cy: p[3] };
    dist = { k1: p[4], k2: p[5], p1: p[6], p2: p[7], k3: p[8] };
    refr = {
      d: p[9],
      tg: p[10],
      ng: p[11],
      nw: opts.fixNw ? refr.nw : p[12],
      tiltX: 0,
      tiltY: 0,
    };
    let cursor = opts.fixNw ? 12 : 13;
    if (includeTilt) {
      refr.tiltX = p[cursor];
      refr.tiltY = p[cursor + 1];
      cursor += 2;
    }
    for (let k = 0; k < obs.length; k++) {
      const q = p.slice(cursor + k * 6, cursor + k * 6 + 6);
      obs[k].rod = q.slice(0, 3);
      obs[k].t = q.slice(3);
      obs[k].pose = paramsToPose(q);
    }
  };

  const globalResiduals = (p: number[], includeTilt: boolean) => {
    unpackGlobal(p, includeTilt);
    const out: number[] = [];
    for (const o of obs) {
      for (let i = 0; i < o.board.length; i++) {
        const proj = projectWorldPoint(o.board[i], o.pose, refr, K, dist);
        if (!proj) {
          out.push(1e4, 1e4);
          continue;
        }
        out.push(
          proj.x - o.img.corners[i].x,
          proj.y - o.img.corners[i].y,
        );
      }
    }
    return out;
  };

  const globalScales = (p: number[], includeTilt: boolean) => {
    const s = [
      100, 100, 10, 10, // K
      0.1, 0.1, 0.01, 0.01, 0.1, // 畸变
      10, 2, 0.05, // d,tg,ng
    ];
    if (!opts.fixNw) s.push(0.01);
    if (includeTilt) s.push(0.01, 0.01);
    for (let k = 0; k < obs.length; k++) {
      s.push(0.1, 0.1, 0.1, 10, 10, 100);
    }
    return s.slice(0, p.length);
  };

  const runGlobal = (includeTilt: boolean) => {
    const p0 = buildGlobal(includeTilt);
    const r = levenbergMarquardt(
      p0,
      (p) => globalResiduals(p, includeTilt),
      {
        maxIterations: includeTilt ? 60 : 40,
        scales: globalScales(p0, includeTilt),
      },
    );
    unpackGlobal(r.params, includeTilt);
    return r;
  };

  runGlobal(false);
  syncPoses();
  if (opts.estimateTilt) {
    runGlobal(true);
    syncPoses();
  }

  // 物理量约束：折射率与厚度必须为正
  refr.d = Math.max(refr.d, 1e-3);
  refr.tg = Math.max(refr.tg, 0.1);
  refr.ng = Math.max(refr.ng, 1.01);
  refr.nw = Math.max(refr.nw, 1.01);
  K.fx = Math.max(K.fx, 1);
  K.fy = Math.max(K.fy, 1);

  const stats0 = rmsOf(obs, K, dist, refr);
  const pointCount = obs.reduce((s, o) => s + o.board.length, 0);
  const stats: CalibrationStats = {
    reprojectionRms: stats0.rms,
    maxError: stats0.max,
    imageCount: obs.length,
    pointCount,
    perImageRms: stats0.perImage,
  };

  return {
    id: `cal_${Date.now().toString(36)}`,
    name: opts.name ?? '水下折射标定',
    createdAt: new Date().toISOString(),
    imageWidth: w,
    imageHeight: h,
    intrinsics: K,
    distortion: dist,
    refraction: refr,
    stats,
  };
}

/** 用已有标定计算新图像的重投影误差（验收用） */
export function evaluateCalibration(
  cal: Calibration,
  images: CheckerboardImage[],
  squareMm: number,
): CalibrationStats {
  const tmp: ImageObservation[] = images.map((img) => {
    const board = checkerboardPoints(img.cols, img.rows, squareMm);
    const H = findHomography(
      board.map((p) => ({ x: p.x, y: p.y })),
      img.corners,
    );
    const pose = H
      ? poseFromHomography(H, cal.intrinsics)
      : {
          R: [1, 0, 0, 0, 1, 0, 0, 0, 1],
          t: { x: 0, y: 0, z: 1000 },
        };
    return {
      img,
      board,
      pose,
      rod: [0, 0, 0],
      t: [pose.t.x, pose.t.y, pose.t.z],
    };
  });
  // 位姿精修后再统计
  for (const o of tmp) {
    const resFn = (p: number[]) => {
      const pose = paramsToPose(p);
      const out: number[] = [];
      for (let i = 0; i < o.board.length; i++) {
        const proj = projectWorldPoint(
          o.board[i],
          pose,
          cal.refraction,
          cal.intrinsics,
          cal.distortion,
        );
        if (!proj) {
          out.push(1e4, 1e4);
          continue;
        }
        out.push(proj.x - o.img.corners[i].x, proj.y - o.img.corners[i].y);
      }
      return out;
    };
    const r = levenbergMarquardt(
      [0, 0, 0, o.pose.t.x, o.pose.t.y, o.pose.t.z],
      resFn,
      { maxIterations: 40, scales: [0.1, 0.1, 0.1, 10, 10, 100] },
    );
    o.pose = paramsToPose(r.params);
  }
  const s = rmsOf(tmp, cal.intrinsics, cal.distortion, cal.refraction);
  return {
    reprojectionRms: s.rms,
    maxError: s.max,
    imageCount: images.length,
    pointCount: tmp.reduce((a, o) => a + o.board.length, 0),
    perImageRms: s.perImage,
  };
}
