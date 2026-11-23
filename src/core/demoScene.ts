/**
 * 合成演示/测试场景：真实的双界面折射几何 + 多视角棋盘格 + 半埋比例尺 + 已知器物。
 * 棋盘格、比例尺、器物都严格位于探方平面内；窗口法向带装配倾斜。
 * 同时供单元测试与浏览器内置演示使用。
 */
import { Vec3, v3, add, scale, rodrigues, mv, dot } from './vec3';
import { Intrinsics, PortParams } from './model';
import { Plane3D } from './rectify';
import { SyntheticScene } from './synthetic';
import { refractiveIndices } from './water';

export interface GroundTruth {
  intrinsics: Intrinsics;
  port: PortParams;
  plane: Plane3D;
}

export function makeGroundTruth(): GroundTruth {
  const n = refractiveIndices({ temperatureC: 18, salinityPSU: 32 });
  const intrinsics: Intrinsics = { fx: 1200, fy: 1200, cx: 500, cy: 375 };
  const port: PortParams = {
    nx: 0.012,
    ny: -0.008,
    nz: 1,
    d0: 28,
    t: 6,
    nAir: n.nAir,
    nGlass: n.nGlass,
    nWater: n.nWater,
  };
  const plane: Plane3D = { q: v3(0, 0, 1500), n: v3(0, 0, 1) };
  return { intrinsics, port, plane };
}

function sub3(a: Vec3, b: Vec3): Vec3 {
  return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}
function add3(a: Vec3, b: Vec3): Vec3 {
  return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}
function norm3(a: Vec3): Vec3 {
  const l = Math.hypot(a.x, a.y, a.z);
  return v3(a.x / l, a.y / l, a.z / l);
}
function cross3(a: Vec3, b: Vec3): Vec3 {
  return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/** 在给定平面内放置棋盘格：以平面点 q+shift 为中心，绕平面法向旋转 */
function placeBoard(
  plane: Plane3D,
  rows: number,
  cols: number,
  squareMm: number,
  shift: Vec3,
  angleRad: number,
): SyntheticScene['board'] {
  const helper = Math.abs(plane.n.z) < 0.9 ? v3(0, 0, 1) : v3(0, 1, 0);
  const ex0 = norm3(sub3(helper, scale(plane.n, dot(helper, plane.n))));
  const ey0 = cross3(plane.n, ex0);
  const ex = add3(scale(ex0, Math.cos(angleRad)), scale(ey0, Math.sin(angleRad)));
  const ey = add3(scale(ex0, -Math.sin(angleRad)), scale(ey0, Math.cos(angleRad)));
  const center = add(plane.q, shift);
  const origin = add(
    center,
    add(scale(ex, -((cols - 1) * squareMm) / 2), scale(ey, -((rows - 1) * squareMm) / 2)),
  );
  return { rows, cols, squareMm, origin, ex, ey };
}

export function makeScene(viewIndex: number): SyntheticScene {
  const gt = makeGroundTruth();
  const width = 1000;
  const height = 750;
  const rows = 9;
  const cols = 12;
  const squareMm = 20;

  // 6 个视角：不同平移、平面内旋转与拍摄深度，模拟手持多姿态拍摄。
  // 深度多样性是打破焦距-平移尺度耦合的关键。
  const shifts: Vec3[] = [
    v3(-120, -60, -250),
    v3(140, 40, 150),
    v3(0, 0, 0),
    v3(-60, 90, 350),
    v3(90, -100, -100),
    v3(40, 20, 250),
  ];
  const angles = [0.08, -0.12, 0.03, 0.18, -0.05, 0.25];
  const board = placeBoard(gt.plane, rows, cols, squareMm, shifts[viewIndex], angles[viewIndex]);

  // 半埋比例尺：全长 300mm，40% 被覆盖（可见 180mm）
  const barStart = add(gt.plane.q, v3(300, 220, 0));
  const barEnd = add(gt.plane.q, v3(600, 220, 0));
  // 已知器物：半径 55mm 圆盘（直径 110mm）
  const artCenter = add(gt.plane.q, v3(-260, 180, 0));

  return {
    width,
    height,
    intrinsics: gt.intrinsics,
    port: gt.port,
    plane: gt.plane,
    board,
    scaleBars: [
      { start: barStart, end: barEnd, widthMm: 12, fullLengthMm: 300, buriedFraction: 0.4 },
    ],
    artifacts: [{ center: artCenter, radiusMm: 55, shade: { r: 160, g: 90, b: 60 } }],
  };
}

/** 倾斜探方平面场景：法向绕 x/y 轴旋转，板/比例尺/器物都置于旋转后平面内 */
export function makeTiltedPlaneScene(): SyntheticScene {
  const scene = makeScene(0);
  const r = rodrigues(v3(0.18, -0.1, 0));
  const n = mv(r, v3(0, 0, 1));
  const plane: Plane3D = { q: scene.plane.q, n };
  const lift = (offset: Vec3): Vec3 => add(plane.q, mv(r, offset));
  scene.plane = plane;
  scene.board = placeBoard(plane, scene.board.rows, scene.board.cols, scene.board.squareMm, v3(-120, -60, 0), 0.08);
  scene.scaleBars = [
    {
      start: lift(v3(300, 220, 0)),
      end: lift(v3(600, 220, 0)),
      widthMm: 12,
      fullLengthMm: 300,
      buriedFraction: 0.4,
    },
  ];
  scene.artifacts = [{ center: lift(v3(-260, 180, 0)), radiusMm: 55, shade: { r: 160, g: 90, b: 60 } }];
  return scene;
}
