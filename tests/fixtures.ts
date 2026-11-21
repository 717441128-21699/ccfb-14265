import { mulberry32, synthesizeCheckerboard, makePoses } from '../src/core/synthetic';
import { toGray } from '../src/core/image';
import type {
  BrownDistortion,
  Intrinsics,
  RefractionParams,
} from '../src/core/types';

export const FIXTURE = {
  width: 720,
  height: 540,
  cols: 9,
  rows: 7,
  squareMm: 30,
};

export const TRUE_INTRINSICS: Intrinsics = {
  fx: 820,
  fy: 815,
  cx: 360,
  cy: 270,
};

export const TRUE_DISTORTION: BrownDistortion = {
  k1: -0.08,
  k2: 0.03,
  p1: 0.0006,
  p2: -0.0004,
  k3: -0.005,
};

export const TRUE_REFRACTION: RefractionParams = {
  d: 28,
  tg: 6,
  ng: 1.49,
  nw: 1.335,
  tiltX: 0,
  tiltY: 0,
};

export function makeSyntheticSet(count = 10, seed = 42, withSediment = false) {
  const rng = mulberry32(seed);
  const poses = makePoses(count, rng, 850);
  return poses.map((pose, i) => {
    const rgba = synthesizeCheckerboard({
      width: FIXTURE.width,
      height: FIXTURE.height,
      intrinsics: TRUE_INTRINSICS,
      distortion: TRUE_DISTORTION,
      refraction: TRUE_REFRACTION,
      pose,
      cols: FIXTURE.cols,
      rows: FIXTURE.rows,
      squareMm: FIXTURE.squareMm,
      noiseSigma: 2.5,
      sedimentFraction: withSediment ? 0.06 : 0,
      seed: seed * 100 + i,
    });
    return { rgba, gray: toGray(rgba), pose };
  });
}
