/**
 * 测试夹具：统一从核心演示场景模块再导出，保持测试与浏览器演示同源几何。
 */
export {
  makeGroundTruth,
  makeScene,
  makeTiltedPlaneScene,
  type GroundTruth,
} from '../src/core/demoScene';
