/**
 * 浏览器端演示数据：用合成折射几何生成棋盘 ImageData 与角点观测，
 * 使应用无需真实水下照片即可走通「检测→标定→校正→测量」全流程。
 * 几何与测试夹具同源（core/synthetic）。
 */
import { renderSynthetic, detectCheckerboardCorners, SyntheticScene } from '../core/synthetic';
import { makeScene } from '../core/demoScene';
import { CalibView } from '../core/calibration';

export interface DemoBundle {
  views: { name: string; imageData: ImageData; view: CalibView }[];
  scene: SyntheticScene;
}

export function buildDemoBundle(viewCount = 6): DemoBundle {
  const views: DemoBundle['views'] = [];
  let scene0!: SyntheticScene;
  for (let i = 0; i < viewCount; i++) {
    const scene = makeScene(i);
    if (i === 0) scene0 = scene;
    const img = renderSynthetic(scene, 1);
    const imageData = new ImageData(
      new Uint8ClampedArray(img.data),
      img.width,
      img.height,
    );
    const corners = detectCheckerboardCorners(scene);
    views.push({
      name: `synthetic-view-${i}.png`,
      imageData,
      view: { name: `synthetic-view-${i}.png`, width: scene.width, height: scene.height, corners },
    });
  }
  return { views, scene: scene0 };
}

/** 单张演示场景（含比例尺与器物），用于测量页 */
export function buildDemoMeasurementScene(): { imageData: ImageData; scene: SyntheticScene } {
  const scene = makeScene(0);
  const img = renderSynthetic(scene, 1);
  return {
    imageData: new ImageData(new Uint8ClampedArray(img.data), img.width, img.height),
    scene,
  };
}
