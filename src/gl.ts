// WebGL2 图像校正: 逐像素反折射重采样
// 对输出(校正后)图像的每个像素:
//   虚拟针孔内参 Kv -> 水中方向 w -> 反折射得空气斜率 (a,b) -> 镜头畸变 -> 原图采样
// 全部为闭式运算, 24MP(6000x4000)在普通 GPU 上远低于 2 秒。
import { Intrinsics } from './refraction';

const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uImg;
uniform vec2 uInSize;
uniform vec2 uOutSize;
uniform float uFx, uFy, uCx, uCy, uK1, uK2;
uniform float uFvx, uFvy, uCvx, uCvy;
uniform float uNW;
out vec4 frag;
void main() {
  float u = gl_FragCoord.x;
  float v = uOutSize.y - gl_FragCoord.y;   // 图像行(自上而下)
  float wx = (u - uCvx) / uFvx;
  float wy = (v - uCvy) / uFvy;
  float wn = inversesqrt(wx * wx + wy * wy + 1.0);
  wx *= wn; wy *= wn;
  float s = uNW;
  float t2 = s * s * (wx * wx + wy * wy);
  if (t2 >= 1.0) { frag = vec4(0.0, 0.0, 0.0, 1.0); return; }  // 全反射临界角之外
  float az = sqrt(1.0 - t2);
  float a = s * wx / az;
  float b = s * wy / az;
  float r2 = a * a + b * b;
  float dist = 1.0 + uK1 * r2 + uK2 * r2 * r2;
  float ui = uFx * a * dist + uCx;
  float vi = uFy * b * dist + uCy;
  if (ui < 0.0 || vi < 0.0 || ui >= uInSize.x || vi >= uInSize.y) {
    frag = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  frag = texture(uImg, vec2(ui / uInSize.x, vi / uInSize.y));
}`;

export interface CorrectionParams {
  K: Intrinsics;    // 原相机内参+畸变
  Kv: Intrinsics;   // 校正后虚拟相机内参 (k1=k2=0)
  nWater: number;
}

export class Corrector {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private tex: WebGLTexture;
  private inW = 0;
  private inH = 0;

  constructor() {
    this.canvas = document.createElement('canvas');
    const gl = this.canvas.getContext('webgl2', { preserveDrawingBuffer: true });
    if (!gl) throw new Error('当前浏览器不支持 WebGL2');
    this.gl = gl;
    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error('shader 编译失败: ' + gl.getShaderInfoLog(sh));
      }
      return sh;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('program 链接失败: ' + gl.getProgramInfoLog(prog));
    }
    this.prog = prog;
    this.tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  get maxTextureSize(): number {
    return this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE) as number;
  }

  /** 载入原始水下照片 */
  setImage(src: TexImageSource, width: number, height: number): void {
    const gl = this.gl;
    if (width > this.maxTextureSize || height > this.maxTextureSize) {
      throw new Error(`图像 ${width}x${height} 超过 GPU 纹理上限 ${this.maxTextureSize}`);
    }
    this.inW = width;
    this.inH = height;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
  }

  /** 执行校正, 返回校正后的画布 (outW x outH) */
  correct(outW: number, outH: number, p: CorrectionParams): HTMLCanvasElement {
    const gl = this.gl;
    this.canvas.width = outW;
    this.canvas.height = outH;
    gl.viewport(0, 0, outW, outH);
    gl.useProgram(this.prog);
    const u = (name: string) => gl.getUniformLocation(this.prog, name);
    gl.uniform2f(u('uInSize'), this.inW, this.inH);
    gl.uniform2f(u('uOutSize'), outW, outH);
    gl.uniform1f(u('uFx'), p.K.fx);
    gl.uniform1f(u('uFy'), p.K.fy);
    gl.uniform1f(u('uCx'), p.K.cx);
    gl.uniform1f(u('uCy'), p.K.cy);
    gl.uniform1f(u('uK1'), p.K.k1);
    gl.uniform1f(u('uK2'), p.K.k2);
    gl.uniform1f(u('uFvx'), p.Kv.fx);
    gl.uniform1f(u('uFvy'), p.Kv.fy);
    gl.uniform1f(u('uCvx'), p.Kv.cx);
    gl.uniform1f(u('uCvy'), p.Kv.cy);
    gl.uniform1f(u('uNW'), p.nWater);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(u('uImg'), 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return this.canvas;
  }
}

/** 由标定结果构造虚拟内参: 保持焦距与主点, 输出尺寸同原图 */
export function makeVirtualIntrinsics(K: Intrinsics, w: number, h: number): Intrinsics {
  return { fx: K.fx, fy: K.fy, cx: w / 2, cy: h / 2, k1: 0, k2: 0 };
}
