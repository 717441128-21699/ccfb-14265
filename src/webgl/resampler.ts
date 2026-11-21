import type { RemapTable } from '../core/remap';
import type { RgbaImage } from '../core/image';

/**
 * WebGL2 校正重采样。
 * - 源图：RGBA8 纹理，线性过滤
 * - 映射：RG32F 浮点纹理（WebGL2 核心可选格式，桌面/移动 GPU 普遍支持），
 *   存源像素坐标；NEAREST 采样后在着色器内手动双线性查映射
 * - 重采样：4x4 Catmull-Rom 双三次，抑制折射放大边缘的锯齿
 * 全分辨率直接渲染，输出可读 canvas / toBlob 下载，无 CPU 回读。
 */

const VERT = `#version 300 es
in vec2 aPos;
out vec2 vTex;
void main() {
  vTex = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform sampler2D uMap;
uniform vec2 uMapSize;
uniform vec2 uSrcSize;
uniform float uValid;
in vec2 vTex;
out vec4 outColor;

vec2 sampleMap(vec2 uv) {
  vec2 px = uv * uMapSize - 0.5;
  vec2 base = floor(px);
  vec2 f = fract(px);
  vec2 acc = vec2(0.0);
  for (int j = 0; j < 2; j++) {
    for (int i = 0; i < 2; i++) {
      vec2 ij = vec2(float(i), float(j));
      vec2 coord = (base + ij + 0.5) / uMapSize;
      coord = clamp(coord, vec2(0.0), vec2(1.0) - 1e-6);
      vec2 m = texture(uMap, coord).xy;
      float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
      acc += m * w;
    }
  }
  return acc;
}

float catmull(float x) {
  float ax = abs(x);
  if (ax < 1.0) return 1.5*ax*ax*ax - 2.5*ax*ax + 1.0;
  if (ax < 2.0) return -0.5*ax*ax*ax + 2.5*ax*ax - 4.0*ax + 2.0;
  return 0.0;
}

void main() {
  vec2 srcPx = sampleMap(vTex);
  vec2 center = srcPx + 0.5;
  if (srcPx.x < -1.0 || srcPx.y < -1.0 ||
      srcPx.x > uSrcSize.x || srcPx.y > uSrcSize.y) {
    outColor = vec4(0.02, 0.03, 0.04, 1.0);
    return;
  }
  vec4 col = vec4(0.0);
  float wsum = 0.0;
  for (int j = -1; j <= 2; j++) {
    for (int i = -1; i <= 2; i++) {
      vec2 sp = floor(center) + vec2(float(i), float(j));
      float w = catmull(center.x - sp.x) * catmull(center.y - sp.y);
      vec2 uv = (sp + 0.5) / uSrcSize;
      col += texture(uSrc, uv) * w;
      wsum += w;
    }
  }
  col /= max(wsum, 1e-6);
  outColor = vec4(clamp(col.rgb, 0.0, 1.0), 1.0);
}`;

export class GlResampler {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private srcTex: WebGLTexture;
  private mapTex: WebGLTexture;
  private fbo: WebGLFramebuffer;
  private mapW = 0;
  private mapH = 0;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      preserveDrawingBuffer: true,
      premultipliedAlpha: false,
    });
    if (!gl) throw new Error('当前浏览器不支持 WebGL2');
    this.gl = gl;
    this.program = this.buildProgram(VERT, FRAG);
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW,
    );
    const loc = gl.getAttribLocation(this.program, 'aPos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.srcTex = gl.createTexture()!;
    this.mapTex = gl.createTexture()!;
    this.fbo = gl.createFramebuffer()!;
  }

  private buildProgram(vs: string, fs: string): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error('着色器编译失败: ' + gl.getShaderInfoLog(sh));
      }
      return sh;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error('程序链接失败: ' + gl.getProgramInfoLog(p));
    }
    return p;
  }

  /** 上传或更新源图（ImageBitmap / Canvas / Image） */
  uploadSource(
    source: TexImageSource,
    width: number,
    height: number,
  ): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8,
      width,
      height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      source,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** 测试/离线路径：直接从 RgbaImage 上传 */
  uploadRgba(img: RgbaImage): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8,
      img.width,
      img.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      img.data,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  uploadMap(table: RemapTable): void {
    const gl = this.gl;
    // RG32F：x 存 mapX，y 存 mapY，无效点写 -10000
    const pixels = new Float32Array(table.width * table.height * 2);
    for (let i = 0; i < table.width * table.height; i++) {
      if (table.valid[i]) {
        pixels[i * 2] = table.mapX[i];
        pixels[i * 2 + 1] = table.mapY[i];
      } else {
        pixels[i * 2] = -10000;
        pixels[i * 2 + 1] = -10000;
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.mapTex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RG32F,
      table.width,
      table.height,
      0,
      gl.RG,
      gl.FLOAT,
      pixels,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.mapW = table.width;
    this.mapH = table.height;
  }

  /** 执行校正渲染到 canvas，返回 GPU 耗时（ms，含上传外的绘制时间） */
  render(srcWidth: number, srcHeight: number): number {
    const gl = this.gl;
    const t0 = performance.now();
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.uniform1i(gl.getUniformLocation(this.program, 'uSrc'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.mapTex);
    gl.uniform1i(gl.getUniformLocation(this.program, 'uMap'), 1);
    gl.uniform2f(
      gl.getUniformLocation(this.program, 'uMapSize'),
      this.mapW,
      this.mapH,
    );
    gl.uniform2f(
      gl.getUniformLocation(this.program, 'uSrcSize'),
      srcWidth,
      srcHeight,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.finish();
    return performance.now() - t0;
  }

  /** 渲染到离屏 framebuffer 并读回（测试用，测量真实像素） */
  renderToImage(srcWidth: number, srcHeight: number): RgbaImage {
    const gl = this.gl;
    const w = gl.canvas.width;
    const h = gl.canvas.height;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8,
      w,
      h,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      tex,
      0,
    );
    this.render(srcWidth, srcHeight);
    const data = new Uint8ClampedArray(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(tex);
    // GL 原点在左下，翻转到行主序左上原点
    const flipped = new Uint8ClampedArray(data.length);
    const rowBytes = w * 4;
    for (let y = 0; y < h; y++) {
      flipped.set(
        data.subarray((h - 1 - y) * rowBytes, (h - y) * rowBytes),
        y * rowBytes,
      );
    }
    return { width: w, height: h, data: flipped };
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteTexture(this.srcTex);
    gl.deleteTexture(this.mapTex);
    gl.deleteFramebuffer(this.fbo);
    gl.deleteVertexArray(this.vao);
  }
}

export function isWebGL2Available(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!c.getContext('webgl2');
  } catch {
    return false;
  }
}
