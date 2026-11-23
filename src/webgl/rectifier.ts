/**
 * WebGL2 正射校正重采样器。
 *
 * 对输出正射影像的每个像素：
 *   平面 3D 点（OrthoFrame）→ 含双界面折射的前向投影 → 源图采样坐标
 * 前向投影在着色器里用牛顿迭代（与 CPU projectPoint 同一数学），
 * 双线性采样。离屏 framebuffer + 32F 纹理保证大图像素不溢出。
 */
import type { Intrinsics, PortParams } from '../core/model';
import type { OrthoFrame } from '../core/rectify';

const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_src;
uniform vec2 u_srcSize;
uniform vec2 u_outSize;
uniform vec4 u_intr;   // fx, fy, cx, cy
uniform vec4 u_port1;  // nx, ny, nz, d0
uniform vec3 u_port2;  // t, nAir/nGlass eta1 placeholder, ...
uniform vec3 u_eta;    // etaAirGlass, etaGlassWater, nAir
uniform vec3 u_origin;
uniform vec3 u_ex;
uniform vec3 u_ey;
uniform float u_mmPerPx;

vec3 normalize3(vec3 v){ return v / length(v); }

// Hecht 约定折射：n 指向入射侧
vec3 refractH(vec3 I, vec3 n, float eta) {
  float cosi = -dot(I, n);
  float sin2t = eta * eta * (1.0 - cosi * cosi);
  if (sin2t >= 1.0) return vec3(0.0);
  float cost = sqrt(1.0 - sin2t);
  return normalize3(eta * I + (eta * cosi - cost) * n);
}

// 像素 -> 水中光线（起点在外表面）
void traceBackward(vec2 px, out vec3 origin, out vec3 dir) {
  vec3 dAir = normalize3(vec3((px.x - u_intr.z) / u_intr.x,
                              (px.y - u_intr.w) / u_intr.y, 1.0));
  vec3 n = normalize3(u_port1.xyz);
  vec3 nb = -n;
  float s0 = u_port1.w / dot(dAir, n);
  vec3 x0 = dAir * s0;
  vec3 dGlass = refractH(dAir, nb, u_eta.x);
  float s1 = u_port2.x / dot(dGlass, n);
  vec3 x1 = x0 + dGlass * s1;
  vec3 dWater = refractH(dGlass, nb, u_eta.y);
  origin = x1;
  dir = dWater;
}

// 水中光线上距 P 最近点
vec3 closestOnRay(vec2 px, vec3 P) {
  vec3 o, d;
  traceBackward(px, o, d);
  float lambda = dot(P - o, d);
  return o + d * lambda;
}

// 前向投影：相机系 3D 点 -> 源像素（牛顿迭代）
vec2 projectPoint(vec3 P) {
  vec3 g = normalize3(P);
  vec2 uv = vec2(u_intr.x * g.x / g.z + u_intr.z,
                 u_intr.y * g.y / g.z + u_intr.w);
  float eps = 0.05;
  for (int i = 0; i < 24; i++) {
    vec3 c0 = closestOnRay(uv, P);
    vec3 res = P - c0;
    if (length(res) < 1e-6) break;
    vec3 cu = closestOnRay(uv + vec2(eps, 0.0), P);
    vec3 cv = closestOnRay(uv + vec2(0.0, eps), P);
    // 雅可比两列（对 u、v 的偏导），显式 vec3 避免 mat 构造歧义
    vec3 j0 = (cu - c0) / eps;
    vec3 j1 = (cv - c0) / eps;
    // 最小二乘 (JᵀJ) δ = Jᵀ res
    float a00 = dot(j0, j0), a01 = dot(j0, j1), a11 = dot(j1, j1);
    vec2 bv = vec2(dot(j0, res), dot(j1, res));
    float det = a00 * a11 - a01 * a01;
    if (abs(det) < 1e-18) break;
    vec2 delta = vec2((a11 * bv.x - a01 * bv.y) / det,
                      (a00 * bv.y - a01 * bv.x) / det);
    uv += delta;
    if (length(delta) < 1e-5) break;
  }
  return uv;
}

void main() {
  vec2 outPx = v_uv * u_outSize;
  vec3 X = u_origin
         + u_ex * (outPx.x * u_mmPerPx)
         + u_ey * (outPx.y * u_mmPerPx);
  vec2 src = projectPoint(X);
  vec2 suv = src / u_srcSize;
  if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
    outColor = vec4(0.0, 0.0, 0.0, 0.0);
    return;
  }
  outColor = texture(u_src, suv);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('着色器编译失败: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

export interface RectifyUniforms {
  intr: Intrinsics;
  port: PortParams;
  frame: OrthoFrame;
  srcWidth: number;
  srcHeight: number;
  outWidth: number;
  outHeight: number;
}

export class WebGLRectifier {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private tex: WebGLTexture | null = null;
  private fbo: WebGLFramebuffer | null = null;
  private texW = 0;
  private texH = 0;

  constructor(canvas?: HTMLCanvasElement) {
    const c = canvas ?? document.createElement('canvas');
    const gl = c.getContext('webgl2', { preserveDrawingBuffer: true });
    if (!gl) throw new Error('当前环境不支持 WebGL2');
    this.gl = gl;
    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('着色器链接失败: ' + gl.getProgramInfoLog(prog));
    }
    this.program = prog;
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW,
    );
    const loc = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }

  private ensureTexture(w: number, h: number) {
    const gl = this.gl;
    if (this.tex && this.texW === w && this.texH === h) return;
    if (this.tex) gl.deleteTexture(this.tex);
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
    this.texW = w;
    this.texH = h;
  }

  /** 上传源图（ImageBitmap/HTMLImageElement/ImageData/Canvas）并校正到输出尺寸 */
  rectify(source: TexImageSource, u: RectifyUniforms): Uint8ClampedArray {
    const gl = this.gl;
    const { outWidth: W, outHeight: H } = u;
    this.ensureTexture(W, H);

    // 源纹理
    const srcTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    const set = (name: string, fn: 'uniform1f' | 'uniform2f' | 'uniform3f' | 'uniform4f', ...v: number[]) => {
      const loc = gl.getUniformLocation(this.program, name);
      if (loc) (gl as unknown as Record<string, (...a: unknown[]) => void>)[fn](loc, ...v);
    };
    const srcLoc = gl.getUniformLocation(this.program, 'u_src');
    if (srcLoc) gl.uniform1i(srcLoc, 0);
    set('u_srcSize', 'uniform2f', u.srcWidth, u.srcHeight);
    set('u_outSize', 'uniform2f', W, H);
    set('u_intr', 'uniform4f', u.intr.fx, u.intr.fy, u.intr.cx, u.intr.cy);
    set('u_port1', 'uniform4f', u.port.nx, u.port.ny, u.port.nz, u.port.d0);
    set('u_port2', 'uniform3f', u.port.t, 0, 0);
    set(
      'u_eta',
      'uniform3f',
      u.port.nAir / u.port.nGlass,
      u.port.nGlass / u.port.nWater,
      u.port.nAir,
    );
    set('u_origin', 'uniform3f', u.frame.origin.x, u.frame.origin.y, u.frame.origin.z);
    set('u_ex', 'uniform3f', u.frame.ex.x, u.frame.ex.y, u.frame.ex.z);
    set('u_ey', 'uniform3f', u.frame.ey.x, u.frame.ey.y, u.frame.ey.z);
    set('u_mmPerPx', 'uniform1f', u.frame.mmPerPx);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const pixels = new Uint8ClampedArray(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.deleteTexture(srcTex);
    // WebGL 原点在左下，翻转回图像坐标（左上原点）
    const flipped = new Uint8ClampedArray(pixels.length);
    const rowBytes = W * 4;
    for (let y = 0; y < H; y++) {
      flipped.set(pixels.subarray((H - 1 - y) * rowBytes, (H - y) * rowBytes), y * rowBytes);
    }
    return flipped;
  }

  dispose() {
    const gl = this.gl;
    if (this.tex) gl.deleteTexture(this.tex);
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
  }
}

export function isWebGL2Supported(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!c.getContext('webgl2');
  } catch {
    return false;
  }
}
