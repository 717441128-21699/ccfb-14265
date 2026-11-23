import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, '..', 'src', 'webgl', 'rectifier.ts'), 'utf8');

describe('WebGL2 着色器健全性', () => {
  it('包含与 CPU 同源的折射/投影关键逻辑', () => {
    expect(src).toContain('refractH');
    expect(src).toContain('traceBackward');
    expect(src).toContain('projectPoint');
    expect(src).toContain('#version 300 es');
    // 双界面 eta：空气/玻璃、玻璃/水
    expect(src).toContain('u_eta.x');
    expect(src).toContain('u_eta.y');
    // 牛顿迭代与最小二乘
    expect(src).toContain('for (int i = 0; i < 24; i++)');
    expect(src).toContain('a00 * a11 - a01 * a01');
  });

  it('GLSL 片段括号与花括号配平', () => {
    const fragMatch = src.match(/const FRAG = `([\s\S]*?)`;/);
    expect(fragMatch).not.toBeNull();
    const frag = fragMatch![1];
    const count = (ch: string) => frag.split(ch).length - 1;
    expect(count('(')).toBe(count(')'));
    expect(count('{')).toBe(count('}'));
    // 每个 uniform 都有宿主端设置
    for (const u of ['u_intr', 'u_port1', 'u_port2', 'u_eta', 'u_origin', 'u_ex', 'u_ey', 'u_mmPerPx', 'u_srcSize', 'u_outSize']) {
      expect(src).toContain(`'${u}'`);
    }
  });

  it('着色器折射 eta 与 CPU PortParams 字段约定一致', () => {
    // 宿主端必须以 nAir/nGlass、nGlass/nWater 顺序填充 u_eta
    expect(src).toContain('u.port.nAir / u.port.nGlass');
    expect(src).toContain('u.port.nGlass / u.port.nWater');
  });
});
