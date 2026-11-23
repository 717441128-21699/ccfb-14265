/**
 * 零依赖 PNG 编码器（RGBA 8-bit）。
 * deflate 使用 stored block（zlib 包装），体积略大但实现简单、完全确定性，
 * 用于合成测试图导出；生产环境浏览器侧走 canvas.toBlob。
 */

const CRC_TABLE: Uint32Array = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function adler32(data: Uint8Array): number {
  let a = 1;
  let b = 0;
  const MOD = 65521;
  for (let i = 0; i < data.length; i++) {
    a = (a + data[i]) % MOD;
    b = (b + a) % MOD;
  }
  return ((b << 16) | a) >>> 0;
}

function u32be(n: number): Uint8Array {
  return Uint8Array.from([(n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff]);
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const typeBuf = Uint8Array.from([...type].map((ch) => ch.charCodeAt(0)));
  const len = u32be(data.length);
  const body = new Uint8Array(typeBuf.length + data.length);
  body.set(typeBuf, 0);
  body.set(data, typeBuf.length);
  const crc = u32be(crc32(body));
  const out = new Uint8Array(4 + body.length + 4);
  out.set(len, 0);
  out.set(body, 4);
  out.set(crc, 4 + body.length);
  return out;
}

/** zlib 流（CMF/FLG + stored deflate blocks + adler32） */
function zlibStore(data: Uint8Array): Uint8Array {
  const MAX = 65535;
  const blocks = Math.ceil(data.length / MAX) || 1;
  // 每块 5 字节头
  const out = new Uint8Array(2 + data.length + blocks * 5 + 4);
  out[0] = 0x78; // CM=8 deflate, CINFO=7 (32K window)
  out[1] = 0x01; // FLG（0x78 0x01 的 adler 可被 31 整除）
  let p = 2;
  for (let off = 0; off < data.length || data.length === 0; ) {
    const remain = data.length - off;
    const n = Math.min(remain, MAX);
    const last = off + n >= data.length ? 1 : 0;
    out[p++] = last;
    out[p++] = n & 0xff;
    out[p++] = (n >> 8) & 0xff;
    out[p++] = (~n) & 0xff;
    out[p++] = ((~n) >> 8) & 0xff;
    out.set(data.subarray(off, off + n), p);
    p += n;
    off += n;
    if (last) break;
  }
  out.set(u32be(adler32(data)), p);
  return out;
}

export interface RawImage {
  width: number;
  height: number;
  /** RGBA */
  data: Uint8Array | Uint8ClampedArray;
}

export function encodePng(img: RawImage): Uint8Array {
  const { width, height, data } = img;
  // 加 filter byte（每行 0）
  const raw = new Uint8Array((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 4 + 1)] = 0;
    raw.set(data.subarray(y * width * 4, (y + 1) * width * 4), y * (width * 4 + 1) + 1);
  }
  const ihdr = new Uint8Array(13);
  ihdr.set(u32be(width), 0);
  ihdr.set(u32be(height), 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type RGBA
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const sig = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const c1 = chunk('IHDR', ihdr);
  const c2 = chunk('IDAT', zlibStore(raw));
  const c3 = chunk('IEND', new Uint8Array(0));
  const out = new Uint8Array(sig.length + c1.length + c2.length + c3.length);
  let o = 0;
  out.set(sig, o); o += sig.length;
  out.set(c1, o); o += c1.length;
  out.set(c2, o); o += c2.length;
  out.set(c3, o);
  return out;
}
