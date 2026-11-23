// 档案管理: 按 探方号/日期 组织目录
//   <根目录>/<探方号>/<YYYY-MM-DD>/calibration.json
//   <根目录>/<探方号>/<YYYY-MM-DD>/corrected/<照片名>
//   <根目录>/<探方号>/<YYYY-MM-DD>/measurements.json
// 使用 File System Access API (Chrome/Edge)。不可用时退化为浏览器下载。

export interface DiveFolder {
  root: FileSystemDirectoryHandle;
  tanfang: string;
  date: string; // YYYY-MM-DD
}

export const fsAccessSupported = (): boolean =>
  typeof (window as any).showDirectoryPicker === 'function';

export async function pickRoot(): Promise<FileSystemDirectoryHandle> {
  return await (window as any).showDirectoryPicker({ mode: 'readwrite' });
}

async function ensureDir(parent: FileSystemDirectoryHandle, name: string): Promise<FileSystemDirectoryHandle> {
  return await parent.getDirectoryHandle(name, { create: true });
}

export async function diveFolder(root: FileSystemDirectoryHandle, tanfang: string, date: string): Promise<FileSystemDirectoryHandle> {
  const safeTanfang = tanfang.replace(/[\\/:*?"<>|]/g, '_').trim() || '未命名探方';
  const t = await ensureDir(root, safeTanfang);
  return await ensureDir(t, date);
}

export async function saveJSON(dir: FileSystemDirectoryHandle, name: string, obj: unknown): Promise<void> {
  const fh = await dir.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(JSON.stringify(obj, null, 2));
  await w.close();
}

export async function loadJSON<T>(dir: FileSystemDirectoryHandle, name: string): Promise<T | null> {
  try {
    const fh = await dir.getFileHandle(name);
    const f = await fh.getFile();
    return JSON.parse(await f.text()) as T;
  } catch {
    return null;
  }
}

export async function saveBlob(dir: FileSystemDirectoryHandle, name: string, blob: Blob): Promise<void> {
  const fh = await dir.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(blob);
  await w.close();
}

export async function listDirs(dir: FileSystemDirectoryHandle): Promise<string[]> {
  const names: string[] = [];
  for await (const [name, handle] of (dir as any).entries()) {
    if (handle.kind === 'directory') names.push(name);
  }
  return names.sort();
}

/** 无 File System Access API 时的退化保存: 触发浏览器下载 */
export function downloadBlob(name: string, blob: Blob): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}
