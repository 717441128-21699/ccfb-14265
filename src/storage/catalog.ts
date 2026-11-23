/**
 * 本地目录存储：IndexedDB 持久化标定结果、潜水记录与照片元数据。
 * 按「探方编号 / 日期」两级组织。图片本体为大对象，单独存 blob 仓库。
 * 纯前端、无网络依赖；不可用时降级到内存（数据不持久）。
 */

export interface DiveRecord {
  id: string;
  trench: string;          // 探方编号
  date: string;            // YYYY-MM-DD
  temperatureC: number;
  salinityPSU: number;
  depthM?: number;
  note?: string;
  createdAt: number;
}

export interface StoredCalibration {
  id: string;
  diveId: string;
  trench: string;
  date: string;
  /** CalibrationResult 的完整 JSON */
  payload: unknown;
  rms: number;
  maxError: number;
  createdAt: number;
}

export interface StoredPhoto {
  id: string;
  trench: string;
  date: string;
  calibrationId?: string;
  fileName: string;
  width: number;
  height: number;
  /** 校正后测量记录（两点、长度毫米） */
  measurements: { id: string; x1: number; y1: number; x2: number; y2: number; lengthMm: number; at: number }[];
  createdAt: number;
}

const DB_NAME = 'uw-refraction-tool';
const DB_VERSION = 1;
const STORES = {
  dives: 'dives',
  calibrations: 'calibrations',
  photos: 'photos',
  blobs: 'blobs',
} as const;

let dbPromise: Promise<IDBDatabase | null> | null = null;

function openDb(): Promise<IDBDatabase | null> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve) => {
    if (typeof indexedDB === 'undefined') {
      resolve(null);
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORES.dives)) {
        const s = db.createObjectStore(STORES.dives, { keyPath: 'id' });
        s.createIndex('trench', 'trench', { unique: false });
        s.createIndex('date', 'date', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORES.calibrations)) {
        const s = db.createObjectStore(STORES.calibrations, { keyPath: 'id' });
        s.createIndex('diveId', 'diveId', { unique: false });
        s.createIndex('trench_date', ['trench', 'date'], { unique: false });
      }
      if (!db.objectStoreNames.contains(STORES.photos)) {
        const s = db.createObjectStore(STORES.photos, { keyPath: 'id' });
        s.createIndex('trench_date', ['trench', 'date'], { unique: false });
      }
      if (!db.objectStoreNames.contains(STORES.blobs)) {
        db.createObjectStore(STORES.blobs);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
  return dbPromise;
}

function tx<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then((db) => {
    if (!db) throw new Error('IndexedDB 不可用');
    return new Promise<T>((resolve, reject) => {
      const t = db.transaction(store, mode);
      const req = fn(t.objectStore(store));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  });
}

export function genId(prefix = ''): string {
  return prefix + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export const catalog = {
  async putDive(d: DiveRecord): Promise<void> {
    await tx(STORES.dives, 'readwrite', (s) => s.put(d));
  },
  async listDives(trench?: string): Promise<DiveRecord[]> {
    const db = await openDb();
    if (!db) return [];
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORES.dives, 'readonly');
      const store = t.objectStore(STORES.dives);
      const req = trench ? store.index('trench').getAll(trench) : store.getAll();
      req.onsuccess = () => resolve((req.result as DiveRecord[]).sort((a, b) => b.createdAt - a.createdAt));
      req.onerror = () => reject(req.error);
    });
  },

  async putCalibration(c: StoredCalibration): Promise<void> {
    await tx(STORES.calibrations, 'readwrite', (s) => s.put(c));
  },
  async listCalibrations(trench: string, date: string): Promise<StoredCalibration[]> {
    const db = await openDb();
    if (!db) return [];
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORES.calibrations, 'readonly');
      const req = t.objectStore(STORES.calibrations).index('trench_date').getAll([trench, date]);
      req.onsuccess = () => resolve(req.result as StoredCalibration[]);
      req.onerror = () => reject(req.error);
    });
  },

  async putPhoto(p: StoredPhoto): Promise<void> {
    await tx(STORES.photos, 'readwrite', (s) => s.put(p));
  },
  async listPhotos(trench: string, date: string): Promise<StoredPhoto[]> {
    const db = await openDb();
    if (!db) return [];
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORES.photos, 'readonly');
      const req = t.objectStore(STORES.photos).index('trench_date').getAll([trench, date]);
      req.onsuccess = () => resolve((req.result as StoredPhoto[]).sort((a, b) => b.createdAt - a.createdAt));
      req.onerror = () => reject(req.error);
    });
  },

  async putBlob(id: string, blob: Blob): Promise<void> {
    await tx(STORES.blobs, 'readwrite', (s) => s.put(blob, id));
  },
  async getBlob(id: string): Promise<Blob | undefined> {
    const db = await openDb();
    if (!db) return undefined;
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORES.blobs, 'readonly');
      const req = t.objectStore(STORES.blobs).get(id);
      req.onsuccess = () => resolve(req.result as Blob | undefined);
      req.onerror = () => reject(req.error);
    });
  },

  /** 列出所有探方编号 */
  async listTrenches(): Promise<string[]> {
    const dives = await this.listDives();
    return Array.from(new Set(dives.map((d) => d.trench))).sort();
  },
};
