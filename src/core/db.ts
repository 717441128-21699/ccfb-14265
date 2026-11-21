import type { Calibration, Measurement, ScaleBar } from './types';

/**
 * 本地目录（IndexedDB）。
 * 层次：探方 site -> 一次下水日期 diveDate -> 标定/图像/测量。
 * 所有数据仅保存在本机浏览器，不上传任何服务器。
 */

export interface ImageRecord {
  id: string;
  site: string;
  diveDate: string; // YYYY-MM-DD
  fileName: string;
  calibrationId: string;
  width: number;
  height: number;
  referenceDepthMm: number;
  scaleBar: ScaleBar | null;
  measurements: Measurement[];
  /** 原图与校正图 Blob（JPEG/PNG） */
  originalBlob?: Blob;
  correctedBlob?: Blob;
  createdAt: string;
}

const DB_NAME = 'uw-refraction-calibrator';
const DB_VERSION = 1;
const STORE_CAL = 'calibrations';
const STORE_IMG = 'images';

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_CAL)) {
        const s = db.createObjectStore(STORE_CAL, { keyPath: 'id' });
        s.createIndex('site', 'site', { unique: false });
        s.createIndex('diveDate', 'diveDate', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORE_IMG)) {
        const s = db.createObjectStore(STORE_IMG, { keyPath: 'id' });
        s.createIndex('site', 'site', { unique: false });
        s.createIndex('diveDate', 'diveDate', { unique: false });
        s.createIndex('calibrationId', 'calibrationId', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx<T>(
  store: string,
  mode: IDBTransactionMode,
  fn: (s: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(store, mode);
        const req = fn(t.objectStore(store));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      }),
  );
}

export interface StoredCalibration extends Calibration {
  site: string;
  diveDate: string;
}

export const db = {
  async putCalibration(cal: StoredCalibration): Promise<void> {
    await tx(STORE_CAL, 'readwrite', (s) => s.put(cal));
  },
  async getCalibration(id: string): Promise<StoredCalibration | undefined> {
    return tx(STORE_CAL, 'readonly', (s) =>
      s.get(id),
    ) as Promise<StoredCalibration | undefined>;
  },
  async listCalibrations(site?: string): Promise<StoredCalibration[]> {
    const all = (await tx(STORE_CAL, 'readonly', (s) =>
      s.getAll(),
    )) as StoredCalibration[];
    const rows = site ? all.filter((r) => r.site === site) : all;
    return rows.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  },
  async deleteCalibration(id: string): Promise<void> {
    await tx(STORE_CAL, 'readwrite', (s) => s.delete(id));
  },
  async putImage(rec: ImageRecord): Promise<void> {
    await tx(STORE_IMG, 'readwrite', (s) => s.put(rec));
  },
  async getImage(id: string): Promise<ImageRecord | undefined> {
    return tx(STORE_IMG, 'readonly', (s) => s.get(id)) as Promise<
      ImageRecord | undefined
    >;
  },
  async listImages(site?: string, diveDate?: string): Promise<ImageRecord[]> {
    const all = (await tx(STORE_IMG, 'readonly', (s) =>
      s.getAll(),
    )) as ImageRecord[];
    return all
      .filter(
        (r) =>
          (!site || r.site === site) &&
          (!diveDate || r.diveDate === diveDate),
      )
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  },
  async deleteImage(id: string): Promise<void> {
    await tx(STORE_IMG, 'readwrite', (s) => s.delete(id));
  },
  async listSites(): Promise<{ site: string; dates: string[] }[]> {
    const cals = (await tx(STORE_CAL, 'readonly', (s) =>
      s.getAll(),
    )) as StoredCalibration[];
    const imgs = (await tx(STORE_IMG, 'readonly', (s) =>
      s.getAll(),
    )) as ImageRecord[];
    const map = new Map<string, Set<string>>();
    for (const r of [...cals, ...imgs]) {
      if (!map.has(r.site)) map.set(r.site, new Set());
      map.get(r.site)!.add(r.diveDate);
    }
    return [...map.entries()]
      .map(([site, dates]) => ({
        site,
        dates: [...dates].sort().reverse(),
      }))
      .sort((a, b) => a.site.localeCompare(b.site));
  },
};
