// ============================================================
// M7 — Agent persistence (IndexedDB 'aiface_agent').
// Model != Memory:
//   model          : per-language MultinomialNB artifacts (key = lang)
//   learningEvents : feedback/learning audit trail (auto-incremented)
//   memory         : long-term event memory (M8 grows this)
//   metadata       : { modelVersion, corpusHash, lastTrainedAt, schema }
// Mirrors the openDB/withStore pattern from chatStore.js.
// Pure helpers (no IDB) are importable under Node for Vitest.
// ============================================================

const DB_NAME = 'aiface_agent';
const DB_VERSION = 1;
const MODEL_STORE = 'model';
const EVENT_STORE = 'learningEvents';
const MEMORY_STORE = 'memory';
const META_STORE = 'metadata';

export const AGENT_STORE_NAMES = [MODEL_STORE, EVENT_STORE, MEMORY_STORE, META_STORE];

function openDB() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') { reject(new Error('IndexedDB unavailable')); return; }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      for (const name of AGENT_STORE_NAMES) {
        if (!db.objectStoreNames.contains(name)) {
          db.createObjectStore(name, name === EVENT_STORE ? { keyPath: 'id', autoIncrement: true } : {});
        }
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => {};
  });
}

function txDone(t) {
  return new Promise((resolve, reject) => {
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error || new Error('transaction aborted'));
  });
}

function reqResult(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function withStore(storeName, mode, fn) {
  const db = await openDB();
  try {
    const t = db.transaction(storeName, mode);
    const store = t.objectStore(storeName);
    const result = await fn(store, reqResult);
    await txDone(t);
    return result;
  } finally {
    db.close();
  }
}

// ---- model ----
export async function saveModel(language, artifact) {
  return withStore(MODEL_STORE, 'readwrite', (store, get) => get(store.put({ lang: language, artifact })));
}
export async function loadModel(language) {
  const row = await withStore(MODEL_STORE, 'readonly', (store, get) => get(store.get(language)));
  return row ? row.artifact : null;
}

// ---- learningEvents ----
export async function addLearningEvent(event) {
  const ev = { ts: Date.now(), ...(event || {}) };
  return withStore(EVENT_STORE, 'readwrite', (store, get) => get(store.put(ev)));
}
export async function listLearningEvents() {
  let all = [];
  try { all = await withStore(EVENT_STORE, 'readonly', (store, get) => get(store.getAll())); }
  catch (e) { all = []; }
  return Array.isArray(all) ? all : [];
}

// ---- memory (long-term, M8) ----
export async function saveMemoryItem(key, value) {
  return withStore(MEMORY_STORE, 'readwrite', (store, get) => get(store.put({ key, value })));
}
export async function loadMemoryItem(key) {
  const row = await withStore(MEMORY_STORE, 'readonly', (store, get) => get(store.get(key)));
  return row ? row.value : null;
}
export async function listMemoryItems() {
  let all = [];
  try { all = await withStore(MEMORY_STORE, 'readonly', (store, get) => get(store.getAll())); }
  catch (e) { all = []; }
  return Array.isArray(all) ? all : [];
}
export async function deleteMemoryItem(key) {
  return withStore(MEMORY_STORE, 'readwrite', (store, get) => get(store.delete(key)));
}

// ---- metadata ----
export async function saveMeta(obj) {
  return withStore(META_STORE, 'readwrite', (store, get) => get(store.put({ key: 'meta', ...obj })));
}
export async function loadMeta() {
  const row = await withStore(META_STORE, 'readonly', (store, get) => get(store.get('meta')));
  return row ? row : null;
}

// Full reset (used by the corrupted-model recovery path and by tests).
export async function nukeAgentStore() {
  try {
    const db = await openDB();
    const names = [...db.objectStoreNames];
    for (const name of names) db.deleteObjectStore(name);
  } catch (e) { /* not fatal */ }
}