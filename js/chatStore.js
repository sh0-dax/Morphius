// ============================================================
// AI Face — Chat Session Store
// Conversational history lives in IndexedDB (db 'aiface_chat',
// store 'sessions'). Only roles + text content are persisted;
// API keys / secrets are never stored here. Pure helpers are
// kept importable under Node so Vitest can unit-test them.
// ============================================================

const DB_NAME = 'aiface_chat';
const DB_VERSION = 1;
const STORE = 'sessions';

const VALID_ROLES = new Set(['user', 'assistant', 'system', 'error']);

// ---- Pure helpers (unit-tested) ----

// Persistence quotas: IndexedDB has no server-side trim, so cap at the
// sanitize layer. A single 5MB base64 photo x 50 turns would otherwise
// blow the per-origin quota and break ALL future saves with QuotaExceeded.
export const CHAT_MAX_MESSAGES = 200;
export const CHAT_MAX_TEXT_CHARS = 20000;
export const CHAT_MAX_IMAGE_BYTES = 500 * 1024;
export const CHAT_MAX_IMAGES_PER_MESSAGE = 4;

// Approximate decoded bytes of a data: URL (base64 ~4/3 overhead).
export function dataUrlBytes(url) {
  if (typeof url !== 'string') return 0;
  const comma = url.indexOf(',');
  const b64 = comma >= 0 ? url.slice(comma + 1) : url;
  // Strip whitespace; non-base64 chars (e.g. non-data URLs) count 0.
  const clean = b64.replace(/\s+/g, '');
  if (!clean || /[^A-Za-z0-9+\/=]/.test(clean)) return 0;
  const pad = (clean.match(/=+$/) || [''])[0].length;
  return Math.max(0, Math.floor((clean.length * 3) / 4) - pad);
}

function clipText(s) {
  return s.length > CHAT_MAX_TEXT_CHARS ? s.slice(0, CHAT_MAX_TEXT_CHARS) : s;
}

// Normalizes an arbitrary messages list into persistable {role, content}
// records. Text content is kept as a string; future multimodal parts
// arrive as an array of {type, text|image_url} objects.
export function sanitizeMessages(messages) {
  if (!Array.isArray(messages)) return [];
  const out = [];
  for (const m of messages) {
    if (!m || typeof m !== 'object' || typeof m.role !== 'string') continue;
    if (!VALID_ROLES.has(m.role)) continue;
    if (typeof m.content === 'string') {
      if (m.content.trim()) out.push({ role: m.role, content: clipText(m.content) });
    } else if (Array.isArray(m.content)) {
      const parts = [];
      let images = 0;
      for (const p of m.content) {
        if (!p || typeof p !== 'object') continue;
        if (typeof p.text === 'string' && p.text.trim()) parts.push({ type: 'text', text: clipText(p.text) });
        else if (p.type === 'image_url' && p.image_url && typeof p.image_url === 'object') {
          const url = p.image_url.url;
          if (typeof url !== 'string' || !url.startsWith('data:')) continue;
          if (dataUrlBytes(url) > CHAT_MAX_IMAGE_BYTES) continue;
          if (images >= CHAT_MAX_IMAGES_PER_MESSAGE) continue;
          images += 1;
          parts.push({ type: 'image_url', image_url: { url } });
        }
      }
      if (parts.length) out.push({ role: m.role, content: parts });
    }
  }
  // Keep the most recent turns when the history overflows the cap.
  return out.length > CHAT_MAX_MESSAGES ? out.slice(out.length - CHAT_MAX_MESSAGES) : out;
}

// Derives a display title from the first user message.
export function sessionTitle(messages) {
  const list = sanitizeMessages(messages);
  const first = list.find((m) => m.role === 'user');
  const text = first ? (typeof first.content === 'string' ? first.content : '') : '';
  const t = String(text).replace(/\s+/g, ' ').trim();
  if (!t) return 'New Chat';
  return t.length > 60 ? t.slice(0, 60) + '\u2026' : t;
}

// Collision-resistant session id (timestamp + random tail).
export function makeSessionId(ts, rand) {
  const time = typeof ts === 'number' && ts ? ts : Date.now();
  const r = typeof rand === 'function' ? rand() : Math.random();
  return 's_' + time.toString(36) + '_' + Math.floor(r * 0xffffffff).toString(36).padStart(8, '0');
}

// Builds a well-formed session record for persistence.
export function buildSession({ id, provider, model, messages, created, updated, title }) {
  const now = Date.now();
  const clean = sanitizeMessages(messages);
  return {
    id: id || makeSessionId(),
    provider: String(provider || 'gemini'),
    model: String(model || ''),
    title: title || sessionTitle(clean),
    created: created || now,
    updated: updated || now,
    messages: clean,
  };
}

// ---- IndexedDB layer ----

function openDB() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') { reject(new Error('IndexedDB unavailable')); return; }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id' });
        store.createIndex('updated', 'updated');
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

// IDBRequest objects are not native promises; wrap them so callers can await
// the actual result value.
function reqResult(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function withStore(mode, fn) {
  const db = await openDB();
  try {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    const result = await fn(store, reqResult);
    await txDone(t);
    return result;
  } finally {
    db.close();
  }
}

export async function saveSession(session) {
  const s = buildSession(session);
  await withStore('readwrite', (store, list) => list(store.put(s)));
  return s;
}

export async function loadSession(id) {
  if (!id) return null;
  return withStore('readonly', (store, list) => list(store.get(id)));
}

export async function deleteSession(id) {
  if (!id) return;
  return withStore('readwrite', (store, list) => list(store.delete(id)));
}

// Lists session metadata (no message bodies) newest-first.
export async function listSessions() {
  let all = [];
  try { all = await withStore('readonly', (store, list) => list(store.getAll())); }
  catch (e) { all = []; }
  return (all || [])
    .map((s) => ({
      id: s.id, provider: s.provider || '', model: s.model || '',
      title: s.title || '', created: s.created || 0, updated: s.updated || 0,
      count: Array.isArray(s.messages) ? s.messages.length : 0,
    }))
    .sort((a, b) => (b.updated || 0) - (a.updated || 0));
}

// Returns the full most-recent session (for auto-restore), or null.
export async function getLastSession() {
  const list = await listSessions();
  if (!list.length) return null;
  return loadSession(list[0].id);
}