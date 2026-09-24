/**
 * js/core/secretStore.js
 *
 * Fail-closed encrypted key vault for provider API keys.
 *
 * Keys are encrypted with AES-GCM-256 via a non-extractable master key kept
 * in IndexedDB (`aiface_secure_store/keys`). The DOM never touches this
 * module: environment (subtle crypto, key slot, log fn) arrives via
 * constructor args, so Node tests can use fakes.
 *
 * FAIL-CLOSED CONTRACT (the point of this module)
 * ------------------------------------------------
 * The old inline code in js/app.js returned the PLAINTEXT key when
 * `crypto.subtle` was unavailable (file://, old browser) and wrote it to
 * localStorage silently. This module instead:
 *   - encrypt():  throws EncryptionUnavailableError -- NEVER returns plaintext
 *   - persistSettings(): stores '' for key fields when encryption is
 *     unavailable and reports { keysPersisted: false } so the UI can warn
 *   - decrypt(): returns '' on any failure (wrong format, bad IV, missing key)
 *   - clearAllKeys(): wipes the IDB master key + every enc:/v1: field
 *
 * WIRE FORMAT: v1:<b64-iv12>:<b64-cipher> (current, versioned)
 *              enc:<b64-iv12>:<b64-cipher> (legacy, still readable)
 *
 * @module core/secretStore
 */

/** Thrown by encrypt()/persistSettings() when no WebCrypto is available. */
export class EncryptionUnavailableError extends Error {
  constructor(message) {
    super(message || 'WebCrypto unavailable -- refusing to store API keys in plaintext. Serve over http://localhost or https.');
    this.name = 'EncryptionUnavailableError';
  }
}

export const SECURE_DB = 'aiface_secure_store';
export const SECURE_STORE = 'keys';
export const KEY_VERSION = 1;

/** localStorage fields that hold (encrypted) secrets. */
export const SECRET_FIELDS = ['key', 'geminiTtsKey', 'liveKey'];
/**
 * @param {string} b64 base64 string
 * @returns {Uint8Array}
 */
function b64ToBytes(b64) {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

/**
 * @param {Uint8Array} bytes
 * @returns {string} base64
 */
function bytesToB64(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s);
}

function textToBytes(text) {
  return new TextEncoder().encode(text);
}

function bytesToText(bytes) {
  return new TextDecoder().decode(bytes);
}

/**
 * Split "v1:iv:data" / "enc:iv:data" into parts. Null when not that shape.
 * @param {string} stored
 * @returns {{version: number, ivB64: string, dataB64: string} | null}
 */
export function parseStoredKey(stored) {
  if (typeof stored !== 'string' || !stored) return null;
  let version = -1;
  if (stored.startsWith('v1:')) version = 1;
  else if (stored.startsWith('enc:')) version = 0;
  else return null;
  const parts = stored.split(':');
  if (parts.length !== 3 || !parts[1] || !parts[2]) return null;
  return { version, ivB64: parts[1], dataB64: parts[2] };
}

/**
 * @param {*} subtle WebCrypto subtle (or null/undefined when unavailable)
 * @returns {boolean}
 */
export function isCryptoAvailable(subtle) {
  return !!(subtle && typeof subtle.generateKey === 'function' && typeof subtle.encrypt === 'function');
}

/**
 * In-memory stand-in for the IndexedDB master-key slot (tests + no-IDB envs).
 */
export function createMemoryKeySlot() {
  let key = null;
  return {
    get: async () => key,
    put: async (k) => { key = k; },
    clear: async () => { key = null; },
  };
}

/**
 * IndexedDB-backed master-key slot (browser only).
 */
export function createIdbKeySlot() {
  const open = () => new Promise((resolve, reject) => {
    const req = indexedDB.open(SECURE_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(SECURE_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return {
    get: async () => {
      const db = await open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(SECURE_STORE, 'readonly');
        const req = tx.objectStore(SECURE_STORE).get('masterKey');
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = () => reject(req.error);
      });
    },
    put: async (key) => {
      const db = await open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(SECURE_STORE, 'readwrite');
        const req = tx.objectStore(SECURE_STORE).put(key, 'masterKey');
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    },
    clear: () => new Promise((resolve) => {
      try {
        const del = indexedDB.deleteDatabase(SECURE_DB);
        del.onsuccess = () => resolve();
        del.onerror = () => resolve();
        del.onblocked = () => resolve();
      } catch (e) { resolve(); }
    }),
  };
}


/**
 * Create the vault.
 * @param {object} [deps]
 * @param {*} [deps.subtle] crypto.subtle (null in tests that simulate file://)
 * @param {{get: Function, put: Function, clear: Function}} [deps.keySlot] master-key slot
 * @param {(msg: string, kind: string) => void} [deps.log] diagnostic sink
 */
export function createSecretStore(deps) {
  const d = deps || {};
  const subtle = d.subtle !== undefined
    ? d.subtle
    : (typeof crypto !== 'undefined' ? crypto.subtle : null);
  const keySlot = d.keySlot || createMemoryKeySlot();
  const log = typeof d.log === 'function' ? d.log : () => {};
  let cachedKey = null;
  let lastPersisted = true;

  async function getOrCreateKey() {
    if (cachedKey) return cachedKey;
    const existing = await keySlot.get();
    if (existing) { cachedKey = existing; return cachedKey; }
    if (!isCryptoAvailable(subtle)) throw new EncryptionUnavailableError();
    cachedKey = await subtle.generateKey({ name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
    await keySlot.put(cachedKey);
    return cachedKey;
  }

  function randomIvFallback() {
    const iv = new Uint8Array(12);
    for (let i = 0; i < 12; i++) iv[i] = Math.floor(Math.random() * 256);
    return iv;
  }

  /**
   * Encrypt plaintext. '' stays ''. Throws when WebCrypto is missing --
   * callers must NOT fall back to plaintext.
   * @param {string} plain
   * @returns {Promise<string>} 'v1:iv:data'
   */
  async function encrypt(plain) {
    if (!plain) return '';
    if (!isCryptoAvailable(subtle)) throw new EncryptionUnavailableError();
    const key = await getOrCreateKey();
    let iv = null;
    try {
      if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
        iv = crypto.getRandomValues(new Uint8Array(12));
      }
    } catch (e) { iv = null; }
    if (!iv || iv.length !== 12) iv = randomIvFallback();
    const cipher = await subtle.encrypt({ name: 'AES-GCM', iv }, key, textToBytes(plain));
    return 'v1:' + bytesToB64(iv) + ':' + bytesToB64(new Uint8Array(cipher));
  }

  /**
   * Decrypt a stored value. '' on any failure. Legacy 'enc:' still decrypts.
   * @param {string} stored
   * @returns {Promise<string>}
   */
  async function decrypt(stored) {
    if (!stored) return '';
    const parsed = parseStoredKey(stored);
    if (!parsed) return '';
    try {
      const iv = b64ToBytes(parsed.ivB64);
      const data = b64ToBytes(parsed.dataB64);
      if (iv.length !== 12 || !data.length) return '';
      const key = await getOrCreateKey();
      const plainBuf = await subtle.decrypt({ name: 'AES-GCM', iv }, key, data);
      return bytesToText(new Uint8Array(plainBuf));
    } catch (e) {
      log('Key decryption failed: ' + (e && e.message), 'err');
      return '';
    }
  }

  /** True when keys can (and will) be persisted encrypted right now. */
  async function ensureKeyAvailable() {
    try {
      await getOrCreateKey();
      return true;
    } catch (e) {
      return false;
    }
  }

  /**
   * Build the persistable settings object from raw field values.
   * Key fields are encrypted; when unavailable they become '' and
   * keysPersisted=false so the UI warns instead of leaking.
   * @param {Record<string, string>} values raw field values (plaintext keys)
   * @param {Record<string, *>} rest non-secret settings to carry through
   */
  async function persistSettings(values, rest) {
    const settings = { ...(rest || {}) };
    let keysPersistedFlag = true;
    for (const f of SECRET_FIELDS) {
      const plain = values && typeof values[f] === 'string' ? values[f] : '';
      if (!plain) { settings[f] = ''; continue; }
      try {
        settings[f] = await encrypt(plain);
      } catch (e) {
        if (e instanceof EncryptionUnavailableError) {
          keysPersistedFlag = false;
          settings[f] = '';
          log('Key encryption unavailable -- key for "' + f + '" NOT saved (fail-closed)', 'warn');
        } else { throw e; }
      }
    }
    lastPersisted = keysPersistedFlag;
    return { settings, keysPersisted: keysPersistedFlag };
  }

  /**
   * Decrypt the secret fields of a loaded settings object.
   * @param {Record<string, *>} stored parsed localStorage settings
   * @returns {Promise<Record<string, string>>} plaintext key values
   */
  async function restoreSettings(stored) {
    /** @type {Record<string, string>} */
    const out = {};
    for (const f of SECRET_FIELDS) {
      out[f] = stored && stored[f] ? await decrypt(stored[f]) : '';
    }
    return out;
  }

  /**
   * Wipe the master key and every secret field of the given settings object.
   * @param {Record<string, *>} [stored] parsed settings (mutated in place)
   */
  async function clearAllKeys(stored) {
    cachedKey = null;
    try { await keySlot.clear(); } catch (e) { /* best-effort */ }
    if (stored && typeof stored === 'object') {
      for (const f of SECRET_FIELDS) stored[f] = '';
    }
    lastPersisted = true;
    log('All stored API keys cleared', 'ok');
  }

  function keysPersisted() {
    return lastPersisted;
  }

  return { encrypt, decrypt, ensureKeyAvailable, persistSettings, restoreSettings, clearAllKeys, keysPersisted };
}
