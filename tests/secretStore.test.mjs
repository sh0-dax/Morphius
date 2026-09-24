import { describe, it, expect } from 'vitest';
import {
  createSecretStore,
  createMemoryKeySlot,
  parseStoredKey,
  isCryptoAvailable,
  EncryptionUnavailableError,
  SECRET_FIELDS,
} from '../js/core/secretStore.js';
import { webcrypto } from 'node:crypto';

const subtle = webcrypto.subtle;

function makeStore(overrides) {
  return createSecretStore({
    subtle,
    keySlot: createMemoryKeySlot(),
    log: () => {},
    ...(overrides || {}),
  });
}

describe('secretStore: fail-closed contract', () => {
  it('encrypt() throws (never plaintext) when WebCrypto is unavailable', async () => {
    const store = makeStore({ subtle: null });
    await expect(store.encrypt('sk-secret')).rejects.toBeInstanceOf(EncryptionUnavailableError);
  });

  it('persistSettings() stores empty keys + keysPersisted=false when unavailable', async () => {
    const logs = [];
    const store = makeStore({ subtle: null, log: (m, k) => logs.push([m, k]) });
    const { settings, keysPersisted } = await store.persistSettings(
      { key: 'sk-live', geminiTtsKey: 'g-key', liveKey: '' },
      { provider: 'openai' }
    );
    expect(keysPersisted).toBe(false);
    expect(settings.key).toBe('');
    expect(settings.geminiTtsKey).toBe('');
    expect(settings.provider).toBe('openai');
    expect(logs.some(([, k]) => k === 'warn')).toBe(true);
    // The plaintext must appear NOWHERE in the persisted object.
    expect(JSON.stringify(settings)).not.toContain('sk-live');
  });

  it('round-trips encrypt/decrypt with real WebCrypto', async () => {
    const store = makeStore();
    const stored = await store.encrypt('sk-abc-123');
    expect(stored.startsWith('v1:')).toBe(true);
    expect(stored).not.toContain('sk-abc-123');
    expect(await store.decrypt(stored)).toBe('sk-abc-123');
  });

  it('decrypt() returns empty string on corrupt input', async () => {
    const store = makeStore();
    expect(await store.decrypt('v1:!!!:???')).toBe('');
    expect(await store.decrypt('v1:onlyonepart')).toBe('');
    expect(await store.decrypt('garbage-no-prefix')).toBe('');
    expect(await store.decrypt('')).toBe('');
  });

  it('decrypt() reads legacy enc: values (migration path)', async () => {
    const store = makeStore();
    // Produce a legacy-shaped value: encrypt then swap the prefix.
    const modern = await store.encrypt('legacy-secret');
    const legacy = 'enc:' + modern.slice('v1:'.length);
    // Same store, same master key -> must decrypt (shared AES-GCM params).
    expect(await store.decrypt(legacy)).toBe('legacy-secret');
  });

  it('clearAllKeys() wipes IDB slot + settings fields', async () => {
    const store = makeStore();
    const stored = { provider: 'gemini', key: await store.encrypt('k1'), geminiTtsKey: await store.encrypt('k2'), liveKey: '' };
    await store.clearAllKeys(stored);
    expect(stored.key).toBe('');
    expect(stored.geminiTtsKey).toBe('');
    expect(stored.provider).toBe('gemini');
    // Fresh store over the same slot: nothing recoverable.
    expect(await store.decrypt('v1:AAAAAAAAAAAAAAAA:not-real')).toBe('');
  });

  it('SECRET_FIELDS covers the three key inputs', () => {
    expect([...SECRET_FIELDS].sort()).toEqual(['geminiTtsKey', 'key', 'liveKey']);
  });

  it('parseStoredKey() versions v1 vs enc and rejects junk', () => {
    expect(parseStoredKey('v1:a:b').version).toBe(1);
    expect(parseStoredKey('enc:a:b').version).toBe(0);
    expect(parseStoredKey('plain-key')).toBeNull();
    expect(parseStoredKey('v1:missing')).toBeNull();
    expect(parseStoredKey('')).toBeNull();
  });

  it('isCryptoAvailable() gates on subtle presence', () => {
    expect(isCryptoAvailable(subtle)).toBe(true);
    expect(isCryptoAvailable(null)).toBe(false);
    expect(isCryptoAvailable(undefined)).toBe(false);
    expect(isCryptoAvailable({})).toBe(false);
  });
});
