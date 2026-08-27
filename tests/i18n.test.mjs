import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const LOCALES = ['en', 'ar', 'fr', 'de', 'es', 'ja'];
const load = (l) => {
  const raw = readFileSync(join(here, '..', 'i18n', l + '.json'), 'utf8');
  return { keys: new Set(Object.keys(JSON.parse(raw))), map: JSON.parse(raw) };
};
const en = load('en');

describe('i18n locale parity', () => {
  for (const l of LOCALES) {
    it(l + '.json parses and matches en key set', () => {
      const lc = load(l);
      expect(new Set([...lc.keys].sort())).toEqual(new Set([...en.keys].sort()));
    });
  }

  it('every value is a non-empty string', () => {
    for (const l of LOCALES) {
      const lc = load(l);
      for (const k of en.keys) {
        expect(typeof lc.map[k], `${l}.${k}`).toBe('string');
        expect(lc.map[k].trim().length, `${l}.${k} not empty`).toBeGreaterThan(0);
      }
    }
  });
});