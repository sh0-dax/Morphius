import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
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

  it('sw.js SHELL_FILES covers every first-party js module (offline parity)', () => {
    const sw = readFileSync(join(root, 'sw.js'), 'utf8');
    const m = /const SHELL_FILES = \[([\s\S]*?)\];/.exec(sw);
    expect(m, 'SHELL_FILES block').not.toBeNull();
    const pinned = new Set(
      [...m[1].matchAll(/'(\.\/[^']+)'/g)].map((x) => x[1].replace(/^\.\//, '').split('?')[0])
    );
    const walk = (dir) => {
      const out = [];
      for (const e of readdirSync(dir)) {
        const p = join(dir, e);
        if (statSync(p).isDirectory()) out.push(...walk(p));
        else if (e.endsWith('.js')) out.push(relative(root, p).split(sep).join('/'));
      }
      return out;
    };
    const missing = walk(join(root, 'js')).filter((f) => !pinned.has(f));
    expect(missing, 'js modules missing from sw SHELL_FILES: ' + missing.join(', ')).toEqual([]);
  });
});