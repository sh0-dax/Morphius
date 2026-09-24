import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inlineScriptHashes, cspContent, scriptSrcDirective, dynamicImportOrigins } from '../scripts/gen-csp-hash.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const html = readFileSync(join(root, 'index.html'), 'utf8');
const sw = readFileSync(join(root, 'sw.js'), 'utf8');
const csp = cspContent(html);
const scriptSrc = scriptSrcDirective(csp);

/** Directives of a CSP string, e.g. { 'script-src': "'self' ..." }. */
function directives(cspString) {
  const out = {};
  for (const part of cspString.split(';')) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const sp = trimmed.indexOf(' ');
    const name = (sp === -1 ? trimmed : trimmed.slice(0, sp)).toLowerCase();
    out[name] = sp === -1 ? '' : trimmed.slice(sp + 1).trim();
  }
  return out;
}
const dirs = directives(csp);

/** Paths listed in sw.js SHELL_FILES (without the leading './'). */
const shellFiles = (() => {
  const m = /const SHELL_FILES = \[([\s\S]*?)\];/.exec(sw);
  return new Set(
    [...(m ? m[1] : '').matchAll(/'(\.\/[^']+)'/g)].map((x) => x[1].replace(/^\.\//, '').split('?')[0])
  );
})();

describe('CSP: index.html has no un-hashed inline script', () => {
  it('declares a Content-Security-Policy meta tag', () => {
    expect(csp, 'CSP meta tag missing').not.toBe('');
  });

  it("script-src has no 'unsafe-inline'", () => {
    expect(scriptSrc).not.toContain("'unsafe-inline'");
  });

  it('every inline <script> block is allow-listed by its sha256 hash', () => {
    const blocks = inlineScriptHashes(html);
    // External scripts reference files; only genuinely inline blocks need hashes.
    const missing = blocks.filter((b) => !scriptSrc.includes(b.hash));
    expect(
      missing,
      'inline <script> not hashed in CSP: ' +
        missing.map((b) => `line ${b.line} (${b.preview}) -> ${b.hash}`).join(' | ') +
        '\nRun: node scripts/gen-csp-hash.mjs and paste the suggested script-src.'
    ).toEqual([]);
  });

  it('keeps the load-bearing hardening directives', () => {
    expect(dirs['default-src']).toBe("'none'");
    expect(dirs['object-src']).toBe("'none'");
    expect(dirs['base-uri']).toBe("'self'");
    expect(dirs['frame-src']).toBe("'none'");
    expect(dirs['worker-src']).toContain('blob:');
    expect(scriptSrc).toContain("'self'");
  });

  it('allows every host referenced by the import map', () => {
    const m = /<script\s+type="importmap"[^>]*>([\s\S]*?)<\/script>/i.exec(html);
    expect(m, 'import map block not found').not.toBeNull();
    const map = JSON.parse(m[1]);
    const origins = [...new Set(Object.values(map.imports || {}).map((u) => new URL(u).origin))];
    expect(origins.length).toBeGreaterThan(0);
    for (const origin of origins) {
      expect(scriptSrc, `import map host not allowed by script-src: ${origin}`).toContain(origin);
    }
  });

  it('allows every host pulled by dynamic import() in first-party js', () => {
    const origins = dynamicImportOrigins(root);
    expect(origins).toContain('https://esm.run');
    for (const origin of origins) {
      expect(scriptSrc, `dynamic import host not allowed by script-src: ${origin}`).toContain(origin);
    }
  });
});

describe('Boot scripts extracted from index.html', () => {
  const bootDir = join(root, 'js', 'boot');
  const expected = ['theme.js', 'nosw.js', 'watchdog.js'];

  it('exists for every extracted block', () => {
    for (const f of expected) {
      expect(existsSync(join(bootDir, f)), `missing js/boot/${f}`).toBe(true);
    }
  });

  it('is referenced by index.html (nothing left inline)', () => {
    for (const f of expected) {
      expect(html, `index.html does not load js/boot/${f}`).toContain(`./js/boot/${f}`);
    }
  });

  it('keeps boot-time ordering guarantees', () => {
    // nosw.js must run before theme/app so window.__NOSW is set first,
    // and watchdog.js must come after the app module.
    const nosw = html.indexOf('./js/boot/nosw.js');
    const theme = html.indexOf('./js/boot/theme.js');
    const app = html.indexOf('./js/app.js');
    const watchdog = html.indexOf('./js/boot/watchdog.js');
    expect(nosw).toBeGreaterThan(-1);
    expect(theme).toBeGreaterThan(-1);
    expect(app).toBeGreaterThan(-1);
    expect(watchdog).toBeGreaterThan(-1);
    expect(nosw).toBeLessThan(app);
    expect(watchdog).toBeGreaterThan(app);
  });

  it('works without module syntax (classic scripts, no imports)', () => {
    for (const f of expected) {
      const src = readFileSync(join(bootDir, f), 'utf8');
      expect(src, `js/boot/${f} must not use ESM import/export`).not.toMatch(/^\s*(import|export)\s/m);
    }
  });
});

describe('Service worker shell parity', () => {
  const jsFiles = (() => {
    const walk = (dir) => {
      const out = [];
      for (const e of readdirSync(dir)) {
        const p = join(dir, e);
        if (statSync(p).isDirectory()) out.push(...walk(p));
        else if (e.endsWith('.js')) out.push(relative(root, p).split(sep).join('/'));
      }
      return out;
    };
    return walk(join(root, 'js'));
  })();

  it('precaches every first-party js module (incl. js/boot)', () => {
    const missing = jsFiles.filter((f) => !shellFiles.has(f));
    expect(missing, 'missing from sw.js SHELL_FILES: ' + missing.join(', ')).toEqual([]);
  });

  it('precaches every local <script src> used by index.html', () => {
    const srcs = [...html.matchAll(/<script[^>]+src="(\.\/[^"]+)"/g)].map((m) => m[1].replace(/^\.\//, ''));
    expect(srcs.length).toBeGreaterThan(0);
    for (const s of srcs) {
      expect(shellFiles, `${s} not in SHELL_FILES`).toContain(s);
    }
  });
});
