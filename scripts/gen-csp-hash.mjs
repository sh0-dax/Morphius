/**
 * scripts/gen-csp-hash.mjs
 *
 * Computes the CSP `sha256-…` hashes for every inline <script> block in
 * index.html and prints a ready-to-paste `script-src` line. Static hosting
 * (GitHub Pages) cannot mint per-response nonces, so hashes are the correct
 * primitive for the few inline blocks we still need (the import map).
 *
 * Usage:
 *   node scripts/gen-csp-hash.mjs            # print hashes + current CSP status
 *   node scripts/gen-csp-hash.mjs --strict   # also print a variant without 'unsafe-eval'
 *
 * tests/csp.test.mjs runs the same scan in CI: if you edit an inline block and
 * forget to update the CSP meta tag, the suite fails and tells you the hash.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const SCRIPT_RE = /<script([^>]*)>([\s\S]*?)<\/script>/gid;
const CSP_META_RE = /<meta\s+http-equiv="Content-Security-Policy"\s+content="([^"]*)"/i;

/**
 * Blank out HTML comments while preserving length and line numbers, so the
 * scanner can't be fooled by `<script>` text that appears inside a comment
 * (the CSP rationale comment in index.html contains exactly that string).
 * @param {string} html
 * @returns {string}
 */
export function stripHtmlComments(html) {
  return html.replace(/<!--[\s\S]*?-->/g, (m) => m.replace(/[^\n]/g, ' '));
}

/**
 * Every inline <script> (no src attribute) with the exact sha256 the browser
 * will compute for it. Hashes are taken from the ORIGINAL html (comment
 * blanking preserves offsets), matching the CSP spec: the hash covers the raw
 * text content between the tags.
 * @param {string} html
 * @returns {Array<{line: number, type: string, hash: string, preview: string}>}
 */
export function inlineScriptHashes(html) {
  const scan = stripHtmlComments(html);
  const out = [];
  for (const m of scan.matchAll(SCRIPT_RE)) {
    const attrs = m[1] || '';
    if (/\bsrc\s*=/i.test(attrs)) continue;
    const idx = m.indices && m.indices[2];
    const body = idx ? html.slice(idx[0], idx[1]) : m[2];
    out.push({
      line: html.slice(0, m.index).split('\n').length,
      type: (/type\s*=\s*"([^"]*)"/i.exec(attrs) || [, 'classic'])[1],
      hash: 'sha256-' + createHash('sha256').update(body, 'utf8').digest('base64'),
      preview: body.trim().split('\n')[0].slice(0, 60),
    });
  }
  return out;
}

/** The content="…" string of the CSP meta tag ('' when missing). */
export function cspContent(html) {
  const m = CSP_META_RE.exec(html);
  return m ? m[1] : '';
}

/** The script-src directive of a CSP string ('' when missing). */
export function scriptSrcDirective(csp) {
  const m = /(?:^|;)\s*script-src\s+([^;]*)/i.exec(csp);
  return m ? m[1].trim() : '';
}

/** sha256-… tokens of a script-src directive. */
export function cspHashes(scriptSrc) {
  return [...String(scriptSrc || '').matchAll(/'(sha256-[A-Za-z0-9+/=]+)'/g)].map((m) => m[1]);
}

/**
 * Every https:// origin a first-party module pulls at runtime via dynamic
 * import() / static from (e.g. WebLLM via https://esm.run). These need a
 * script-src host entry just like import-map entries.
 * @param {string} root repo root
 * @returns {string[]} origins
 */
export function dynamicImportOrigins(root) {
  const out = [];
  const walk = (dir) => {
    for (const e of readdirSync(dir)) {
      const p = join(dir, e);
      if (statSync(p).isDirectory()) { walk(p); continue; }
      if (!e.endsWith('.js')) continue;
      const src = readFileSync(p, 'utf8');
      for (const m of src.matchAll(/(?:import\s*\(\s*|from\s+)(['"])https:\/\/[^'"]+\1/g)) {
        const raw = m[0].replace(/^(?:import\s*\(\s*|from\s+)(['"])/, '').replace(/['"]$/, '');
        try {
          const origin = new URL(raw).origin;
          if (origin && !out.includes(origin)) out.push(origin);
        } catch (e) { /* ignore */ }
      }
    }
  };
  walk(join(root, 'js'));
  return out;
}

function main() {
  const strict = process.argv.includes('--strict');
  const html = readFileSync(join(root, 'index.html'), 'utf8');
  const blocks = inlineScriptHashes(html);
  const scriptSrc = scriptSrcDirective(cspContent(html));
  const hashes = cspHashes(scriptSrc);

  console.log(`inline <script> blocks in index.html: ${blocks.length}\n`);
  for (const b of blocks) {
    const listed = scriptSrc.includes(b.hash) ? 'OK  ' : 'MISS';
    console.log(`  [${listed}] line ${b.line}  type=${b.type}  ${b.hash}`);
    console.log(`         ${b.preview}`);
  }

  const stale = hashes.filter((h) => !blocks.some((b) => b.hash === h));
  if (stale.length) {
    console.log('\n  [STALE] CSP hashes with no matching inline block:');
    for (const h of stale) console.log('         ' + h);
  }

  const origins = [];
  const mapSrc = /<script\s+type="importmap"[^>]*>([\s\S]*?)<\/script>/i.exec(html);
  if (mapSrc) {
    try {
      const map = JSON.parse(mapSrc[1]);
      for (const url of Object.values(map.imports || {})) {
        const origin = new URL(url).origin;
        if (!origins.includes(origin)) origins.push(origin);
      }
    } catch (e) { /* malformed importmap: the test suite reports it */ }
  }

  // Dynamic imports of remote modules (e.g. WebLLM via https://esm.run)
  // also need a script-src host entry.
  for (const o of dynamicImportOrigins(root)) {
    if (!origins.includes(o)) origins.push(o);
  }

  console.log('\nscript-src currently:');
  console.log('  ' + (scriptSrc || '(missing)'));
  console.log('\nsuggested script-src:');
  const base = `script-src 'self' ${blocks.map((b) => `'${b.hash}'`).join(' ')} 'wasm-unsafe-eval' 'unsafe-eval' ${origins.join(' ')}`
    .replace(/\s+/g, ' ').trim();
  console.log('  ' + base);
  if (strict) {
    console.log('\nstrict variant (drop ONLY if no model needs KTX2/Basis textures):');
    console.log('  ' + base.replace(" 'unsafe-eval'", ''));
  }
  console.log("\nNOTE: 'unsafe-eval' is required by the basis transcoder that KTX2Loader\n" +
    'fetches from the CDN (see index.html CSP note). Removing it disables\n' +
    'KTX2/Basis-compressed textures.');
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1] === fileURLToPath(import.meta.url)) main();

