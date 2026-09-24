import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  AGENT_INTENT_IDS,
  AGENT_TO_AFFECT,
  agentIntentToAffect,
  unifyIntents,
  unifyTextWithAgent,
  affectIntentKeys,
  MIN_STRONG_OVERRIDE,
} from '../js/core/intelligence/intentMap.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

function corpusIntentIds() {
  const ids = new Set();
  for (const lang of ['en', 'fr', 'ar']) {
    const json = JSON.parse(readFileSync(join(root, 'data', 'agent', `${lang}.json`), 'utf8'));
    for (const it of json.intents || []) {
      if (it && it.id) ids.add(it.id);
    }
  }
  return [...ids].sort();
}

describe('intentMap: corpus parity (anti-drift guard)', () => {
  it('covers every intent id present in data/agent/{en,fr,ar}.json', () => {
    const corpus = corpusIntentIds();
    expect(corpus.length).toBeGreaterThan(0);
    const missing = corpus.filter((id) => !(id in AGENT_TO_AFFECT));
    expect(missing, 'agent intents with no affect mapping: ' + missing.join(', ')).toEqual([]);
  });

  it('maps nothing that the corpus does not define', () => {
    const corpus = new Set(corpusIntentIds());
    const extra = Object.keys(AGENT_TO_AFFECT).filter((id) => !corpus.has(id));
    expect(extra, 'stale mappings for removed intents: ' + extra.join(', ')).toEqual([]);
  });

  it('AGENT_INTENT_IDS matches the mapped keys exactly', () => {
    expect([...AGENT_INTENT_IDS].sort()).toEqual(Object.keys(AGENT_TO_AFFECT).sort());
  });

  it('every mapping target is a valid affect intent', () => {
    const valid = new Set(affectIntentKeys());
    for (const [from, to] of Object.entries(AGENT_TO_AFFECT)) {
      expect(valid.has(to), `${from} -> ${to} is not a valid affect intent`).toBe(true);
    }
  });
});

describe('intentMap: unification policy', () => {
  it('agent wins when it maps to a real affect', () => {
    expect(unifyIntents({ agentIntent: 'thanks', affectIntent: 'none', affectConfidence: 0 }))
      .toMatchObject({ intent: 'positive', source: 'agent' });
    expect(unifyIntents({ agentIntent: 'greeting', affectIntent: 'confusion', affectConfidence: 0.9 }))
      .toMatchObject({ intent: 'greeting', source: 'agent' });
    expect(unifyIntents({ agentIntent: 'help', affectIntent: 'none', affectConfidence: 0 }))
      .toMatchObject({ intent: 'help', source: 'agent' });
  });

  it('device/memory commands keep the face neutral', () => {
    for (const id of ['lighting', 'mirror', 'vision', 'time', 'settings']) {
      const u = unifyIntents({ agentIntent: id, affectIntent: 'none', affectConfidence: 0 });
      expect(u.intent).toBe('none');
      expect(u.source).toBe('agent-unmapped');
    }
  });

  it('strong user emotion still shows through a neutral command', () => {
    const u = unifyIntents({ agentIntent: 'lighting', affectIntent: 'negative', affectConfidence: MIN_STRONG_OVERRIDE });
    expect(u).toMatchObject({ intent: 'negative', source: 'heuristic-override' });
    // Weak emotion does not override the command.
    const w = unifyIntents({ agentIntent: 'lighting', affectIntent: 'negative', affectConfidence: 0.3 });
    expect(w).toMatchObject({ intent: 'none', source: 'agent-unmapped' });
  });

  it('falls back to the heuristic when the agent abstains', () => {
    expect(unifyIntents({ agentIntent: null, affectIntent: 'confusion', affectConfidence: 0.6 }))
      .toMatchObject({ intent: 'confusion', source: 'heuristic-only' });
  });

  it('agentIntentToAffect() is total (never throws, never undefined)', () => {
    expect(agentIntentToAffect(null)).toBe('none');
    expect(agentIntentToAffect(undefined)).toBe('none');
    expect(agentIntentToAffect('no-such-intent')).toBe('none');
    for (const id of AGENT_INTENT_IDS) {
      expect(typeof agentIntentToAffect(id)).toBe('string');
    }
  });

  it('unifyTextWithAgent() end-to-end on representative sentences', () => {
    // Agent says "lighting", text is emotionally flat -> neutral face.
    const flat = unifyTextWithAgent('set the lighting to warm', 'lighting');
    expect(flat.intent).toBe('none');
    expect(flat.heuristic).toBeDefined();
    // Agent abstains, heuristic reads a greeting -> greeting shows.
    const greet = unifyTextWithAgent('hello there', null);
    expect(greet.intent).toBe('greeting');
    expect(greet.source).toBe('heuristic-only');
  });
});
