/**
 * js/core/intelligence/intentMap.js
 *
 * Single bridge between the two decision brains of Morphius:
 *
 *   - Local Cognitive Agent (js/agent/agent.js): 26 action intents, per-lang
 *     NaiveBayes, thresholds 0.35/0.12 -- answers "WHAT should the app DO?"
 *   - Heuristic classifier (./intent.js): 7 affect intents, deterministic
 *     seed lexicon, MIN_CONFIDENCE 0.5 -- answers "HOW should the face FEEL?"
 *
 * Previously app.js called only the heuristic brain for the face, while chat
 * actions came from the agent brain; a sentence could therefore produce two
 * contradictory readings (e.g. affect=confusion while action=lighting).
 * This module is the ONE place that maps agent intents -> affect intents, so
 * both layers can agree. It is pure + DOM-free + unit-testable.
 *
 * USAGE
 * -----
 *   import { agentIntentToAffect, unifyIntents } from './intentMap.js';
 *   const a = agentIntentToAffect('thanks');            // 'positive'
 *   unifyIntents({ agentIntent: 'lighting', affectIntent: 'none' });
 *   // -> { intent: 'none', source: 'agent-unmapped', ... } (face stays neutral)
 *
 * @module core/intelligence/intentMap
 */

import { classifyIntent, INTENT_KEYS } from './intent.js';

/**
 * Canonical list of the 26 agent action intents (data/agent/{en,fr,ar}.json).
 * Kept as a literal so the test suite can fail loudly when the corpus gains
 * an intent that has no affect mapping yet.
 */
export const AGENT_INTENT_IDS = [
  'greeting',
  'farewell',
  'thanks',
  'how_are_you',
  'who_are_you',
  'name_inquiry',
  'capabilities',
  'help',
  'time',
  'date',
  'small_talk',
  'compliment',
  'weather_offline',
  'joke',
  'repeat',
  'lighting',
  'mirror',
  'vision',
  'stop_talking',
  'change_face',
  'projection',
  'memory_store',
  'memory_fact',
  'memory_recall',
  'settings',
  'privacy',
];

/**
 * Agent intent -> affect intent. 'none' means "no emotional content; the face
 * stays neutral" (device commands, memory lookups, settings...).
 */
export const AGENT_TO_AFFECT = {
  greeting: 'greeting',
  farewell: 'farewell',
  thanks: 'positive',
  how_are_you: 'greeting',
  who_are_you: 'question',
  name_inquiry: 'question',
  capabilities: 'question',
  help: 'help',
  time: 'none',
  date: 'none',
  small_talk: 'positive',
  compliment: 'positive',
  weather_offline: 'none',
  joke: 'positive',
  repeat: 'none',
  lighting: 'none',
  mirror: 'none',
  vision: 'none',
  stop_talking: 'negative',
  change_face: 'none',
  projection: 'none',
  memory_store: 'none',
  memory_fact: 'question',
  memory_recall: 'question',
  settings: 'none',
  privacy: 'question',
};

/**
 * @param {string|null|undefined} agentIntent
 * @returns {string} affect intent ('none' when unknown/unmapped)
 */
export function agentIntentToAffect(agentIntent) {
  if (!agentIntent) return 'none';
  return AGENT_TO_AFFECT[agentIntent] || 'none';
}

/**
 * Affect intents that the heuristic layer may produce.
 * @returns {string[]} ['greeting','farewell','help','confusion','positive','negative','question','none']
 */
export function affectIntentKeys() {
  return [...INTENT_KEYS, 'none'];
}

/**
 * Reconcile the two brains for one user turn.
 *
 * Policy (documented, deterministic):
 *  - agent intent is null/unknown  -> trust the heuristic reading as-is.
 *  - agent maps to a real affect   -> agent wins (it saw the full sentence
 *    through the trained NB model); heuristic confidence is preserved for
 *    face-blend scaling.
 *  - agent maps to 'none'          -> the turn is a device/memory command;
 *    the face stays neutral UNLESS the heuristic is strongly confident
 *    (>= MIN_STRONG_OVERRIDE) about positive/negative/confusion, in which
 *    case the user's emotion still shows (e.g. "stupid lighting, fix it!").
 *
 * @param {object} [opts]
 * @param {string|null} [opts.agentIntent] agent action intent (or null)
 * @param {string} [opts.affectIntent] heuristic reading ('none' default)
 * @param {number} [opts.affectConfidence] heuristic confidence (0 default)
 * @returns {{intent: string, source: string, agentAffect: string}}
 */
export const MIN_STRONG_OVERRIDE = 0.8;

export function unifyIntents(opts) {
  const o = opts || {};
  const affectIntent = o.affectIntent || 'none';
  const affectConfidence = typeof o.affectConfidence === 'number' ? o.affectConfidence : 0;
  const agentAffect = agentIntentToAffect(o.agentIntent || null);

  if (!o.agentIntent) {
    return { intent: affectIntent, source: 'heuristic-only', agentAffect };
  }
  if (agentAffect !== 'none') {
    return { intent: agentAffect, source: 'agent', agentAffect };
  }
  if (
    affectConfidence >= MIN_STRONG_OVERRIDE &&
    (affectIntent === 'positive' || affectIntent === 'negative' || affectIntent === 'confusion')
  ) {
    return { intent: affectIntent, source: 'heuristic-override', agentAffect };
  }
  return { intent: 'none', source: 'agent-unmapped', agentAffect };
}

/**
 * Convenience: classify the text heuristically, then unify with the agent's
 * decision for the same turn. Pure -- takes values, returns a decision.
 * @param {string} text
 * @param {string|null} agentIntent
 * @param {object} [features] pre-computed features (avoids re-extraction)
 * @param {Record<string, Record<string, number>>} [extraLexicon] learned weights
 */
export function unifyTextWithAgent(text, agentIntent, features, extraLexicon) {
  const h = classifyIntent(text, features, extraLexicon);
  const u = unifyIntents({ agentIntent, affectIntent: h.intent, affectConfidence: h.confidence });
  return { ...u, heuristic: h };
}
