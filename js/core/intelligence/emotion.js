/**
 * core/intelligence/emotion.js
 *
 * Interaction affect mapping. IMPORTANT: this does NOT detect the user's
 * psychological state. It maps a classified interaction intent plus the
 * app's existing lexicon feeling (detectFeeling) onto an *avatar affect*
 * using the same FEELING_TARGETS keys the face engine already supports.
 *
 * The existing raw lexicon feeling wins when present; the intent only shapes
 * the avatar's affect (and an "attentive" posture flag) when confidence is
 * high. Pure + DOM-free + unit-testable.
 *
 * @module core/intelligence/emotion
 */

import { MIN_CONFIDENCE } from './intent.js';

/**
 * Map an interaction to an avatar affect.
 *
 * @param {object} opts
 * @param {string} opts.intent         classified intent key
 * @param {number} opts.confidence     classifier confidence (0..1)
 * @param {string} [opts.baseFeeling]  existing detectFeeling result ('neutral' | feeling key)
 * @returns {{ feeling: string|null, attentive: boolean }}
 *   feeling is a FEELING_TARGETS key to apply, or null to leave the current
 *   face affect untouched; attentive marks curiosity/helplessness postures.
 */
export function composeEmotion({ intent, confidence, baseFeeling = 'neutral' }) {
  const attentive = confidence >= MIN_CONFIDENCE && (intent === 'confusion' || intent === 'help');

  let feeling = null;
  if (baseFeeling && baseFeeling !== 'neutral') {
    // Raw user lexicon (detectFeeling) always leads so we never fight it.
    feeling = baseFeeling;
  } else if (confidence >= MIN_CONFIDENCE) {
    if (intent === 'positive' || intent === 'greeting') feeling = 'happy';
    else if (intent === 'negative') feeling = 'sad';
  }
  return { feeling, attentive };
}