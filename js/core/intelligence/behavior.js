/**
 * core/intelligence/behavior.js
 *
 * Behavior policy for the avatar. Given the classified interaction (intent +
 * confidence + affect) and the accumulated user model, it produces a small,
 * bounded set of *recommendations* — never direct provider/LLM control.
 *
 * All knobs default to fully neutral (scale 1 / hint 0 / length 0.5). Weak
 * classifications and tiny sample sizes keep the output neutral, and passing
 * `intelligenceOn: false` returns the exact neutral policy so behaviour is
 * identical to a build without the intelligence layer.
 *
 * @module core/intelligence/behavior
 */

import { MIN_CONFIDENCE } from './intent.js';

/** Minimum number of user turns before user-model adaptation kicks in. */
export const MIN_TURNS_FOR_ADAPT = 5;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/**
 * Fully neutral behavior directive (also the OFF-state parity baseline).
 * @returns {BehaviorDirective}
 */
export function neutralBehavior() {
  return {
    feeling: null,
    attentive: false,
    blinkScale: 1,
    headMoveScale: 1,
    speechRateHint: 0,
    responseLengthHint: 0.5,
  };
}

/**
 * Resolve a behavior directive for the current interaction.
 *
 * @typedef {{
 *   feeling: string|null,
 *   attentive: boolean,
 *   blinkScale: number,
 *   headMoveScale: number,
 *   speechRateHint: number,
 *   responseLengthHint: number
 * }} BehaviorDirective
 *
 * @param {object} opts
 * @param {boolean} [opts.intelligenceOn]  master switch (default true)
 * @param {string}  [opts.state]           avatar drive state
 * @param {string}  [opts.intent]          classified intent key
 * @param {number}  [opts.confidence]      classifier confidence (0..1)
 * @param {object}  [opts.emotion]         { feeling, attentive } from composeEmotion
 * @param {object}  [opts.userModel]       user-model snapshot (turnCount, interruptionRate, ...)
 * @returns {BehaviorDirective}
 */
export function behaviorPolicy({
  intelligenceOn = true,
  state = 'idle',
  intent = 'none',
  confidence = 0,
  emotion = null,
  userModel = null,
}) {
  const out = neutralBehavior();
  if (!intelligenceOn) return out;
  void state;

  const strong = confidence >= MIN_CONFIDENCE;
  const attentive = !!(emotion && emotion.attentive);

  out.attentive = attentive;
  if (emotion && emotion.feeling && strong) {
    out.feeling = emotion.feeling;
  }

  if (attentive) {
    // Calmer, focused posture for confusion/help interactions.
    out.blinkScale = 0.9;
    out.headMoveScale = 0.7;
    out.speechRateHint = -0.2; // slow down slightly
  } else if (strong) {
    if (intent === 'positive') out.speechRateHint = 0.2; // a little more energy
    else if (intent === 'confusion') out.speechRateHint = -0.1;
  }

  if (userModel && userModel.turnCount >= MIN_TURNS_FOR_ADAPT) {
    const ir = clamp(Number(userModel.interruptionRate) || 0, 0, 1);
    // Frequent interruption → gently hint at shorter responses (recommendation
    // only; it never constrains or mutates the provider call).
    out.responseLengthHint = 0.5 - ir * 0.4;
  }

  out.blinkScale = clamp(out.blinkScale, 0.4, 1.2);
  out.headMoveScale = clamp(out.headMoveScale, 0.4, 1.2);
  out.speechRateHint = clamp(out.speechRateHint, -0.4, 0.4);
  out.responseLengthHint = clamp(out.responseLengthHint, 0.2, 0.8);
  return out;
}