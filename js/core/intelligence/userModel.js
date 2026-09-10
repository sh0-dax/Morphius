/**
 * core/intelligence/userModel.js
 *
 * Session-scoped user behavior model. Tracks lightweight interaction metrics
 * over the whole session and derives read-only rates for the behavior policy.
 * Every rate is combined with `turnCount` (sample size) so a policy can refuse
 * to adapt on tiny samples. Pure + DOM-free + unit-testable; persistence lives
 * in the caller (app.js/localStorage), not here.
 *
 * @module core/intelligence/userModel
 */

const clamp01 = (v) => Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0));

/**
 * @typedef {object} IntelligenceEvent
 * @property {'userTurn'|'assistantEnd'|'interruption'|'visionEvent'|'setVoicePreference'} type
 * @property {boolean} [isQuestion] userTurn: whether the utterance was a question
 * @property {number}  [textLen]    assistantEnd: assistant text length in chars
 * @property {string}  [mode]       setVoicePreference: tts engine/hint label
 */

/**
 * Create a fresh user model.
 * @param {object} [initial] optional prior state (from toJSON) to restore
 * @returns {{ reconcile: Function, snapshot: Function, toJSON: Function, fromJSON: Function, reset: Function }}
 */
export function createUserModel(initial) {
  const s = {
    turnCount: 0,
    questionCount: 0,
    followUpCount: 0,
    interruptionCount: 0,
    assistantLenSum: 0,
    assistantLenCount: 0,
    visionEvents: 0,
    voicePreference: null,
    lastWasQuestion: false,
  };

  function fromJSON(plain) {
    if (!plain || typeof plain !== 'object') return s;
    for (const k of Object.keys(s)) {
      if (k === 'voicePreference') {
        if (typeof plain[k] === 'string') s[k] = plain[k];
      } else if (k === 'lastWasQuestion') {
        s[k] = !!plain[k];
      } else if (Number.isFinite(Number(plain[k]))) {
        s[k] = Number(plain[k]);
      }
    }
    return s;
  }
  fromJSON(initial);

  /**
   * Apply one observed interaction event.
   * @param {IntelligenceEvent} [event]
   */
  function reconcile(event) {
    const e = event || { type: '' };
    switch (e.type) {
      case 'userTurn': {
        s.turnCount += 1;
        if (e.isQuestion) {
          if (s.lastWasQuestion) s.followUpCount += 1;
          s.questionCount += 1;
          s.lastWasQuestion = true;
        } else {
          s.lastWasQuestion = false;
        }
        break;
      }
      case 'assistantEnd': {
        if (Number.isFinite(e.textLen) && e.textLen >= 0) {
          s.assistantLenSum += e.textLen;
          s.assistantLenCount += 1;
        }
        break;
      }
      case 'interruption':
        s.interruptionCount += 1;
        break;
      case 'visionEvent':
        s.visionEvents += 1;
        break;
      case 'setVoicePreference':
        if (typeof e.mode === 'string') s.voicePreference = e.mode;
        break;
      default:
        break;
    }
    return snapshot();
  }

  /** Read-only derived metrics. */
  function snapshot() {
    const interruptionRate = s.turnCount ? s.interruptionCount / s.turnCount : 0;
    const followUpRate = s.questionCount ? s.followUpCount / s.questionCount : 0;
    const visionUsage = Math.min(1, s.visionEvents / 20);
    const engagement = clamp01(0.5 + followUpRate * 0.5 - interruptionRate * 0.5);
    return {
      turnCount: s.turnCount,
      sampleSize: s.turnCount,
      questionCount: s.questionCount,
      followUpCount: s.followUpCount,
      interruptionCount: s.interruptionCount,
      interruptionRate: round3(interruptionRate),
      followUpRate: round3(followUpRate),
      avgAssistantLen: s.assistantLenCount ? Math.round(s.assistantLenSum / s.assistantLenCount) : 0,
      visionUsage: round3(visionUsage),
      engagement: round3(engagement),
      voicePreference: s.voicePreference,
    };
  }

  function toJSON() {
    const snap = snapshot();
    return Object.assign({}, s, snap);
  }

  function reset() {
    s.turnCount = 0;
    s.questionCount = 0;
    s.followUpCount = 0;
    s.interruptionCount = 0;
    s.assistantLenSum = 0;
    s.assistantLenCount = 0;
    s.visionEvents = 0;
    s.voicePreference = null;
    s.lastWasQuestion = false;
    return snapshot();
  }

  return { reconcile, snapshot, toJSON, fromJSON, reset };
}

function round3(v) {
  return Math.round(v * 1000) / 1000;
}