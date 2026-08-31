/**
 * core/morphEngine.js
 *
 * Pure, DOM/Three-free helpers for the avatar face-morphing engine. The
 * render loop feeds the current emotional/behavioural state + mirror weights
 * through these functions to produce per-slot morph target amounts, blended
 * toward the current weights with per-target dampening rates.
 *
 * @module core/morphEngine
 */

/**
 * Combine the active state/emotion/feeling/vision target tables into one goal
 * map and blend the existing weights toward it.
 *
 * Order matters: later tables override earlier ones (state < emotion < feeling
 * < vision), matching the pre-extraction behaviour.
 *
 * @param {object} opts
 * @param {string} opts.state           canonical state ('idle' | 'listening' | 'thinking' | 'responding' | 'alert')
 * @param {string} opts.feeling         feeling key ('neutral', 'happy', ...)
 * @param {string} [opts.emotion]       mirror-emotion key when emotion mode is active
 * @param {boolean} [opts.emotionMode]  whether mirror emotion targetting is on
 * @param {boolean} [opts.copyMode]     whether mirror copy/puppet mode is on
 * @param {Object} [opts.mirrorWeights] raw per-slot weights from the mirror (copy mode)
 * @param {boolean} [opts.visionTargetActive] whether the vision "active/idle-life" target applies
 * @param {string[]} [opts.idleLifeStates]   states that count as idle life (default ['idle','thinking'])
 * @param {Object} [opts.stateTargets]  STATE_TARGETS table
 * @param {Object} [opts.emotionTargets] EMOTION_TARGETS table
 * @param {Object} [opts.feelingTargets] FEELING_TARGETS table
 * @param {Object} [opts.visionTargets] VISION_TARGETS table
 * @param {boolean} [opts.isVrm]        whether the current avatar uses the VRM expression manager
 * @param {string[]} [opts.allSlots]    canonical slot names (VRM)
 * @param {string[]} [opts.dictKeys]    morph target names present on the current mesh
 * @param {Object} [opts.currentWeights] current per-slot weights (read + blended in place to a new object)
 * @param {Function} [opts.lerp]        damping function: (current, goal, rate) => next
 * @returns {{ weights: Object, slots: string[] }} blended weights + full slot list
 */
export function computeBlendedWeights({
  state, feeling, emotion,
  emotionMode = false, copyMode = false,
  mirrorWeights = null,
  visionTargetActive = false,
  idleLifeStates = ['idle', 'thinking'],
  stateTargets = {}, emotionTargets = {}, feelingTargets = {}, visionTargets = {},
  isVrm = false, allSlots = [], dictKeys = [],
  currentWeights = {},
  lerp = defaultLerp,
}) {
  const stateTarget = stateTargets[state] || {};
  const emotionTarget = (emotionMode && emotion && emotionTargets[emotion]) || {};
  const feelingTarget = feelingTargets[feeling] || {};
  const visionTarget = (visionTargetActive && idleLifeStates.indexOf(state) !== -1) ? (visionTargets.active || {}) : {};
  const target = Object.assign({}, stateTarget, emotionTarget, feelingTarget, visionTarget);

  // For VRM avatars we also blend non-canonical target keys which aren't in the
  // mesh's morph list; otherwise those poses never appear on VRM models.
  const targetSlots = new Set(isVrm ? allSlots : dictKeys);
  for (const k of Object.keys(target)) targetSlots.add(k);
  const slots = [...targetSlots];

  const weights = Object.assign({}, currentWeights);
  for (const key of slots) {
    let goal = target[key] !== undefined ? target[key] : 0;
    let rate = 0.08;
    if (copyMode && mirrorWeights && mirrorWeights[key] !== undefined) {
      goal = mirrorWeights[key];
      rate = 0.45;
    }
    weights[key] = lerp(weights[key] ?? 0, goal, rate);
  }
  return { weights, slots };
}

/**
 * Decide whether autonomous idle life (blinking + breathing) should run.
 * @param {object} opts
 * @param {boolean} opts.speaking
 * @param {*}       opts.visemePreview   truthy while a viseme preview is active
 * @param {string}  opts.state
 * @param {string[]} [opts.idleLifeStates]
 * @returns {boolean}
 */
export function shouldIdleLife({ speaking, visemePreview, state, idleLifeStates = ['idle', 'thinking'] }) {
  return !speaking && !visemePreview && idleLifeStates.indexOf(state) !== -1;
}

/**
 * Default damping function: move `current` a fraction `rate` toward `goal`.
 * @param {number} current
 * @param {number} goal
 * @param {number} rate
 * @returns {number}
 */
function defaultLerp(current, goal, rate) {
  return current + (goal - current) * rate;
}
