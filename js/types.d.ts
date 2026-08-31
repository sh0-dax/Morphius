/**
 * Ambient type declarations for the avatar's shared runtime state object `S`
 * ("AvatarState"). Consumed by the core/ modules via JSDoc @typedef/@param so
 * `tsc --checkJs` can type-check them without a .js type annotation.
 */

export {};

/**
 * Per-slot morph weight map (slot name -> 0..1 target amount).
 * @typedef {Object<string, number>} WeightMap
 */

/**
 * The main avatar runtime state bag (exported as `window.AIFace.getStatus()`).
 * @typedef {Object} AvatarState
 * @property {string} currentState       canonical drive state ('idle'|'listening'|'thinking'|'responding'|'alert')
 * @property {string} feeling            feeling key ('neutral','happy','sad','angry','surprised','scared','love')
 * @property {number} feelingBlend       0..1 current blend toward feelingTarget
 * @property {number} feelingBlendTarget target blend value the state ramps toward
 * @property {number} feelingExpire      timestamp (ms) when a transient feeling expires
 * @property {WeightMap} currentWeights  current per-slot morph weights
 * @property {boolean} speaking          whether speech is active
 * @property {boolean} lipSyncActive     whether lip-sync is modulating the mouth
 * @property {boolean} isTtsSpeaking     whether TTS audio is currently playing
 * @property {boolean} streamDriven      whether the last response used streaming
 * @property {number|null} speakStartedAt timestamp the current speech began
 * @property {boolean} isLongResponse    whether the current reply is "long"
 * @property {number} lastActivityAt     timestamp of last user/AI activity
 * @property {number} talkPulse          decaying pulse for talking "energy"
 * @property {number} beatPulse          decaying pulse for beat/rhythm
 * @property {number} phonemeIntensity   0..1 mouth openness from phonemes
 * @property {number} bass              audio low-frequency band level
 * @property {number} mid               audio mid-frequency band level
 * @property {number} treble            audio high-frequency band level
 * @property {*} visemePreview           active viseme-preview spec, or null
 */
