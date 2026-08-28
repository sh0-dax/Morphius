// Pure presentation/data tables for Morphius (no DOM, no THREE, no side effects).
// Central home for the tuneable face-expression, emotion, feeling, color, and
// lighting presets so the app logic module does not have to mix config data
// with behavior. Imported by app.js and kept dependency-free for testability.

// Expression targets per top-level avatar state (blend weights in [0..1]).
export const STATE_TARGETS = {
  idle: { eyeBlinkLeft: 0.12, eyeBlinkRight: 0.12, mouthClose: 0.15, jawOpen: 0.0 },
  listening: { eyeWideLeft: 0.55, eyeWideRight: 0.55, browInnerUp: 0.45, browOuterUpLeft: 0.3, browOuterUpRight: 0.3, jawOpen: 0.12, mouthPucker: 0.1 },
  thinking: { eyeLookUpLeft: 0.6, eyeLookUpRight: 0.6, browDownLeft: 0.4, browDownRight: 0.35, mouthPucker: 0.35, mouthRollLower: 0.2, jawOpen: 0.05 },
  responding: { mouthSmileLeft: 0.55, mouthSmileRight: 0.55, jawOpen: 0.35, cheekPuff: 0.15, eyeSquintLeft: 0.2, eyeSquintRight: 0.2, browInnerUp: 0.25 },
  alert: { eyeWideLeft: 0.95, eyeWideRight: 0.95, browDownLeft: 0.75, browDownRight: 0.75, mouthFrownLeft: 0.55, mouthFrownRight: 0.55, jawOpen: 0.25, noseSneerLeft: 0.3, noseSneerRight: 0.3 },
};

// Expression overlay used in "emotion detect" mirror mode (layer on top of STATE_TARGETS).
export const EMOTION_TARGETS = {
  happy: { mouthSmileLeft: 0.5, mouthSmileRight: 0.5, cheekPuff: 0.12, eyeSquintLeft: 0.25, eyeSquintRight: 0.25 },
  upset: { browDownLeft: 0.5, browDownRight: 0.45, mouthFrownLeft: 0.45, mouthFrownRight: 0.45, jawOpen: 0.1 },
  surprised: { eyeWideLeft: 0.6, eyeWideRight: 0.6, browInnerUp: 0.5, browOuterUpLeft: 0.4, browOuterUpRight: 0.4, jawOpen: 0.25 },
  curious: { browInnerUp: 0.45, eyeLookUpLeft: 0.35, eyeLookUpRight: 0.35, mouthPucker: 0.2 },
  neutral: {},
};

// Expression overlay used by sentiment-driven feeling colors (layered last,
// so it wins over state/mirror defaults while the sentiment is active).
export const FEELING_TARGETS = {
  happy: { mouthSmileLeft: 0.5, mouthSmileRight: 0.5, cheekPuff: 0.12, eyeSquintLeft: 0.25, eyeSquintRight: 0.25 },
  sad: { browInnerUp: 0.35, browDownLeft: 0.3, browDownRight: 0.3, mouthFrownLeft: 0.35, mouthFrownRight: 0.35, eyeSquintLeft: 0.2, eyeSquintRight: 0.2 },
  angry: { browDownLeft: 0.65, browDownRight: 0.6, mouthFrownLeft: 0.45, mouthFrownRight: 0.45, eyeSquintLeft: 0.3, eyeSquintRight: 0.3, noseSneerLeft: 0.25, noseSneerRight: 0.25 },
  surprised: EMOTION_TARGETS.surprised,
  scared: { eyeWideLeft: 0.7, eyeWideRight: 0.7, browInnerUp: 0.5, browOuterUpLeft: 0.4, browOuterUpRight: 0.4, jawOpen: 0.2, mouthStretchLeft: 0.15, mouthStretchRight: 0.15 },
  love: { mouthSmileLeft: 0.45, mouthSmileRight: 0.45, cheekPuff: 0.1, browInnerUp: 0.2, eyeSquintLeft: 0.2, eyeSquintRight: 0.2 },
  neutral: {},
};

// Expression overlay applied when AI Vision detects a person while the avatar
// is idle. Subtle "engaged" look layered like emotion/feeling targets.
export const VISION_TARGETS = {
  active: { eyeLookUpLeft: 0.3, eyeLookUpRight: 0.3, browInnerUp: 0.25, mouthSmileLeft: 0.2, mouthSmileRight: 0.2 },
};

// State accent colors (blend weights as hex numbers / CSS strings).
export const STATE_COLORS = {
  idle:     { emissive: 0x2f81f7, wire: 0x7dd3fc, ambient: 0x020810, pulse: '#070c16', pulseOp: 0.0 },
  listening:{ emissive: 0x00d4aa, wire: 0x5fffd4, ambient: 0x001a12, pulse: '#003d2e', pulseOp: 0.15 },
  thinking: { emissive: 0xffaa00, wire: 0xffd700, ambient: 0x1a1200, pulse: '#3d2e00', pulseOp: 0.2 },
  responding:{ emissive: 0x00aaff, wire: 0x88ddff, ambient: 0x001a33, pulse: '#003d66', pulseOp: 0.25 },
  alert:    { emissive: 0xff2244, wire: 0xff6b7a, ambient: 0x1a0005, pulse: '#3d000a', pulseOp: 0.4 },
};

// Sentiment-driven feeling colors (blended over STATE_COLORS while active).
export const FEELING_COLORS = {
  happy:     { emissive: 0xffb84d, wire: 0xffe0a3, ambient: 0x3a2a08, pulse: '#ffb84d', pulseOp: 0.30 },
  sad:       { emissive: 0x5b8bf7, wire: 0xa9c4ff, ambient: 0x08123a, pulse: '#3f5bd9', pulseOp: 0.25 },
  angry:     { emissive: 0xff4747, wire: 0xff9a8a, ambient: 0x3a0808, pulse: '#ff4747', pulseOp: 0.35 },
  surprised: { emissive: 0xc46bff, wire: 0xe6c2ff, ambient: 0x26083a, pulse: '#c46bff', pulseOp: 0.30 },
  scared:    { emissive: 0x7fd4ff, wire: 0xc2ecff, ambient: 0x08263a, pulse: '#7fd4ff', pulseOp: 0.25 },
  love:      { emissive: 0xff7ab5, wire: 0xffc2dc, ambient: 0x3a0820, pulse: '#ff7ab5', pulseOp: 0.30 },
  neutral:   { emissive: 0x2f81f7, wire: 0x7dd3fc, ambient: 0x020810, pulse: '#070c16', pulseOp: 0.0 },
};

// Global lighting presets (configurable). The ambient light lerps toward these values.
export const LIGHT_PRESETS = {
  blueprint: { color: 0x2f81f7, intensity: 0.4 },
  matrix:    { color: 0x00ff7f, intensity: 0.5 },
  warm:      { color: 0xffb56b, intensity: 0.62 },
  soft:      { color: 0xf4f7ff, intensity: 0.5 },
  noir:      { color: 0xffffff, intensity: 0.22 },
};

// Default preset name applied at startup.
export const DEFAULT_LIGHT_PRESET = 'blueprint';
