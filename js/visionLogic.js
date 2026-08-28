// Pure, side-effect-free decision logic for the AI Vision feature.
// Kept separate from app.js so it can be unit-tested deterministically:
// it never touches the DOM, the TTS engine, the Web Speech API, or state.
// app.js maps the decisions this module returns onto real side effects.

// Map seen object classes to an avatar sentiment (feeling) label.
export const VISION_FEELING_MAP = {
  'person': 'love',
  'cat': 'happy', 'dog': 'happy', 'horse': 'happy', 'sheep': 'happy',
  'cow': 'happy', 'elephant': 'happy', 'bear': 'happy', 'zebra': 'happy',
  'giraffe': 'happy', 'bird': 'happy',
};

// Classes the avatar will proactively call out (out loud) when they appear.
export const VISION_SPEAK_CLASSES = new Set([
  'person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bird',
]);

export const VISION_SURPRISE_LINGER_MS = 1600;

// Resolve the sentiment decision for a frame of detections.
// prevClasses is the set of classes from the previous frame (use null/undefined
// on the first frame so everything counts as "new").
// Returns one of:
//   { action: 'none' }                        -> keep current state
//   { action: 'clear' }                       -> drop back to neutral
//   { action: 'feeling', label, lingerMs }    -> set feeling (lingerMs 0 = hold)
export function computeVisionFeeling({ personPresent, classes, prevClasses }) {
  const prev = prevClasses instanceof Set ? prevClasses : new Set();
  const newClasses = new Set([...classes].filter((c) => !prev.has(c)));

  if (personPresent && VISION_FEELING_MAP.person) {
    return { action: 'feeling', label: VISION_FEELING_MAP.person, lingerMs: 0 };
  }
  if (newClasses.size > 0) {
    const firstNew = [...newClasses][0];
    if (VISION_FEELING_MAP[firstNew]) {
      return { action: 'feeling', label: VISION_FEELING_MAP[firstNew], lingerMs: 0 };
    }
    return { action: 'feeling', label: 'surprised', lingerMs: VISION_SURPRISE_LINGER_MS };
  }
  if (classes.size === 0) {
    return { action: 'clear' };
  }
  const anyMapped = [...classes].find((c) => VISION_FEELING_MAP[c]);
  if (anyMapped) {
    return { action: 'feeling', label: VISION_FEELING_MAP[anyMapped], lingerMs: 0 };
  }
  return { action: 'none' };
}

// Which classes to announce, and the human hint used to build the line.
export function getSpeakHint(className, speakClasses) {
  const set = speakClasses instanceof Set ? speakClasses : VISION_SPEAK_CLASSES;
  if (className === 'person') return { type: 'person', obj: 'person' };
  if (set.has(className)) return { type: 'animal', obj: className };
  return { type: 'generic', obj: className };
}

// Decide whether the avatar should proactively speak about a scene change.
// Pure: the caller supplies current time/cooldown and speaking flags, so this
// stays deterministic and unit-testable.
// Returns { speak: boolean, className?: string }.
export function decideVisionCommentary({
  active,
  paused,
  commentaryEnabled,
  state,
  speaking,
  ttsSpeaking,
  ttsMuted,
  newClasses,
  nowMs,
  lastCommentAt,
  cooldownMs = 25000,
}) {
  if (!active || paused || !commentaryEnabled) return { speak: false };
  if (state !== 'idle' || speaking || ttsSpeaking || ttsMuted) return { speak: false };
  if (!(newClasses instanceof Set) || newClasses.size === 0) return { speak: false };
  if (nowMs - lastCommentAt < cooldownMs) return { speak: false };
  const firstNew = [...newClasses][0];
  const hint = getSpeakHint(firstNew, VISION_SPEAK_CLASSES);
  if (hint.type === 'generic') return { speak: false };
  return { speak: true, className: firstNew };
}

// Strip a class name to a display-friendly object label.
export function displayClass(className) {
  return (className || 'object').replace(/[_-]+/g, ' ');
}

// Device-tier GPU concurrency gating. On low-end devices (weak/mobile GPUs)
// running several camera/vision GPU pipelines at once (Three.js renderer +
// MediaPipe face mirror + YOLO WebGPU) starves the GPU frame budget and adds
// heat. This pure decision decides whether a camera pipeline may start given
// the device tier and which pipelines are already active. Pure + unit-tested.
//   tier: 'low' | 'mid' | 'high'
//   requested: 'vision' | 'mirror'
//   otherActive: whether the OTHER camera pipeline is currently running
// Returns { allowed: boolean, reason: 'none' | 'conflict' }.
export function canRunCameraPipeline({ tier, requested, otherActive }) {
  if (requested !== 'vision' && requested !== 'mirror') {
    return { allowed: false, reason: 'invalid' };
  }
  if (!otherActive) return { allowed: true, reason: 'none' };
  if (tier === 'low') return { allowed: false, reason: 'conflict' };
  // mid/high devices can generally run both, but flag it as a warning.
  return { allowed: true, reason: 'none' };
}

