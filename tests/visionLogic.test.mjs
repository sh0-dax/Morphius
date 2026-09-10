import { describe, it, expect } from 'vitest';
import {
  VISION_FEELING_MAP,
  VISION_SPEAK_CLASSES,
  VISION_SURPRISE_LINGER_MS,
  computeVisionFeeling,
  decideVisionCommentary,
  getSpeakHint,
  displayClass,
  canRunCameraPipeline,
  shouldRunVisionFrame,
  clampVisionFps,
  computeNewClasses,
  splitBands,
  DEFAULT_VISION_FPS,
  MIN_VISION_FPS,
  MAX_VISION_FPS,
  lumaMotionScore,
  classifySceneMotion,
  decideDetectionGate,
} from '../js/visionLogic.js';

describe('VISION_FEELING_MAP / VISION_SPEAK_CLASSES', () => {
  it('maps person to love and animals to happy', () => {
    expect(VISION_FEELING_MAP.person).toBe('love');
    expect(VISION_FEELING_MAP.cat).toBe('happy');
    expect(VISION_FEELING_MAP.dog).toBe('happy');
    expect(VISION_FEELING_MAP['tv']).toBeUndefined();
  });
  it('keeps speak classes and the map in sync for person/animals', () => {
    expect(VISION_SPEAK_CLASSES.has('person')).toBe(true);
    for (const c of ['cat', 'dog', 'horse', 'bird']) {
      expect(VISION_SPEAK_CLASSES.has(c)).toBe(true);
    }
    for (const c of VISION_SPEAK_CLASSES) {
      expect(VISION_FEELING_MAP[c]).toBeDefined();
    }
  });
});

describe('computeVisionFeeling', () => {
  const S = (arr) => new Set(arr);

  it('returns clear when nothing is detected', () => {
    expect(computeVisionFeeling({ personPresent: false, classes: S([]), prevClasses: S([]) }))
      .toEqual({ action: 'clear' });
  });

  it('person present -> love with linger 0 (hold)', () => {
    expect(computeVisionFeeling({ personPresent: true, classes: S(['person']), prevClasses: S([]) }))
      .toEqual({ action: 'feeling', label: 'love', lingerMs: 0 });
  });

  it('new mapped animal -> its mapped feeling with linger 0', () => {
    expect(computeVisionFeeling({ personPresent: false, classes: S(['cat']), prevClasses: S([]) }))
      .toEqual({ action: 'feeling', label: 'happy', lingerMs: 0 });
  });

  it('new unmapped object -> surprised burst with linger', () => {
    expect(computeVisionFeeling({ personPresent: false, classes: S(['laptop']), prevClasses: S([]) }))
      .toEqual({ action: 'feeling', label: 'surprised', lingerMs: VISION_SURPRISE_LINGER_MS });
  });

  it('no new classes and nothing detected -> clear', () => {
    expect(computeVisionFeeling({ personPresent: false, classes: S([]), prevClasses: S(['cat']) }))
      .toEqual({ action: 'clear' });
  });

  it('persistent unmapped object (not new) does not clear a lingering burst', () => {
    const prev = S(['laptop']);
    const cur = S(['laptop']);
    expect(computeVisionFeeling({ personPresent: false, classes: cur, prevClasses: prev }))
      .toEqual({ action: 'none' });
  });

  it('persistent mapped object -> keeps its feeling', () => {
    expect(computeVisionFeeling({ personPresent: false, classes: S(['dog']), prevClasses: S(['dog']) }))
      .toEqual({ action: 'feeling', label: 'happy', lingerMs: 0 });
  });

  it('fires the surprised burst once, then goes quiet for a persistent unmapped object', () => {
    const first = computeVisionFeeling({ personPresent: false, classes: S(['laptop']), prevClasses: S([]) });
    expect(first).toEqual({ action: 'feeling', label: 'surprised', lingerMs: VISION_SURPRISE_LINGER_MS });
    const second = computeVisionFeeling({ personPresent: false, classes: S(['laptop']), prevClasses: S(['laptop']) });
    expect(second).toEqual({ action: 'none' });
  });
});

describe('getSpeakHint', () => {
  it('person -> person type', () => {
    expect(getSpeakHint('person', VISION_SPEAK_CLASSES)).toEqual({ type: 'person', obj: 'person' });
  });
  it('speakable animal -> animal type', () => {
    expect(getSpeakHint('cat', VISION_SPEAK_CLASSES)).toEqual({ type: 'animal', obj: 'cat' });
  });
  it('unmapped/unspoken -> generic type', () => {
    expect(getSpeakHint('laptop', VISION_SPEAK_CLASSES)).toEqual({ type: 'generic', obj: 'laptop' });
  });
});

describe('decideVisionCommentary', () => {
  const base = {
    active: true, paused: false, commentaryEnabled: true,
    state: 'idle', speaking: false, ttsSpeaking: false, ttsMuted: false,
    newClasses: new Set(['cat']), nowMs: 30000, lastCommentAt: 0, cooldownMs: 25000,
  };

  it('speaks for a new speakable class when idle', () => {
    expect(decideVisionCommentary(base)).toEqual({ speak: true, className: 'cat' });
  });

  it('does not speak when vision inactive', () => {
    expect(decideVisionCommentary({ ...base, active: false }).speak).toBe(false);
  });
  it('does not speak while paused', () => {
    expect(decideVisionCommentary({ ...base, paused: true }).speak).toBe(false);
  });
  it('does not speak when commentary is disabled', () => {
    expect(decideVisionCommentary({ ...base, commentaryEnabled: false }).speak).toBe(false);
  });
  it('does not speak when not idle', () => {
    expect(decideVisionCommentary({ ...base, state: 'thinking' }).speak).toBe(false);
  });
  it('does not speak while already speaking', () => {
    expect(decideVisionCommentary({ ...base, speaking: true }).speak).toBe(false);
    expect(decideVisionCommentary({ ...base, ttsSpeaking: true }).speak).toBe(false);
    expect(decideVisionCommentary({ ...base, ttsMuted: true }).speak).toBe(false);
  });
  it('does not speak with no new classes', () => {
    expect(decideVisionCommentary({ ...base, newClasses: new Set() }).speak).toBe(false);
  });
  it('respects the cooldown window', () => {
    const soon = decideVisionCommentary({ ...base, nowMs: 24000, lastCommentAt: 0 });
    expect(soon.speak).toBe(false);
    const after = decideVisionCommentary({ ...base, nowMs: 25001, lastCommentAt: 0 });
    expect(after.speak).toBe(true);
  });
  it('does not speak for generic (unspeakable / unmapped) classes', () => {
    expect(decideVisionCommentary({ ...base, newClasses: new Set(['laptop']) }).speak).toBe(false);
  });
  it('defaults cooldown to 25s when not provided', () => {
    const { cooldownMs, ...noCooldown } = base;
    expect(decideVisionCommentary(noCooldown)).toEqual({ speak: true, className: 'cat' });
  });
});

describe('displayClass', () => {
  it('replaces separators with spaces and falls back to object', () => {
    expect(displayClass('tv-remote')).toBe('tv remote');
    expect(displayClass('sports_ball')).toBe('sports ball');
    expect(displayClass('')).toBe('object');
    expect(displayClass()).toBe('object');
  });
});

describe('canRunCameraPipeline (device-tier GPU concurrency gate)', () => {
  it('allows vision when no other pipeline is active', () => {
    expect(canRunCameraPipeline({ tier: 'low', requested: 'vision', otherActive: false }))
      .toEqual({ allowed: true, reason: 'none' });
  });
  it('allows mirror when no other pipeline is active', () => {
    expect(canRunCameraPipeline({ tier: 'low', requested: 'mirror', otherActive: false }))
      .toEqual({ allowed: true, reason: 'none' });
  });
  it('blocks vision while mirror runs on a low-tier device', () => {
    expect(canRunCameraPipeline({ tier: 'low', requested: 'vision', otherActive: true }))
      .toEqual({ allowed: false, reason: 'conflict' });
  });
  it('blocks mirror while vision runs on a low-tier device', () => {
    expect(canRunCameraPipeline({ tier: 'low', requested: 'mirror', otherActive: true }))
      .toEqual({ allowed: false, reason: 'conflict' });
  });
  it('allows concurrency on mid-tier devices', () => {
    expect(canRunCameraPipeline({ tier: 'mid', requested: 'vision', otherActive: true }))
      .toEqual({ allowed: true, reason: 'none' });
  });
  it('allows concurrency on high-tier devices', () => {
    expect(canRunCameraPipeline({ tier: 'high', requested: 'mirror', otherActive: true }))
      .toEqual({ allowed: true, reason: 'none' });
  });
  it('rejects an unknown requested pipeline', () => {
    expect(canRunCameraPipeline({ tier: 'high', requested: 'unknown', otherActive: false }))
      .toEqual({ allowed: false, reason: 'invalid' });
  });
});

describe('shouldRunVisionFrame (rAF-free detection throttling)', () => {
  it('runs the very first frame unconditionally', () => {
    const r = shouldRunVisionFrame({ lastRun: 0, now: 0, fps: 8 });
    expect(r.shouldRun).toBe(true);
    expect(r.intervalMs).toBe(Math.round(1000 / 8));
  });
  it('runs immediately when the interval has elapsed', () => {
    const r = shouldRunVisionFrame({ lastRun: 1000, now: 1125, fps: 8 });
    expect(r.shouldRun).toBe(true);
    expect(r.nextDelayMs).toBe(0);
  });
  it('skips when the interval has not yet elapsed and offers a wait', () => {
    const r = shouldRunVisionFrame({ lastRun: 1000, now: 1060, fps: 8 });
    expect(r.shouldRun).toBe(false);
    expect(r.nextDelayMs).toBe(Math.round(1000 / 8) - 60);
  });
  it('defaults to DEFAULT_VISION_FPS when fps is omitted', () => {
    const r = shouldRunVisionFrame({ lastRun: 0, now: 0 });
    expect(r.intervalMs).toBe(Math.round(1000 / DEFAULT_VISION_FPS));
  });
  it('never throttles when fps is Infinity', () => {
    const r = shouldRunVisionFrame({ lastRun: 1000, now: 1001, fps: Infinity });
    expect(r.shouldRun).toBe(true);
    expect(r.intervalMs).toBe(0);
  });
  it('is rate-limited by the target fps', () => {
    const r = shouldRunVisionFrame({ lastRun: 0, now: 1000, fps: 10 });
    expect(r.intervalMs).toBe(100);
    expect(r.shouldRun).toBe(true);
  });
});

describe('clampVisionFps', () => {
  it('pins 0 / undefined / NaN to the default', () => {
    expect(clampVisionFps(0)).toBe(DEFAULT_VISION_FPS);
    expect(clampVisionFps(undefined)).toBe(DEFAULT_VISION_FPS);
    expect(clampVisionFps(NaN)).toBe(DEFAULT_VISION_FPS);
  });
  it('clamps to the min/max window', () => {
    expect(clampVisionFps(0.5)).toBe(MIN_VISION_FPS);
    expect(clampVisionFps(999)).toBe(MAX_VISION_FPS);
  });
  it('rounds fractional fps into the window', () => {
    expect(clampVisionFps(8.6)).toBe(9);
    expect(clampVisionFps(8.2)).toBe(8);
  });
  it('passes Infinity through as unlimited', () => {
    expect(clampVisionFps(Infinity)).toBe(Infinity);
  });
});

describe('computeNewClasses', () => {
  const S = (arr) => new Set(arr);

  it('treats everything as new on the first frame (null prev)', () => {
    expect([...computeNewClasses(S(['cat', 'tv']), null)].sort()).toEqual(['cat', 'tv']);
  });
  it('only reports classes not already present in the previous frame', () => {
    const next = computeNewClasses(S(['cat', 'tv', 'laptop']), S(['cat']));
    expect([...next].sort()).toEqual(['laptop', 'tv']);
  });
  it('returns empty set when nothing is new', () => {
    expect(computeNewClasses(S(['cat']), S(['cat'])).size).toBe(0);
  });
  it('does not mutate the passed prev set', () => {
    const prev = S(['cat']);
    computeNewClasses(S(['cat', 'dog']), prev);
    expect(prev.has('cat')).toBe(true);
    expect(prev.has('dog')).toBe(false);
  });
});

// The app's handleVisionDetections previously passed prevClasses == classes
// (it overwrote the "previous" set before decisioning), which made the
// surprised/new-object branch dead. This regression test pins the contract
// used by the app-side fix: compute new classes from the PREVIOUS frame.
describe('I1 regression: new-class reaction pipeline', () => {
  const S = (arr) => new Set(arr);
  const prevClasses = S(['cat']);
  const classes = S(['cat', 'laptop']);
  const newClasses = computeNewClasses(classes, prevClasses);
  const feeling = computeVisionFeeling({ personPresent: false, classes, prevClasses });
  const firstNew = [...newClasses][0];

  it('derives a non-empty new set from the previous frame', () => {
    expect(newClasses.size).toBeGreaterThan(0);
    expect(firstNew).toBe('laptop');
  });
  it('fires the surprised burst because the new object is unmapped', () => {
    expect(feeling).toEqual({ action: 'feeling', label: 'surprised', lingerMs: VISION_SURPRISE_LINGER_MS });
  });
});

describe('splitBands', () => {
  // 512-bin FFT at 48kHz -> 93.75 Hz per bin.
  const bins = 256;
  const sampleRate = 48000;
  const fftSize = 512;

  it('returns zeros for falsy/empty input', () => {
    expect(splitBands(null, sampleRate, fftSize)).toEqual({ bass: 0, mid: 0, treble: 0 });
    expect(splitBands([], sampleRate, fftSize)).toEqual({ bass: 0, mid: 0, treble: 0 });
  });
  it('ignores bins at or above the 8kHz treble cutoff', () => {
    const arr = new Uint8Array(bins).fill(255); // all bins loud
    const b = splitBands(arr, sampleRate, fftSize);
    expect(b.bass).toBeGreaterThan(0);
    expect(b.mid).toBeGreaterThan(0);
  });
  it('is dominated by bass for a low-frequency bin', () => {
    const arr = new Uint8Array(bins);       // all zero
    arr[0] = 255;                            // 0 Hz -> bass
    const b = splitBands(arr, sampleRate, fftSize);
    expect(b.bass).toBeGreaterThan(b.mid);
    expect(b.mid).toBe(0);
  });
  it('treats a mid-frequency bin as mid', () => {
    const arr = new Uint8Array(bins);
    const midBin = Math.round(1000 / (sampleRate / fftSize)); // ~1000 Hz
    arr[midBin] = 255;
    const b = splitBands(arr, sampleRate, fftSize);
    expect(b.mid).toBeGreaterThan(0);
    expect(b.bass).toBe(0);
  });
});

describe('lumaMotionScore / classifySceneMotion (16x12 gate probe)', () => {
  const grid = (w, h, fill = (x, y) => 0) => {
    const a = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) a[y * w + x] = fill(x, y);
    }
    return a;
  };

  it('returns 0 for null/shape-mismatched input', () => {
    expect(lumaMotionScore(null, new Uint8Array(4))).toBe(0);
    expect(lumaMotionScore(new Uint8Array(4), new Uint8Array(8))).toBe(0);
  });

  it('scores 0 for identical probes', () => {
    const a = grid(16, 12, () => 100);
    expect(lumaMotionScore(a, new Uint8Array(a))).toBe(0);
    expect(classifySceneMotion(a, new Uint8Array(a))).toBe(false);
  });

  it('ignores sub-threshold sensor noise', () => {
    const a = grid(16, 12, (x) => x % 7);
    const b = new Uint8Array(a);
    b[0] += 3; // below the default cellThreshold of 6
    expect(classifySceneMotion(a, b)).toBe(false);
  });

  it('detects motion when enough cells move past the threshold', () => {
    const a = grid(16, 12, () => 60);
    const b = new Uint8Array(a).fill(200); // whole scene changed
    expect(classifySceneMotion(a, b)).toBe(true);
    const c = new Uint8Array(a);
    for (let i = 0; i < 8; i++) c[i] = 200; // only 8 of 192 cells moved
    expect(classifySceneMotion(a, c, { cellsRequired: 8 })).toBe(true);
    expect(classifySceneMotion(a, c, { cellsRequired: 16 })).toBe(false);
  });
});

describe('decideDetectionGate (idle-floor scene gate)', () => {
  // Helpers that mimic the vision.js loop contract.
  const HOT = { personPresent: true, motion: false };
  const COLD_IDLE = { personPresent: false, motion: false };
  const COLD_MOTION = { personPresent: false, motion: true };
  const GATE = { baseFps: 8, floorFps: 2, burstMs: 3000, stableMs: 3000 };
  const base = () => Math.round(1000 / 8); // 125
  const floor = () => Math.round(1000 / 2); // 500

  it('runs on the very first wake (lastRun 0)', () => {
    const r = decideDetectionGate({ now: 0, lastRun: 0, ...COLD_IDLE, ...GATE, state: null });
    expect(r.shouldRun).toBe(true);
    expect(r.state.stableSince).toBe(0); // seeds the stability epoch
  });

  it('keeps full rate while a person is present', () => {
    const state = { burstUntil: -Infinity, stableSince: -1 };
    let lastRun = 0;
    let r;
    for (let i = 0; i < 3; i++) {
      const now = lastRun + base();
      r = decideDetectionGate({ now, lastRun, ...HOT, ...GATE, state });
      expect(r.shouldRun).toBe(true);
      lastRun = now;
    }
    expect(r.state.burstUntil).toBeGreaterThan(lastRun - GATE.burstMs);
  });

  it('holds full rate inside the post-person burst tail', () => {
    const now = 10000;
    const state = { burstUntil: now + 1500, stableSince: -1 };
    const r = decideDetectionGate({ now, lastRun: now - base(), ...COLD_IDLE, ...GATE, state });
    expect(r.state.burstUntil).toBe(now + 1500); // burst is not re-armed without a person
    expect(r.intervalMs).toBe(base());
  });

  it('reacts at full rate while the cold scene keeps moving', () => {
    const state = { burstUntil: -Infinity, stableSince: -1 };
    const now = 5000;
    const r = decideDetectionGate({ now, lastRun: now - 400, ...COLD_MOTION, ...GATE, state });
    expect(r.intervalMs).toBe(base());
    expect(r.shouldRun).toBe(true); // 400ms >= 125ms
    expect(r.state.stableSince).toBe(-1);
  });

  it('stays at base rate during the unstable settle grace period', () => {
    const state = { burstUntil: -Infinity, stableSince: 2000 };
    const now = 4000; // 2000ms static, still below stableMs=3000
    const r = decideDetectionGate({ now, lastRun: now - 120, ...COLD_IDLE, ...GATE, state });
    expect(r.intervalMs).toBe(base());
  });

  it('drops to the floor after the scene has been static past stableMs', () => {
    const state = { burstUntil: -Infinity, stableSince: 1000 };
    const now = 4200; // 3200ms static >= stableMs
    const r = decideDetectionGate({ now, lastRun: now - 120, ...COLD_IDLE, ...GATE, state });
    expect(r.intervalMs).toBe(floor());
  });

  it('schedules the exact remaining delay when not due yet', () => {
    const state = { burstUntil: -Infinity, stableSince: 0 };
    const now = 8000; // stable >= stableMs -> floor cadence
    // last run 380ms ago, floor interval 500ms -> next due in 120ms
    const r = decideDetectionGate({ now, lastRun: now - 380, ...COLD_IDLE, ...GATE, state });
    expect(r.shouldRun).toBe(false);
    expect(r.nextDelayMs).toBe(floor() - 380);
  });

  it('never lets the floor exceed the base (user set 1fps)', () => {
    const state = { burstUntil: -Infinity, stableSince: 0 };
    const now = 5000;
    const r = decideDetectionGate({ now, lastRun: now - 500, ...COLD_IDLE, ...GATE, baseFps: 1, floorFps: 2, state });
    expect(r.intervalMs).toBe(1000); // floor clamped to base
  });

  it('runs everything when baseFps is Infinity (unthrottled)', () => {
    const state = { burstUntil: -Infinity, stableSince: -1 };
    const r = decideDetectionGate({ now: 1234, lastRun: 1233, ...COLD_IDLE, baseFps: Infinity, floorFps: 2, state });
    expect(r.shouldRun).toBe(true);
    expect(r.nextDelayMs).toBe(0);
  });

  it('does not mutate the caller state object (pure)', () => {
    const state = { burstUntil: -Infinity, stableSince: 0 };
    const snapshot = { ...state };
    decideDetectionGate({ now: 5000, lastRun: 4900, ...HOT, ...GATE, state });
    expect(state).toEqual(snapshot);
  });
});

