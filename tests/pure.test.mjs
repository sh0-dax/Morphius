import { describe, it, expect } from 'vitest';
import {
  detectFeeling,
  visemeFor,
  DEFAULT_VISEME,
  detectEmotion,
  contentToText,
  contentImages,
  buildUserContent,
  dataUrlMeta,
  geminiContentParts,
  WEBLLM_TIER_MODELS,
  detectDeviceTier,
  recommendedWebLlmModel,
  createEventBus,
  iou,
  nonMaxSuppression,
  nonMaxSuppressionPerClass,
  parseYoloOneToOne,
  lerpWeight,
  clampProjectionScale,
  projectionFitAspect,
  classifyProjectionGesture,
} from '../js/pure.js';
import { getMasterVolume, setMasterVolume, getOutputDevice, setOutputDevice, routeOutput } from '../js/masterBus.js';

describe('detectFeeling', () => {
  it('returns neutral for empty or missing input', () => {
    expect(detectFeeling('')).toBe('neutral');
    expect(detectFeeling(null)).toBe('neutral');
    expect(detectFeeling(undefined)).toBe('neutral');
  });

  it('detects happy from English tokens', () => {
    expect(detectFeeling('I feel happy and great today awesome')).toBe('happy');
  });

  it('detects sad from English tokens', () => {
    expect(detectFeeling('this is so bad, I am terribly upset')).toBe('sad');
  });

  it('detects Arabic feelings', () => {
    expect(detectFeeling('أحبك يا قمر')).toBe('love');
    expect(detectFeeling('أنا سعيد جدا')).toBe('happy');
    expect(detectFeeling('أنا غاضب منه')).toBe('angry');
  });

  it('is case-insensitive and ignores punctuation', () => {
    expect(detectFeeling('WOW!!! Incredible!!!')).toBe('surprised');
    expect(detectFeeling('Happy, happy, and HAPPY')).toBe('happy');
  });

  it('falls back to neutral on unknown words', () => {
    expect(detectFeeling('random gibberish words')).toBe('neutral');
  });
});

describe('visemeFor', () => {
  it('maps bilabial consonants correctly', () => {
    const s = visemeFor('b');
    expect(s.jawOpen).toBeLessThan(0.1);
    expect(s.mouthClose).toBeGreaterThan(0.5);
  });

  it('maps open vowels to a wide jaw', () => {
    expect(visemeFor('ا').jawOpen).toBeCloseTo(0.42, 5);
    expect(visemeFor('a').jawOpen).toBeCloseTo(0.42, 5);
    expect(visemeFor('A').jawOpen).toBeCloseTo(0.42, 5);
  });

  it('maps fricatives to mouth stretch', () => {
    for (const ch of ['s', 'S', 'ز', 'ش']) {
      expect(visemeFor(ch).mouthStretchLeft).toBeGreaterThan(0.3);
    }
  });

  it('returns the default shape for unknown characters', () => {
    expect(visemeFor('#')).toBe(DEFAULT_VISEME);
    expect(visemeFor('')).toBe(DEFAULT_VISEME);
  });
});

describe('detectEmotion', () => {
  it('returns neutral with no weights', () => {
    expect(detectEmotion(null)).toBe('neutral');
    expect(detectEmotion({})).toBe('neutral');
  });

  it('detects a smile as happy', () => {
    const w = { mouthSmileLeft: 0.6, mouthSmileRight: 0.6 };
    expect(detectEmotion(w)).toBe('happy');
  });

  it('detects surprise from wide eyes and open jaw', () => {
    const w = { eyeWideLeft: 0.7, eyeWideRight: 0.7, jawOpen: 0.7 };
    expect(detectEmotion(w)).toBe('surprised');
  });

  it('detects upset from a frown or lowered brows', () => {
    expect(detectEmotion({ mouthFrownLeft: 0.5, mouthFrownRight: 0.5 })).toBe('upset');
    expect(detectEmotion({ browDownLeft: 0.6, browDownRight: 0.6 })).toBe('upset');
  });

  it('detects curiosity from raised brows', () => {
    const w = { browInnerUp: 0.5, browOuterUpLeft: 0.3, browOuterUpRight: 0.3 };
    expect(detectEmotion(w)).toBe('curious');
  });
});

describe('multimodal content helpers', () => {
  const DATA_URL = 'data:image/png;base64,AAAA';

  it('contentToText handles strings and parts arrays', () => {
    expect(contentToText('plain')).toBe('plain');
    expect(contentToText([{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }])).toBe('a\nb');
    expect(contentToText([{ type: 'image_url', image_url: { url: DATA_URL } }])).toBe('');
    expect(contentToText(null)).toBe('');
  });

  it('contentImages extracts data urls in order', () => {
    expect(contentImages('plain')).toEqual([]);
    expect(contentImages(null)).toEqual([]);
    const parts = [
      { type: 'text', text: 'x' },
      { type: 'image_url', image_url: { url: DATA_URL } },
      { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,BBBB' } },
    ];
    expect(contentImages(parts)).toEqual([DATA_URL, 'data:image/jpeg;base64,BBBB']);
  });

  it('buildUserContent returns text when there is no image', () => {
    expect(buildUserContent('hello', null)).toBe('hello');
    expect(buildUserContent('', null)).toBe('');
  });

  it('buildUserContent builds parts when an image is attached', () => {
    const c = buildUserContent('what is this', DATA_URL);
    expect(Array.isArray(c)).toBe(true);
    expect(c).toEqual([
      { type: 'text', text: 'what is this' },
      { type: 'image_url', image_url: { url: DATA_URL } },
    ]);
  });

  it('dataUrlMeta parses mime and payload', () => {
    expect(dataUrlMeta(DATA_URL)).toEqual({ mimeType: 'image/png', payload: 'AAAA' });
    expect(dataUrlMeta('https://example.com/x.png')).toBeNull();
    expect(dataUrlMeta(null)).toBeNull();
  });

  it('geminiContentParts builds inline_data parts from a data url', () => {
    const parts = geminiContentParts([{ type: 'text', text: 'see' }, { type: 'image_url', image_url: { url: DATA_URL } }]);
    expect(parts).toEqual([
      { text: 'see' },
      { inline_data: { mime_type: 'image/png', data: 'AAAA' } },
    ]);
  });

  it('geminiContentParts falls back to text only for strings', () => {
    expect(geminiContentParts('just text')).toEqual([{ text: 'just text' }]);
  });
});

describe('master audio bus', () => {
  it('clamps volume to 0-1 range', () => {
    setMasterVolume(500);
    expect(getMasterVolume()).toBe(1);
    setMasterVolume(-3);
    expect(getMasterVolume()).toBe(0);
    setMasterVolume(0.45);
    expect(getMasterVolume()).toBeCloseTo(0.45);
  });

  it('falls back to 1 on garbage', () => {
    setMasterVolume('nope');
    expect(getMasterVolume()).toBe(1);
  });

  it('stores non-empty output device ids only', () => {
    setOutputDevice('');
    expect(getOutputDevice()).toBe('default');
    setOutputDevice('id-abc');
    expect(getOutputDevice()).toBe('id-abc');
  });

  it('routes output through setSinkId when supported', async () => {
    let routed = null;
    const fakeCtx = { setSinkId: async (id) => { routed = id; } };
    setOutputDevice('id-abc');
    expect(await routeOutput(fakeCtx)).toBe(true);
    expect(routed).toBe('id-abc');
  });

  it('skips routing for non-audio nodes and default device', async () => {
    setOutputDevice('default');
    expect(await routeOutput({})).toBe(false);
    expect(await routeOutput(null)).toBe(false);
  });
});

describe('detectDeviceTier / recommendedWebLlmModel (WebLLM Lite mode)', () => {
  it('recommends "low" for a low-memory device', () => {
    expect(detectDeviceTier({ deviceMemory: 2, hardwareConcurrency: 4 })).toBe('low');
  });

  it('recommends "low" for a low-core device even without deviceMemory support', () => {
    expect(detectDeviceTier({ deviceMemory: undefined, hardwareConcurrency: 2 })).toBe('low');
  });

  it('recommends "high" for a well-resourced device', () => {
    expect(detectDeviceTier({ deviceMemory: 8, hardwareConcurrency: 12 })).toBe('high');
  });

  it('falls back to "mid" when no signal is available at all', () => {
    expect(detectDeviceTier({})).toBe('mid');
  });

  it('does not crash when called with no argument (falls back to global navigator, if any)', () => {
    // Real value depends on the host running the test (Node's navigator vs a
    // browser's) -- the contract under test is "never throws, always returns
    // a valid tier", not a specific tier for this environment.
    expect(['low', 'mid', 'high']).toContain(detectDeviceTier(undefined));
  });

  it('maps each tier to the corresponding model id', () => {
    expect(recommendedWebLlmModel({ deviceMemory: 2 })).toBe(WEBLLM_TIER_MODELS.low);
    expect(recommendedWebLlmModel({ deviceMemory: 8, hardwareConcurrency: 8 })).toBe(WEBLLM_TIER_MODELS.high);
    expect(recommendedWebLlmModel({ deviceMemory: 6, hardwareConcurrency: 6 })).toBe(WEBLLM_TIER_MODELS.mid);
  });
});

describe('createEventBus', () => {
  it('calls subscribed handlers with the emitted detail', () => {
    const bus = createEventBus();
    let received = null;
    bus.on('stateChange', (detail) => { received = detail; });
    bus.emit('stateChange', { state: 'thinking', previous: 'idle' });
    expect(received).toEqual({ state: 'thinking', previous: 'idle' });
  });

  it('supports multiple handlers on the same event', () => {
    const bus = createEventBus();
    let a = 0, b = 0;
    bus.on('sessionUpdate', () => { a++; });
    bus.on('sessionUpdate', () => { b++; });
    bus.emit('sessionUpdate', {});
    expect(a).toBe(1);
    expect(b).toBe(1);
  });

  it('unsubscribes via the returned function', () => {
    const bus = createEventBus();
    let calls = 0;
    const unsubscribe = bus.on('providerSwitch', () => { calls++; });
    bus.emit('providerSwitch', {});
    unsubscribe();
    bus.emit('providerSwitch', {});
    expect(calls).toBe(1);
  });

  it('unsubscribes via off()', () => {
    const bus = createEventBus();
    let calls = 0;
    const handler = () => { calls++; };
    bus.on('speakingChange', handler);
    bus.off('speakingChange', handler);
    bus.emit('speakingChange', { speaking: true });
    expect(calls).toBe(0);
  });

  it('isolates a throwing handler and still calls the others, reporting the error', () => {
    const bus = createEventBus();
    const errors = [];
    bus.on('stateChange', () => { throw new Error('boom'); });
    let secondCalled = false;
    bus.on('stateChange', () => { secondCalled = true; });
    const busWithReporter = createEventBus((event, e) => errors.push([event, e.message]));
    busWithReporter.on('stateChange', () => { throw new Error('boom'); });
    expect(() => bus.emit('stateChange', {})).not.toThrow();
    expect(secondCalled).toBe(true);
    busWithReporter.emit('stateChange', {});
    expect(errors).toEqual([['stateChange', 'boom']]);
  });

  it('does nothing when emitting an event with no listeners', () => {
    const bus = createEventBus();
    expect(() => bus.emit('unknownEvent', {})).not.toThrow();
  });
});

describe('iou / nonMaxSuppression (vision helpers)', () => {
  it('computes IoU = 1 for identical boxes', () => {
    expect(iou({ bbox: [0, 0, 10, 10] }, { bbox: [0, 0, 10, 10] })).toBeCloseTo(1, 5);
  });

  it('computes IoU = 0 for non-overlapping boxes', () => {
    expect(iou({ bbox: [0, 0, 10, 10] }, { bbox: [20, 20, 10, 10] })).toBe(0);
  });

  it('computes a fractional IoU for partial overlap', () => {
    // box A 10x10; box B w=5,h=10 shifted right by 5 → overlap 5x10=50,
    // union = 100 + 50 - 50 = 100 → IoU = 0.5
    expect(iou({ bbox: [0, 0, 10, 10] }, { bbox: [5, 0, 5, 10] })).toBeCloseTo(0.5, 5);
  });

  it('drops overlapping lower-scored boxes (NMS)', () => {
    const keep = nonMaxSuppression([
      { bbox: [0, 0, 10, 10], score: 0.9 },
      { bbox: [1, 1, 10, 10], score: 0.3 },
      { bbox: [50, 50, 10, 10], score: 0.8 },
    ]);
    expect(keep.length).toBe(2);
    expect(keep.map((d) => d.score).sort((a, b) => b - a)).toEqual([0.9, 0.8]);
  });

  it('respects a custom IoU threshold', () => {
    const keep = nonMaxSuppression([
      { bbox: [0, 0, 10, 10], score: 0.9 },
      { bbox: [1, 1, 10, 10], score: 0.3 },
    ], 0.05); // tiny threshold → even small overlap collapses the pair
    expect(keep.length).toBe(1);
  });

  it('suppresses overlaps only within the same class (per-class NMS)', () => {
    // Two different classes overlapping heavily must BOTH survive per-class NMS
    // (class-agnostic NMS would wrongly drop one).
    const keep = nonMaxSuppressionPerClass([
      { bbox: [0, 0, 10, 10], score: 0.9, class: 0 },
      { bbox: [1, 1, 10, 10], score: 0.8, class: 1 },
    ]);
    expect(keep.length).toBe(2); // different classes → both kept
  });

  it('drops overlapping same-class boxes (per-class NMS)', () => {
    const keep = nonMaxSuppressionPerClass([
      { bbox: [0, 0, 10, 10], score: 0.9, class: 0 },
      { bbox: [1, 1, 10, 10], score: 0.3, class: 0 }, // same class, low score
      { bbox: [50, 50, 10, 10], score: 0.8, class: 0 },
      { bbox: [0, 0, 10, 10], score: 0.7, class: 2 },
    ]);
    expect(keep.length).toBe(3); // 0.9, 0.8, and class-2 box
    expect(keep.map((d) => d.score).sort((a, b) => b - a)).toEqual([0.9, 0.8, 0.7]);
  });
});

describe('parseYoloOneToOne (YOLO26 one-to-one output)', () => {
  // Square 640x640 video → scale=1, dx=0, dy=0; normalized == absolute/640.
  const lb = { scale: 1, dx: 0, dy: 0, videoW: 640, videoH: 640 };

  it('parses [1, 300, 6] rows into normalized [x, y, w, h] bboxes in [0..1]', () => {
    // One confident box "person" (class 0) covering [160,160,320,320] (→ 0.25,0.25,0.25,0.25)
    const data = new Float32Array(300 * 6);
    data.fill(0);
    // box 0: score 0.9, class 0
    data[0 + 0] = 160; data[0 + 1] = 160; data[0 + 2] = 320; data[0 + 3] = 320;
    data[0 + 4] = 0.9; data[0 + 5] = 0;

    const dets = parseYoloOneToOne(data, [1, 300, 6], lb);
    expect(dets.length).toBe(1);
    expect(dets[0].score).toBeCloseTo(0.9, 5);
    expect(dets[0].class).toBe(0);
    expect(dets[0].bbox[0]).toBeCloseTo(0.25, 5);
    expect(dets[0].bbox[1]).toBeCloseTo(0.25, 5);
    expect(dets[0].bbox[2]).toBeCloseTo(0.25, 5);
    expect(dets[0].bbox[3]).toBeCloseTo(0.25, 5);
    expect(dets[0].bbox.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it('filters out boxes below the score threshold', () => {
    const data = new Float32Array(300 * 6);
    data.fill(0);
    data[0 + 4] = 0.1; // low confidence
    data[6 + 4] = 0.99; // high confidence (box 1)
    const dets = parseYoloOneToOne(data, [1, 300, 6], lb);
    expect(dets.length).toBe(1);
    expect(dets[0].score).toBeCloseTo(0.99, 5);
  });

  it('undoes letterbox so coords land in original video space', () => {
    // 1280x720 video scaled to 640x640 → scale=0.5, dx=0, dy=140
    const letterbox = { scale: 0.5, dx: 0, dy: 140, videoW: 1280, videoH: 720 };
    const data = new Float32Array(300 * 6);
    data.fill(0);
    // a box [0,140,640,500] in letterbox space covers the full real frame
    data[0 + 0] = 0; data[0 + 1] = 140; data[0 + 2] = 640; data[0 + 3] = 500;
    data[0 + 4] = 0.8; data[0 + 5] = 1; // "bicycle"

    const dets = parseYoloOneToOne(data, [1, 300, 6], letterbox);
    expect(dets.length).toBe(1);
    // x1=(0-0)/0.5/1280=0 ; y1=(140-140)/0.5/720=0 ; w=640/0.5/1280=1 ; h=360/0.5/720=1
    expect(dets[0].bbox[0]).toBeCloseTo(0, 5);
    expect(dets[0].bbox[1]).toBeCloseTo(0, 5);
    expect(dets[0].bbox[2]).toBeCloseTo(1, 5);
    expect(dets[0].bbox[3]).toBeCloseTo(1, 5);
  });
});

// C1 regression: the expression blender lerp must never produce NaN, even when
// a slot's current weight is unseeded (undefined) after a model switch.
describe('lerpWeight', () => {
  it('keeps an already-set weight unchanged when goal equals current', () => {
    expect(lerpWeight(0.5, 0.5, 0.08)).toBe(0.5);
  });
  it('moves toward the goal by the given rate', () => {
    expect(lerpWeight(0, 1, 0.25)).toBeCloseTo(0.25, 5);
    expect(lerpWeight(0.5, 0, 0.5)).toBeCloseTo(0.25, 5);
  });
  it('treats an undefined/NaN current weight as 0 (no NaN corruption)', () => {
    const r = lerpWeight(undefined, 0.5, 0.08);
    expect(Number.isNaN(r)).toBe(false);
    expect(r).toBeCloseTo(0.04, 5);
    expect(Number.isNaN(lerpWeight(NaN, 0.5, 0.08))).toBe(false);
  });
});

describe('clampProjectionScale', () => {
  it('clamps into the allowed range', () => {
    expect(clampProjectionScale(0.1)).toBe(0.3);
    expect(clampProjectionScale(99)).toBe(4);
    expect(clampProjectionScale(1.5)).toBe(1.5);
  });
  it('honours custom min/max', () => {
    expect(clampProjectionScale(0.01, 0.05, 6)).toBe(0.05);
    expect(clampProjectionScale(50, 0.05, 6)).toBe(6);
  });
  it('falls back to 1 for NaN/undefined (no corruption)', () => {
    expect(clampProjectionScale(NaN)).toBe(1);
    expect(clampProjectionScale(undefined)).toBe(1);
  });
});

describe('projectionFitAspect', () => {
  it('fits landscape and portrait preserving aspect', () => {
    const l = projectionFitAspect(2, 1, 1.6);
    expect(l.w).toBeCloseTo(1.6, 5);
    expect(l.h).toBeCloseTo(0.8, 5);
    const p = projectionFitAspect(1, 2, 1.6);
    expect(p.w).toBeCloseTo(1.6, 5);
    expect(p.h).toBeCloseTo(3.2, 5);
  });
  it('falls back to 1:1 square for missing/zero dims', () => {
    expect(projectionFitAspect(0, 0)).toEqual({ w: 1.6, h: 1.6 });
  });
});

describe('classifyProjectionGesture', () => {
  it('classifies touch start', () => {
    expect(classifyProjectionGesture('touchstart', 1, false, null)).toBe('pan');
    expect(classifyProjectionGesture('touchstart', 2, false, null)).toBe('pinch');
  });
  it('classifies touch end as tap only when not moved', () => {
    expect(classifyProjectionGesture('touchend', 1, false, null)).toBe('tap');
    expect(classifyProjectionGesture('touchend', 1, true, null)).toBe('none');
    expect(classifyProjectionGesture('touchend', 2, false, null)).toBe('none');
  });
  it('returns none for everything else', () => {
    expect(classifyProjectionGesture('pointermove', 1, false, null)).toBe('none');
    expect(classifyProjectionGesture('wheel', 0, false, null)).toBe('none');
  });
});