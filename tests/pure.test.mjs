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