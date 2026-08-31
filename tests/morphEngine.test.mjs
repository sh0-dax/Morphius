import { describe, it, expect } from 'vitest';
import { computeBlendedWeights, shouldIdleLife } from '../js/core/morphEngine.js';
import { lerpWeight } from '../js/pure.js';

const lerp = lerpWeight;

describe('computeBlendedWeights', () => {
  const STATE = { idle: { jawOpen: 0.1, mouthClose: 0.2 }, alert: { eyeWideLeft: 0.9 } };
  const FEELING = { happy: { mouthSmileLeft: 0.5 } };
  const EMOTION = { surprised: { browInnerUp: 0.5 } };
  const VISION = { active: { browInnerUp: 0.25 } };

  it('blends state targets toward current weights with the default rate', () => {
    const { weights } = computeBlendedWeights({
      state: 'idle', feeling: 'neutral',
      stateTargets: STATE, feelingTargets: FEELING,
      dictKeys: ['jawOpen', 'mouthClose'], currentWeights: { jawOpen: 0, mouthClose: 0 },
      lerp,
    });
    // first frame at rate 0.08 => 0 + (0.1-0)*0.08
    expect(weights.jawOpen).toBeCloseTo(0.1 * 0.08, 10);
    expect(weights.mouthClose).toBeCloseTo(0.2 * 0.08, 10);
  });

  it('merges feeling over state for shared keys', () => {
    // feeling.happy sets mouthSmileLeft; state idle doesn't, so it appears.
    const { weights, slots } = computeBlendedWeights({
      state: 'idle', feeling: 'happy',
      stateTargets: STATE, feelingTargets: FEELING,
      dictKeys: ['jawOpen', 'mouthSmileLeft'], currentWeights: {},
      lerp,
    });
    expect(slots).toContain('mouthSmileLeft');
    expect(weights.mouthSmileLeft).toBeGreaterThan(0);
  });

  it('applies mirror copy-mode weights at the fast rate', () => {
    // start jawOpen at 0; mirror copy says 0.8, rate 0.45
    const { weights } = computeBlendedWeights({
      state: 'idle', feeling: 'neutral', copyMode: true,
      mirrorWeights: { jawOpen: 0.8 },
      dictKeys: ['jawOpen'], currentWeights: { jawOpen: 0 },
      stateTargets: STATE, feelingTargets: FEELING,
      lerp,
    });
    expect(weights.jawOpen).toBeCloseTo(0.8 * 0.45, 10);
  });

  it('ignores emotion unless emotionMode is active', () => {
    const base = computeBlendedWeights({
      state: 'idle', feeling: 'neutral',
      emotionTargets: EMOTION, dictKeys: ['browInnerUp'], currentWeights: {},
      stateTargets: STATE, feelingTargets: FEELING, lerp,
    });
    // not in any target map, it's zero-filled (matches original zero-fill of all slots)
    expect(base.weights.browInnerUp).toBe(0);

    const active = computeBlendedWeights({
      state: 'idle', feeling: 'neutral', emotionMode: true, emotion: 'surprised',
      emotionTargets: EMOTION, dictKeys: ['browInnerUp'], currentWeights: {},
      stateTargets: STATE, feelingTargets: FEELING, lerp,
    });
    expect(active.weights.browInnerUp).toBeGreaterThan(0);
  });

  it('applies vision target only for idle-life states', () => {
    const inIdle = computeBlendedWeights({
      state: 'idle', feeling: 'neutral', visionTargetActive: true,
      dictKeys: ['browInnerUp'], currentWeights: {},
      stateTargets: STATE, feelingTargets: FEELING, visionTargets: VISION, lerp,
    });
    expect(inIdle.weights.browInnerUp).toBeGreaterThan(0);

    const inAlert = computeBlendedWeights({
      state: 'alert', feeling: 'neutral', visionTargetActive: true,
      idleLifeStates: ['idle'], // alert not in idle-life
      dictKeys: ['browInnerUp'], currentWeights: {},
      stateTargets: STATE, feelingTargets: FEELING, visionTargets: VISION, lerp,
    });
    // alert not in idle-life states, so vision target does not apply; slot zero-filled
    expect(inAlert.weights.browInnerUp).toBe(0);
  });

  it('normalizes non-finite current weights to 0', () => {
    const { weights } = computeBlendedWeights({
      state: 'idle', feeling: 'neutral',
      stateTargets: STATE, feelingTargets: FEELING,
      dictKeys: ['jawOpen'], currentWeights: { jawOpen: undefined },
      lerp,
    });
    expect(Number.isFinite(weights.jawOpen)).toBe(true);
  });
});

describe('shouldIdleLife', () => {
  it('is true while idle and not speaking', () => {
    expect(shouldIdleLife({ speaking: false, visemePreview: null, state: 'idle' })).toBe(true);
  });
  it('is false while speaking or previewing a viseme', () => {
    expect(shouldIdleLife({ speaking: true, visemePreview: null, state: 'idle' })).toBe(false);
    expect(shouldIdleLife({ speaking: false, visemePreview: {}, state: 'idle' })).toBe(false);
  });
  it('is false outside idle-life states', () => {
    expect(shouldIdleLife({ speaking: false, visemePreview: null, state: 'alert' })).toBe(false);
  });
});
