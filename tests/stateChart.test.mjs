import { describe, it, expect } from 'vitest';
import { STATES, STATE_BODY_CLASS, stateBodyClass, isValidState } from '../js/core/stateChart.js';

describe('stateChart', () => {
  it('declares the canonical drive states', () => {
    expect(STATES).toContain('idle');
    expect(STATES).toContain('listening');
    expect(STATES).toContain('thinking');
    expect(STATES).toContain('responding');
    expect(STATES).toContain('alert');
  });

  it('maps responding/listening to a body class and others to empty', () => {
    expect(stateBodyClass('responding')).toBe('state-responding');
    expect(stateBodyClass('listening')).toBe('state-listening');
    expect(stateBodyClass('idle')).toBe('');
    expect(stateBodyClass('alert')).toBe('');
  });

  it('agrees with the STATE_BODY_CLASS table', () => {
    for (const s of STATES) {
      expect(stateBodyClass(s)).toBe(STATE_BODY_CLASS[s] || '');
    }
  });

  it('validates canonical states and rejects unknowns', () => {
    expect(isValidState('idle')).toBe(true);
    expect(isValidState('responding')).toBe(true);
    expect(isValidState('bogus')).toBe(false);
    expect(isValidState('')).toBe(false);
  });
});
