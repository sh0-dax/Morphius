import { describe, it, expect } from 'vitest';
import { extractFeatures, QUESTION_WORDS, NEGATION_WORDS } from '../js/core/intelligence/features.js';
import { classifyIntent, scoreTokens, MIN_CONFIDENCE, INTENT_LEXICON } from '../js/core/intelligence/intent.js';
import { composeEmotion } from '../js/core/intelligence/emotion.js';
import { behaviorPolicy, neutralBehavior, MIN_TURNS_FOR_ADAPT } from '../js/core/intelligence/behavior.js';
import { createUserModel } from '../js/core/intelligence/userModel.js';
import { createAnomalyTracker } from '../js/core/intelligence/anomaly.js';
import { createIntentLearner } from '../js/core/intelligence/learn.js';

describe('features', () => {
  it('tokenizes and strips English stop words', () => {
    const f = extractFeatures('hello the world');
    expect(f.tokens).toEqual(['hello', 'world']);
    expect(f.tokenCount).toBe(2);
    expect(f.firstWord).toBe('hello');
  });

  it('detects question marks and bilingual question words', () => {
    expect(extractFeatures('can you help me?').hasQuestionMark).toBe(true);
    expect(extractFeatures('can you help me?').isQuestion).toBe(true);
    expect(extractFeatures('هل يمكنك مساعدتي؟').hasQuestionMark).toBe(true);
    expect(extractFeatures('how does this work').questionWordCount).toBe(1);
    expect(extractFeatures('كيف يعمل هذا؟').isQuestion).toBe(true);
    expect(extractFeatures('hello').isQuestion).toBe(false);
  });

  it('counts negation markers', () => {
    expect(extractFeatures('I am not sure').negationCount).toBe(1);
    expect(extractFeatures('لا أفهم').negationCount).toBe(1);
    expect(extractFeatures('never').negationCount).toBe(1);
  });

  it('computes lexical statistics', () => {
    const f = extractFeatures('abcdef! hi. Won der?');
    expect(f.avgTokenLength).toBe(3.5); // (6+2+3+3)/4
    expect(f.punctuationEnergy).toBe(2);
    expect(f.sentenceCount).toBe(3);
    expect(f.rawLength).toBeGreaterThan(0);
  });

  it('handles empty input safely', () => {
    const f = extractFeatures('');
    expect(f.tokens).toEqual([]);
    expect(f.avgTokenLength).toBe(0);
    expect(f.isQuestion).toBe(false);
  });
});

describe('intent', () => {
  it('classifies clear help-like utterances', () => {
    expect(classifyIntent('I need help').intent).toBe('help');
    expect(classifyIntent('ساعدني من فضلك').intent).toBe('help');
  });

  it('classifies strong greetings and farewells with high confidence', () => {
    const hi = classifyIntent('hello hi there');
    expect(hi.intent).toBe('greeting');
    expect(hi.confidence).toBeGreaterThanOrEqual(0.6);
    const bye = classifyIntent('goodbye see you');
    expect(bye.intent).toBe('farewell');
    expect(bye.confidence).toBeGreaterThanOrEqual(0.6);
  });

  it('signals confusion', () => {
    const r = classifyIntent('I am confused');
    expect(r.intent).toBe('confusion');
    expect(r.confidence).toBeLessThan(MIN_CONFIDENCE);
  });

  it('returns none + zero confidence on empty/no-signal input', () => {
    expect(classifyIntent('')).toEqual({ intent: 'none', confidence: 0 });
    expect(classifyIntent('the and or').intent).toBe('none');
  });

  it('is deterministic and bilingual', () => {
    const a = classifyIntent('uau زمان');
    const b = classifyIntent('uau زمان');
    expect(a).toEqual(b);
    expect(INTENT_LEXICON.positive.has('رائع')).toBe(true);
    expect(INTENT_LEXICON.greeting.has('مرحبا')).toBe(true);
  });

  it('scores token bags against a lexicon', () => {
    expect(scoreTokens(['help'], INTENT_LEXICON.help)).toBe(1);
    expect(scoreTokens([], INTENT_LEXICON.help)).toBe(0);
  });

  it('resolves weak multi-intent ties deterministically', () => {
    const r = classifyIntent('can you help me?');
    expect(['help', 'question']).toContain(r.intent);
    expect(r.confidence).toBeLessThan(MIN_CONFIDENCE); // weak, never acted on strongly
  });
});

describe('emotion (interaction affect mapping)', () => {
  it('does not detect user psychology — only maps intent to avatar affect', () => {
    const r = composeEmotion({ intent: 'confusion', confidence: 0.9, baseFeeling: 'neutral' });
    expect(r).toEqual({ feeling: null, attentive: true });
  });

  it('stays neutral on weak confidence', () => {
    expect(composeEmotion({ intent: 'confusion', confidence: 0.2 })).toEqual({ feeling: null, attentive: false });
    expect(composeEmotion({ intent: 'positive', confidence: 0.1 })).toEqual({ feeling: null, attentive: false });
  });

  it('maps positive/greeting to happy and negative to sad', () => {
    expect(composeEmotion({ intent: 'positive', confidence: 0.9 }).feeling).toBe('happy');
    expect(composeEmotion({ intent: 'greeting', confidence: 0.9 }).feeling).toBe('happy');
    expect(composeEmotion({ intent: 'negative', confidence: 0.9 }).feeling).toBe('sad');
  });

  it('lets the raw lexicon feeling win when present', () => {
    expect(composeEmotion({ intent: 'positive', confidence: 0.9, baseFeeling: 'angry' }).feeling).toBe('angry');
  });
});

describe('behavior policy', () => {
  it('OFF state returns the exact neutral baseline (parity guarantee)', () => {
    const off = behaviorPolicy({ intelligenceOn: false, intent: 'confusion', confidence: 0.9 });
    expect(off).toEqual(neutralBehavior());
  });

  it('is fully neutral for weak classifications', () => {
    const r = behaviorPolicy({ intent: 'confusion', confidence: 0.3 });
    expect(r).toEqual(neutralBehavior());
  });

  it('adopts a calmer attentive posture for confusion/help', () => {
    const r = behaviorPolicy({ intent: 'confusion', confidence: 0.9, emotion: composeEmotion({ intent: 'confusion', confidence: 0.9 }) });
    expect(r.attentive).toBe(true);
    expect(r.blinkScale).toBe(0.9);
    expect(r.headMoveScale).toBe(0.7);
    expect(r.speechRateHint).toBe(-0.2);
  });

  it('adds a little energy for positive interactions', () => {
    expect(behaviorPolicy({ intent: 'positive', confidence: 0.9 }).speechRateHint).toBe(0.2);
  });

  it('only adapts response length hints after enough turns', () => {
    const young = behaviorPolicy({ userModel: { turnCount: MIN_TURNS_FOR_ADAPT - 1, interruptionRate: 0.8 } });
    expect(young.responseLengthHint).toBe(0.5);
    const adult = behaviorPolicy({ userModel: { turnCount: 8, interruptionRate: 0.5 } });
    expect(adult.responseLengthHint).toBeCloseTo(0.3); // 0.5 - 0.4*0.5
  });

  it('clamps every numeric knob to its bounds', () => {
    const r = behaviorPolicy({ confidence: 0.9, intent: 'confusion', emotion: { feeling: null, attentive: true }, userModel: { turnCount: 50, interruptionRate: 2 } });
    expect(r.blinkScale).toBeGreaterThanOrEqual(0.4);
    expect(r.blinkScale).toBeLessThanOrEqual(1.2);
    expect(r.blinkScale).toBe(0.9); // attentive posture
    expect(r.responseLengthHint).toBeGreaterThanOrEqual(0.2);
    expect(r.speechRateHint).toBeGreaterThanOrEqual(-0.4);
  });

  it('OFF = exact behavior parity over a pathological input matrix', () => {
    // Every case that could trip the policy must produce the exact neutral
    // baseline when intelligence is disabled (the M2 parity guarantee).
    const cases = [
      {},
      { intent: 'confusion', confidence: 0.99, emotion: { feeling: 'sad', attentive: true } },
      { intent: 'positive', confidence: 1, emotion: { feeling: 'happy' } },
      { userModel: { turnCount: 999, interruptionRate: 0.95, sampleSize: 999 } },
      { intent: 'none', confidence: 0, state: 'alert' },
    ];
    const neutral = neutralBehavior();
    for (const c of cases) {
      expect(behaviorPolicy(Object.assign({ intelligenceOn: false }, c))).toEqual(neutral);
    }
  });
});

describe('userModel', () => {
  it('starts empty with neutral engagement baseline', () => {
    const m = createUserModel().snapshot();
    expect(m.turnCount).toBe(0);
    expect(m.sampleSize).toBe(0);
    expect(m.interruptionRate).toBe(0);
    expect(m.followUpRate).toBe(0);
    expect(m.engagement).toBe(0.5);
    expect(m.avgAssistantLen).toBe(0);
  });

  it('tracks turns, questions and follow-ups', () => {
    const m = createUserModel();
    m.reconcile({ type: 'userTurn', isQuestion: true });
    m.reconcile({ type: 'assistantEnd', textLen: 100 });
    m.reconcile({ type: 'userTurn', isQuestion: true });
    const s = m.snapshot();
    expect(s.turnCount).toBe(2);
    expect(s.questionCount).toBe(2);
    expect(s.followUpCount).toBe(1);
    expect(s.followUpRate).toBe(0.5);
    expect(s.avgAssistantLen).toBe(100);
  });

  it('counts interruptions with a rate against turns', () => {
    const m = createUserModel();
    m.reconcile({ type: 'userTurn', isQuestion: false });
    m.reconcile({ type: 'interruption' });
    m.reconcile({ type: 'userTurn', isQuestion: false });
    const s = m.snapshot();
    expect(s.interruptionCount).toBe(1);
    expect(s.interruptionRate).toBe(0.5);
    expect(s.engagement).toBeLessThan(0.5);
  });

  it('computes vision usage saturating at 1', () => {
    const m = createUserModel();
    for (let i = 0; i < 5; i++) m.reconcile({ type: 'visionEvent' });
    expect(m.snapshot().visionUsage).toBe(0.25);
    const m2 = createUserModel();
    for (let i = 0; i < 60; i++) m2.reconcile({ type: 'visionEvent' });
    expect(m2.snapshot().visionUsage).toBe(1);
  });

  it('round-trips through toJSON/fromJSON and resets', () => {
    const m = createUserModel();
    m.reconcile({ type: 'userTurn', isQuestion: true });
    m.reconcile({ type: 'userTurn', isQuestion: true });
    m.reconcile({ type: 'setVoicePreference', mode: 'gemini' });
    const json = m.toJSON();
    const restored = createUserModel(json);
    expect(restored.snapshot()).toEqual(m.snapshot());
    const fresh = restored.reset();
    expect(fresh.turnCount).toBe(0);
  });

  it('ignores malformed events', () => {
    const m = createUserModel();
    m.reconcile({});
    m.reconcile({ type: 'assistantEnd', textLen: NaN });
    expect(m.snapshot().turnCount).toBe(0);
  });
});

describe('anomaly', () => {
  it('scores 0 before enough observations', () => {
    const t = createAnomalyTracker();
    t.observe('x', 1);
    t.observe('x', 2);
    expect(t.score('x', 10)).toBe(0);
  });

  it('uses a real rolling window (old values drop out)', () => {
    const t = createAnomalyTracker({ windowSize: 3 });
    for (const v of [1, 2, 3, 100]) t.observe('m', v);
    expect(t.mean('m')).toBe(35); // only 2, 3, 100 remain
    expect(t.summary().m.count).toBe(3);
  });

  it('flags genuine outliers after enough samples', () => {
    const t = createAnomalyTracker();
    for (const v of [0.3, 0.4, 0.2, 0.35, 0.25, 0.3, 0.45, 0.15, 0.3, 0.4]) t.observe('r', v);
    const s = t.score('r', 0.9);
    expect(s).toBeGreaterThan(2);
  });

  it('returns 0 for degenerate constant windows', () => {
    const t = createAnomalyTracker();
    for (let i = 0; i < 5; i++) t.observe('c', 7);
    expect(t.score('c', 7)).toBe(0);
    expect(t.score('c', 9)).toBe(0);
  });

  it('scores the candidate against prior observations only (self-exclusion)', () => {
    const t = createAnomalyTracker();
    for (let i = 0; i < 10; i++) { t.observe('m', 0.25); t.observe('m', 0.35); }
    expect(t.score('m', 0.3)).toBeLessThan(0.5);       // in-distribution ≈ 0
    const before = t.score('m', 0.9);                  // 0.9 not yet in the window
    t.observe('m', 0.9);                               // now it is part of its own window
    const after = t.score('m', 0.9);
    expect(before).toBeGreaterThan(5);                 // clearly anomalous vs prior norm
    expect(after).toBeLessThan(before);                // dampened by self-inclusion
    expect(after).toBeLessThan(5);
  });

  it('ignores non-finite observations and resets', () => {
    const t = createAnomalyTracker();
    t.observe('m', NaN);
    t.observe('m', 1);
    expect(t.summary().m.count).toBe(1);
    t.reset();
    expect(t.summary()).toEqual({});
  });
});

describe('learn', () => {
  it('parses English corrections and extracts the corrected phrase', () => {
    const l = createIntentLearner();
    const c = l.parseCorrection('no, I meant fusing the vrm');
    expect(c.hadCorrection).toBe(true);
    expect(c.hadRejection).toBe(true);
    expect(c.phrase).toContain('fusing');
    expect(l.parseCorrection('can you help me?').hadCorrection).toBe(false);
  });

  it('parses Arabic corrections', () => {
    const l = createIntentLearner();
    const c = l.parseCorrection('لا أقصد الكاميرا');
    expect(c.hadCorrection).toBe(true);
    expect(c.phrase).toContain('كاميرا');
    expect(l.getStats().correctionsDetected).toBeGreaterThan(0);
  });

  it('teaches tokens toward an intent and refuses bad input', () => {
    const l = createIntentLearner();
    const r = l.teach('help', 'fusing the vrm');
    expect(r).not.toBeNull();
    expect(r.tokens).toContain('fusing');
    expect(r.tokens).not.toContain('the');            // stopword stripped
    expect(l.getExtraLexicon().help.fusing).toBe(1);
    expect(l.getLearnedTokens().length).toBe(2);
    expect(l.teach('none', 'whatever')).toBeNull();   // can't teach 'none'
    expect(l.teach('help', 'the a an')).toBeNull();   // stopwords only
    expect(l.teach('help', '')).toBeNull();
  });

  it('caps a learned token at MAX_TOKEN_WEIGHT', () => {
    const l = createIntentLearner();
    for (let i = 0; i < 5; i++) l.teach('help', 'zigzag');
    expect(l.getExtraLexicon().help.zigzag).toBe(3);
  });

  it('merges learned weights into classification, only when requested', () => {
    const before = classifyIntent('fusing vrm');
    expect(before.intent).toBe('none');
    const l = createIntentLearner();
    l.teach('help', 'fusing the vrm');
    const after = classifyIntent('fusing vrm', undefined, l.getExtraLexicon());
    expect(after.intent).toBe('help');
    expect(after.confidence).toBeGreaterThanOrEqual(0.6);
  });

  it('auto-learns repeated novel tokens from consistent intents', () => {
    const l = createIntentLearner();
    const f = extractFeatures('zigzag');
    expect(l.observe('zigzag', f, 'help')).toBe(false);   // still below MIN_REPEATS
    expect(l.observe('zigzag', f, 'help')).toBe(false);
    expect(l.observe('zigzag', f, 'help')).toBe(true);    // 3rd appearance learns
    expect(l.getExtraLexicon().help.zigzag).toBe(0.5);
    expect(l.getStats().repetitionsLearned).toBe(1);
    // learned token now classifies upcoming turns
    expect(classifyIntent('the zigzag setup', undefined, l.getExtraLexicon()).intent).toBe('help');
  });

  it('recent-turn ring and reset', () => {
    const l = createIntentLearner();
    const f1 = extractFeatures('hello');
    l.observe('hello', f1, 'greeting');
    expect(l.getRecent().length).toBe(1);
    l.observe('help', extractFeatures('help'), 'help');
    expect(l.getRecent().length).toBe(2);
    l.reset();
    expect(l.getStats().utterances).toBe(0);
    expect(l.getLearnedTokens()).toEqual([]);
    expect(l.getRecent()).toEqual([]);
  });
});