import { describe, it, expect } from 'vitest';
import {
  sanitizeMessages,
  sessionTitle,
  makeSessionId,
  buildSession,
  dataUrlBytes,
  CHAT_MAX_MESSAGES,
  CHAT_MAX_TEXT_CHARS,
  CHAT_MAX_IMAGE_BYTES,
} from '../js/chatStore.js';

describe('sanitizeMessages', () => {
  it('returns [] for non-array input', () => {
    expect(sanitizeMessages(null)).toEqual([]);
    expect(sanitizeMessages(undefined)).toEqual([]);
    expect(sanitizeMessages('hello')).toEqual([]);
  });

  it('keeps only valid roles with non-empty text', () => {
    const sample = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
      { role: 'system', content: 'sys' },
      { role: 'bot', content: 'ignored' },
      { role: 'user', content: '   ' },
      null,
      'nope',
      { role: 'user' },
    ];
    expect(sanitizeMessages(sample)).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
      { role: 'system', content: 'sys' },
    ]);
  });

  it('preserves multimodal text parts', () => {
    const sample = [{ role: 'user', content: [{ type: 'text', text: 'see this' }] }];
    expect(sanitizeMessages(sample)).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'see this' }] },
    ]);
  });

  it('drops malformed parts and empty content arrays', () => {
    const sample = [
      { role: 'user', content: [{ type: 'text', text: '  ' }, { type: 'image_url', image_url: { url: 'data:x' } }, 'junk'] },
      { role: 'assistant', content: [] },
    ];
    expect(sanitizeMessages(sample)).toEqual([
      { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:x' } }] },
    ]);
  });

  it('never mutates the source array', () => {
    const src = [{ role: 'user', content: 'x' }, { role: 'junk', content: 'y' }];
    const copy = JSON.parse(JSON.stringify(src));
    sanitizeMessages(src);
    expect(src).toEqual(copy);
  });

  it('drops oversized/non-data images and caps text + history', () => {
    const big = 'data:image/png;base64,' + 'A'.repeat(CHAT_MAX_IMAGE_BYTES * 2);
    const small = 'data:image/png;base64,' + 'QQ==';
    expect(dataUrlBytes(small)).toBe(1);
    expect(dataUrlBytes(big)).toBeGreaterThan(CHAT_MAX_IMAGE_BYTES);
    const sample = [{
      role: 'user',
      content: [
        { type: 'text', text: 'ok' },
        { type: 'image_url', image_url: { url: big } },
        { type: 'image_url', image_url: { url: 'https://x/y.png' } },
        { type: 'image_url', image_url: { url: small } },
      ],
    }];
    const out = sanitizeMessages(sample);
    expect(out[0].content).toEqual([
      { type: 'text', text: 'ok' },
      { type: 'image_url', image_url: { url: small } },
    ]);
    const long = [{ role: 'user', content: 'x'.repeat(CHAT_MAX_TEXT_CHARS + 10) }];
    expect(sanitizeMessages(long)[0].content).toHaveLength(CHAT_MAX_TEXT_CHARS);
    const many = Array.from({ length: CHAT_MAX_MESSAGES + 10 }, (_, i) => ({ role: 'user', content: 'm' + i }));
    const capped = sanitizeMessages(many);
    expect(capped).toHaveLength(CHAT_MAX_MESSAGES);
    expect(capped[capped.length - 1].content).toBe('m' + (CHAT_MAX_MESSAGES + 9));
  });
});

describe('sessionTitle', () => {
  it('returns New Chat when there is no user message', () => {
    expect(sessionTitle([])).toBe('New Chat');
    expect(sessionTitle([{ role: 'assistant', content: 'yo' }])).toBe('New Chat');
    expect(sessionTitle(null)).toBe('New Chat');
  });

  it('uses the first user message and normalizes whitespace', () => {
    expect(sessionTitle([{ role: 'user', content: '  hello   world  ' }])).toBe('hello world');
  });

  it('truncates titles longer than 60 characters', () => {
    const long = 'a'.repeat(80);
    const title = sessionTitle([{ role: 'user', content: long }]);
    expect(title).toHaveLength(61);
    expect(title.endsWith('\u2026')).toBe(true);
  });
});

describe('makeSessionId', () => {
  it('always starts with the s_ prefix', () => {
    expect(makeSessionId(123, () => 0.5)).toMatch(/^s_/);
  });

  it('produces distinct ids for identical timestamps via injected rand', () => {
    expect(makeSessionId(1000, () => 0.1)).not.toBe(makeSessionId(1000, () => 0.9));
  });
});

describe('buildSession', () => {
  it('fills defaults for empty input', () => {
    const s = buildSession({ messages: [] });
    expect(s.id).toMatch(/^s_/);
    expect(s.provider).toBe('gemini');
    expect(s.model).toBe('');
    expect(s.title).toBe('New Chat');
    expect(s.messages).toEqual([]);
    expect(typeof s.created).toBe('number');
    expect(typeof s.updated).toBe('number');
  });

  it('sanitizes messages and derives the title', () => {
    const s = buildSession({ id: 'abc', provider: 'ollama', model: 'llama3.2', messages: [{ role: 'user', content: 'first question' }, { role: 'junk', content: 'nope' }] });
    expect(s.id).toBe('abc');
    expect(s.provider).toBe('ollama');
    expect(s.title).toBe('first question');
    expect(s.messages).toEqual([{ role: 'user', content: 'first question' }]);
  });

  it('keeps an explicit title over the derived one', () => {
    const s = buildSession({ title: 'Custom', messages: [{ role: 'user', content: 'x' }] });
    expect(s.title).toBe('Custom');
  });
});