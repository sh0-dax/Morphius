// ============================================================
// AI Face v6 — Pure helpers (DOM-free) shared by app.js and
// mirror.js. Kept side-effect free so they can be unit-tested
// with Node/Vitest directly.
// ============================================================

export const FEELING_LEXICON = {
  happy: ['happy', 'joy', 'joyful', 'glad', 'great', 'awesome', 'amazing', 'wonderful', 'excited', 'fun', 'nice', 'yay', 'haha', 'perfect', 'fantastic', 'سعيد', 'سعيدة', 'مبسوط', 'مبسوطة', 'فرحان', 'فرحانة', 'رائع', 'رائعة', 'ممتاز', 'جميل', 'جميلة', 'مرح', 'هههه'],
  sad: ['sad', 'sorry', 'unfortunately', 'bad', 'hurt', 'crying', 'cry', 'upset', 'terrible', 'miss', 'missed', 'alone', 'lonely', 'حزين', 'حزينة', 'متأسف', 'آسف', 'سيء', 'سيئة', 'يؤسفني', 'مستاء', 'مستاءة', 'ابكي', 'ابقى وحيدا'],
  angry: ['angry', 'mad', 'furious', 'hate', 'damn', 'stupid', 'wrong', 'never', 'annoying', 'غاضب', 'غاضبة', 'زعلان', 'زعلانة', 'أكره', 'مستفز', 'فظيع', 'سخيف'],
  surprised: ['wow', 'whoa', 'wha', 'surprised', 'shocking', 'unbelievable', 'incredible', 'really?', 'what!', 'مذهل', 'مفاجأة', 'مصدم', 'يا للهول', 'لا تصدق'],
  scared: ['afraid', 'scared', 'fear', 'danger', 'dangerous', 'worry', 'worried', 'terrified', 'panic', 'خائف', 'خائفة', 'أخاف', 'خطر', 'قلق', 'قلقة', 'مرعوب', 'مرعوبة'],
  love: ['love', 'lovely', 'beautiful', 'gorgeous', 'precious', 'dear', 'sweetheart', 'حبيبي', 'حبيبتي', 'أحبك', 'أحب', 'أعشق', 'قمر', 'غاليتي'],
};

const FEELING_MAP = {};
for (const f of Object.keys(FEELING_LEXICON)) {
  for (const w of FEELING_LEXICON[f]) FEELING_MAP[w] = f;
}

export function detectFeeling(text) {
  if (!text) return 'neutral';
  const clean = String(text).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ');
  const words = clean.split(/\s+/);
  let best = 'neutral', bestN = 0;
  const scores = {};
  for (const w of words) {
    const f = FEELING_MAP[w];
    if (f) { scores[f] = (scores[f] || 0) + 1; if (scores[f] > bestN) { bestN = scores[f]; best = f; } }
  }
  return best;
}

export const VISEME_GROUPS = [
  { chars: 'bmpبمپ',             shape: { jawOpen: 0.05, mouthClose: 0.6, mouthPucker: 0.08 } },
  { chars: 'ouwOUWوؤ',           shape: { jawOpen: 0.24, mouthPucker: 0.55 } },
  { chars: 'iyeIYEيإئ',          shape: { jawOpen: 0.13, mouthSmileLeft: 0.4, mouthSmileRight: 0.4 } },
  { chars: 'aAاآأىة',            shape: { jawOpen: 0.42 } },
  { chars: 'sSzZشسصزژ',          shape: { jawOpen: 0.09, mouthStretchLeft: 0.35, mouthStretchRight: 0.35 } },
  { chars: 'tdnTDNتدنطض',        shape: { jawOpen: 0.15, mouthClose: 0.08 } },
  { chars: 'lrLRلر',             shape: { jawOpen: 0.2, mouthRollLower: 0.15 } },
  { chars: 'kgqحخعغقكهKGQH',    shape: { jawOpen: 0.27, mouthPucker: 0.08 } },
];

export const DEFAULT_VISEME = { jawOpen: 0.17, mouthClose: 0.06 };

export const VISEME_KEYS = ['jawOpen', 'mouthClose', 'mouthPucker', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthRollLower'];

export function visemeFor(ch) {
  if (!ch) return DEFAULT_VISEME;
  for (const g of VISEME_GROUPS) { if (g.chars.indexOf(ch) !== -1) return g.shape; }
  return DEFAULT_VISEME;
}

export function detectEmotion(w) {
  const g = (k) => (w && w[k]) || 0;
  const smile = (g('mouthSmileLeft') + g('mouthSmileRight')) / 2;
  const frown = (g('mouthFrownLeft') + g('mouthFrownRight')) / 2;
  const browUp = (g('browInnerUp') + g('browOuterUpLeft') + g('browOuterUpRight')) / 3;
  const browDown = (g('browDownLeft') + g('browDownRight')) / 2;
  const wide = (g('eyeWideLeft') + g('eyeWideRight')) / 2;
  const jaw = g('jawOpen');
  if (wide > 0.55 && jaw > 0.55) return 'surprised';
  if (frown > 0.4 || browDown > 0.5) return 'upset';
  if (smile > 0.4) return 'happy';
  if (browUp > 0.35) return 'curious';
  return 'neutral';
}

// ============================================================
// Multimodal content helpers. Our persisted message "content" is
// either a plain string (text-only) or an OpenAI-style parts array:
//   [{ type: 'text', text }, { type: 'image_url', image_url: { url: dataUrl } }]
// These stay DOM-free so they can be unit-tested.
// ============================================================

// Extracts the plain-text portion of a message content.
export function contentToText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((p) => p && p.type === 'text' && typeof p.text === 'string')
      .map((p) => p.text)
      .join('\n');
  }
  return '';
}

// Extracts data-URL images from a message content (in order).
export function contentImages(content) {
  if (!Array.isArray(content)) return [];
  return content
    .filter((p) => p && p.type === 'image_url' && p.image_url && typeof p.image_url.url === 'string')
    .map((p) => p.image_url.url);
}

// Builds the content a user turn should carry given optional text + image.
export function buildUserContent(text, image) {
  const t = String(text || '').trim();
  if (!image) return t;
  const parts = [];
  if (t) parts.push({ type: 'text', text: t });
  parts.push({ type: 'image_url', image_url: { url: image } });
  return parts;
}

// Parses a data URL into mime type + raw payload (for Gemini inline_data
// and Ollama base64 images). Returns null when the URL is not a data URL.
export function dataUrlMeta(dataUrl) {
  if (typeof dataUrl !== 'string') return null;
  const m = /^data:([^;,]+)(?:;base64)?,(.*)$/s.exec(dataUrl.trim());
  if (!m) return null;
  return { mimeType: m[1], payload: m[2] };
}

// Converts any message content into a Gemini-style parts array.
export function geminiContentParts(content) {
  const parts = [];
  const text = contentToText(content);
  if (text) parts.push({ text });
  for (const url of contentImages(content)) {
    const meta = dataUrlMeta(url);
    if (meta) parts.push({ inline_data: { mime_type: meta.mimeType, data: meta.payload } });
  }
  return parts;
}