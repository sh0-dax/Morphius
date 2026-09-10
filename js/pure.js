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

// ============================================================
// WebLLM "Lite mode" — recommend a model tier from rough device
// capability. Heuristic only: navigator.deviceMemory is capped at 8GB
// by the spec and unavailable in some browsers (notably Safari/Firefox),
// so this errs toward "mid" whenever signal is missing rather than
// guessing "low" or "high". Good enough to steer clearly low-end
// devices away from downloading a 5GB+ model by default -- not a
// precise VRAM measurement.
// ============================================================
export const WEBLLM_TIER_MODELS = {
  low: 'TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC',   // ~0.7 GB
  mid: 'Qwen2.5-1.5B-Instruct-q4f32_1-MLC',       // ~1.1 GB
  high: 'Llama-3.2-3B-Instruct-q4f32_1-MLC'       // ~2 GB (previous hardcoded default)
};

// One-time synchronous probe of the WebGL renderer string, used to classify
// the GPU when detectDeviceTier() runs at boot. Creates a throwaway 1x1
// context and immediately loses it. Never throws; empty string when the
// renderer is unknown (Node, driver blocked, etc.).
let _glRendererHintCache = null;
export function glRendererHint() {
  if (_glRendererHintCache !== null) return _glRendererHintCache;
  let hint = '';
  try {
    if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
      const canvas = document.createElement('canvas');
      /** @type {WebGLRenderingContext | WebGL2RenderingContext | null} */
      const gl = /** @type {any} */ (
        canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
      );
      if (gl) {
        const dbg = gl.getExtension('WEBGL_debug_renderer_info');
        if (dbg) hint = String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || '');
        const loseCtx = gl.getExtension('WEBGL_lose_context');
        if (loseCtx) loseCtx.loseContext();
      }
    }
  } catch (e) {
    hint = '';
  }
  _glRendererHintCache = hint;
  return _glRendererHintCache;
}

const WEAK_GPU_RE = /(llvmpipe|swiftshader|microsoft basic renderer|software)/i;
const INTEGRATED_GPU_RE = /(intel|uhd|hd graphics|\biris\b|radeon|adreno|\bmali\b|apple m\d|arm)/i;
const STRONG_GPU_RE = /(geforce|rtx|gtx|quadro|tesla|radeon rx|\bvega\b|apple m[2-9]\b)/i;

// Classify a WEBGL_debug_renderer_info string into a GPU-strength hint:
//   'low'  -> software rasterizer
//   'mid'  -> integrated/unknown-ish (capped at mid)
//   'high' -> discrete/strong
//   null   -> no usable signal (let the pure core heuristics decide)
export function gpuTierHint(glInfo) {
  const g = String(glInfo || '').toLowerCase();
  if (!g) return null;
  if (WEAK_GPU_RE.test(g)) return 'low';
  if (STRONG_GPU_RE.test(g)) return 'high';
  if (INTEGRATED_GPU_RE.test(g)) return 'mid';
  return null;
}

// Accepts a navigator-like object so it's testable without a real DOM:
// detectDeviceTier({ deviceMemory: 4, hardwareConcurrency: 4 })
//   nav must have { deviceMemory?, hardwareConcurrency? }
//   glInfo optional WEBGL renderer string (defaults to glRendererHint()).
// The classification now knows the GPU, so a desktop/Laptop that reports
// 8 cores + 8 GB but drives an INTEGRATED UHD/iGPU is correctly kept at
// 'mid' (its GPU is the bottleneck, not its CPU/RAM).
export function detectDeviceTier(nav, glInfo) {
  const n = nav || (typeof navigator !== 'undefined' ? navigator : {});
  const mem = typeof n.deviceMemory === 'number' ? n.deviceMemory : null; // GB
  const cores = typeof n.hardwareConcurrency === 'number' ? n.hardwareConcurrency : null;
  if (mem !== null && mem <= 4) return 'low';
  if (cores !== null && cores <= 4 && (mem === null || mem <= 4)) return 'low';

  const hint = glInfo === undefined ? glRendererHint() : gpuTierHint(glInfo);
  if (hint === 'low') return 'low';
  if (hint === 'high') return (mem !== null && mem >= 8 && cores !== null && cores >= 8) ? 'high' : 'mid';
  if (hint === 'mid') return 'mid'; // integrated GPU caps the tier at 'mid'

  // No GPU signal: a full 8+ GB / 8+ cores machine can still be 'high', but
  // only when BOTH signals agree (tightened from OR so a RAM-heavy, weak-core
  // box isn't over-ranked).
  if (mem !== null && mem >= 8 && cores !== null && cores >= 8) return 'high';
  return 'mid';
}

// Face render loop quality profile per device tier. This is the SINGLE knob
// that decides how much GPU work each frame costs (and whether the face runs
// at 60 or 30 fps on an idle screen). Kept out of app.js so it's pure +
// unit-testable.
//   pixelRatioCap  - max window.devicePixelRatio multiplier for the canvas
//   ibl            - scene.environment (RoomEnvironment PMREM) on
//   wireframe      - second pass drawing the same mesh as wireframe lines
//   antialias      - WebGL MSAA (fixed at renderer creation time)
//   fps            - face render target when idle (always 60 on 'high')
export function faceRenderQuality(tier) {
  const t = tier === 'high' ? 'high' : (tier === 'low' ? 'low' : 'mid');
  if (t === 'high') return { fps: 60, pixelRatioCap: 2, ibl: true, wireframe: true, antialias: true };
  if (t === 'mid') return { fps: 30, pixelRatioCap: 1.5, ibl: false, wireframe: true, antialias: true };
  return { fps: 30, pixelRatioCap: 1, ibl: false, wireframe: false, antialias: false };
}

// Face render-loop framerate gate. On low/mid devices the face renders at
// 30fps even with no camera pipeline (the wireframe+glow double draw is the
// heaviest regular GPU work), halving frame cost vs 60fps. High-tier devices
// and reduced-motion users stay at 60.
// Returns the target render rate in Hz: 60 | 30.
/**
 * @param {{ tier?: 'low'|'mid'|'high', reduceMotion?: boolean }} opts
 */
export function computeFaceRenderCap(opts) {
  const { tier, reduceMotion } = opts || {};
  if (reduceMotion) return 60;
  return faceRenderQuality(tier).fps;
}

// Whether frame `frameIndex` (0-based) should actually be drawn at `capHz`.
// At 60Hz every frame renders; at 30Hz alternate frames are skipped so the
// visible cadence stays constant while GPU work halves.
export function shouldRenderFaceFrame(frameIndex, capHz) {
  const cap = Math.floor(Number(capHz) || 60);
  if (cap >= 60) return true;
  return (frameIndex % 2) === 0;
}

export function recommendedWebLlmModel(nav) {
  return WEBLLM_TIER_MODELS[detectDeviceTier(nav)];
}

// ============================================================
// Minimal pub/sub event bus used by window.AIFace so embedders can
// react to state/provider/session changes without polling getStatus().
// Kept DOM-free and dependency-free so it's independently unit-testable.
// ============================================================
export function createEventBus(onHandlerError) {
  const listeners = Object.create(null);
  function on(event, handler) {
    if (typeof handler !== 'function') return () => {};
    (listeners[event] || (listeners[event] = new Set())).add(handler);
    return () => off(event, handler);
  }
  function off(event, handler) {
    const set = listeners[event];
    if (set) set.delete(handler);
  }
  function emit(event, detail) {
    const set = listeners[event];
    if (!set || !set.size) return;
    for (const handler of set) {
      try { handler(detail); }
      catch (e) { if (typeof onHandlerError === 'function') onHandlerError(event, e); }
    }
  }
  return { on, off, emit };
}

// ============================================================
// Vision helpers (DOM-free) — used by js/vision.js and unit-tested
// in tests so the parsing math doesn't need a browser.
// ============================================================

// Frame-rate/weight lerp used by the expression blender. Guards against NaN:
// an unseeded current value (undefined) is treated as 0 so a model switch or a
// first frame can never corrupt all weights to NaN. DOM-free + unit-tested.
export function lerpWeight(current, goal, rate) {
  const c = Number.isFinite(current) ? current : 0;
  return c + (goal - c) * rate;
}

// ---- Projection math (DOM-free, unit-tested) ----

// Clamps a projection scale into the allowed range (0.3..4 in the UI).
export function clampProjectionScale(scale, min = 0.3, max = 4) {
  const s = Number.isFinite(scale) ? scale : 1;
  return Math.min(max, Math.max(min, s));
}

// Fits an image of (w,h) to a target max width preserving aspect ratio.
// Returns { w, h }. Falls back to a 1:1 plane for missing/zero dims.
export function projectionFitAspect(imageW, imageH, maxWidth = 1.6) {
  const w = Number.isFinite(imageW) && imageW > 0 ? imageW : 1;
  const h = Number.isFinite(imageH) && imageH > 0 ? imageH : 1;
  const aspect = w / h || 1;
  const pw = maxWidth;
  const ph = pw / aspect;
  if (!Number.isFinite(ph) || ph <= 0) return { w: pw, h: pw };
  return { w: pw, h: ph };
}

// Maps a raw pointer/touch state to a projection gesture intent.
// Returns 'none' | 'pan' | 'pinch' | 'tap'. Used by the gesture layer to
// decide whether to handle an event or pass it through to OrbitControls.
export function classifyProjectionGesture(type, touchCount, moved, previous) {
  if (type === 'pointerdown' || type === 'touchstart') {
    if (touchCount >= 2) return 'pinch';
    return 'pan';
  }
  if (type === 'pointerup' || type === 'touchend') {
    if (touchCount < 2 && !moved) return 'tap';
    return 'none';
  }
  return 'none';
}

// Intersection-over-Union of two [x, y, w, h] boxes.
export function iou(a, b) {
  const [ax, ay, aw, ah] = a.bbox;
  const [bx, by, bw, bh] = b.bbox;
  const x1 = Math.max(ax, bx), y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw), y2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = aw * ah + bw * bh - inter;
  return union <= 0 ? 0 : inter / union;
}

// Greedy non-max-suppression. NOTE: currently UNUSED by the live YOLO26
// one-to-one path in vision.js (one-to-one already returns final boxes, so no
// NMS is needed). Kept here only for a possible future one-to-many export or
// for tests — don't assume it's part of the active pipeline.
export function nonMaxSuppression(boxes, iouThreshold = 0.5) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep = [];
  while (sorted.length) {
    const best = sorted.shift();
    keep.push(best);
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(best, sorted[i]) > iouThreshold) sorted.splice(i, 1);
    }
  }
  return keep;
}

// Class-aware NMS. The generic nonMaxSuppression() is class-agnostic, so two
// highly-overlapping boxes of DIFFERENT classes could suppress each other.
// Object detectors should only suppress overlaps within the SAME class, so we
// group boxes by their `class` field and run nonMaxSuppression() per group.
// Detections must carry a numeric `class` (e.g. parseYoloOneToOne output).
export function nonMaxSuppressionPerClass(boxes, iouThreshold = 0.5) {
  const byClass = new Map();
  for (const b of boxes) {
    const key = b.class;
    if (!byClass.has(key)) byClass.set(key, []);
    byClass.get(key).push(b);
  }
  const out = [];
  for (const group of byClass.values()) {
    for (const kept of nonMaxSuppression(group, iouThreshold)) out.push(kept);
  }
  return out;
}

// Parse a YOLO26 ONNX **one-to-one** output tensor into normalized detections.
// Contract (see vision.js for the full rationale / Ultralytics link):
//   - output is a flat array shaped [1, 300, 6]  (300 boxes per image)
//   - each box row is [x1, y1, x2, y2, score, class] in ABSOLUTE pixel coords
//     in letterboxed (640x640) input space
//   - one-to-one outputs are already end-to-end NMS-free
// Returns detections of the form { class, score, bbox: [x, y, w, h] } where
// bbox is normalized to [0..1] in the ORIGINAL video's coordinate space.
// `letterbox` = { scale, dx, dy } from preprocessForYolo() so we can undo the
// letterbox; `videoW`/`videoH` are the source video's pixel dimensions.
export function parseYoloOneToOne(data, dims, { scale, dx, dy, videoW, videoH, scoreThreshold = 0.45 }) {
  // dims is [1, 300, 6] → 300 boxes, 6 values each. The box count is the
  // middle dimension (dims[1]); each iteration reads a 6-wide row at i*6.
  const numBoxes = dims[1];
  const out = [];
  for (let i = 0; i < numBoxes; i++) {
    const row = i * 6;
    const score = data[row + 4];
    if (score < scoreThreshold) continue;
    const x1 = data[row + 0];
    const y1 = data[row + 1];
    const x2 = data[row + 2];
    const y2 = data[row + 3];
    const cls = data[row + 5];
    // undo letterbox to original video space, then normalize
    const x = (x1 - dx) / scale;
    const y = (y1 - dy) / scale;
    const w = (x2 - x1) / scale;
    const h = (y2 - y1) / scale;
    out.push({
      class: Math.round(cls),
      score,
      bbox: [x / videoW, y / videoH, w / videoW, h / videoH],
    });
  }
  return out;
}

// Retry/backoff helpers now live in core/retry.js and are re-exported here for
// backwards-compatible imports from './pure.js' (and existing unit tests).
export { fetchWithRetry, abortableDelay } from './core/retry.js';