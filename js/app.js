// ============================================================
// AI Face v6 — App Logic
// ============================================================

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { Mirror, startMirror, stopMirror } from './mirror.js';
import { Vision } from './vision.js';
import { LocalSpeech, startLocalSTT, stopLocalSTT, generateLocalAudio, playLocalAudio, setLocalCallbacks, stopLocalAudio, setWhisperModel, applyMasterSettings } from './localSpeech.js';
import { modelProgress } from './progress.js';
import { getMasterVolume, setMasterVolume, setOutputDevice, routeOutput } from './masterBus.js';
import { detectFeeling, visemeFor, DEFAULT_VISEME, VISEME_KEYS, contentToText, contentImages, buildUserContent, geminiContentParts, detectDeviceTier, recommendedWebLlmModel, createEventBus } from './pure.js';
import { saveSession, loadSession, listSessions, deleteSession, getLastSession, buildSession, makeSessionId, sanitizeMessages, sessionTitle } from './chatStore.js';

// ---- i18n (lightweight loader, EN fallback, dir flip for RTL) ----
const I18N_FILES = { en: 'i18n/en.json', ar: 'i18n/ar.json', fr: 'i18n/fr.json', de: 'i18n/de.json', es: 'i18n/es.json', ja: 'i18n/ja.json' };
let I18N = null;
let I18N_LOCALE = 'en';

function t(key, fallback) {
  const v = I18N && I18N[key];
  return v != null ? v : (fallback !== undefined ? fallback : key);
}

function detectLocale() {
  const nl = (navigator.language || 'en').toLowerCase();
  if (nl.startsWith('ar')) return 'ar';
  if (nl.startsWith('fr')) return 'fr';
  if (nl.startsWith('de')) return 'de';
  if (nl.startsWith('es')) return 'es';
  if (nl.startsWith('ja')) return 'ja';
  return 'en';
}

async function applyI18n(locale) {
  I18N_LOCALE = I18N_FILES[locale] ? locale : 'en';
  try {
    const res = await fetch(I18N_FILES[I18N_LOCALE], { cache: 'no-store' });
    if (res.ok) I18N = await res.json();
    else I18N = null;
  } catch (e) { I18N = null; }
  document.documentElement.lang = I18N_LOCALE;
  document.documentElement.dir = I18N_LOCALE === 'ar' ? 'rtl' : 'ltr';
  if (I18N) {
    document.querySelectorAll('[data-i18n]').forEach((el) => {
      const key = el.getAttribute('data-i18n');
      if (I18N[key] != null) el.textContent = I18N[key];
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
      const key = el.getAttribute('data-i18n-placeholder');
      if (I18N[key] != null) el.setAttribute('placeholder', I18N[key]);
    });
  }
  dbg('i18n: ' + I18N_LOCALE, 'ok');
}

// applyI18n() is invoked from loadSettings (and on Locale change) once module
// consts it depends on are initialized.

// ---- UI Helpers ----
const debugEl = document.getElementById('debugLog');
const termEl = document.getElementById('terminalOutput');
const TERM_MAX = 50;

function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function term(msg, type) {
  const t = new Date().toTimeString().slice(0, 8);
  const cls = 't-' + (type || 'info');
  const line = document.createElement('div');
  line.className = 't-line';
  line.innerHTML = '<span class="t-time">' + esc(t) + '</span> <span class="' + esc(cls) + '">' + esc(msg) + '</span>';
  termEl.appendChild(line);
  if (termEl.children.length > TERM_MAX) termEl.removeChild(termEl.firstChild);
  termEl.scrollTop = termEl.scrollHeight;
}

function dbg(msg, type) {
  const t = new Date().toTimeString().slice(0, 8);
  const span = document.createElement('div');
  let color = '#8aa';
  if (type === 'err') color = 'var(--alert)';
  if (type === 'ok') color = 'var(--success)';
  if (type === 'warn') color = 'var(--warn)';
  span.innerHTML = '<span style="color:var(--dim)">[' + esc(t) + ']</span> <span style="color:' + esc(color) + '">' + esc(msg) + '</span>';
  debugEl.appendChild(span);
  debugEl.scrollTop = debugEl.scrollHeight;
  term(msg, type);
  console.log('[' + type.toUpperCase() + '] ' + msg);
}
window.onerror = (msg, src, line, col, err) => { dbg('ERROR: ' + msg + ' @ ' + line + ':' + col, 'err'); };
window.onunhandledrejection = (e) => { dbg('PROMISE REJECTION: ' + e.reason, 'err'); };

dbg('Module started', 'ok');

// ---- Session Node ID ----
const hudNode = document.getElementById('hudNode');
if (hudNode) hudNode.textContent = 'SYS.NODE // 0x' + Math.floor(Math.random() * 0xFFFF).toString(16).toUpperCase().padStart(4, '0');

// ---- Debug Log Toggle ----
const debugToggle = document.getElementById('debugToggle');
function toggleDebugLog() {
  debugEl.classList.toggle('visible');
}
debugToggle.addEventListener('click', toggleDebugLog);
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'd') {
    e.preventDefault();
    toggleDebugLog();
  }
});

// ---- Data Panel Waveform ----
const waveformCanvas = document.getElementById('waveform-canvas');
const wCtx = waveformCanvas.getContext('2d');
let waveformData = new Float32Array(64);

function updateWaveform(dataArray) {
  if (dataArray && dataArray.length) {
    const step = Math.max(1, Math.floor(dataArray.length / 64));
    const newData = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      let sum = 0;
      for (let j = 0; j < step; j++) {
        const idx = i * step + j;
        if (idx < dataArray.length) sum += dataArray[idx];
      }
      newData[i] = sum / step;
    }
    waveformData = newData;
  }
}

function drawWaveform() {
  const w = waveformCanvas.width = waveformCanvas.clientWidth * (window.devicePixelRatio || 1);
  const h = waveformCanvas.height = waveformCanvas.clientHeight * (window.devicePixelRatio || 1);
  wCtx.clearRect(0, 0, w, h);

  const mid = h / 2;
  const amp = h * 0.4;
  const step = w / waveformData.length;

  wCtx.beginPath();
  wCtx.strokeStyle = '#7dd3fc';
  wCtx.lineWidth = 2;
  wCtx.shadowColor = 'rgba(125,211,252,0.3)';
  wCtx.shadowBlur = 10;

  for (let i = 0; i < waveformData.length; i++) {
    const x = i * step;
    const y = mid + waveformData[i] * amp;
    if (i === 0) wCtx.moveTo(x, y);
    else wCtx.lineTo(x, y);
  }
  wCtx.stroke();

  wCtx.shadowBlur = 0;
  wCtx.strokeStyle = 'rgba(125,211,252,0.15)';
  wCtx.lineWidth = 6;
  wCtx.stroke();
}
drawWaveform();

// ---- Core state ----
const S = {
  currentState: 'idle',
  speaking: false,
  talkPulse: 0,
  phonemeIntensity: 0,
  isTtsSpeaking: false,
  ttsMuted: false,
  lipSyncActive: false,
  currentWeights: {},
  lastActivityAt: performance.now(),
  streamDriven: false,
  speakStartedAt: null,
  isLongResponse: false,
  listening: false,
  phonemeQueue: [],
  phonemeQueueIndex: 0,
  phonemeQueueTimer: 0,
  bass: 0, mid: 0, treble: 0,
  beat: false, beatPulse: 0,
  feeling: 'neutral',
  feelingBlend: 0,
  feelingBlendTarget: 0,
  feelingExpire: 0,
  feelingEndless: false,
  visemePreview: null,
};
window._AIFaceState = S;

// ---- Event bus: lets embedders/plugins react to internal changes ----
// Public events: 'stateChange', 'providerSwitch', 'sessionUpdate', 'speakingChange'
const _bus = createEventBus((event, e) => dbg('AIFace event handler for "' + event + '" threw: ' + e.message, 'warn'));
const onAIFaceEvent = _bus.on;
const offAIFaceEvent = _bus.off;
const emitAIFaceEvent = _bus.emit;

// ---- Idle life: autonomous blink + breathing (cheap procedural animation) ----
const IDLE_BLINK_LEFT = 'eyeBlinkLeft';
const IDLE_BLINK_RIGHT = 'eyeBlinkRight';
const IDLE_LIFE_STATES = ['idle', 'thinking'];
let idleTime = 0;
let idleBlinkAt = 2.5 + Math.random() * 3;
let idleBlinkT = -1;                // -1 = eyes open; else 0..1 blink progress
let idleBreathPhase = Math.random() * 6.28;
let headBaseY = null;
let blinkLKey = null;
let blinkRKey = null;

function resolveSlotKey(slot, avoid) {
  const nrm = (k) => k.toLowerCase().replace(/[^a-z0-9]/g, '');
  if (dict) {
    if (dict[slot] !== undefined) return slot;
    const mapped = morphMap && morphMap[slot];
    if (mapped && dict[mapped] !== undefined) return mapped;
    const exact = Object.keys(dict).find((k) => nrm(k) === nrm(slot));
    if (exact && exact !== avoid) return exact;
    if (nrm(slot).includes('blink')) {
      const wantRight = /Right$/.test(slot);
      for (const k of Object.keys(dict)) {
        if (k === avoid) continue;
        if (!nrm(k).includes('blink')) continue;
        const dir = wantRight
          ? /(right|[^a-z0-9]r|^r|r$)/i.test(k)
          : /(left|[^a-z0-9]l|^l|l$)/i.test(k);
        if (dir) return k;
      }
    }
    const meta = MORPH_SLOTS.find((m) => m.slot === slot);
    if (meta) {
      for (const hint of meta.hint) {
        const h = nrm(hint);
        if (h.length >= 4) {
          const found = Object.keys(dict).find((k) => nrm(k).includes(h) && k !== avoid);
          if (found) return found;
        }
      }
    }
  }
  return null;
}

function refreshBlinkKeys() {
  blinkLKey = resolveSlotKey('eyeBlinkLeft');
  blinkRKey = resolveSlotKey('eyeBlinkRight', blinkLKey);
}

function resetIdleLife() {
  idleTime = 0;
  idleBlinkT = -1;
  idleBlinkAt = 2 + Math.random() * 3;
  headBaseY = null;
}

const STATE_TARGETS = {
  idle: { eyeBlinkLeft: 0.12, eyeBlinkRight: 0.12, mouthClose: 0.15, jawOpen: 0.0 },
  listening: { eyeWideLeft: 0.55, eyeWideRight: 0.55, browInnerUp: 0.45, browOuterUpLeft: 0.3, browOuterUpRight: 0.3, jawOpen: 0.12, mouthPucker: 0.1 },
  thinking: { eyeLookUpLeft: 0.6, eyeLookUpRight: 0.6, browDownLeft: 0.4, browDownRight: 0.35, mouthPucker: 0.35, mouthRollLower: 0.2, jawOpen: 0.05 },
  responding: { mouthSmileLeft: 0.55, mouthSmileRight: 0.55, jawOpen: 0.35, cheekPuff: 0.15, eyeSquintLeft: 0.2, eyeSquintRight: 0.2, browInnerUp: 0.25 },
  alert: { eyeWideLeft: 0.95, eyeWideRight: 0.95, browDownLeft: 0.75, browDownRight: 0.75, mouthFrownLeft: 0.55, mouthFrownRight: 0.55, jawOpen: 0.25, noseSneerLeft: 0.3, noseSneerRight: 0.3 },
};

// Expression overlay used in "emotion detect" mirror mode (layer on top of STATE_TARGETS).
const EMOTION_TARGETS = {
  happy: { mouthSmileLeft: 0.5, mouthSmileRight: 0.5, cheekPuff: 0.12, eyeSquintLeft: 0.25, eyeSquintRight: 0.25 },
  upset: { browDownLeft: 0.5, browDownRight: 0.45, mouthFrownLeft: 0.45, mouthFrownRight: 0.45, jawOpen: 0.1 },
  surprised: { eyeWideLeft: 0.6, eyeWideRight: 0.6, browInnerUp: 0.5, browOuterUpLeft: 0.4, browOuterUpRight: 0.4, jawOpen: 0.25 },
  curious: { browInnerUp: 0.45, eyeLookUpLeft: 0.35, eyeLookUpRight: 0.35, mouthPucker: 0.2 },
  neutral: {},
};

// Expression overlay used by sentiment-driven feeling colors (layered last,
// so it wins over state/mirror defaults while the sentiment is active).
const FEELING_TARGETS = {
  happy: { mouthSmileLeft: 0.5, mouthSmileRight: 0.5, cheekPuff: 0.12, eyeSquintLeft: 0.25, eyeSquintRight: 0.25 },
  sad: { browInnerUp: 0.35, browDownLeft: 0.3, browDownRight: 0.3, mouthFrownLeft: 0.35, mouthFrownRight: 0.35, eyeSquintLeft: 0.2, eyeSquintRight: 0.2 },
  angry: { browDownLeft: 0.65, browDownRight: 0.6, mouthFrownLeft: 0.45, mouthFrownRight: 0.45, eyeSquintLeft: 0.3, eyeSquintRight: 0.3, noseSneerLeft: 0.25, noseSneerRight: 0.25 },
  surprised: EMOTION_TARGETS.surprised,
  scared: { eyeWideLeft: 0.7, eyeWideRight: 0.7, browInnerUp: 0.5, browOuterUpLeft: 0.4, browOuterUpRight: 0.4, jawOpen: 0.2, mouthStretchLeft: 0.15, mouthStretchRight: 0.15 },
  love: { mouthSmileLeft: 0.45, mouthSmileRight: 0.45, cheekPuff: 0.1, browInnerUp: 0.2, eyeSquintLeft: 0.2, eyeSquintRight: 0.2 },
  neutral: {},
};

// Expression overlay applied when AI Vision detects a person while the avatar
// is idle. Subtle "engaged" look layered like emotion/feeling targets.
const VISION_TARGETS = {
  active: { eyeLookUpLeft: 0.3, eyeLookUpRight: 0.3, browInnerUp: 0.25, mouthSmileLeft: 0.2, mouthSmileRight: 0.2 },
};

// Optional manual morph-name remap (set by the morph mapping screen).
// Maps a canonical slot name (e.g. 'jawOpen') to a different morph name on the model.
let morphMap = null;
let defaultVrmMap = null;
let _mirrorPillTick = -1;
function morphIndex(k) {
  if (dict) {
    const mapped = morphMap && morphMap[k];
    if (mapped && dict[mapped] !== undefined) return dict[mapped];
    if (dict[k] !== undefined) return dict[k];
  }
  if (currentVRM && ALL_SLOT_NAMES.indexOf(k) !== -1) return k; // truthy marker: handled via VRM expression manager
  return undefined;
}

// Canonical avatar slots, used for the morph-mapping screen and VRM expression driving.
const MORPH_SLOTS = [
  { slot: 'jawOpen', label: 'Mouth open', hint: ['jaw', 'open', 'mouth_o', 'ah', 'aa'] },
  { slot: 'mouthSmileLeft', label: 'Smile left', hint: ['smile', 'left', 'ee', 'ih'] },
  { slot: 'mouthSmileRight', label: 'Smile right', hint: ['smile', 'right', 'ee', 'ih'] },
  { slot: 'mouthFrownLeft', label: 'Frown left', hint: ['frown', 'left', 'sad'] },
  { slot: 'mouthFrownRight', label: 'Frown right', hint: ['frown', 'right', 'sad'] },
  { slot: 'mouthPucker', label: 'Pucker', hint: ['puck', 'ooh', 'kiss', 'ou', 'u'] },
  { slot: 'mouthClose', label: 'Mouth close', hint: ['close', 'shut', 'm'] },
  { slot: 'mouthStretchLeft', label: 'Stretch left', hint: ['stretch', 'left', 'oh', 'ee'] },
  { slot: 'mouthStretchRight', label: 'Stretch right', hint: ['stretch', 'right', 'oh', 'ee'] },
  { slot: 'mouthRollLower', label: 'Roll lower lip', hint: ['roll', 'lower', 'oh'] },
  { slot: 'eyeBlinkLeft', label: 'Blink left', hint: ['blink', 'left', 'eye_l', 'blinkLeft'] },
  { slot: 'eyeBlinkRight', label: 'Blink right', hint: ['blink', 'right', 'eye_r', 'blinkRight'] },
  { slot: 'eyeWideLeft', label: 'Eyes wide left', hint: ['wide', 'left', 'surprised'] },
  { slot: 'eyeWideRight', label: 'Eyes wide right', hint: ['wide', 'right', 'surprised'] },
  { slot: 'eyeSquintLeft', label: 'Squint left', hint: ['squint', 'left', 'relaxed'] },
  { slot: 'eyeSquintRight', label: 'Squint right', hint: ['squint', 'right', 'relaxed'] },
  { slot: 'browInnerUp', label: 'Brow inner up', hint: ['brow', 'inner', 'sad', 'sorrow'] },
  { slot: 'browOuterUpLeft', label: 'Brow outer up left', hint: ['brow', 'outer', 'up', 'left', 'surprised'] },
  { slot: 'browOuterUpRight', label: 'Brow outer up right', hint: ['brow', 'outer', 'up', 'right', 'surprised'] },
  { slot: 'browDownLeft', label: 'Brow down left', hint: ['brow', 'down', 'left', 'angry'] },
  { slot: 'browDownRight', label: 'Brow down right', hint: ['brow', 'down', 'right', 'angry'] },
  { slot: 'cheekPuff', label: 'Cheek puff', hint: ['cheek', 'puff'] },
  { slot: 'noseSneerLeft', label: 'Sneer left', hint: ['sneer', 'left', 'angry'] },
  { slot: 'noseSneerRight', label: 'Sneer right', hint: ['sneer', 'right', 'angry'] },
];
const ALL_SLOT_NAMES = MORPH_SLOTS.map((s) => s.slot);

function detectMorphNames() {
  const names = new Set();
  if (dict) for (const k of Object.keys(dict)) names.add(k);
  if (currentVRM && currentVRM.expressionManager) {
    try {
      for (const e of currentVRM.expressionManager.expressions) names.add(e.name);
    } catch (err) {}
  }
  return Array.from(names);
}

function morphMapStorageKey() {
  const u = localStorage.getItem('aiface_lastModelUrl') || '';
  return 'aiface_morphmap_' + u.replace(/^.*[\\/]/, '').replace(/[^a-zA-Z0-9_.-]/g, '_');
}
function saveMorphMap() {
  try { localStorage.setItem(morphMapStorageKey(), JSON.stringify(morphMap || {})); } catch (e) {}
}
function loadMorphMap(url) {
  try {
    if (url) localStorage.setItem('aiface_lastModelUrl', url);
    const raw = localStorage.getItem(morphMapStorageKey());
    morphMap = raw ? JSON.parse(raw) : {};
  } catch (e) { morphMap = {}; }
}
function autoSuggestMorphs() {
  const names = detectMorphNames();
  const lower = names.map((n) => n.toLowerCase());
  for (const { slot, hint } of MORPH_SLOTS) {
    let best = '';
    let bestScore = 0;
    for (let i = 0; i < names.length; i++) {
      let score = 0;
      for (const h of hint) if (lower[i].includes(h)) score += 1;
      if (score > bestScore) { bestScore = score; best = names[i]; }
    }
    if (bestScore > 0) morphMap[slot] = best;
  }
  saveMorphMap();
}

// Built-in fallback map for VRM avatars: maps our canonical slots to the
// avatar's real expression names so lips/blinks work without manual mapping.
// User-mapped slots (morphMap) always override this fallback.
function buildDefaultVrmMap(vrm) {
  defaultVrmMap = null;
  try {
    if (!vrm || !vrm.expressionManager || !vrm.expressionManager.expressions) return;
    const names = vrm.expressionManager.expressions.map((e) => e.name);
    const has = (n) => names.indexOf(n) !== -1;
    const RULES = [
      ['jawOpen',       ['aa', 'ah', 'oh', 'ou']],
      ['mouthPucker',   ['ou', 'ooh', 'u']],
      ['mouthClose',    ['m', 'close', 'shut']],
      ['mouthSmileLeft',  ['ee', 'ih', 'happy', 'smile']],
      ['mouthSmileRight', ['ee', 'ih', 'happy', 'smile']],
      ['mouthStretchLeft',  ['oh', 'ee', 'o']],
      ['mouthStretchRight', ['oh', 'ee', 'o']],
      ['mouthRollLower', ['oh', 'roll']],
      ['mouthFrownLeft',  ['sad', 'frown']],
      ['mouthFrownRight', ['sad', 'frown']],
      ['eyeBlinkLeft',  ['blinkLeft', 'blink', 'eyeBlinkLeft']],
      ['eyeBlinkRight', ['blinkRight', 'blink', 'eyeBlinkRight']],
      ['eyeWideLeft',  ['surprised', 'wideLeft', 'EyeBlinkLeft']],
      ['eyeWideRight', ['surprised', 'wideRight', 'EyeBlinkRight']],
      ['eyeSquintLeft',  ['relaxed', 'squint']],
      ['eyeSquintRight', ['relaxed', 'squint']],
      ['browDownLeft',  ['angry', 'browDownLeft']],
      ['browDownRight', ['angry', 'browDownRight']],
      ['browInnerUp',   ['sad', 'sorrow', 'browInnerUp']],
      ['browOuterUpLeft',  ['surprised', 'browOuterUpLeft']],
      ['browOuterUpRight', ['surprised', 'browOuterUpRight']],
      ['cheekPuff',     ['cheekPuff', 'puff']],
      ['noseSneerLeft',  ['angry', 'sneer']],
      ['noseSneerRight', ['angry', 'sneer']],
    ];
    defaultVrmMap = {};
    for (const [slot, opts] of RULES) {
      for (const o of opts) {
        if (has(o)) { if (!defaultVrmMap[slot]) defaultVrmMap[slot] = o; break; }
      }
    }
    if (Object.keys(defaultVrmMap).length) {
      dbg('VRM expression fallback applied (' + Object.keys(defaultVrmMap).length + ' slots)', 'ok');
    }
  } catch (e) { defaultVrmMap = null; }
}

const STATE_COLORS = {
  idle:     { emissive: 0x2f81f7, wire: 0x7dd3fc, ambient: 0x020810, pulse: '#070c16', pulseOp: 0.0 },
  listening:{ emissive: 0x00d4aa, wire: 0x5fffd4, ambient: 0x001a12, pulse: '#003d2e', pulseOp: 0.15 },
  thinking: { emissive: 0xffaa00, wire: 0xffd700, ambient: 0x1a1200, pulse: '#3d2e00', pulseOp: 0.2 },
  responding:{ emissive: 0x00aaff, wire: 0x88ddff, ambient: 0x001a33, pulse: '#003d66', pulseOp: 0.25 },
  alert:    { emissive: 0xff2244, wire: 0xff6b7a, ambient: 0x1a0005, pulse: '#3d000a', pulseOp: 0.4 },
};

// Sentiment-driven feeling colors (blended over STATE_COLORS while active).
const FEELING_COLORS = {
  happy:     { emissive: 0xffb84d, wire: 0xffe0a3, ambient: 0x3a2a08, pulse: '#ffb84d', pulseOp: 0.30 },
  sad:       { emissive: 0x5b8bf7, wire: 0xa9c4ff, ambient: 0x08123a, pulse: '#3f5bd9', pulseOp: 0.25 },
  angry:     { emissive: 0xff4747, wire: 0xff9a8a, ambient: 0x3a0808, pulse: '#ff4747', pulseOp: 0.35 },
  surprised: { emissive: 0xc46bff, wire: 0xe6c2ff, ambient: 0x26083a, pulse: '#c46bff', pulseOp: 0.30 },
  scared:    { emissive: 0x7fd4ff, wire: 0xc2ecff, ambient: 0x08263a, pulse: '#7fd4ff', pulseOp: 0.25 },
  love:      { emissive: 0xff7ab5, wire: 0xffc2dc, ambient: 0x3a0820, pulse: '#ff7ab5', pulseOp: 0.30 },
  neutral:   { emissive: 0x2f81f7, wire: 0x7dd3fc, ambient: 0x020810, pulse: '#070c16', pulseOp: 0.0 },
};

// Global lighting presets (configurable). The ambient light lerps toward these values.
const LIGHT_PRESETS = {
  blueprint: { color: 0x2f81f7, intensity: 0.4 },
  matrix:    { color: 0x00ff7f, intensity: 0.5 },
  warm:      { color: 0xffb56b, intensity: 0.62 },
  soft:      { color: 0xf4f7ff, intensity: 0.5 },
  noir:      { color: 0xffffff, intensity: 0.22 },
};
let lightPresetName = 'blueprint';
const lightTargetColor = new THREE.Color(LIGHT_PRESETS.blueprint.color);
let lightTargetIntensity = LIGHT_PRESETS.blueprint.intensity;

function applyLightPreset(name) {
  const preset = LIGHT_PRESETS[name] || LIGHT_PRESETS.blueprint;
  lightPresetName = name;
  lightTargetColor.setHex(preset.color);
  lightTargetIntensity = preset.intensity;
}

let _feelingPillEl = null;
function getFeelingEl() {
  if (!_feelingPillEl) _feelingPillEl = document.getElementById('feelingPill');
  return _feelingPillEl;
}
function setFeeling(label, lingerMs) {
  const was = S.feeling;
  S.feeling = label || 'neutral';
  if (S.feeling === 'neutral') {
    S.feelingBlendTarget = 0;
    S.feelingEndless = false;
  } else {
    S.feelingBlendTarget = 1;
    if (lingerMs) {
      S.feelingEndless = false;
      S.feelingExpire = performance.now() + lingerMs;
    } else if (!S.feelingEndless) {
      S.feelingEndless = true;
      S.feelingExpire = 0;
    }
  }
  if (S.feeling !== was) {
    const pill = getFeelingEl();
    if (pill) pill.textContent = t('feeling.prefix') + ': ' + S.feeling.toUpperCase();
    revealFeelingPill();
  }
}
function scheduleFeelingDecay() {
  if (S.feeling === 'neutral') return;
  S.feelingEndless = false;
  S.feelingExpire = performance.now() + 2000;
}
function applyFeelingPulse() {
  const sc = STATE_COLORS[S.currentState] || STATE_COLORS.idle;
  const fc = FEELING_COLORS[S.feeling] || FEELING_COLORS.neutral;
  const b = S.feelingBlend;
  const pulseEl = document.getElementById('bgPulse');
  if (!pulseEl) return;
  const mix = (hexA, hexB) => {
    const a = new THREE.Color(hexA), c = a.clone().lerp(new THREE.Color(hexB), b);
    return '#' + c.getHexString();
  };
  pulseEl.style.background = 'radial-gradient(circle at 50% 42%, ' + mix(sc.pulse, fc.pulse) + ' 0%, transparent 70%)';
  pulseEl.style.opacity = (sc.pulseOp + (fc.pulseOp - sc.pulseOp) * b).toFixed(2);
}

const _feelingColorCache = {};
function _hexC(h) { return _feelingColorCache[h] || (_feelingColorCache[h] = new THREE.Color(h)); }
let _lastPulseBlend = -1;
const _feelingDotTmp = new THREE.Color();

let userInteracted = false;
let audioPrimed = false;
function primeAudio() {
  if (audioPrimed || !window.speechSynthesis) return;
  audioPrimed = true;
  try {
    const primer = new SpeechSynthesisUtterance(' ');
    primer.volume = 0;
    window.speechSynthesis.speak(primer);
    dbg('Audio engine primed on user gesture', 'ok');
  } catch (e) { dbg('Audio priming failed: ' + e.message, 'warn'); }
}
document.addEventListener('click', () => { userInteracted = true; primeAudio(); }, { once: true });
document.addEventListener('keydown', () => { userInteracted = true; primeAudio(); }, { once: true });
document.addEventListener('touchstart', () => { userInteracted = true; primeAudio(); }, { once: true });

// ---- Viseme mapping ----
function driveWordPhonemes(fullText, charIndex) {
  const rest = fullText.slice(charIndex);
  const match = rest.match(/[\p{L}]+/u);
  if (!match) { S.phonemeQueue = []; return; }
  S.phonemeQueue = match[0].split('').map(visemeFor);
  S.phonemeQueueIndex = 0;
  S.phonemeQueueTimer = 0;
  const fm = cfgFeelingMode ? cfgFeelingMode.value : 'auto';
  if (fm === 'auto' && match[0]) {
    const wf = detectFeeling(match[0]);
    if (wf !== 'neutral') setFeeling(wf, 2600);
  }
}

function applyPhonemeShape(t) {
  if (S.phonemeQueue && S.phonemeQueue.length) {
    S.phonemeQueueTimer -= 1 / 60;
    if (S.phonemeQueueTimer <= 0) {
      S.phonemeQueueIndex = Math.min(S.phonemeQueueIndex + 1, S.phonemeQueue.length - 1);
      S.phonemeQueueTimer = 0.085 + Math.random() * 0.03;
    }
  }
  const shape = (S.phonemeQueue && S.phonemeQueue.length) ? S.phonemeQueue[S.phonemeQueueIndex] : DEFAULT_VISEME;
  const intensity = S.phonemeIntensity;
  const noise = (Math.random() * 0.02 - 0.01) * intensity;
  const flap = 0.4 + 0.6 * Math.abs(Math.sin(t * 9.5));

  if (morphIndex('jawOpen') !== undefined) {
    const cur = S.currentWeights['jawOpen'] || 0;
    const baseGoal = shape.jawOpen !== undefined ? shape.jawOpen : DEFAULT_VISEME.jawOpen;
    const goal = Math.min(0.42, baseGoal * intensity * flap * 2.8 + noise);
    S.currentWeights['jawOpen'] = cur + (Math.max(0, goal) - cur) * 0.45;
  }
  for (const key of VISEME_KEYS) {
    if (key === 'jawOpen' || key === 'mouthClose' || morphIndex(key) === undefined) continue;
    const cur = S.currentWeights[key] || 0;
    const base = shape[key] !== undefined ? shape[key] : 0;
    const goal = base * intensity * 1.6;
    S.currentWeights[key] = cur + (goal - cur) * 0.3;
  }
  if (morphIndex('mouthClose') !== undefined) {
    const cur = S.currentWeights['mouthClose'] || 0;
    const base = shape.mouthClose !== undefined ? shape.mouthClose : DEFAULT_VISEME.mouthClose;
    const goal = base * intensity + (1 - flap) * intensity * 0.35;
    S.currentWeights['mouthClose'] = cur + (goal - cur) * 0.35;
  }
  if (morphIndex('cheekPuff') !== undefined) S.currentWeights['cheekPuff'] = intensity * 0.08;
}

let _lastTtsFailureAt = 0;
function notifyTtsFailure(msg) {
  const now = performance.now();
  if (now - _lastTtsFailureAt < 4000) return;
  _lastTtsFailureAt = now;
  addMessage('error', msg, true);
}

// ---- Gemini TTS ----
let activeGeminiAudioEl = null;
let activeGeminiPhonemeTimer = null;

function base64ToUint8(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function pcm16ToWavBlob(pcmBytes, sampleRate, channels) {
  const dataLen = pcmBytes.length;
  const buffer = new ArrayBuffer(44 + dataLen);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
  writeStr(0, 'RIFF'); view.setUint32(4, 36 + dataLen, true); writeStr(8, 'WAVE');
  writeStr(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, channels, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true); view.setUint16(32, channels * 2, true); view.setUint16(34, 16, true);
  writeStr(36, 'data'); view.setUint32(40, dataLen, true);
  new Uint8Array(buffer, 44).set(pcmBytes);
  return new Blob([buffer], { type: 'audio/wav' });
}

function stopGeminiAudio() {
  if (activeGeminiPhonemeTimer) { clearInterval(activeGeminiPhonemeTimer); activeGeminiPhonemeTimer = null; }
  if (activeGeminiAudioEl) {
    try { activeGeminiAudioEl.pause(); } catch (e) {}
    activeGeminiAudioEl = null;
  }
}

async function speakGemini(text, lang, apiKey, voiceName, model) {
  stopGeminiAudio();
  if (S.ttsMuted) { dbg('Gemini TTS muted', 'warn'); setState('idle'); return; }
  if (!text || !text.trim()) { dbg('Gemini TTS empty', 'warn'); setState('idle'); return; }
  if (!apiKey) {
    notifyTtsFailure('Gemini voice enabled but no API key provided. Add one in Voice Settings.');
    setState('idle');
    return;
  }
  dbg('speakGemini() called: ' + text.substring(0, 30), 'info');
  const url = 'https://generativelanguage.googleapis.com/v1beta/models/' + (model || 'gemini-2.5-flash-preview-tts') + ':generateContent?key=' + encodeURIComponent(apiKey);
  const body = {
    contents: [{ parts: [{ text: text.trim() }] }],
    generationConfig: {
      responseModalities: ['AUDIO'],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: voiceName || 'Kore' } },
        languageCode: lang || undefined,
      }
    }
  };
  try {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res.ok) {
      let errMsg = 'Gemini TTS HTTP ' + res.status;
      try { const errBody = await res.json(); errMsg = errBody.error?.message || errBody.error?.status || errMsg; } catch (e) {}
      throw new Error(errMsg);
    }
    const json = await res.json();
    const part = json.candidates && json.candidates[0] && json.candidates[0].content && json.candidates[0].content.parts && json.candidates[0].content.parts[0] ? json.candidates[0].content.parts[0].inlineData : null;
    if (!part || !part.data) throw new Error('Gemini returned no audio data');

    const pcmBytes = base64ToUint8(part.data);
    const rateMatch = (part.mimeType || '').match(/rate=(\d+)/);
    const sampleRate = rateMatch ? parseInt(rateMatch[1], 10) : 24000;
    const wavBlob = pcm16ToWavBlob(pcmBytes, sampleRate, 1);
    const objUrl = URL.createObjectURL(wavBlob);
    const audioEl = new Audio(objUrl);
    audioEl.volume = getMasterVolume();
    routeOutput(audioEl);
    activeGeminiAudioEl = audioEl;

    audioEl.onplay = () => {
      S.isTtsSpeaking = true;
      setState('responding');
      setSpeaking(true);
      dbg('Gemini TTS STARTED', 'ok');
      updateVoiceUI();
      activeGeminiPhonemeTimer = setInterval(() => {
        if (!audioEl.duration || isNaN(audioEl.duration)) return;
        const frac = Math.min(1, audioEl.currentTime / audioEl.duration);
        driveWordPhonemes(text, Math.floor(frac * text.length));
        S.talkPulse = 0.42;
        S.lastActivityAt = performance.now();
      }, 90);
    };
    audioEl.onended = () => {
      stopGeminiAudio();
      S.isTtsSpeaking = false;
      setSpeaking(false);
      setState('idle');
      dbg('Gemini TTS ENDED', 'ok');
      updateVoiceUI();
      URL.revokeObjectURL(objUrl);
    };
    audioEl.onerror = () => {
      stopGeminiAudio();
      S.isTtsSpeaking = false;
      setSpeaking(false);
      setState('idle');
      dbg('Gemini TTS playback ERROR', 'err');
      updateVoiceUI();
      notifyTtsFailure('Failed to play Gemini-generated audio.');
      URL.revokeObjectURL(objUrl);
    };
    await audioEl.play();
  } catch (err) {
    dbg('Gemini TTS error: ' + err.message, 'err');
    setState('idle');
    notifyTtsFailure('Gemini TTS failed: ' + err.message);
  }
}

// ---- Browser TTS ----
function speak(text, lang, _retried) {
  if (!lang) lang = 'en-US';
  dbg('speak() called: ' + text.substring(0, 30), 'info');
  if (S.ttsMuted) { dbg('TTS muted', 'warn'); setState('idle'); return; }
  if (!text || !text.trim()) { dbg('TTS empty', 'warn'); setState('idle'); return; }
  if (!window.speechSynthesis) { dbg('TTS NOT AVAILABLE', 'err'); setState('idle'); return; }

  const availableVoices = window.speechSynthesis.getVoices() || [];
  if (availableVoices.length === 0 && !_retried) {
    dbg('No voices yet -- retrying speak() shortly', 'warn');
    setTimeout(() => speak(text, lang, true), 350);
    return;
  }

  try { window.speechSynthesis.cancel(); } catch (e) {}

  const utterance = new SpeechSynthesisUtterance(text.trim());
  const voices = window.speechSynthesis.getVoices() || [];
  let voice = voices.find(v => v.lang === lang);
  if (!voice) voice = voices.find(v => v.lang && v.lang.startsWith(lang.split('-')[0]));
  if (!voice && voices.length) voice = voices[0];
  if (voice) { utterance.voice = voice; dbg('TTS voice: ' + voice.name, 'ok'); }
  else { dbg('TTS no voice, using default', 'warn'); }
  utterance.lang = lang;
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.volume = getMasterVolume();

  utterance.onstart = () => {
    S.isTtsSpeaking = true;
    setState('responding');
    setSpeaking(true);
    dbg('TTS STARTED', 'ok');
    updateVoiceUI();
  };
  utterance.onend = () => {
    S.isTtsSpeaking = false;
    setSpeaking(false);
    setState('idle');
    dbg('TTS ENDED', 'ok');
    updateVoiceUI();
  };
  utterance.onerror = (e) => {
    S.isTtsSpeaking = false;
    setSpeaking(false);
    setState('idle');
    dbg('TTS ERROR: ' + e.error, 'err');
    updateVoiceUI();
    notifyTtsFailure('Speech output failed (' + e.error + '). Try the speaker button next to the message, or open the file in a standalone browser tab.');
  };
  utterance.onboundary = (e) => {
    if (e.name === 'word' || e.name === 'sentence') {
      S.talkPulse = 0.42;
      S.lastActivityAt = performance.now();
      driveWordPhonemes(text, e.charIndex);
    }
  };

  try { window.speechSynthesis.resume(); } catch (e) {}
  window.speechSynthesis.speak(utterance);
  dbg('speechSynthesis.speak() done', 'ok');

  setTimeout(() => {
    if (!S.isTtsSpeaking && window.speechSynthesis.speaking === false) {
      dbg('TTS did not start -- likely blocked by browser/preview', 'err');
      notifyTtsFailure('Speech did not start within 2 seconds -- the browser or preview window may be blocking auto-play. Click the speaker button next to the AI response, or open the file in a standalone browser tab.');
    }
  }, 1200);
}

function stopTTS() {
  if (window.speechSynthesis) { try { window.speechSynthesis.cancel(); } catch (e) {} S.isTtsSpeaking = false; updateVoiceUI(); }
  stopGeminiAudio();
  stopLocalAudio();
}

function toggleMute() {
  S.ttsMuted = !S.ttsMuted;
  if (S.ttsMuted) stopTTS();
  updateVoiceUI();
}

function speakUnified(text, lang) {
  const fm = cfgFeelingMode ? cfgFeelingMode.value : 'auto';
  if ((fm === 'auto' || fm === 'response') && text) {
    const f = detectFeeling(text);
    if (f !== 'neutral') setFeeling(f, 8000);
  }
  if (cfgUseGeminiTts.checked) {
    const key = cfgGeminiTtsKey.value.trim() || (cfgProvider.value === 'gemini' ? cfgKey.value.trim() : '');
    speakGemini(text, lang, key, cfgGeminiVoice.value, cfgGeminiTtsModel.value.trim() || 'gemini-2.5-flash-preview-tts');
  } else if (cfgTtsEngine && cfgTtsEngine.value === 'local') {
    speakLocal(text, lang);
  } else {
    speak(text, lang);
  }
}

function finishLocalSpeak() {
  S.isTtsSpeaking = false;
  setSpeaking(false);
  setState('idle');
  updateVoiceUI();
}

async function speakLocal(text, lang) {
  if (S.ttsMuted) { dbg('TTS muted', 'warn'); setState('idle'); return; }
  if (!text || !text.trim()) { setState('idle'); return; }
  setState('responding');
  setSpeaking(true);
  S.isTtsSpeaking = true;
  S.speakStartedAt = performance.now();
  updateVoiceUI();
  dbg('Local TTS generating...', 'info');
  try {
    const audio = await generateLocalAudio(text.trim(), lang);
    setLocalCallbacks({
      onWord: (charIndex) => {
        S.talkPulse = 0.42;
        S.lastActivityAt = performance.now();
        driveWordPhonemes(text, charIndex);
      },
      onEnd: () => finishLocalSpeak(),
    });
    playLocalAudio(text.trim(), audio);
    dbg('Local TTS playing', 'ok');
  } catch (e) {
    dbg('Local TTS failed: ' + e.message, 'err');
    finishLocalSpeak();
    notifyTtsFailure('On-device voice failed (' + e.message + '). Falling back to the browser voice.');
    speak(text, lang);
  }
}

function stateLabelText(s) {
  const st = t('state.' + s, s.toUpperCase());
  return st + (S.speaking ? ' + ' + t('state.talk') : '');
}

function updateVoiceUI() {
  const el = document.getElementById('voiceIndicator');
  const label = document.getElementById('voiceLabel');
  if (S.ttsMuted) { el.className = 'voice-indicator muted'; label.textContent = t('voice.muted', 'MUTED'); }
  else if (S.isTtsSpeaking) { el.className = 'voice-indicator speaking'; label.textContent = t('voice.speaking', 'SPEAKING...'); }
  else { el.className = 'voice-indicator'; label.textContent = t('voice.on', 'Voice ON'); }
}

function updateStateBody() {
  const cls = S.currentState === 'responding' ? 'state-responding'
    : (S.currentState === 'listening' ? 'state-listening' : '');
  if (document.body.className !== cls) document.body.className = cls;
}

function setState(s) {
  if (!STATE_TARGETS[s]) return;
  const previous = S.currentState;
  S.currentState = s;
  if (previous !== s) emitAIFaceEvent('stateChange', { state: s, previous });
  updateStateBody();
  document.getElementById('stateVal').textContent = stateLabelText(s);
  const label = document.getElementById('stateLabel');
  label.className = 'state-label' + (s === 'alert' ? ' alert-mode' : '');
  const colors = STATE_COLORS[s];
  if (colors) {
    document.getElementById('bgPulse').style.background = 'radial-gradient(circle at 50% 42%, ' + colors.pulse + ' 0%, transparent 70%)';
    document.getElementById('bgPulse').style.opacity = colors.pulseOp;
  }
  document.getElementById('state-metric').textContent = t('state.' + s, s.toUpperCase());
  const orb = document.getElementById('statusOrb');
  if (orb) { orb.className = 'status-orb state-' + s; }
}

function setSpeaking(v) {
  const was = S.speaking;
  S.speaking = v;
  S.lipSyncActive = v;
  if (v && !was) { S.speakStartedAt = performance.now(); S.isLongResponse = false; }
  if (!v) { S.speakStartedAt = null; S.isLongResponse = false; document.getElementById('bgPulse').classList.remove('long-mode'); S.talkPulse = 0; scheduleFeelingDecay(); }
  S.lastActivityAt = performance.now();
  document.getElementById('stateVal').textContent = stateLabelText(S.currentState);
  if (v !== was) emitAIFaceEvent('speakingChange', { speaking: v });
}

function onToken() {
  S.streamDriven = true;
  S.lastActivityAt = performance.now();
}

function reset() {
  S.streamDriven = false;
  setState('idle');
  setSpeaking(false);
  stopTTS();
}

window.AIFace = {
  setState, setSpeaking, onToken, reset,
  speak, stopTTS, toggleMute,
  modelProgress,
  getStatus: () => S,
  // Event bus: on('stateChange' | 'speakingChange' | 'providerSwitch' | 'sessionUpdate', handler)
  // Returns an unsubscribe function. handler receives an event-specific detail object.
  on: onAIFaceEvent,
  off: offAIFaceEvent
};
Object.assign(window.AIFace, {
  newChat,
  saveCurrentSession,
  restoreSession,
  openSessions: openSessionsModal,
  getMessages: () => sanitizeMessages(messages),
  sessions: { list: listSessions, load: loadSession, remove: deleteSession },
  registerSessionMessage: (role, content) => {
    if (!VALID_ROLES_UI[role] || typeof content !== 'string' || !content.trim()) return false;
    messages.push({ role, content });
    addMessage(role, content, role === 'error');
    saveCurrentSession();
    return true;
  },
});

window.addEventListener('message', (event) => {
  const d = event.data;
  if (!d || d.source !== 'ai-face-control') return;
  if (event.origin && event.origin !== location.origin) return;
  if (d.type === 'state' && STATE_TARGETS[d.value]) setState(d.value);
  else if (d.type === 'speaking') setSpeaking(!!d.value);
  else if (d.type === 'token') onToken();
  else if (d.type === 'speakText') speak(d.text, d.lang || 'en-US');
});

// ---- Three.js scene ----
let head = null;
let influences = null;
let dict = null;
let mixer = null;
let glowMat = null;
let wireMat = null;
function makeMatPair() {
  glowMat = new THREE.MeshStandardMaterial({ color: 0x020810, emissive: 0x2f81f7, emissiveIntensity: 0.8, metalness: 0.6, roughness: 0.3, transparent: true, opacity: 0.95 });
  wireMat = new THREE.MeshBasicMaterial({ color: 0x7dd3fc, wireframe: true, transparent: true, opacity: 0.9 });
  return { glow: glowMat, wire: wireMat };
}
let ambientLight = null;
let eyeDotMats = [];
let faceCanvas = null;
let _scene = null;
let _renderer = null;
let _gltfLoader = null;
let _ktx2Loader = null;
let _camera = null;
let _currentModelGroup = null;
let currentVRM = null;

// ---- Reusable model loader ----
function loadModel(url, fitFn, buildMatsFn) {
  return new Promise((resolve, reject) => {
    if (!_gltfLoader || !_scene) { reject(new Error('Scene not ready')); return; }

    const loadMsg = document.getElementById('loadMsg');
    if (loadMsg) loadMsg.textContent = 'Loading model...';

    const loadTimeout = setTimeout(() => {
      if (loadMsg && loadMsg.style.display !== 'none') loadMsg.textContent = 'Loading is taking longer -- check connection or open Console (F12)';
    }, 8000);

    loadMorphMap(url);

    if (url.toLowerCase().endsWith('.vrm')) {
      loadModelVRM(url, fitFn, resolve, reject, loadMsg, loadTimeout);
      return;
    }

    _gltfLoader.load(url, (gltf) => {
      clearTimeout(loadTimeout);
      dbg('GLTF loaded: ' + url.substring(0, 60), 'ok');

      if (_currentModelGroup) {
        _scene.remove(_currentModelGroup);
        _currentModelGroup = null;
        head = null; influences = null; dict = null; mixer = null; eyeDotMats = [];
        resetIdleLife();
      }
      if (currentVRM) { try { currentVRM.dispose(); } catch (e) {} currentVRM = null; }
      defaultVrmMap = null;

      _scene.add(gltf.scene);
      gltf.scene.updateMatrixWorld(true);
      _currentModelGroup = gltf.scene;

      gltf.scene.traverse((child) => {
        if (!child.isMesh) return;
        const n = child.name.toLowerCase();
        if (n.includes('eye') || n.includes('oculus') || n.includes('tooth') || n.includes('teeth') || n.includes('jaw') || n.includes('ball') || n.includes('sphere')) {
          child.visible = false; return;
        }
        if (child.material && child.material.color) {
          const c = child.material.color;
          if (c.r > 0.85 && c.g > 0.85 && c.b > 0.85 && n !== 'mesh_2') child.visible = false;
        }
      });

      // Find the main face mesh: prefer 'mesh_2' (FaceCap), then first Mesh with morphs, then any Mesh
      head = gltf.scene.getObjectByName('mesh_2') || null;
      if (!head || !head.isMesh || !head.geometry) {
        let best = null;
        gltf.scene.traverse((child) => {
          if (child.isMesh && child.geometry && !best) best = child;
          if (child.isMesh && child.geometry && child.morphTargetDictionary && Object.keys(child.morphTargetDictionary).length > 0) best = child;
        });
        head = best || null;
      }
      if (!head) { dbg('No usable mesh found in model', 'err'); if (loadMsg) loadMsg.style.display = 'none'; reject(new Error('No mesh')); return; }
      head.visible = true;
      dbg('Head mesh: ' + (head.name || '(unnamed)') + '  isMesh=' + head.isMesh + '  morphs=' + Object.keys(head.morphTargetDictionary || {}).length, 'ok');
      influences = head.morphTargetInfluences;
      dict = head.morphTargetDictionary || {};
      refreshBlinkKeys();
      const morphCount = Object.keys(dict).length;
      dbg('Morph targets: ' + morphCount, 'ok');

      const mats = buildMatsFn();
      head.material = mats.glow;
      const wireMesh = new THREE.Mesh(head.geometry, mats.wire);
      if (influences) wireMesh.morphTargetInfluences = influences;
      if (dict) wireMesh.morphTargetDictionary = dict;
      head.add(wireMesh);

      if (morphCount > 0) Object.keys(dict).forEach(k => S.currentWeights[k] = 0);

      function eyeCentroidFromMorph(morphName) {
        if (!dict || !morphCount) return null;
        const idx = dict[morphName];
        if (idx === undefined) return null;
        const morphAttr = head.geometry.morphAttributes.position ? head.geometry.morphAttributes.position[idx] : null;
        if (!morphAttr) return null;
        const baseAttr = head.geometry.attributes.position;
        const d = new THREE.Vector3();
        let sx = 0, sy = 0, sz = 0, sw = 0;
        for (let i = 0; i < morphAttr.count; i++) {
          d.fromBufferAttribute(morphAttr, i);
          const w = d.length();
          if (w < 1e-5) continue;
          sx += baseAttr.getX(i) * w;
          sy += baseAttr.getY(i) * w;
          sz += baseAttr.getZ(i) * w;
          sw += w;
        }
        return sw > 0 ? new THREE.Vector3(sx / sw, sy / sw, sz / sw) : null;
      }

      const leftEyePos = eyeCentroidFromMorph('eyeBlinkLeft');
      const rightEyePos = eyeCentroidFromMorph('eyeBlinkRight');
      if (leftEyePos && rightEyePos) {
        const dotGeo = new THREE.SphereGeometry(0.012, 16, 16);
        [leftEyePos, rightEyePos].forEach((pos) => {
          const mat = new THREE.MeshBasicMaterial({ color: 0x7dd3fc, toneMapped: false, depthTest: false });
          const dot = new THREE.Mesh(dotGeo, mat);
          dot.position.copy(pos);
          dot.renderOrder = 999;
          head.add(dot);
          eyeDotMats.push(mat);
        });
      }

      if (gltf.animations && gltf.animations.length > 0) {
        mixer = new THREE.AnimationMixer(gltf.scene);
        mixer.clipAction(gltf.animations[0]).play();
      }

      if (loadMsg) loadMsg.style.display = 'none';
      if (fitFn) fitFn();
      toggleFaceCanvas(cfgShowFace.checked);
      resolve();
    }, (xhr) => {
      const pct = xhr.total ? Math.round(xhr.loaded / xhr.total * 100) : 0;
      if (loadMsg) loadMsg.textContent = 'Loading model... ' + pct + '%';
    }, (err) => {
      clearTimeout(loadTimeout);
      if (loadMsg) loadMsg.textContent = 'Failed to load model -- open Console (F12)';
      dbg('GLTF error: ' + err.message, 'err');
      reject(err);
    });
  });
}

function loadModelVRM(url, fitFn, resolve, reject, loadMsg, loadTimeout) {
  defaultVrmMap = null;
  import('@pixiv/three-vrm').then(({ VRMLoaderPlugin, VRMUtils }) => {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    loader.load(url, (gltf) => {
      clearTimeout(loadTimeout);
      const vrm = gltf.userData.vrm;
      if (!vrm) {
        if (loadMsg) loadMsg.style.display = 'none';
        reject(new Error('File is not a valid VRM'));
        return;
      }
      if (_currentModelGroup) {
        _scene.remove(_currentModelGroup);
        _currentModelGroup = null;
        head = null; influences = null; dict = null; mixer = null; eyeDotMats = [];
        resetIdleLife();
      }
      if (currentVRM) { try { currentVRM.dispose(); } catch (e) {} currentVRM = null; }

      VRMUtils.removeUnnecessaryVertices(gltf.scene);
      VRMUtils.combineSkeletons(gltf.scene);
      _scene.add(vrm.scene);
      vrm.scene.updateMatrixWorld(true);
      _currentModelGroup = vrm.scene;
      currentVRM = vrm;
      vrm.update(0);
      buildDefaultVrmMap(vrm);

      let best = null;
      vrm.scene.traverse((c) => {
        if (!best && c.isMesh && c.geometry && c.morphTargetDictionary && Object.keys(c.morphTargetDictionary).length) best = c;
      });
      head = vrm.scene;
      influences = best ? best.morphTargetInfluences : null;
      dict = best ? (best.morphTargetDictionary || {}) : null;

      const vrmName = (vrm.meta && vrm.meta.name) || url.substring(url.lastIndexOf('/') + 1);
      dbg('VRM loaded: ' + vrmName, 'ok');
      if (loadMsg) loadMsg.style.display = 'none';
      if (fitFn) fitFn();
      toggleFaceCanvas(cfgShowFace.checked);
      resolve();
    }, (xhr) => {
      if (loadMsg && loadMsg.style.display !== 'none') loadMsg.textContent = 'Loading model... ' + (xhr.total ? Math.round(xhr.loaded / xhr.total * 100) : 0) + '%';
    }, (err) => {
      clearTimeout(loadTimeout);
      if (loadMsg) loadMsg.style.display = 'none';
      dbg('VRM error: ' + err.message, 'err');
      reject(err);
    });
  }).catch((e) => {
    clearTimeout(loadTimeout);
    if (loadMsg) loadMsg.style.display = 'none';
    dbg('VRM loader import failed: ' + e.message, 'err');
    reject(e);
  });
}

// ---- File browse / URL load for custom models ----
const cfgModelFile = document.getElementById('cfgModelFile');
const browseModelBtn = document.getElementById('browseModelBtn');
const loadModelBtn = document.getElementById('loadModelBtn');
const modelNameEl = document.getElementById('modelName');

if (browseModelBtn) {
  browseModelBtn.addEventListener('click', () => cfgModelFile.click());
}

if (cfgModelFile) {
  cfgModelFile.addEventListener('change', () => {
    const file = cfgModelFile.files[0];
    if (!file) return;
    const blobUrl = URL.createObjectURL(file);
    cfgCustomModelUrl.value = '';
    cfgModel3d.value = 'custom';
    customModelGroup.style.display = 'block';
    if (modelNameEl) modelNameEl.textContent = file.name;
    dbg('Loading file: ' + file.name, 'ok');
    loadModel(blobUrl, null, () => {
      return makeMatPair();
    }).catch(e => dbg('Model load failed: ' + e.message, 'err'));
    saveSettings();
  });
}

// ---- Morph mapping screen ----
const mapMorphsBtn = document.getElementById('mapMorphsBtn');
const morphMapOverlay = document.getElementById('morphMapOverlay');
const morphMapList = document.getElementById('morphMapList');

function rebuildMorphRows() {
  const names = detectMorphNames();
  morphMapList.innerHTML = '';
  if (!names.length) {
    const row = document.createElement('div');
    row.className = 'morph-row';
    const lab = document.createElement('label');
    lab.style.color = 'var(--warn)';
    lab.textContent = 'No morph targets / VRM expressions found on this model. Load a face model with morphs first, then re-open this screen.';
    row.appendChild(lab);
    morphMapList.appendChild(row);
    return;
  }
  for (const { slot, label } of MORPH_SLOTS) {
    const row = document.createElement('div');
    row.className = 'morph-row';
    const lab = document.createElement('label');
    lab.textContent = label;
    const sel = document.createElement('select');
    const oAuto = document.createElement('option');
    oAuto.value = '';
    oAuto.textContent = '-- default / none --';
    sel.appendChild(oAuto);
    for (const name of names) {
      const o = document.createElement('option');
      o.value = name;
      o.textContent = name;
      sel.appendChild(o);
    }
    sel.value = (morphMap && morphMap[slot]) || '';
    sel.addEventListener('change', () => {
      morphMap = morphMap || {};
      if (sel.value) morphMap[slot] = sel.value;
      else delete morphMap[slot];
      saveMorphMap();
    });
    row.appendChild(lab);
    row.appendChild(sel);
    morphMapList.appendChild(row);
  }
}

function openMorphMapper() {
  rebuildMorphRows();
  morphMapOverlay.classList.add('show');
}

if (mapMorphsBtn) {
  mapMorphsBtn.addEventListener('click', openMorphMapper);
  const closeMorph = document.getElementById('morphClose');
  const autoMorph = document.getElementById('morphAutoMap');
  const resetMorph = document.getElementById('morphResetMap');
  if (closeMorph) closeMorph.addEventListener('click', () => morphMapOverlay.classList.remove('show'));
  if (resetMorph) resetMorph.addEventListener('click', () => { morphMap = {}; saveMorphMap(); rebuildMorphRows(); });
  if (autoMorph) autoMorph.addEventListener('click', () => { autoSuggestMorphs(); rebuildMorphRows(); });
  morphMapOverlay.addEventListener('click', (e) => { if (e.target === morphMapOverlay) morphMapOverlay.classList.remove('show'); });
}

if (loadModelBtn) {
  loadModelBtn.addEventListener('click', () => {
    const url = cfgCustomModelUrl.value.trim();
    if (!url) { showStatus('Enter a model URL first', 'warn'); return; }
    cfgModel3d.value = 'custom';
    customModelGroup.style.display = 'block';
    if (modelNameEl) modelNameEl.textContent = url.split('/').pop() || url;
    loadModel(url, null, () => {
      return makeMatPair();
    }).catch(e => dbg('Model load failed: ' + e.message, 'err'));
    saveSettings();
  });
}

async function initScene() {
  dbg('initScene() starting', 'info');
  const timer = new THREE.Timer();
  const stage = document.querySelector('.stage');
  const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 50);
  camera.position.set(0, 0.05, 3.2);
  _camera = camera;
  const scene = new THREE.Scene();
  _scene = scene;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  stage.appendChild(renderer.domElement);
  renderer.domElement.classList.add('gl');
  faceCanvas = renderer.domElement;
  _renderer = renderer;

  const environment = new RoomEnvironment();
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = pmremGenerator.fromScene(environment, 0.04).texture;

  ambientLight = new THREE.AmbientLight(0x2f81f7, 0.4);
  scene.add(ambientLight);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 2.0;
  controls.maxDistance = 6;
  controls.minAzimuthAngle = -Math.PI / 2;
  controls.maxAzimuthAngle = Math.PI / 2;
  controls.maxPolarAngle = Math.PI / 1.8;
  controls.target.set(0, 0.15, -0.2);

  function fit() {
    const rect = stage.getBoundingClientRect();
    renderer.setSize(rect.width, rect.height, false);
    camera.aspect = rect.width / rect.height;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', () => { fit(); resizeWaveCanvas(); });

  function buildBlueprintMaterials() {
    return makeMatPair();
  }

  const ktx2Loader = new KTX2Loader();
  ktx2Loader.setTranscoderPath('https://unpkg.com/three@0.183.0/examples/jsm/libs/basis/');
  await ktx2Loader.detectSupport(renderer);
  _ktx2Loader = ktx2Loader;

  const gltfLoader = new GLTFLoader().setKTX2Loader(ktx2Loader).setMeshoptDecoder(MeshoptDecoder);
  _gltfLoader = gltfLoader;

  const MODEL_MAP = {
    facecap: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@master/examples/models/gltf/facecap.glb',
    custom: cfgCustomModelUrl.value || 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@master/examples/models/gltf/facecap.glb'
  };
  const MODEL_URL = MODEL_MAP[cfgModel3d.value] || MODEL_MAP.facecap;
  await loadModel(MODEL_URL, fit, buildBlueprintMaterials);
  if (modelNameEl && cfgModel3d.value === 'facecap') modelNameEl.textContent = 'FaceCap (Classic)';

  setState('idle');

  const colorA = new THREE.Color();
  const colorB = new THREE.Color();
  let t = 0;

  function animate() {
    timer.update();
    const delta = timer.getDelta();
    t += delta;
    if (mixer) mixer.update(delta);

    S.talkPulse = Math.max(0, S.talkPulse - delta * 1.8);
    S.beatPulse = Math.max(0, S.beatPulse - delta * 4);
    const targetIntensity = Math.max(S.talkPulse, S.isTtsSpeaking ? 0.16 : 0);
    S.phonemeIntensity += (targetIntensity - S.phonemeIntensity) * 0.09;

    if ((influences && dict) || currentVRM) {
      const mirrorActive = Mirror.active && Mirror.hasFace;
      const copyMode = mirrorActive && cfgMirrorMode && cfgMirrorMode.value === 'copy';
      const emotionMode = mirrorActive && cfgMirrorMode && cfgMirrorMode.value === 'emotion';

      const stateTarget = STATE_TARGETS[S.currentState] || {};
      const emotionTarget = (emotionMode && EMOTION_TARGETS[Mirror.emotion]) || {};
      const feelingTarget = FEELING_TARGETS[S.feeling] || {};
      const visionTarget = (visionReact && visionPersonPresent && IDLE_LIFE_STATES.includes(S.currentState)) ? VISION_TARGETS.active : {};
      const target = Object.assign({}, stateTarget, emotionTarget, feelingTarget, visionTarget);

      const slots = currentVRM ? ALL_SLOT_NAMES : Object.keys(dict);

      for (const key of slots) {
        let goal = target[key] !== undefined ? target[key] : 0;
        let rate = 0.08;
        if (copyMode && Mirror.weights && Mirror.weights[key] !== undefined) {
          goal = Mirror.weights[key];
          rate = 0.45;
        }
        S.currentWeights[key] += (goal - S.currentWeights[key]) * rate;
      }

      // ---- Idle life: autonomous blinking + soft breathing ----
      const idleLife = !S.speaking && !S.phonemePreview && IDLE_LIFE_STATES.includes(S.currentState);
      if (idleLife) {
        idleTime += delta;
        if (idleBlinkT < 0) {
          if (idleTime >= idleBlinkAt) {
            idleBlinkT = 0;
            idleBlinkAt = idleTime + 2.5 + Math.random() * 4;
          }
        } else {
          const step = Math.min(delta / 0.16, 0.22);
          idleBlinkT = Math.min(1, idleBlinkT + step);
          const blinkAmt = 0.95 * Math.sin(idleBlinkT * Math.PI);
          if (blinkLKey) S.currentWeights[blinkLKey] = Math.max(S.currentWeights[blinkLKey] || 0, blinkAmt);
          else S.currentWeights[IDLE_BLINK_LEFT] = Math.max(S.currentWeights[IDLE_BLINK_LEFT] || 0, blinkAmt);
          if (blinkRKey) S.currentWeights[blinkRKey] = Math.max(S.currentWeights[blinkRKey] || 0, blinkAmt);
          else S.currentWeights[IDLE_BLINK_RIGHT] = Math.max(S.currentWeights[IDLE_BLINK_RIGHT] || 0, blinkAmt);
          if (idleBlinkT >= 1) idleBlinkT = -1;
        }
        if (!S.lipSyncActive && S.phonemeIntensity <= 0.03) {
          idleBreathPhase += delta;
          const b = 0.5 + 0.5 * Math.sin(idleBreathPhase * 2.2);
          S.currentWeights.jawOpen = Math.max(S.currentWeights.jawOpen || 0, 0.03 + 0.05 * b);
          if (head) {
            if (headBaseY === null) headBaseY = head.position.y;
            head.position.y = headBaseY + 0.003 * b;
          }
        }
      } else if (head && headBaseY !== null) {
        head.position.y += (headBaseY - head.position.y) * 0.08;
      }

      // ---- Viseme playground preview override (mouth shapes win while held) ----
      if (S.visemePreview) {
        if (performance.now() < S.visemePreview.until) {
          for (const k of VISEME_KEYS) {
            if (S.visemePreview.target[k] !== undefined) {
              S.currentWeights[k] = Math.max(S.currentWeights[k] || 0, S.visemePreview.target[k]);
            }
          }
        } else {
          S.visemePreview = null;
        }
      }

      if (S.speaking || S.phonemeIntensity > 0.03) {
        applyPhonemeShape(t);
      }

      if (currentVRM && currentVRM.expressionManager) {
        const em = currentVRM.expressionManager;
        for (const slot of slots) {
          const mapped = (morphMap && morphMap[slot]) || (defaultVrmMap && defaultVrmMap[slot]) || slot;
          em.setValue(mapped, S.currentWeights[slot] || 0);
        }
        currentVRM.update(delta);
      } else {
        for (const key of Object.keys(dict)) {
          const idx = morphIndex(key);
          if (idx !== undefined && influences) influences[idx] = S.currentWeights[key] || 0;
        }
      }

      const pillTick = Math.floor(t * 2);
      if (_mirrorPillTick !== pillTick && Mirror.active && cfgMirror && cfgMirror.checked && mirrorStatus) {
        _mirrorPillTick = pillTick;
        updateMirrorUI();
      }

      if (S.currentState === 'alert' && influences) {
        const li = morphIndex('eyeWideLeft');
        const ri = morphIndex('eyeWideRight');
        if (li !== undefined && ri !== undefined) {
          const twitch = Math.sin(t * 15) * 0.08;
          influences[li] += twitch;
          influences[ri] += twitch;
        }
      }

      if (head) {
        const intensity = S.currentState === 'alert' ? 2.5 : (S.currentState === 'responding' ? 1.5 : 1);
        head.rotation.y = Math.sin(t * 0.4 * intensity) * 0.06;
        head.rotation.x = Math.sin(t * 0.28 * intensity) * 0.03;
      }
    }

    // Sentiment blend ramp + expiry
    if (S.feelingExpire && performance.now() > S.feelingExpire) {
      S.feelingExpire = 0;
      setFeeling('neutral', 0);
    }
    S.feelingBlend += (S.feelingBlendTarget - S.feelingBlend) * 0.05;
    if (S.feelingBlend < 0.01 && S.feelingBlendTarget === 0) S.feelingBlend = 0;

    if (glowMat && wireMat && ambientLight) {
      const colors = STATE_COLORS[S.currentState];
      const fc = FEELING_COLORS[S.feeling] || FEELING_COLORS.neutral;
      const fb = S.feelingBlend;
      const speakingNow = S.speaking || S.isTtsSpeaking;
      const excite = speakingNow ? Math.min(1, (audioAnalyser || LocalSpeech.analyser)
        ? (S.bass * 0.9 + S.mid * 0.5 + S.treble * 0.3) * 2.4
        : S.phonemeIntensity * 2.2) : 0;
      if (colors) {
        colorA.copy(_hexC(colors.emissive)).lerp(_hexC(fc.emissive), fb);
        if (excite > 0.01) colorA.multiplyScalar(1 + excite * 0.5);
        colorB.setHex(glowMat.emissive.getHex()); colorB.lerp(colorA, 0.04); glowMat.emissive.copy(colorB);
        colorA.copy(_hexC(colors.wire)).lerp(_hexC(fc.wire), fb);
        colorB.setHex(wireMat.color.getHex()); colorB.lerp(colorA, 0.04); wireMat.color.copy(colorB);
        colorA.copy(_hexC(colors.ambient)).lerp(_hexC(fc.ambient), fb);
        colorB.setHex(ambientLight.color.getHex()); colorB.lerp(colorA, 0.03); ambientLight.color.copy(colorB);
        ambientLight.intensity += (lightTargetIntensity - ambientLight.intensity) * 0.04;
        ambientLight.color.lerp(lightTargetColor, 0.04);
        if (eyeDotMats.length) {
          _feelingDotTmp.copy(_hexC(colors.wire)).lerp(_hexC(fc.wire), fb);
          eyeDotMats.forEach((m) => { colorB.setHex(m.color.getHex()); colorB.lerp(_feelingDotTmp, 0.06); m.color.copy(colorB); });
        }
        if (Math.abs(fb - _lastPulseBlend) > 0.03) {
          _lastPulseBlend = fb;
          applyFeelingPulse();
        }
      }
    }

    if (S.speaking && S.speakStartedAt !== null) {
      const elapsedSec = (performance.now() - S.speakStartedAt) / 1000;
      const wasLong = S.isLongResponse;
      S.isLongResponse = elapsedSec >= 12;
      if (S.isLongResponse !== wasLong) {
        document.getElementById('bgPulse').classList.toggle('long-mode', S.isLongResponse);
        document.getElementById('stateLabel').className = 'state-label' + (S.currentState === 'alert' ? ' alert-mode' : '') + (S.isLongResponse ? ' long-mode' : '');
      }
      document.getElementById('elapsedLabel').textContent = S.isLongResponse ? t('status.extended', 'EXTENDED // ') + elapsedSec.toFixed(0) + 'S' : elapsedSec.toFixed(0) + 'S';
      document.getElementById('elapsedLabel').className = 'elapsed-label' + (S.isLongResponse ? ' long-mode' : '');
      if (!S.isTtsSpeaking && (performance.now() - S.lastActivityAt) / 1000 > 8) {
        setSpeaking(false);
      }
    } else {
      document.getElementById('elapsedLabel').textContent = '';
    }

    document.getElementById('hudCoord').textContent = 'X:' + (Math.sin(t * 0.7) * 99 + 150).toFixed(0).padStart(3, '0') + ' Y:' + (Math.cos(t * 0.5) * 99 + 150).toFixed(0).padStart(3, '0');
    document.getElementById('hudClock').textContent = new Date().toTimeString().slice(0, 8);
    const freqLive = audioAnalyser || LocalSpeech.analyser;
    const hudFreqEl = document.getElementById('hudFreq');
    if (freqLive) {
      const dominant = S.bass > S.mid && S.bass > S.treble ? S.bass * 250 : S.mid > S.treble ? S.mid * 2000 : S.treble * 8000;
      hudFreqEl.textContent = String.fromCharCode(216) + ' ' + dominant.toFixed(1) + ' Hz';
    } else {
      hudFreqEl.textContent = String.fromCharCode(216) + ' --';
    }

    controls.update();
    renderer.render(scene, camera);
  }

  renderer.setAnimationLoop(animate);
  fit();
  dbg('Animation loop started', 'ok');
}

if (window.speechSynthesis) {
  const loadV = () => {
    const voices = window.speechSynthesis.getVoices() || [];
    dbg('Voices loaded: ' + voices.length, 'ok');
  };
  loadV();
  if (window.speechSynthesis.onvoiceschanged !== undefined) {
    window.speechSynthesis.onvoiceschanged = loadV;
  }
} else {
  dbg('speechSynthesis NOT AVAILABLE -- TTS disabled', 'warn');
}

// ============================================================
// Settings UI
// ============================================================
const panel = document.getElementById('sidePanel');
const overlay = document.getElementById('overlay');
const toggleBtn = document.getElementById('togglePanel');
const closeBtn = document.getElementById('closePanel');
const cfgProvider = document.getElementById('cfgProvider');
const cfgModel = document.getElementById('cfgModel');
const cfgKey = document.getElementById('cfgKey');
const cfgUrl = document.getElementById('cfgUrl');
const cfgTemp = document.getElementById('cfgTemp');
const cfgMaxTokens = document.getElementById('cfgMaxTokens');
const cfgSystem = document.getElementById('cfgSystem');
const cfgTtsLang = document.getElementById('cfgTtsLang');
const cfgAutoSend = document.getElementById('cfgAutoSend');
const cfgUseGeminiTts = document.getElementById('cfgUseGeminiTts');
const geminiTtsFields = document.getElementById('geminiTtsFields');
const cfgGeminiTtsKey = document.getElementById('cfgGeminiTtsKey');
const cfgGeminiTtsModel = document.getElementById('cfgGeminiTtsModel');
const cfgGeminiVoice = document.getElementById('cfgGeminiVoice');
const cfgLiveVoice = document.getElementById('cfgLiveVoice');
const liveVoiceFields = document.getElementById('liveVoiceFields');
const cfgLiveKey = document.getElementById('cfgLiveKey');
const cfgLiveModel = document.getElementById('cfgLiveModel');
const cfgLiveTranscript = document.getElementById('cfgLiveTranscript');
const cfgModel3d = document.getElementById('cfgModel3d');
const cfgCustomModelUrl = document.getElementById('cfgCustomModelUrl');
const customModelGroup = document.getElementById('customModelGroup');
const cfgWaveStyle = document.getElementById('cfgWaveStyle');
const cfgTheme = document.getElementById('cfgTheme');
const cfgLightPreset = document.getElementById('cfgLightPreset');
if (cfgLightPreset) {
  cfgLightPreset.addEventListener('change', () => {
    applyLightPreset(cfgLightPreset.value);
    saveSettings();
  });
}
const cfgShowFace = document.getElementById('cfgShowFace');
const cfgMicEngine = document.getElementById('cfgMicEngine');
const cfgTtsEngine = document.getElementById('cfgTtsEngine');
const cfgAutoRestoreChat = document.getElementById('cfgAutoRestoreChat');
const cfgWhisperTier = document.getElementById('cfgWhisperTier');
const whisperTierGroup = document.getElementById('whisperTierGroup');
const cfgWebLlmModel = document.getElementById('cfgWebLlmModel');
const cfgWebLlmCustom = document.getElementById('cfgWebLlmCustom');
const cfgMasterVolume = document.getElementById('cfgMasterVolume');
const cfgMasterVolumeValue = document.getElementById('masterVolumeValue');
const cfgOutputDevice = document.getElementById('cfgOutputDevice');
const cfgLocale = document.getElementById('cfgLocale');

// ---- Viseme playground ----
const visemeInput = document.getElementById('visemeInput');
const visemePreviewBtn = document.getElementById('visemePreviewBtn');
const VISEME_PREVIEW_MS = 300;
let visemeHoldUntil = null;
function engageVisemePreview() {
  if (visemeInput && !S.visemePreview) {
    const ch = (visemeInput.value || '').trim().toUpperCase();
    if (!ch) return;
    S.visemePreview = { target: visemeFor(ch) || DEFAULT_VISEME, until: performance.now() + 300 };
    if (visemePreviewBtn) visemePreviewBtn.setAttribute('data-viseme', ch);
  }
  visemeHoldUntil = performance.now() + 10000;
  if (S.visemePreview) S.visemePreview.until = visemeHoldUntil;
}
function releaseVisemePreview() {
  visemeHoldUntil = null;
  if (S.visemePreview) S.visemePreview.until = 0;
}
if (visemeInput && visemePreviewBtn) {
  visemeInput.addEventListener('input', () => {
    const ch = (visemeInput.value || '').trim().toUpperCase();
    if (!ch) { S.visemePreview = null; return; }
    const target = visemeFor(ch) || DEFAULT_VISEME;
    S.visemePreview = { target, until: performance.now() + VISEME_PREVIEW_MS };
    visemePreviewBtn.setAttribute('data-viseme', ch);
  });
  visemePreviewBtn.addEventListener('pointerdown', engageVisemePreview);
  visemePreviewBtn.addEventListener('pointerup', releaseVisemePreview);
  visemePreviewBtn.addEventListener('pointerleave', releaseVisemePreview);
}

// ---- PWA install ----
let deferredInstallPrompt = null;
let installBtn = document.getElementById('installBtn');
if (installBtn) {
  const inStandalone = window.matchMedia && window.matchMedia('(display-mode: standalone)').matches;
  if (inStandalone) installBtn.style.display = 'none';
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredInstallPrompt = e;
    if (!inStandalone) installBtn.style.display = 'block';
  });
  installBtn.addEventListener('click', async () => {
    if (!deferredInstallPrompt) return;
    deferredInstallPrompt.prompt();
    const choice = await deferredInstallPrompt.userChoice.catch(() => null);
    if (choice && choice.outcome === 'accepted') installBtn.style.display = 'none';
    deferredInstallPrompt = null;
  });
  window.addEventListener('appinstalled', () => { installBtn.style.display = 'none'; });
}

function updateMasterVolumeUI() {
  if (cfgMasterVolumeValue) cfgMasterVolumeValue.textContent = parseInt(cfgMasterVolume.value, 10) + '%';
  setMasterVolume(parseInt(cfgMasterVolume.value, 10) / 100);
  applyMasterSettings();
}

async function populateOutputDevices() {
  if (!cfgOutputDevice || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const prev = cfgOutputDevice.value;
    cfgOutputDevice.innerHTML = '<option value="default">System default</option>';
    const devices = await navigator.mediaDevices.enumerateDevices();
    const seen = new Set(['default']);
    devices.filter((d) => d.kind === 'audiooutput' && d.deviceId).forEach((d) => {
      if (seen.has(d.deviceId)) return;
      seen.add(d.deviceId);
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || ('Output ' + (seen.size - 1));
      cfgOutputDevice.appendChild(opt);
    });
    const good = Array.from(cfgOutputDevice.options).some((o) => o.value === prev);
    cfgOutputDevice.value = good ? prev : 'default';
    setOutputDevice(cfgOutputDevice.value);
  } catch (e) {
    dbg('Output device enumeration failed: ' + e.message, 'warn');
  }
}

// ---- WebLLM Lite mode: local-storage flag for "user made an explicit choice" ----
// (detectDeviceTier / recommendedWebLlmModel are imported from pure.js so
// the recommendation heuristic itself stays unit-testable without a DOM.)
const WEBLLM_MODEL_OVERRIDE_KEY = 'aiface_webllm_model_manual';
function hasManualWebLlmModelChoice() {
  try { return !!localStorage.getItem(WEBLLM_MODEL_OVERRIDE_KEY); } catch (e) { return false; }
}
function markManualWebLlmModelChoice() {
  try { localStorage.setItem(WEBLLM_MODEL_OVERRIDE_KEY, '1'); } catch (e) {}
}

function syncWebLlmModel() {
  if (!cfgWebLlmModel) return;
  const v = cfgModel ? cfgModel.value : '';
  const has = Array.from(cfgWebLlmModel.options).some((o) => o.value === v);
  if (v && has) {
    cfgWebLlmModel.value = v;
    cfgWebLlmCustom.style.display = 'none';
  } else {
    cfgWebLlmModel.value = '';
    cfgWebLlmCustom.style.display = 'block';
    if (v) cfgWebLlmCustom.value = v;
  }
}

function updateWhisperUI() {
  if (!whisperTierGroup) return;
  whisperTierGroup.style.display = cfgMicEngine.value === 'local' ? 'block' : 'none';
}

if (cfgWhisperTier) {
  cfgWhisperTier.addEventListener('change', () => {
    setWhisperModel(cfgWhisperTier.value);
    saveSettings();
  });
}

if (cfgMasterVolume) {
  cfgMasterVolume.addEventListener('input', () => { updateMasterVolumeUI(); });
  cfgMasterVolume.addEventListener('change', () => { updateMasterVolumeUI(); saveSettings(); });
}
if (cfgOutputDevice) {
  cfgOutputDevice.addEventListener('change', () => {
    setOutputDevice(cfgOutputDevice.value);
    applyMasterSettings();
    saveSettings();
  });
}
populateOutputDevices();

function toggleFaceCanvas(show) {
  if (faceCanvas) faceCanvas.style.display = show ? 'block' : 'none';
  dbg('Face canvas: ' + (show ? 'visible' : 'hidden'), 'ok');
}

cfgShowFace.addEventListener('change', () => {
  toggleFaceCanvas(cfgShowFace.checked);
  saveSettings();
});

// ---- Live Mirror (MediaPipe FaceLandmarker) ----
const cfgMirror = document.getElementById('cfgMirror');
const cfgMirrorMode = document.getElementById('cfgMirrorMode');
const cfgFeelingMode = document.getElementById('cfgFeelingMode');
const mirrorStatus = document.getElementById('mirrorStatus');
const mirrorModeGroup = document.getElementById('mirrorModeGroup');

function revealFeelingPill() {
  const pill = getFeelingEl();
  if (!pill) return;
  const fm = cfgFeelingMode ? cfgFeelingMode.value : 'auto';
  const active = fm !== 'off' && S.feeling !== 'neutral';
  pill.style.display = active ? 'block' : 'none';
}

function updateMirrorUI() {
  const on = cfgMirror.checked;
  if (mirrorModeGroup) mirrorModeGroup.style.display = on ? 'block' : 'none';
  if (!on) {
    if (mirrorStatus) { mirrorStatus.className = 'status-pill warn'; mirrorStatus.textContent = 'MIRROR: OFF'; }
    return;
  }
  if (!mirrorStatus) return;
  if (Mirror.active) {
    mirrorStatus.className = 'status-pill ' + (Mirror.hasFace ? 'ok' : 'warn');
    mirrorStatus.textContent = Mirror.hasFace ? ('MIRROR: ' + Mirror.emotion.toUpperCase()) : 'MIRROR: SEARCHING FOR FACE...';
  } else {
    mirrorStatus.className = 'status-pill warn';
    mirrorStatus.textContent = 'MIRROR: STARTING...';
  }
}

cfgMirror.addEventListener('change', async () => {
  if (cfgMirror.checked) {
    const ok = await startMirror();
    if (!ok) {
      cfgMirror.checked = false;
      if (mirrorStatus) { mirrorStatus.className = 'status-pill err'; mirrorStatus.textContent = 'MIRROR: CAMERA/PROCESSING UNAVAILABLE'; }
    }
  } else {
    stopMirror();
  }
  updateMirrorUI();
  saveSettings();
});
cfgMirrorMode.addEventListener('change', () => { saveSettings(); });
cfgFeelingMode.addEventListener('change', () => { saveSettings(); if (S.feeling === 'neutral') revealFeelingPill(); });
cfgMicEngine.addEventListener('change', () => { updateWhisperUI(); saveSettings(); });
cfgTtsEngine.addEventListener('change', () => { saveSettings(); });
window.addEventListener('pagehide', () => { if (Mirror.active) stopMirror(); });

// ---- AI Vision (on-device object detection) ----
const cfgVision = document.getElementById('cfgVision');
const cfgVisionBackend = document.getElementById('cfgVisionBackend');
const cfgVisionMaxFps = document.getElementById('cfgVisionMaxFps');
const cfgVisionReact = document.getElementById('cfgVisionReact');
const btnVisionPause = document.getElementById('btnVisionPause');
const visionStatus = document.getElementById('visionStatus');
const visionBackendGroup = document.getElementById('visionBackendGroup');
const visionFpsGroup = document.getElementById('visionFpsGroup');
const visionReactGroup = document.getElementById('visionReactGroup');
const visionPauseGroup = document.getElementById('visionPauseGroup');
let visionActive = false;
let visionPaused = false;
let visionReact = true;
let visionPersonPresent = false;
let visionBackendName = '';
let visionVideo = null;
let visionStream = null;

function handleVisionDetections(detections) {
  visionPersonPresent = detections.some((d) => d.class === 'person');
  let summary = '';
  const counts = {};
  for (const d of detections) counts[d.class] = (counts[d.class] || 0) + 1;
  const labels = Object.keys(counts).slice(0, 3).map((c) => (counts[c] > 1 ? counts[c] + ' ' + c : c));
  if (labels.length) summary = labels.join(', ');
  const label = t('vision.status.label', 'VISION: {backend} \u00b7 {summary}')
    .replace('{backend}', visionBackendName)
    .replace('{summary}', summary);
  setVisionStatus('on', label);
}

function setVisionStatus(state, extra) {
  if (!visionStatus) return;
  if (state === 'off') {
    visionStatus.className = 'status-pill warn';
    visionStatus.textContent = t('vision.status.off', 'VISION: OFF');
  } else if (state === 'starting') {
    visionStatus.className = 'status-pill warn';
    visionStatus.textContent = t('vision.status.starting', 'VISION: STARTING...');
  } else if (state === 'err') {
    visionStatus.className = 'status-pill err';
    visionStatus.textContent = t('vision.status.error', 'VISION: UNAVAILABLE');
  } else if (state === 'on') {
    visionStatus.className = 'status-pill ok';
    visionStatus.textContent = extra || visionBackendName;
  }
}

function updateVisionUI() {
  const on = cfgVision.checked;
  const hidden = on ? 'block' : 'none';
  if (visionBackendGroup) visionBackendGroup.style.display = hidden;
  if (visionFpsGroup) visionFpsGroup.style.display = hidden;
  if (visionReactGroup) visionReactGroup.style.display = hidden;
  if (visionPauseGroup) visionPauseGroup.style.display = hidden;
  if (!on) {
    setVisionStatus('off');
    if (btnVisionPause) btnVisionPause.textContent = t('settings.visionPause') || 'Pause Vision';
  }
}

async function startVision() {
  if (visionActive) return true;
  try {
    if (!visionVideo) {
      visionVideo = document.createElement('video');
      visionVideo.muted = true;
      visionVideo.setAttribute('playsinline', '');
      visionVideo.setAttribute('autoplay', '');
      visionVideo.style.display = 'none';
      document.body.appendChild(visionVideo);
    }
    visionStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    });
    visionVideo.srcObject = visionStream;
    try { await visionVideo.play(); } catch (e) {}

    const forceBackend = cfgVisionBackend && cfgVisionBackend.value !== 'auto'
      ? (cfgVisionBackend.value === 'yolo' ? 'yolo' : 'coco-ssd')
      : undefined;
    const maxFPS = cfgVisionMaxFps ? Math.max(1, Math.min(30, parseInt(cfgVisionMaxFps.value, 10) || 8)) : 8;
    setVisionStatus('starting');
    const backend = await Vision.init(visionVideo, { forceBackend, maxFPS });
    visionBackendName = backend === 'yolo-webgpu' ? 'YOLO (WebGPU)' : 'COCO-SSD (WebGL)';
    visionActive = true;
    visionPaused = false;
    visionReact = cfgVisionReact ? cfgVisionReact.checked : true;
    Vision.start(handleVisionDetections);
    if (btnVisionPause) btnVisionPause.textContent = t('settings.visionPause') || 'Pause Vision';
    setVisionStatus('on');
    return true;
  } catch (err) {
    console.warn('[vision] start failed:', err);
    stopVision();
    setVisionStatus('err');
    return false;
  }
}

function stopVision() {
  visionActive = false;
  visionPaused = false;
  visionPersonPresent = false;
  try { Vision.stop(); } catch (e) {}
  if (visionStream) { visionStream.getTracks().forEach((t) => t.stop()); visionStream = null; }
  if (visionVideo) { visionVideo.srcObject = null; }
}

function toggleVisionPause() {
  if (!visionActive) return;
  visionPaused = !visionPaused;
  if (visionPaused) {
    Vision.stop();
    if (btnVisionPause) btnVisionPause.textContent = t('settings.visionResume') || 'Resume Vision';
  } else {
    Vision.start(handleVisionDetections);
    if (btnVisionPause) btnVisionPause.textContent = t('settings.visionPause') || 'Pause Vision';
    setVisionStatus('on');
  }
}

cfgVision.addEventListener('change', async () => {
  if (cfgVision.checked) {
    const ok = await startVision();
    if (!ok) cfgVision.checked = false;
  } else {
    stopVision();
  }
  updateVisionUI();
  saveSettings();
});
cfgVisionBackend.addEventListener('change', () => {
  saveSettings();
  if (visionActive) { stopVision(); startVision(); }
});
cfgVisionMaxFps.addEventListener('change', () => {
  saveSettings();
  if (visionActive) { stopVision(); startVision(); }
});
cfgVisionReact.addEventListener('change', () => { visionReact = cfgVisionReact.checked; saveSettings(); });
if (btnVisionPause) btnVisionPause.addEventListener('click', toggleVisionPause);
window.addEventListener('pagehide', () => { if (visionActive) stopVision(); });

cfgLiveVoice.addEventListener('change', () => {
  liveVoiceFields.style.display = cfgLiveVoice.checked ? 'block' : 'none';
  updateMicButtonsMode();
  if (recognizing) recognizer.stop();
  if (liveCallActive) toggleLiveCall();
});
cfgUseGeminiTts.addEventListener('change', () => {
  geminiTtsFields.style.display = cfgUseGeminiTts.checked ? 'block' : 'none';
});
const btnSave = document.getElementById('btnSave');
const btnTest = document.getElementById('btnTest');
const btnTestVoice = document.getElementById('btnTestVoice');
const connStatus = document.getElementById('connStatus');
const chatHistory = document.getElementById('chatHistory');
const chatInput = document.getElementById('chatInput');
const btnSend = document.getElementById('btnSend');
const btnClear = document.getElementById('btnClear');
const btnStop = document.getElementById('btnStop');
const btnNewChat = document.getElementById('btnNewChat');
const btnSessions = document.getElementById('btnSessions');
const attachBtn = document.getElementById('attachBtn');
const imageInput = document.getElementById('imageInput');
const imageChip = document.getElementById('imageChip');
const imageChipThumb = document.getElementById('imageChipThumb');
const imageChipRemove = document.getElementById('imageChipRemove');
let pendingImageDataUrl = '';
const voiceIndicator = document.getElementById('voiceIndicator');
const micBtn = document.getElementById('micBtn');
const micFloatBtn = document.getElementById('micFloatBtn');
const captionOverlay = document.getElementById('captionOverlay');
const captionLabel = document.getElementById('captionLabel');
const captionText = document.getElementById('captionText');

let abortCtrl = null;
let messages = [];
let fullResponse = '';
let assistantTextEl = null;
let currentSessionId = null;

function openPanel() { panel.classList.add('open'); overlay.classList.add('show'); }
function closePanel() { panel.classList.remove('open'); overlay.classList.remove('show'); }
toggleBtn.addEventListener('click', () => panel.classList.contains('open') ? closePanel() : openPanel());
closeBtn.addEventListener('click', closePanel);
overlay.addEventListener('click', closePanel);
voiceIndicator.addEventListener('click', toggleMute);

// ---- Collapsible Settings Sections ----
document.querySelectorAll('.settings-section-header').forEach(header => {
  header.addEventListener('click', () => {
    const section = header.parentElement;
    section.classList.toggle('open');
  });
});

// ---- Onboarding Modal ----
const onboardingModal = document.getElementById('onboardingModal');
function checkOnboarding() {
  if (!localStorage.getItem('aiface_onboarded')) {
    setTimeout(() => onboardingModal.classList.add('show'), 600);
  }
}
function dismissOnboarding(openSettings) {
  localStorage.setItem('aiface_onboarded', '1');
  onboardingModal.classList.remove('show');
  if (openSettings) openPanel();
}
document.getElementById('onboardingGetStarted').addEventListener('click', () => dismissOnboarding(true));
document.getElementById('onboardingSkip').addEventListener('click', () => dismissOnboarding(false));
checkOnboarding();

const DEFAULTS = {
  gemini: { url: 'https://generativelanguage.googleapis.com/v1beta', model: 'gemini-2.5-flash', key: '', needsKey: true },
  openai: { url: 'https://api.openai.com/v1', model: 'gpt-4o-mini', key: '', needsKey: true },
  meta: { url: 'https://api.meta.ai/v1', model: 'muse-spark-1.2', key: '', needsKey: true },
  bazaarlink: { url: 'https://api.bazaarlink.ai/v1', model: 'auto:free', key: '', needsKey: true },
  ollama: { url: 'http://localhost:11434', model: 'llama3.2', key: '', needsKey: false },
  webllm: { url: '', model: 'Llama-3.2-3B-Instruct-q4f32_1-MLC', key: '', needsKey: false },
  custom: { url: 'http://localhost:8000/v1', model: 'default', key: '', needsKey: true }
};

function applyProviderDefaults(resetFields) {
  const p = cfgProvider.value;
  const d = DEFAULTS[p];
  cfgUrl.placeholder = d.url;
  cfgModel.placeholder = d.model;
  document.getElementById('groupKey').style.display = d.needsKey ? 'block' : 'none';
  document.getElementById('groupUrl').style.display = (p === 'webllm') ? 'none' : 'block';
  document.getElementById('groupModel').style.display = (p === 'webllm') ? 'none' : 'block';
  document.getElementById('groupWebLlmModel').style.display = (p === 'webllm') ? 'block' : 'none';
  if (attachBtn) attachBtn.style.display = (p === 'webllm') ? 'none' : 'inline-flex';
  if ((p === 'webllm') && pendingImageDataUrl) clearImageChip();
  if (resetFields) {
    cfgModel.value = (p === 'webllm' && !hasManualWebLlmModelChoice()) ? recommendedWebLlmModel() : d.model;
    cfgUrl.value = '';
    cfgKey.value = '';
  }
  if (p === 'webllm') syncWebLlmModel();
  if (p === 'gemini') {
    document.getElementById('modelHint').textContent = 'Example: gemini-2.5-flash, gemini-2.5-pro';
    document.getElementById('urlHint').textContent = ':streamGenerateContent appended automatically';
  } else if (p === 'openai') {
    document.getElementById('modelHint').textContent = 'Example: gpt-4o-mini, gpt-4o, gpt-4.1-mini';
    document.getElementById('urlHint').textContent = '/chat/completions appended automatically -- paste your sk-... key below';
  } else if (p === 'meta') {
    document.getElementById('modelHint').textContent = 'Example: muse-spark-1.2, muse-spark-1.1';
    document.getElementById('urlHint').textContent = 'Meta Model API (dev.meta.ai) -- paste your key from dev.meta.ai/api-keys below';
  } else if (p === 'bazaarlink') {
    document.getElementById('modelHint').textContent = "Use 'auto:free' for free-tier routing, or e.g. openai/gpt-4o, anthropic/claude-...";
    document.getElementById('urlHint').textContent = 'Unified multi-model gateway -- paste your sk-bl-... key below';
  } else if (p === 'ollama') {
    document.getElementById('modelHint').textContent = 'Example: llama3.2, mistral, phi4';
    document.getElementById('urlHint').textContent = 'Run: ollama serve (local machine)';
  } else if (p === 'webllm') {
    const tier = detectDeviceTier();
    const tierNote = hasManualWebLlmModelChoice()
      ? ''
      : (tier === 'low' ? ' Lite mode: a smaller model was auto-selected for your device.'
        : tier === 'high' ? ' Your device looks capable of larger models too.'
        : '');
    document.getElementById('modelHint').textContent = 'Runs 100% in the browser via WebGPU -- no server, no key. First use downloads the model (~0.7-2GB) then caches it.' + tierNote;
  } else {
    document.getElementById('modelHint').textContent = 'Model name per your server';
    document.getElementById('urlHint').textContent = 'Must support OpenAI Chat Completions';
  }
}
let _lastEmittedProvider = cfgProvider.value;
cfgProvider.addEventListener('change', () => {
  const previous = _lastEmittedProvider;
  applyProviderDefaults(true);
  _lastEmittedProvider = cfgProvider.value;
  if (previous !== _lastEmittedProvider) emitAIFaceEvent('providerSwitch', { provider: _lastEmittedProvider, previous });
});
applyProviderDefaults(false);

if (cfgWebLlmModel) {
  cfgWebLlmModel.addEventListener('change', () => {
    markManualWebLlmModelChoice(); // user made an explicit pick -- stop auto-recommending on future provider switches
    if (cfgWebLlmModel.value) {
      cfgWebLlmCustom.style.display = 'none';
      cfgModel.value = cfgWebLlmModel.value;
    } else {
      cfgWebLlmCustom.style.display = 'block';
      cfgModel.value = cfgWebLlmCustom.value.trim();
    }
    saveSettings();
  });
}
if (cfgWebLlmCustom) {
  cfgWebLlmCustom.addEventListener('input', () => {
    markManualWebLlmModelChoice();
    cfgModel.value = cfgWebLlmCustom.value.trim();
  });
}

// ---- Local key protection ----
const SECURE_DB = 'aiface_secure_store';
const SECURE_STORE = 'keys';
function openKeyDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(SECURE_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(SECURE_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function getOrCreateCryptoKey() {
  const db = await openKeyDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(SECURE_STORE, 'readwrite');
    const store = tx.objectStore(SECURE_STORE);
    const getReq = store.get('masterKey');
    getReq.onsuccess = async () => {
      if (getReq.result) { resolve(getReq.result); return; }
      try {
        const key = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
        const putReq = store.put(key, 'masterKey');
        putReq.onsuccess = () => resolve(key);
        putReq.onerror = () => reject(putReq.error);
      } catch (e) { reject(e); }
    };
    getReq.onerror = () => reject(getReq.error);
  });
}
async function encryptText(plain) {
  if (!plain) return '';
  try {
    const key = await getOrCreateCryptoKey();
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const data = new TextEncoder().encode(plain);
    const cipher = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, data);
    const ivB64 = btoa(String.fromCharCode(...iv));
    const dataB64 = btoa(String.fromCharCode(...new Uint8Array(cipher)));
    return 'enc:' + ivB64 + ':' + dataB64;
  } catch (e) {
    dbg('Key encryption unavailable (' + e.message + ') -- falling back to plain storage', 'warn');
    return plain;
  }
}
async function decryptText(stored) {
  if (!stored) return '';
  if (!stored.startsWith('enc:')) return stored;
  try {
    const parts = stored.split(':');
    const iv = Uint8Array.from(atob(parts[1]), c => c.charCodeAt(0));
    const data = Uint8Array.from(atob(parts[2]), c => c.charCodeAt(0));
    const key = await getOrCreateCryptoKey();
    const plainBuf = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, data);
    return new TextDecoder().decode(plainBuf);
  } catch (e) {
    dbg('Key decryption failed: ' + e.message, 'err');
    return '';
  }
}

async function saveSettings() {
  const [key, geminiTtsKey, liveKey] = await Promise.all([
    encryptText(cfgKey.value), encryptText(cfgGeminiTtsKey.value), encryptText(cfgLiveKey.value)
  ]);
  const settings = {
    provider: cfgProvider.value, model: cfgModel.value, key,
    url: cfgUrl.value, temp: cfgTemp.value, maxTokens: cfgMaxTokens.value,
    system: cfgSystem.value, ttsLang: cfgTtsLang.value, autoSend: cfgAutoSend.checked,
    useGeminiTts: cfgUseGeminiTts.checked, geminiTtsKey,
    geminiTtsModel: cfgGeminiTtsModel.value, geminiVoice: cfgGeminiVoice.value,
    liveVoice: cfgLiveVoice.checked, liveKey,
    liveModel: cfgLiveModel.value, liveTranscript: cfgLiveTranscript.checked,
    model3d: cfgModel3d.value, customModelUrl: cfgCustomModelUrl.value,
    waveStyle: cfgWaveStyle.value, theme: cfgTheme.value,
    lightPreset: cfgLightPreset ? cfgLightPreset.value : 'blueprint',
    showFace: cfgShowFace.checked,
    mirror: cfgMirror.checked, mirrorMode: cfgMirrorMode.value,
    feelingMode: cfgFeelingMode ? cfgFeelingMode.value : 'auto',
    micEngine: cfgMicEngine.value, ttsEngine: cfgTtsEngine.value,
    whisperTier: cfgWhisperTier ? cfgWhisperTier.value : 'Xenova/whisper-tiny',
    masterVolume: cfgMasterVolume ? parseInt(cfgMasterVolume.value, 10) : 100,
    outputDevice: cfgOutputDevice ? cfgOutputDevice.value : 'default',
    autoRestoreChat: cfgAutoRestoreChat.checked,
    locale: cfgLocale ? cfgLocale.value : 'en',
    vision: cfgVision ? cfgVision.checked : false,
    visionBackend: cfgVisionBackend ? cfgVisionBackend.value : 'auto',
    visionMaxFps: cfgVisionMaxFps ? cfgVisionMaxFps.value : '8',
    visionReact: cfgVisionReact ? cfgVisionReact.checked : true,
  };
  localStorage.setItem('aiface_llm_settings', JSON.stringify(settings));
  showStatus('Settings saved (keys encrypted locally)', 'ok');
}

async function loadSettings() {
  try {
    const raw = localStorage.getItem('aiface_llm_settings');
    const det = detectLocale();
    if (cfgLocale) cfgLocale.value = det;
    if (!raw) { applyI18n(det); return; }
    const s = JSON.parse(raw);
    cfgProvider.value = s.provider || 'gemini';
    cfgModel.value = s.model || '';
    cfgKey.value = await decryptText(s.key);
    cfgUrl.value = s.url || '';
    cfgTemp.value = s.temp !== undefined ? s.temp : '0.7';
    cfgMaxTokens.value = s.maxTokens !== undefined ? s.maxTokens : '1024';
    cfgSystem.value = s.system || '';
    cfgTtsLang.value = (s.ttsLang && s.ttsLang !== 'ar-SA') ? s.ttsLang : 'en-US';
    cfgAutoSend.checked = s.autoSend !== undefined ? s.autoSend : true;
    cfgUseGeminiTts.checked = s.useGeminiTts !== undefined ? s.useGeminiTts : false;
    geminiTtsFields.style.display = cfgUseGeminiTts.checked ? 'block' : 'none';
    cfgGeminiTtsKey.value = await decryptText(s.geminiTtsKey);
    cfgGeminiTtsModel.value = s.geminiTtsModel || 'gemini-2.5-flash-preview-tts';
    cfgGeminiVoice.value = s.geminiVoice || 'Kore';
    cfgLiveVoice.checked = s.liveVoice !== undefined ? s.liveVoice : false;
    liveVoiceFields.style.display = cfgLiveVoice.checked ? 'block' : 'none';
    cfgLiveKey.value = await decryptText(s.liveKey);
    let liveModel = s.liveModel || 'gemini-2.5-flash-native-audio-preview-12-2025';
    if (liveModel.includes('2.0-flash-live')) liveModel = 'gemini-2.5-flash-native-audio-preview-12-2025';
    cfgLiveModel.value = liveModel;
    cfgLiveTranscript.checked = s.liveTranscript !== undefined ? s.liveTranscript : true;
    cfgModel3d.value = s.model3d || 'facecap';
    cfgCustomModelUrl.value = s.customModelUrl || '';
    customModelGroup.style.display = cfgModel3d.value === 'custom' ? 'block' : 'none';
    cfgWaveStyle.value = s.waveStyle || 'wave';
    cfgTheme.value = s.theme || 'blueprint';
    if (cfgLightPreset) {
      cfgLightPreset.value = s.lightPreset || 'blueprint';
      applyLightPreset(cfgLightPreset.value);
    }
    applyTheme(cfgTheme.value);
    if (cfgLocale) cfgLocale.value = s.locale || detectLocale();
    applyI18n(cfgLocale ? cfgLocale.value : detectLocale());
    cfgShowFace.checked = s.showFace !== undefined ? s.showFace : true;
    cfgMirror.checked = s.mirror !== undefined ? s.mirror : false;
    cfgMirrorMode.value = s.mirrorMode || 'copy';
    if (cfgFeelingMode) cfgFeelingMode.value = s.feelingMode || 'auto';
    if (!cfgMirror.checked) stopMirror();
    updateMirrorUI();
    if (cfgVision) {
      cfgVision.checked = s.vision !== undefined ? s.vision : false;
      if (cfgVisionBackend) cfgVisionBackend.value = s.visionBackend || 'auto';
      if (cfgVisionMaxFps) cfgVisionMaxFps.value = s.visionMaxFps || '8';
      if (cfgVisionReact) cfgVisionReact.checked = s.visionReact !== undefined ? s.visionReact : true;
      visionReact = cfgVisionReact.checked;
      updateVisionUI();
      if (cfgVision.checked) startVision();
      else stopVision();
    }
    cfgMicEngine.value = s.micEngine || 'browser';
    cfgTtsEngine.value = s.ttsEngine || 'browser';
    if (cfgWhisperTier) cfgWhisperTier.value = s.whisperTier || 'Xenova/whisper-tiny';
    setWhisperModel(cfgWhisperTier ? cfgWhisperTier.value : 'Xenova/whisper-tiny');
    updateWhisperUI();
    if (cfgMasterVolume) {
      const mv = s.masterVolume !== undefined ? s.masterVolume : 100;
      cfgMasterVolume.value = Math.min(100, Math.max(0, mv));
      updateMasterVolumeUI();
      saveSettings();
    }
    if (cfgOutputDevice) cfgOutputDevice.value = s.outputDevice || 'default';
    if (cfgAutoRestoreChat) cfgAutoRestoreChat.checked = s.autoRestoreChat !== undefined ? s.autoRestoreChat : true;
    initWaveStyle();
    applyProviderDefaults();
    if (s.key && !s.key.startsWith('enc:')) saveSettings();
  } catch (e) { dbg('Load settings fail: ' + e.message, 'warn'); }
}
loadSettings().then(() => restoreLastSession());
btnSave.addEventListener('click', saveSettings);

function showStatus(text, type) {
  if (!type) type = 'ok';
  connStatus.innerHTML = '<span class="status-pill ' + esc(type) + '">' + esc(text) + '</span>';
}

btnTestVoice.addEventListener('click', () => {
  dbg('Voice test clicked', 'ok');
  const lang = cfgTtsLang.value || 'en-US';
  const testText = 'Hello, this is a voice and lip-sync test. Can you see the mouth moving in sync with the words?';
  speakUnified(testText, lang);
});

btnTest.addEventListener('click', async () => {
  showStatus('Testing...', 'warn');
  const p = cfgProvider.value;
  const url = cfgUrl.value.trim() || DEFAULTS[p].url;
  const model = cfgModel.value.trim() || DEFAULTS[p].model;
  const key = cfgKey.value.trim();
  try {
    if (p === 'gemini') {
      const testUrl = url + '/models/' + model + '?key=' + encodeURIComponent(key);
      const res = await fetch(testUrl, { method: 'GET' });
      if (!res.ok) throw new Error(await streamErrorMsg(res, 'Gemini'));
      const data = await res.json();
      showStatus('Connected -- ' + (data.displayName || model), 'ok');
    } else if (p === 'ollama') {
      const res = await fetch(url + '/api/tags', { method: 'GET' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      const found = data.models ? data.models.find(m => m.name.startsWith(model)) : null;
      showStatus(found ? 'Connected -- ' + found.name : 'Connected -- model ' + model + ' not found', found ? 'ok' : 'warn');
    } else if (p === 'webllm') {
      if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser/device');
      showStatus('WebGPU available -- model will load on first message', 'ok');
    } else {
      const res = await fetch(url + '/models', { headers: key ? { 'Authorization': 'Bearer ' + key } : {} });
      if (!res.ok && res.status !== 404) throw new Error(await streamErrorMsg(res, 'Custom'));
      showStatus('Connected -- Custom API', 'ok');
    }
  } catch (err) {
    showStatus('Failed: ' + err.message, 'err');
    dbg('Connection test failed: ' + err.message, 'err');
  }
});

// ============================================================
// Chat with UX improvements (timestamps, copy, regenerate, scroll-to-bottom)
// ============================================================
function formatTime(d) {
  return d.toTimeString().slice(0, 5);
}

function addMessage(role, text, isError) {
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role + (isError ? ' error' : '');
  const labels = { user: t('chat.you'), assistant: t('chat.ai'), system: t('chat.system'), error: t('chat.error') };
  const roleLabel = labels[role] || role;
  const timeStr = formatTime(new Date());

  div.innerHTML =
    '<div class="role">' + roleLabel + '<span class="msg-timestamp">' + timeStr + '</span></div>' +
    '<div class="text"></div>';
  const textEl = div.querySelector('.text');
  textEl.textContent = typeof text === 'string' ? text : contentToText(text);
  const imgs = typeof text === 'string' ? [] : contentImages(text);
  imgs.slice(0, 1).forEach((url) => {
    const img = document.createElement('img');
    img.className = 'chat-img';
    img.src = url;
    img.alt = '';
    img.draggable = false;
    textEl.appendChild(img);
  });
  chatHistory.appendChild(div);
  chatHistory.scrollTop = chatHistory.scrollHeight;

  // Add action buttons for assistant messages
  if (role === 'assistant' && !isError) {
    const actions = document.createElement('div');
    actions.className = 'msg-actions';

    const copyBtn = document.createElement('button');
    copyBtn.textContent = t('chat.copy');
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(text).then(() => {
        feedback.textContent = t('chat.copied');
        feedback.classList.add('show');
        setTimeout(() => feedback.classList.remove('show'), 1500);
      });
    });

    const feedback = document.createElement('span');
    feedback.className = 'copy-feedback';

    actions.appendChild(copyBtn);
    actions.appendChild(feedback);
    div.appendChild(actions);
  }

  return div.querySelector('.text');
}

function addReplayButton(textEl, fullText, lang) {
  if (!textEl || !textEl.parentElement) return;
  // Check if replay button already exists
  if (textEl.parentElement.querySelector('.msg-actions .replay-btn')) return;
  const actions = textEl.parentElement.querySelector('.msg-actions');
  if (!actions) return;
  const btn = document.createElement('button');
  btn.className = 'replay-btn';
  btn.textContent = t('chat.play', 'Play audio');
  btn.addEventListener('click', () => { speakUnified(fullText, lang); });
  actions.prepend(btn);
}

function addRegenerateButton(textEl) {
  if (!textEl || !textEl.parentElement) return;
  // Remove existing regenerate buttons
  document.querySelectorAll('.regenerate-btn').forEach(b => b.remove());
  const actions = textEl.parentElement.querySelector('.msg-actions');
  if (!actions) return;
  const btn = document.createElement('button');
  btn.className = 'regenerate-btn';
  btn.textContent = t('chat.regenerate');
  btn.addEventListener('click', () => {
    // Find the last user message
    const lastUserMsg = messages.filter(m => m.role === 'user').pop();
    if (lastUserMsg) {
      chatInput.value = contentToText(lastUserMsg.content);
      const imgs = contentImages(lastUserMsg.content);
      if (imgs.length) showImageChip(imgs[0]);
      sendChat();
    }
  });
  actions.appendChild(btn);
}

// ============================================================
// Chat Persistence (IndexedDB sessions)
// ============================================================
function renderSessionMessages(list) {
  chatHistory.innerHTML = '';
  (Array.isArray(list) ? list : []).forEach((m) => {
    if (!m || typeof m !== 'object') return;
    const role = m.role && VALID_ROLES_UI[m.role] ? m.role : null;
    if (!role) return;
    const content = m.content;
    const textEl = addMessage(role, content, role === 'error');
    if (role === 'assistant' && textEl) {
      const replayText = typeof content === 'string' ? content : contentToText(content);
      if (replayText) addReplayButton(textEl, replayText, cfgTtsLang ? cfgTtsLang.value || 'en-US' : 'en-US');
    }
  });
  chatHistory.scrollTop = chatHistory.scrollHeight;
}
const VALID_ROLES_UI = { user: 1, assistant: 1, system: 1, error: 1 };

async function restoreSession(session) {
  if (!session || !session.id) return false;
  try {
    const list = sanitizeMessages(session.messages);
    currentSessionId = session.id;
    messages = list.map((m) => ({ role: m.role, content: m.content }));
    renderSessionMessages(messages);
    dbg('Restored session "' + sessionTitle(messages) + '" (' + messages.length + ' msgs)', 'ok');
    emitAIFaceEvent('sessionUpdate', { reason: 'restore', sessionId: currentSessionId, messageCount: messages.length });
    return true;
  } catch (e) {
    dbg('Restore fail: ' + e.message, 'warn');
    return false;
  }
}

async function saveCurrentSession() {
  if (!messages.length) return;
  try {
    if (!currentSessionId) currentSessionId = makeSessionId();
    await saveSession(buildSession({
      id: currentSessionId,
      provider: cfgProvider.value,
      model: (cfgModel.value || '').trim(),
      messages
    }));
    emitAIFaceEvent('sessionUpdate', { reason: 'save', sessionId: currentSessionId, messageCount: messages.length });
  } catch (e) {
    dbg('Session save fail: ' + e.message, 'warn');
  }
}

async function restoreLastSession() {
  if (cfgAutoRestoreChat && !cfgAutoRestoreChat.checked) return;
  try {
    const last = await getLastSession();
    if (last && Array.isArray(last.messages) && last.messages.length) {
      await restoreSession(last);
    }
  } catch (e) {
    dbg('Auto-restore fail: ' + e.message, 'warn');
  }
}

function newChat() {
  if (abortCtrl) { try { abortCtrl.abort(); } catch (e) {} abortCtrl = null; }
  stopTTS();
  try { setSpeaking(false); } catch (e) {}
  setState('idle');
  fullResponse = '';
  assistantTextEl = null;
  currentSessionId = null;
  messages = [];
  chatHistory.innerHTML = '';
  showStatus('New chat', 'ok');
  dbg('New chat started', 'ok');
  emitAIFaceEvent('sessionUpdate', { reason: 'new', sessionId: null, messageCount: 0 });
}

async function openSessionsModal() {
  await refreshSessionsList();
  const el = document.getElementById('sessionsOverlay');
  if (el) el.classList.add('show');
}

function closeSessionsModal() {
  const el = document.getElementById('sessionsOverlay');
  if (el) el.classList.remove('show');
}

async function refreshSessionsList() {
  const listEl = document.getElementById('sessionsList');
  if (!listEl) return;
  const list = await listSessions();
  if (!list.length) {
    listEl.innerHTML = '<div style="color:var(--dim);padding:10px 0;font-size:12px;">' + t('sessions.empty', 'No saved conversations yet.') + '</div>';
    return;
  }
  listEl.innerHTML = '';
  list.forEach((s) => {
    const row = document.createElement('div');
    row.className = 'session-row';
    const info = document.createElement('div');
    info.className = 'session-info';
    const title = document.createElement('div');
    title.className = 'session-title';
    title.textContent = s.title || t('sessions.untitled', 'Untitled');
    const meta = document.createElement('div');
    meta.className = 'session-meta';
    meta.textContent = new Date(s.updated || Date.now()).toLocaleString() + ' \u00b7 ' + (s.count || 0) + ' msgs';
    info.appendChild(title);
    info.appendChild(meta);
    const actions = document.createElement('div');
    actions.className = 'session-actions';
    const openBtn = document.createElement('button');
    openBtn.className = 'btn';
    openBtn.textContent = t('btn.resume', 'Open');
    openBtn.addEventListener('click', async () => {
      const full = await loadSession(s.id);
      if (!full) return;
      closeSessionsModal();
      const ok = await restoreSession(full);
      if (ok) saveCurrentSession();
    });
    const delBtn = document.createElement('button');
    delBtn.className = 'btn';
    delBtn.textContent = t('btn.delete', 'Delete');
    delBtn.style.borderColor = 'var(--alert)';
    delBtn.style.color = 'var(--alert)';
    delBtn.addEventListener('click', async () => {
      if (currentSessionId === s.id) { currentSessionId = null; messages = []; chatHistory.innerHTML = ''; }
      await deleteSession(s.id);
      refreshSessionsList();
    });
    actions.appendChild(openBtn);
    actions.appendChild(delBtn);
    row.appendChild(info);
    row.appendChild(actions);
    listEl.appendChild(row);
  });
}

if (btnNewChat) btnNewChat.addEventListener('click', newChat);
if (btnSessions) btnSessions.addEventListener('click', openSessionsModal);
document.getElementById('sessionsNew')?.addEventListener('click', () => { closeSessionsModal(); newChat(); });
document.getElementById('sessionsClose')?.addEventListener('click', closeSessionsModal);
document.getElementById('sessionsOverlay')?.addEventListener('click', (e) => {
  if (e.target === document.getElementById('sessionsOverlay')) closeSessionsModal();
});

// Scroll-to-bottom indicator
const scrollBottomWrap = document.getElementById('chatScrollBottom');
const scrollBottomBtn = document.getElementById('chatScrollBtn');
chatHistory.addEventListener('scroll', () => {
  const atBottom = chatHistory.scrollHeight - chatHistory.scrollTop - chatHistory.clientHeight < 30;
  scrollBottomBtn.classList.toggle('visible', !atBottom);
});
scrollBottomBtn.addEventListener('click', () => {
  chatHistory.scrollTop = chatHistory.scrollHeight;
});

async function sendChat() {
  const text = chatInput.value.trim();
  const image = pendingImageDataUrl;
  const content = buildUserContent(text, image);
  if ((typeof content === 'string' && !content) || (Array.isArray(content) && !content.length)) return;
  if (abortCtrl) { return; }
  primeAudio();
  if (abortCtrl) { abortCtrl.abort(); }

  chatInput.value = '';
  clearImageChip();
  addMessage('user', content);
  messages.push({ role: 'user', content });
  saveCurrentSession();

  const fm = cfgFeelingMode ? cfgFeelingMode.value : 'auto';
  if ((fm === 'auto' || fm === 'user') && detectFeeling(text) !== 'neutral') {
    setFeeling(detectFeeling(text), 2200);
  }

  const p = cfgProvider.value;
  const model = cfgModel.value.trim() || DEFAULTS[p].model;
  const key = cfgKey.value.trim();
  const baseUrl = cfgUrl.value.trim() || DEFAULTS[p].url;
  const temp = parseFloat(cfgTemp.value) || 0.7;
  const maxTokens = parseInt(cfgMaxTokens.value) || 1024;
  const system = cfgSystem.value.trim();
  const ttsLang = cfgTtsLang.value || 'en-US';

  setState('thinking');
  setSpeaking(false);
  stopTTS();
  fullResponse = '';
  assistantTextEl = null;
  btnSend.disabled = true;
  micBtn.disabled = true;
  btnStop.style.display = 'inline-block';
  const localCtrl = new AbortController();
  abortCtrl = localCtrl;

  try {
    if (p === 'gemini') {
      await streamGemini(content, model, key, baseUrl, temp, maxTokens, system, abortCtrl.signal);
    } else if (p === 'ollama') {
      await streamOllama(content, model, baseUrl, temp, maxTokens, system, abortCtrl.signal);
    } else if (p === 'webllm') {
      await streamWebLLM(content, model, temp, maxTokens, system, abortCtrl.signal);
    } else {
      await streamCustom(content, model, key, baseUrl, temp, maxTokens, system, abortCtrl.signal);
    }
    messages.push({ role: 'assistant', content: fullResponse });
    saveCurrentSession();
    if (assistantTextEl) {
      addReplayButton(assistantTextEl, fullResponse, ttsLang);
      addRegenerateButton(assistantTextEl);
    }
    if (!S.ttsMuted && fullResponse) {
      dbg('Starting synced TTS for full reply', 'info');
      speakUnified(fullResponse, ttsLang);
    } else {
      setState('idle');
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      showStatus('Generation stopped', 'warn');
      if (fullResponse) messages.push({ role: 'assistant', content: fullResponse });
      saveCurrentSession();
      setState('idle');
      setSpeaking(false);
    } else {
      dbg('Chat error: ' + err.message, 'err');
      addMessage('error', err.message, true);
      setState('alert');
      setSpeaking(false);
      setTimeout(() => setState('idle'), 3000);
    }
  } finally {
    btnSend.disabled = false;
    micBtn.disabled = false;
    btnStop.style.display = 'none';
    if (abortCtrl === localCtrl) abortCtrl = null;
  }
}

// Unified HTTP error -> user-friendly message (with actionable hints + debug log).
// Call: if (!res.ok) throw new Error(await streamErrorMsg(res, 'ProviderName'));
async function streamErrorMsg(res, provider) {
  let msg = provider + ' HTTP ' + res.status;
  try {
    const body = await res.json();
    const detail = body.error?.message || body.error?.status || '';
    if (detail) msg = detail;
    dbg(provider + ' err: ' + JSON.stringify(body), 'err');
  } catch (e) {
    dbg(provider + ' raw: ' + (await res.text().catch(() => '')), 'err');
  }
  let hint = '';
  if (res.status === 401 || res.status === 403) hint = ' -- Check your API Key in Settings';
  else if (res.status === 400 || res.status === 404) hint = ' -- Check model name and Base URL in Settings';
  else if (res.status === 429) hint = ' -- Rate limited, wait a moment';
  return msg + hint;
}

async function consumeSSE(res, onLine) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      onLine(line);
    }
  }
}

async function streamGemini(content, model, key, baseUrl, temp, maxTokens, system, signal) {
  const contents = [];
  if (system) {
    contents.push({ role: 'user', parts: [{ text: 'System instruction: ' + system }] });
    contents.push({ role: 'model', parts: [{ text: 'OK' }] });
  }
  messages.slice(-6).forEach(m => {
    contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts: geminiContentParts(m.content) });
  });
  contents.push({ role: 'user', parts: geminiContentParts(content) });

  const url = baseUrl + '/models/' + model + ':streamGenerateContent?alt=sse&key=' + encodeURIComponent(key);
  dbg('Gemini: ' + url.substring(0, 60) + '...', 'info');
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents, generationConfig: { temperature: temp, maxOutputTokens: maxTokens } }),
    signal
  });
  if (!res.ok) {
    throw new Error(await streamErrorMsg(res, 'Gemini'));
  }

  let firstToken = true;
  await consumeSSE(res, (line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed === 'data: [DONE]') return;
    try {
      const raw = trimmed.startsWith('data: ') ? trimmed.slice(6) : trimmed;
      const json = JSON.parse(raw);
      const cand = Array.isArray(json) ? json[0]?.candidates?.[0] : json.candidates?.[0];
      const chunk = cand?.content?.parts?.[0]?.text || '';
      if (chunk) {
        if (firstToken) {
          firstToken = false;
          assistantTextEl = addMessage('assistant', '');
          dbg('First token', 'ok');
        }
        fullResponse += chunk;
        if (assistantTextEl) assistantTextEl.textContent = fullResponse;
      }
    } catch (e) { dbg('Gemini parse: ' + e.message, 'warn'); }
  });
  dbg('Gemini done', 'ok');
}

async function streamOllama(content, model, baseUrl, temp, maxTokens, system, signal) {
  const msgs = [];
  if (system) msgs.push({ role: 'system', content: system });
  messages.slice(-6).forEach(m => {
    const c = { role: m.role, content: contentToText(m.content) };
    const imgs = contentImages(m.content);
    if (imgs.length) c.images = imgs;
    msgs.push(c);
  });
  const last = { role: 'user', content: contentToText(content) };
  const lastImgs = contentImages(content);
  if (lastImgs.length) last.images = lastImgs;
  msgs.push(last);

  const res = await fetch(baseUrl + '/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages: msgs, stream: true, options: { temperature: temp, num_predict: maxTokens } }),
    signal
  });
  if (!res.ok) throw new Error(await streamErrorMsg(res, 'Ollama'));

  let firstToken = true;
  await consumeSSE(res, (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      const json = JSON.parse(trimmed);
      const chunk = json.message && json.message.content ? json.message.content : '';
      if (chunk) {
        if (firstToken) {
          firstToken = false;
          assistantTextEl = addMessage('assistant', '');
          dbg('First token', 'ok');
        }
        fullResponse += chunk;
        if (assistantTextEl) assistantTextEl.textContent = fullResponse;
      }
    } catch (e) {}
  });
  dbg('Ollama done', 'ok');
}

// ---- WebLLM ----
let webllmEngine = null;
let webllmLoadedModel = null;
async function streamWebLLM(content, model, temp, maxTokens, system, signal) {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser/device -- try the latest Chrome or Edge on a GPU-capable machine.');
  }
  if (contentImages(content).length) {
    throw new Error('WebLLM is text-only -- images are not supported by local models.');
  }
  if (!webllmEngine || webllmLoadedModel !== model) {
    dbg('WebLLM: loading model ' + model + ' (first time only, then cached)', 'info');
    showStatus('Loading local model...', 'warn');
    const webllm = await import('https://esm.run/@mlc-ai/web-llm');
    webllmEngine = await webllm.CreateMLCEngine(model, {
      initProgressCallback: (p) => {
        const pct = p.progress ? Math.round(p.progress * 100) : 0;
        const msgEl = document.getElementById('loadMsg');
        if (msgEl) msgEl.textContent = 'WebLLM: ' + (p.text || (pct + '%'));
        showStatus('Loading local model... ' + pct + '%', 'warn');
        modelProgress(pct, 'WebLLM model');
      }
    });
    webllmLoadedModel = model;
    dbg('WebLLM model ready', 'ok');
    showStatus('Local model ready', 'ok');
    modelProgress(null);
  }

  const msgs = [];
  if (system) msgs.push({ role: 'system', content: system });
  messages.slice(-6).forEach(m => msgs.push({ role: m.role, content: contentToText(m.content) }));
  msgs.push({ role: 'user', content: contentToText(content) });

  const completion = await webllmEngine.chat.completions.create({
    messages: msgs, temperature: temp, max_tokens: maxTokens, stream: true
  });

  let firstToken = true;
  for await (const chunk of completion) {
    if (signal.aborted) break;
    const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta ? chunk.choices[0].delta.content || '' : '';
    if (delta) {
      if (firstToken) {
        firstToken = false;
        assistantTextEl = addMessage('assistant', '');
        dbg('First token (local)', 'ok');
      }
      fullResponse += delta;
      if (assistantTextEl) assistantTextEl.textContent = fullResponse;
    }
  }
  dbg('WebLLM done', 'ok');
}

async function streamCustom(content, model, key, baseUrl, temp, maxTokens, system, signal) {
  const msgs = [];
  if (system) msgs.push({ role: 'system', content: system });
  messages.slice(-6).forEach(m => msgs.push({ role: m.role, content: m.content }));
  msgs.push({ role: 'user', content });

  const headers = { 'Content-Type': 'application/json' };
  if (key) headers['Authorization'] = 'Bearer ' + key;

  const res = await fetch(baseUrl + '/chat/completions', {
    method: 'POST',
    headers,
    body: JSON.stringify({ model, messages: msgs, stream: true, temperature: temp, max_tokens: maxTokens }),
    signal
  });
  if (!res.ok) throw new Error(await streamErrorMsg(res, 'Custom'));

  let firstToken = true;
  await consumeSSE(res, (line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed === 'data: [DONE]') return;
    if (!trimmed.startsWith('data: ')) return;
    try {
      const json = JSON.parse(trimmed.slice(6));
      const chunk = json.choices && json.choices[0] && json.choices[0].delta ? json.choices[0].delta.content || '' : '';
      if (chunk) {
        if (firstToken) {
          firstToken = false;
          assistantTextEl = addMessage('assistant', '');
          dbg('First token', 'ok');
        }
        fullResponse += chunk;
        if (assistantTextEl) assistantTextEl.textContent = fullResponse;
      }
    } catch (e) {}
  });
  dbg('Custom done', 'ok');
}

btnSend.addEventListener('click', sendChat);
chatInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); } });
btnClear.addEventListener('click', () => {
  if (currentSessionId) { deleteSession(currentSessionId).catch(() => {}); currentSessionId = null; }
  messages = [];
  chatHistory.innerHTML = '';
  showStatus('Chat cleared', 'ok');
});
btnStop.addEventListener('click', () => { if (abortCtrl) { abortCtrl.abort(); } });
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 's') { e.preventDefault(); openPanel(); }
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'e') { e.preventDefault(); exportChat(); }
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'n') { e.preventDefault(); newChat(); }
  if (e.ctrlKey && e.key === 'Escape') { e.preventDefault(); closePanel(); }
});

// ============================================================
// Image attachment (multimodal input)
// ============================================================
function resizedImageDataUrl(file, maxDim) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error('Could not decode image'));
      img.onload = () => {
        const scale = Math.min(1, (maxDim || 1024) / Math.max(img.width, img.height));
        const w = Math.max(1, Math.round(img.width * scale));
        const h = Math.max(1, Math.round(img.height * scale));
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        canvas.getContext('2d').drawImage(img, 0, 0, w, h);
        resolve(canvas.toDataURL('image/jpeg', 0.85));
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

function showImageChip(dataUrl) {
  pendingImageDataUrl = dataUrl;
  if (imageChip) {
    imageChip.style.display = 'flex';
    if (imageChipThumb) imageChipThumb.src = dataUrl;
  }
}

function clearImageChip() {
  pendingImageDataUrl = '';
  if (imageChip) imageChip.style.display = 'none';
  if (imageChipThumb) imageChipThumb.src = '';
  if (imageInput) imageInput.value = '';
}

if (attachBtn) attachBtn.addEventListener('click', () => imageInput && imageInput.click());
if (imageInput) imageInput.addEventListener('change', async () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) return;
  try {
    const dataUrl = await resizedImageDataUrl(file, 1024);
    if (dataUrl.length > 3_000_000) { showStatus('Image too large after resize', 'warn'); return; }
    showImageChip(dataUrl);
    dbg('Image attached (' + (dataUrl.length / 1024).toFixed(0) + ' KB source)', 'ok');
  } catch (e) {
    dbg('Image attach fail: ' + e.message, 'err');
    showStatus('Could not read that image', 'warn');
  }
});
if (imageChipRemove) imageChipRemove.addEventListener('click', clearImageChip);

// ============================================================
// Speech-to-Text
// ============================================================
const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null;
let recognizing = false;
let finalTranscript = '';
const micButtons = [micBtn, micFloatBtn].filter(Boolean);

function setMicVisualState(state) {
  micButtons.forEach(btn => {
    btn.classList.remove('listening', 'denied', 'error');
    if (state !== 'idle') btn.classList.add(state);
  });
}

function showCaption(text, label) {
  captionLabel.textContent = label || t('caption.listening');
  captionText.textContent = text || '';
  captionOverlay.classList.add('show');
}
function hideCaption() { captionOverlay.classList.remove('show'); }

if (!SpeechRecognitionCtor) {
  micButtons.forEach(btn => { btn.disabled = true; btn.title = 'Speech recognition not supported in this browser'; });
  dbg('SpeechRecognition NOT AVAILABLE -- mic disabled', 'warn');
} else if (!window.isSecureContext) {
  micButtons.forEach(btn => { btn.disabled = true; btn.title = 'Microphone requires HTTPS -- open this file in a secure browser tab'; });
  dbg('Insecure context -- mic disabled (needs HTTPS or localhost)', 'warn');
} else {
  recognizer = new SpeechRecognitionCtor();
  recognizer.continuous = false;
  recognizer.interimResults = true;
  recognizer.maxAlternatives = 1;

  recognizer.onstart = () => {
    recognizing = true;
    finalTranscript = '';
    setMicVisualState('listening');
    setState('listening');
    showCaption('');
    dbg('Mic: listening started', 'ok');
  };

  recognizer.onresult = (event) => {
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) finalTranscript += transcript;
      else interim += transcript;
    }
    const combined = (finalTranscript + interim).trim();
    chatInput.value = combined;
    showCaption(combined);
  };

  recognizer.onerror = (e) => {
    dbg('Mic error: ' + e.error, 'err');
    const msgs = {
      'not-allowed': 'Microphone permission denied. Open browser settings and allow mic access for this site.',
      'service-not-allowed': 'The browser or embedding frame is blocking the mic. Try opening the file in a standalone browser tab.',
      'audio-capture': 'No microphone found on this device.',
      'network': 'Network error during speech recognition. Check your internet connection.',
      'no-speech': null,
      'aborted': null,
    };
    const isDenied = (e.error === 'not-allowed' || e.error === 'service-not-allowed');
    const friendly = msgs.hasOwnProperty(e.error) ? msgs[e.error] : ('Mic error: ' + e.error);
    if (isDenied) {
      setMicVisualState('denied');
      showCaption(friendly, 'Permission denied');
      setTimeout(() => { setMicVisualState('idle'); hideCaption(); }, 3500);
    } else if (friendly) {
      setMicVisualState('error');
      showCaption(friendly, 'Error');
      setTimeout(() => { setMicVisualState('idle'); hideCaption(); }, 3000);
    } else {
      setMicVisualState('idle');
    }
    if (friendly) addMessage('error', friendly, true);
  };

  recognizer.onend = () => {
    recognizing = false;
    setState('idle');
    dbg('Mic: listening ended', 'ok');
    const text = chatInput.value.trim();
    if (!micButtons.some(btn => btn.classList.contains('denied') || btn.classList.contains('error'))) {
      setMicVisualState('idle');
      hideCaption();
    }
    if (text && cfgAutoSend.checked) {
      sendChat();
    }
  };

  async function toggleMic() {
    userInteracted = true;
    if (recognizing) {
      recognizer.stop();
      return;
    }
    if (localSttActive) {
      stopLocalSTT();
      return;
    }
    stopTTS();
    setSpeaking(false);

    if (cfgMicEngine && cfgMicEngine.value === 'local') {
      startLocalMicFlow();
      return;
    }

    recognizer.lang = cfgTtsLang.value || 'en-US';

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(tr => tr.stop());
        dbg('Mic permission granted', 'ok');
      } catch (err) {
        dbg('Mic permission failed: ' + err.name + ' -- ' + err.message, 'err');
        let hint = 'Could not access microphone (' + err.name + ').';
        let isDenied = false;
        if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
          hint = 'Microphone permission denied, or the current preview is blocking access. Try opening this file in a standalone browser tab.';
          isDenied = true;
        } else if (err.name === 'NotFoundError') {
          hint = 'No microphone found on this device.';
        }
        setMicVisualState(isDenied ? 'denied' : 'error');
        showCaption(hint, isDenied ? 'Permission denied' : 'Error');
        setTimeout(() => { setMicVisualState('idle'); hideCaption(); }, 3500);
        addMessage('error', hint, true);
        return;
      }
    }

    try {
      recognizer.start();
    } catch (e) {
      dbg('Mic start failed: ' + e.message, 'err');
      setMicVisualState('error');
      showCaption('Could not start speech recognition', 'Error');
      setTimeout(() => { setMicVisualState('idle'); hideCaption(); }, 3000);
      addMessage('error', 'Could not start speech recognition: ' + e.message, true);
    }
  }

  let localSttActive = false;
  async function startLocalMicFlow() {
    setMicVisualState('listening');
    setState('listening');
    showCaption('');
    let ok = false;
    try {
      ok = await startLocalSTT(cfgTtsLang.value || 'en-US', {
        onError: (msg) => {
          localSttActive = false;
          setMicVisualState('error');
          showCaption(msg, 'Error');
          addMessage('error', msg, true);
          setState('idle');
          setTimeout(() => { setMicVisualState('idle'); hideCaption(); }, 3500);
        },
        onResult: (text) => {
          chatInput.value = text;
          showCaption(text);
        },
        onEnd: () => {
          localSttActive = false;
          setMicVisualState('idle');
          hideCaption();
          setState('idle');
          const text = chatInput.value.trim();
          if (text && cfgAutoSend.checked) sendChat();
        },
      });
    } catch (e) { ok = false; }
    if (ok) localSttActive = true;
  }

  function onMicTap() {
    primeAudio();
    if (cfgLiveVoice.checked) toggleLiveCall();
    else toggleMic();
  }
  micButtons.forEach(btn => btn.addEventListener('click', onMicTap));

  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && !e.shiftKey && e.key.toLowerCase() === 'm') {
      e.preventDefault();
      onMicTap();
    }
  });
}

// ============================================================
// Live Voice Call
// ============================================================
let liveWs = null;
let liveCallActive = false;
let liveCaptureCtx = null;
let liveCaptureNode = null;
let liveMicStream = null;
let livePlaybackCtx = null;
let liveNextPlayTime = 0;
let liveAnalyser = null;
let liveAnalyserData = null;
let liveAnalyserRAF = null;
let liveAssistantEl = null;
let liveUserEl = null;
let liveTurnHasAudio = false;

function updateMicButtonsMode() {
  const live = cfgLiveVoice.checked;
  micButtons.forEach(btn => {
    btn.title = live ? 'Start/end live voice call (Ctrl+M)' : 'Talk (Ctrl+M)';
    btn.textContent = live && !liveCallActive ? '\u{1F4DE}' : '\u{1F3A4}';
  });
}

function arrayBufferToBase64(buf) {
  let binary = '';
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function floatTo16BitPCM(float32Array) {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function resampleTo16k(float32Array, inRate) {
  if (inRate === 16000) return float32Array;
  const ratio = inRate / 16000;
  const outLength = Math.round(float32Array.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIdx = i * ratio;
    const i0 = Math.floor(srcIdx);
    const i1 = Math.min(i0 + 1, float32Array.length - 1);
    const frac = srcIdx - i0;
    out[i] = float32Array[i0] * (1 - frac) + float32Array[i1] * frac;
  }
  return out;
}

function liveTearDownAudioGraph() {
  if (liveCaptureNode) { try { liveCaptureNode.disconnect(); } catch (e) {} liveCaptureNode = null; }
  if (liveCaptureCtx) { try { liveCaptureCtx.close(); } catch (e) {} liveCaptureCtx = null; }
  if (liveMicStream) { liveMicStream.getTracks().forEach(tr => tr.stop()); liveMicStream = null; }
  if (liveAnalyserRAF) { cancelAnimationFrame(liveAnalyserRAF); liveAnalyserRAF = null; }
  if (livePlaybackCtx) { try { livePlaybackCtx.close(); } catch (e) {} livePlaybackCtx = null; }
  liveAnalyser = null;
  liveNextPlayTime = 0;
}

async function startLiveCall() {
  if (liveCallActive) return;
  userInteracted = true;
  primeAudio();
  stopTTS();
  if (recognizing) recognizer.stop();

  const key = cfgLiveKey.value.trim() || (cfgProvider.value === 'gemini' ? cfgKey.value.trim() : '');
  if (!key) {
    const msg = 'Live voice enabled but no Gemini API key provided. Add one in Voice Settings.';
    showCaption(msg, 'Error'); addMessage('error', msg, true);
    setTimeout(hideCaption, 3500);
    return;
  }
  const model = cfgLiveModel.value.trim() || 'gemini-2.5-flash-native-audio-preview-12-2025';
  const lang = cfgTtsLang.value || 'en-US';
  const system = cfgSystem.value.trim();
  const voice = cfgGeminiVoice.value || 'Kore';

  try {
    liveMicStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (err) {
    dbg('Live call mic permission failed: ' + err.name, 'err');
    const isDenied = (err.name === 'NotAllowedError' || err.name === 'SecurityError');
    const hint = isDenied
      ? 'Microphone permission denied. Try opening this file in a standalone browser tab.'
      : 'Could not access microphone (' + err.name + ').';
    setMicVisualState(isDenied ? 'denied' : 'error');
    showCaption(hint, 'Error'); addMessage('error', hint, true);
    setTimeout(() => { setMicVisualState('idle'); hideCaption(); }, 3500);
    return;
  }

  liveCallActive = true;
  micButtons.forEach(btn => { btn.classList.add('live-call'); btn.textContent = '\u260E'; });
  captionOverlay.classList.add('live-mode');
  showCaption('', t('live.connecting', 'Live call -- connecting...'));
  setState('listening');
  dbg('Live call: connecting WebSocket', 'info');

  const wsUrl = 'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=' + encodeURIComponent(key);
  liveWs = new WebSocket(wsUrl);
  liveWs.binaryType = 'arraybuffer';

  liveWs.onopen = () => {
    dbg('Live call: WebSocket open, sending setup', 'ok');
    const setup = {
      setup: {
        model: 'models/' + model,
        generationConfig: {
          responseModalities: ['AUDIO'],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } }, languageCode: lang }
        },
        systemInstruction: system ? { parts: [{ text: system }] } : undefined,
        inputAudioTranscription: cfgLiveTranscript.checked ? {} : undefined,
        outputAudioTranscription: cfgLiveTranscript.checked ? {} : undefined,
      }
    };
    liveWs.send(JSON.stringify(setup));
    startLiveMicCapture();
  };

  liveWs.onmessage = async (event) => {
    let text = event.data;
    if (text instanceof ArrayBuffer) text = new TextDecoder().decode(text);
    else if (text instanceof Blob) text = await text.text();
    let msg;
    try { msg = JSON.parse(text); } catch (e) { dbg('Live call: bad JSON from server', 'err'); return; }
    handleLiveServerMessage(msg);
  };

  liveWs.onerror = () => {
    dbg('Live call: WebSocket error', 'err');
  };

  liveWs.onclose = (event) => {
    dbg('Live call: WebSocket closed (' + event.code + ')', event.code === 1000 ? 'ok' : 'warn');
    if (liveCallActive) endLiveCall('Voice call ended.' + (event.reason ? ' (' + event.reason + ')' : ''));
  };
}

function startLiveMicCapture() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  liveCaptureCtx = new AudioCtx();
  const source = liveCaptureCtx.createMediaStreamSource(liveMicStream);
  liveCaptureNode = liveCaptureCtx.createScriptProcessor(4096, 1, 1);
  source.connect(liveCaptureNode);
  const silentGain = liveCaptureCtx.createGain();
  silentGain.gain.value = 0;
  liveCaptureNode.connect(silentGain);
  silentGain.connect(liveCaptureCtx.destination);
  liveCaptureNode.onaudioprocess = (e) => {
    if (!liveCallActive || !liveWs || liveWs.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const resampled = resampleTo16k(input, liveCaptureCtx.sampleRate);
    const pcm16 = floatTo16BitPCM(resampled);
    const b64 = arrayBufferToBase64(pcm16.buffer);
    liveWs.send(JSON.stringify({ realtimeInput: { audio: { data: b64, mimeType: 'audio/pcm;rate=16000' } } }));
  };
  dbg('Live call: mic capture streaming', 'ok');
}

function ensureLivePlaybackCtx() {
  if (livePlaybackCtx) return livePlaybackCtx;
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  livePlaybackCtx = new AudioCtx({ sampleRate: 24000 });
  liveAnalyser = livePlaybackCtx.createAnalyser();
  liveAnalyser.fftSize = 256;
  liveAnalyserData = new Uint8Array(liveAnalyser.frequencyBinCount);
  liveAnalyser.connect(livePlaybackCtx.destination);
  liveNextPlayTime = livePlaybackCtx.currentTime;
  return livePlaybackCtx;
}

function driveLiveLipSync() {
  if (!liveAnalyser) { liveAnalyserRAF = null; return; }
  liveAnalyser.getByteTimeDomainData(liveAnalyserData);
  let sumSq = 0;
  for (let i = 0; i < liveAnalyserData.length; i++) {
    const v = (liveAnalyserData[i] - 128) / 128;
    sumSq += v * v;
  }
  const rms = Math.sqrt(sumSq / liveAnalyserData.length);
  S.talkPulse = Math.min(0.5, rms * 3.2);
  S.lastActivityAt = performance.now();
  liveAnalyserRAF = requestAnimationFrame(driveLiveLipSync);
}

function playLiveAudioChunk(base64Data, mimeType) {
  const ctx = ensureLivePlaybackCtx();
  const rateMatch = (mimeType || '').match(/rate=(\d+)/);
  const sampleRate = rateMatch ? parseInt(rateMatch[1], 10) : 24000;
  const pcmBytes = base64ToUint8(base64Data);
  const pcm16 = new Int16Array(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.length / 2);
  const float32 = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) float32[i] = pcm16[i] / 32768;

  const buffer = ctx.createBuffer(1, float32.length, sampleRate);
  buffer.copyToChannel(float32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(liveAnalyser);

  const startAt = Math.max(ctx.currentTime, liveNextPlayTime);
  src.start(startAt);
  liveNextPlayTime = startAt + buffer.duration;

  if (!liveTurnHasAudio) {
    liveTurnHasAudio = true;
    setState('responding');
    setSpeaking(true);
    if (!liveAnalyserRAF) driveLiveLipSync();
  }
}

function stopLiveAudioPlayback() {
  if (livePlaybackCtx) liveNextPlayTime = livePlaybackCtx.currentTime;
  liveTurnHasAudio = false;
  setSpeaking(false);
  setState('listening');
}

function handleLiveServerMessage(msg) {
  if (msg.setupComplete) {
    dbg('Live call: setup complete', 'ok');
    showCaption('', t('live.speaknow', 'Live call -- speak now'));
    setState('listening');
    return;
  }

  const sc = msg.serverContent;
  if (!sc) return;

  if (sc.interrupted) {
    dbg('Live call: interrupted (barge-in)', 'warn');
    stopLiveAudioPlayback();
  }

  if (sc.modelTurn && sc.modelTurn.parts) {
    for (const part of sc.modelTurn.parts) {
      if (part.inlineData && part.inlineData.data) {
        playLiveAudioChunk(part.inlineData.data, part.inlineData.mimeType);
      }
    }
  }

  if (cfgLiveTranscript.checked) {
    if (sc.inputTranscription && sc.inputTranscription.text) {
      if (!liveUserEl) liveUserEl = addMessage('user', '');
      liveUserEl.textContent = (liveUserEl.textContent || '') + sc.inputTranscription.text;
      showCaption(liveUserEl.textContent, 'You are speaking...');
    }
    if (sc.outputTranscription && sc.outputTranscription.text) {
      if (!liveAssistantEl) liveAssistantEl = addMessage('assistant', '');
      liveAssistantEl.textContent = (liveAssistantEl.textContent || '') + sc.outputTranscription.text;
      showCaption(liveAssistantEl.textContent, 'AI is speaking...');
    }
  }

  if (sc.turnComplete) {
    dbg('Live call: turn complete', 'info');
    liveUserEl = null;
    liveAssistantEl = null;
    if (liveAnalyserRAF) { cancelAnimationFrame(liveAnalyserRAF); liveAnalyserRAF = null; }
    liveTurnHasAudio = false;
    setSpeaking(false);
    setState('listening');
  }
}

function endLiveCall(reasonMsg) {
  liveCallActive = false;
  liveTurnHasAudio = false;
  liveUserEl = null;
  liveAssistantEl = null;
  liveTearDownAudioGraph();
  micButtons.forEach(btn => btn.classList.remove('live-call'));
  captionOverlay.classList.remove('live-mode');
  updateMicButtonsMode();
  hideCaption();
  setSpeaking(false);
  setState('idle');
  if (reasonMsg) addMessage('system', reasonMsg);
  dbg('Live call ended', 'info');
}

function toggleLiveCall() {
  if (liveCallActive) {
    if (liveWs) { try { liveWs.close(1000, 'user ended call'); } catch (e) {} liveWs = null; }
    endLiveCall(null);
  } else {
    startLiveCall();
  }
}

updateMicButtonsMode();

// ============================================================
// Visual Style (Wave / Particles / Rings)
// ============================================================
let waveCanvas = null;
let waveCtx = null;
let waveAnimId = null;
let activeWaveStyle = 'none';

function destroyWaveCanvas() {
  if (waveAnimId) cancelAnimationFrame(waveAnimId);
  waveAnimId = null;
  if (waveCanvas) {
    waveCanvas.remove();
    waveCanvas = null;
    waveCtx = null;
  }
  activeWaveStyle = 'none';
}

function initWaveStyle() {
  const style = cfgWaveStyle.value;
  destroyWaveCanvas();
  if (style === 'none') { dbg('Wave style: none', 'info'); return; }
  const stage = document.querySelector('.stage');
  if (!stage) { dbg('Wave: stage not found', 'warn'); return; }
  waveCanvas = document.createElement('canvas');
  waveCanvas.id = 'waveCanvas';
  waveCanvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:10;';
  stage.appendChild(waveCanvas);
  waveCtx = waveCanvas.getContext('2d');
  activeWaveStyle = style;
  resizeWaveCanvas();
  waveTime = 0;
  animateWave();
  dbg('Wave style: ' + style, 'ok');
}

function resizeWaveCanvas() {
  if (!waveCanvas || !waveCanvas.parentElement) return;
  const rect = waveCanvas.parentElement.getBoundingClientRect();
  waveCanvas.width = rect.width;
  waveCanvas.height = rect.height;
}

let waveTime = 0;
function animateWave() {
  if (!waveCanvas || !waveCtx) return;
  waveAnimId = requestAnimationFrame(animateWave);
  const w = waveCanvas.width;
  const h = waveCanvas.height;
  if (w === 0 || h === 0) return;
  waveCtx.clearRect(0, 0, w, h);
  waveTime += 0.016;
  const bass = S.bass || 0;
  const mid = S.mid || 0;
  const treble = S.treble || 0;
  const loud = Math.max(bass, mid, treble);

  if (activeWaveStyle === 'wave') {
    drawAudioWave(w, h, bass, mid, treble, loud);
  } else if (activeWaveStyle === 'particles') {
    drawParticles(w, h, bass, mid, treble, loud);
  } else if (activeWaveStyle === 'rings') {
    drawPulseRings(w, h, bass, mid, treble, loud);
  }
}

function drawAudioWave(w, h, bass, mid, treble, loud) {
  waveCtx.lineWidth = 2 + S.beatPulse * 3;
  waveCtx.shadowBlur = 12 + S.beatPulse * 20;
  waveCtx.shadowColor = 'rgba(125,211,252,0.5)';
  waveCtx.beginPath();
  const amp = 30 + loud * 80;
  for (let x = 0; x < w; x++) {
    const t = x / w;
    const y = h / 2 + Math.sin(t * 6.28 + waveTime * 2) * amp * (0.5 + bass * 0.5)
                        + Math.sin(t * 12.56 + waveTime * 3.5) * amp * 0.3 * (mid);
    x === 0 ? waveCtx.moveTo(x, y) : waveCtx.lineTo(x, y);
  }
  waveCtx.strokeStyle = `rgba(125,211,252,${0.4 + loud * 0.6})`;
  waveCtx.stroke();
  waveCtx.shadowBlur = 0;
}

function drawParticles(w, h, bass, mid, treble, loud) {
  const count = 120;
  waveCtx.shadowBlur = 6;
  for (let i = 0; i < count; i++) {
    const seed = i * 137.508;
    const px = (Math.sin(seed) * 0.5 + 0.5) * w;
    const py = (Math.cos(seed * 0.7) * 0.5 + 0.5) * h;
    const drift = Math.sin(waveTime + seed) * (20 + bass * 40);
    const driftY = Math.cos(waveTime * 0.8 + seed) * (10 + mid * 30);
    const r = 1.5 + treble * 4 + Math.sin(waveTime + i) * 1 + S.beatPulse * 3;
    waveCtx.beginPath();
    waveCtx.arc(px + drift, py + driftY, r, 0, Math.PI * 2);
    waveCtx.fillStyle = `rgba(125,211,252,${0.2 + loud * 0.5})`;
    waveCtx.fill();
  }
  waveCtx.shadowBlur = 0;
}

function drawPulseRings(w, h, bass, mid, treble, loud) {
  const cx = w / 2;
  const cy = h / 2;
  const maxR = Math.min(w, h) * 0.45;
  for (let i = 0; i < 5; i++) {
    const phase = waveTime * 1.2 + i * 1.256;
    const pulse = (Math.sin(phase) * 0.5 + 0.5);
    const r = maxR * (0.15 + pulse * 0.85) * (1 + bass * 0.3);
    waveCtx.beginPath();
    waveCtx.arc(cx, cy, r, 0, Math.PI * 2);
    waveCtx.strokeStyle = `rgba(125,211,252,${0.08 + loud * 0.25 + S.beatPulse * 0.3})`;
    waveCtx.lineWidth = 1.5 + mid * 2 + S.beatPulse * 4;
    waveCtx.stroke();
  }
}

cfgWaveStyle.addEventListener('change', () => {
  initWaveStyle();
  saveSettings();
});

cfgModel3d.addEventListener('change', () => {
  customModelGroup.style.display = cfgModel3d.value === 'custom' ? 'block' : 'none';
  if (cfgModel3d.value === 'facecap' && modelNameEl) modelNameEl.textContent = 'FaceCap (Classic)';
  saveSettings();
});
cfgCustomModelUrl.addEventListener('change', saveSettings);

// ---- Fetch models/manifest.json + scan models/ folder to populate the dropdown ----
const cfgLocalModels = document.getElementById('cfgLocalModels');
function addModelOption(url, label) {
  if (!url) return;
  for (const opt of cfgLocalModels.options) {
    if (opt.value === url) return;
  }
  const opt = document.createElement('option');
  opt.value = url;
  opt.textContent = label;
  cfgLocalModels.appendChild(opt);
}
function titleCaseFilename(name) {
  return name.replace(/\.glb$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, c => c.toUpperCase());
}
const CACHE_BUST = '?v=' + Date.now();
if (cfgLocalModels) {
  fetch('./models/manifest.json' + CACHE_BUST, { cache: 'no-store' })
    .then(r => r.json())
    .then(data => {
      (data.models || []).forEach(m => addModelOption(m.url, m.name));
      dbg('Loaded ' + (data.models || []).length + ' model(s) from manifest', 'ok');
    })
    .catch(() => dbg('No models/manifest.json found', 'warn'))
    .finally(() => {
      // Scan the models/ folder and append any .glb not already listed.
      fetch('./models/' + CACHE_BUST, { cache: 'no-store' })
        .then(res => res.text())
        .then(html => {
          const re = /href="([^"]+\.glb)"/gi;
          const seen = {};
          for (const opt of cfgLocalModels.options) seen[opt.value] = true;
          let added = 0;
          let match;
          while ((match = re.exec(html))) {
            let name = match[1];
            const url = './models/' + name;
            if (seen[url]) continue;
            if (name.includes('/')) {
              const parts = name.split('/');
              name = parts[parts.length - 1];
            }
            addModelOption(url, titleCaseFilename(name));
            seen[url] = true;
            added++;
          }
          if (added > 0) dbg('Folder scan added ' + added + ' model(s) from ./models/', 'ok');
        })
        .catch(() => dbg('No directory listing for ./models/; using manifest only', 'warn'));
    });

  cfgLocalModels.addEventListener('change', () => {
    const url = cfgLocalModels.value;
    if (!url) return;
    cfgCustomModelUrl.value = url;
    cfgModel3d.value = 'custom';
    customModelGroup.style.display = 'block';
    if (modelNameEl) modelNameEl.textContent = cfgLocalModels.options[cfgLocalModels.selectedIndex].textContent;
    loadModel(url, null, () => {
      return makeMatPair();
    }).catch(e => dbg('Model load failed: ' + e.message, 'err'));
    saveSettings();
  });
}

cfgTheme.addEventListener('change', () => {
  applyTheme(cfgTheme.value);
  saveSettings();
});

if (cfgLocale) {
  cfgLocale.addEventListener('change', () => {
    applyI18n(cfgLocale.value);
    saveSettings();
  });
}

function applyTheme(name) {
  document.documentElement.className = name === 'blueprint' ? '' : 'theme-' + name;
  dbg('Theme: ' + name, 'ok');
}
applyTheme(cfgTheme.value);

// ============================================================
// Data Panel Toggle
// ============================================================
const dataPanel = document.getElementById('dataPanel');
const toggleDataBtn = document.getElementById('toggleDataBtn');
const closeDataBtn = document.getElementById('closeDataBtn');
let dataPanelVisible = true;

const svgBars = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="12" width="4" height="9" rx="1"/><rect x="10" y="7" width="4" height="14" rx="1"/><rect x="17" y="3" width="4" height="18" rx="1"/></svg>';
const svgClose = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';

function toggleDataPanel(show) {
  dataPanelVisible = (show !== undefined) ? show : !dataPanelVisible;
  dataPanel.classList.toggle('collapsed', !dataPanelVisible);
  toggleDataBtn.classList.toggle('hidden', dataPanelVisible);
  toggleDataBtn.innerHTML = dataPanelVisible ? svgClose : svgBars;
}

toggleDataBtn.addEventListener('click', () => toggleDataPanel());
closeDataBtn.addEventListener('click', () => toggleDataPanel(false));

// ---- Audio analyser for waveform + FFT ----
let audioAnalyser = null;
let audioDataArray = null;
let audioFreqArray = null;

function setupAudioAnalyser(stream) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);
  audioAnalyser = analyser;
  audioDataArray = new Uint8Array(analyser.frequencyBinCount);
  audioFreqArray = new Uint8Array(analyser.frequencyBinCount);
  const gain = ctx.createGain();
  gain.gain.value = 0;
  analyser.connect(gain);
  gain.connect(ctx.destination);
}

function extractBands() {
  const src = audioAnalyser || LocalSpeech.analyser || null;
  if (!src || !audioFreqArray) { S.bass = 0; S.mid = 0; S.treble = 0; return; }
  if (audioFreqArray.length !== src.frequencyBinCount) {
    audioDataArray = new Uint8Array(src.frequencyBinCount);
    audioFreqArray = new Uint8Array(src.frequencyBinCount);
  }
  src.getByteFrequencyData(audioFreqArray);
  const bins = audioFreqArray.length;
  const sampleRate = src.context.sampleRate;
  const binHz = sampleRate / (src.fftSize);
  let bassSum = 0, bassCount = 0;
  let midSum = 0, midCount = 0;
  let trebSum = 0, trebCount = 0;
  for (let i = 0; i < bins; i++) {
    const hz = i * binHz;
    const v = audioFreqArray[i] / 255;
    if (hz < 250) { bassSum += v; bassCount++; }
    else if (hz < 2000) { midSum += v; midCount++; }
    else if (hz < 8000) { trebSum += v; trebCount++; }
  }
  S.bass = bassCount ? bassSum / bassCount : 0;
  S.mid = midCount ? midSum / midCount : 0;
  S.treble = trebCount ? trebSum / trebCount : 0;
}

// ---- Beat detection ----
let beatThreshold = 0.55;
let beatDecay = 0.92;
let beatEnergy = 0;
let beatCooldown = 0;
S.beat = false;
S.beatPulse = 0;

function detectBeat() {
  S.beat = false;
  if (!audioAnalyser) return;
  const energy = S.bass * 0.6 + S.mid * 0.3 + S.treble * 0.1;
  beatEnergy = beatEnergy * beatDecay + energy * (1 - beatDecay);
  if (beatCooldown > 0) { beatCooldown--; return; }
  if (energy > beatThreshold && energy > beatEnergy * 1.3) {
    S.beat = true;
    S.beatPulse = 1;
    beatCooldown = 4;
    dbg('Beat detected', 'ok');
  }
}

const originalStartLiveCall = startLiveCall;
startLiveCall = async function () {
  await originalStartLiveCall.call(this);
  if (liveMicStream) {
    setupAudioAnalyser(liveMicStream);
  }
};

const originalEndLiveCall = endLiveCall;
endLiveCall = function (reasonMsg) {
  originalEndLiveCall.call(this, reasonMsg);
  audioAnalyser = null;
  audioDataArray = null;
};

// ---- Data panel update (always live) ----
function updateDataPanelMetrics() {
  extractBands();
  detectBeat();
  const src = audioAnalyser || LocalSpeech.analyser || null;
  const speechActive = S.speaking || S.talkPulse > 0.005;
  const n = performance.now();
  const amp = S.talkPulse;

  // Browser TTS has no analyser (speechSynthesis audio can't be tapped by Web
  // Audio), so synthesize believable live readings from speech activity.
  let bass = S.bass, mid = S.mid, treble = S.treble;
  if (src) {
    // Real FFT values already in S.bass/mid/treble from extractBands().
  } else if (speechActive) {
    const I = Math.min(1, S.phonemeIntensity * 2.6 + S.talkPulse * 1.4);
    bass = I * (0.42 + 0.22 * Math.abs(Math.sin(n / 1000 * 6)));
    mid = I * (0.5 + 0.2 * Math.sin(n / 1000 * 11));
    treble = I * (0.18 + 0.15 * Math.sin(n / 1000 * 17));
  } else {
    // Gentle idle "breathing" so the panel is never a dead flat readout.
    const breath = 0.5 + 0.3 * Math.sin(n / 1000 * 1.4);
    bass = 0.055 * breath;
    mid = 0.05 * breath;
    treble = 0.03 * (0.5 + 0.5 * breath);
  }
  const energy = (bass + mid + treble) / 3 * 100;
  const stability = Math.max(0, 100 - (Math.abs(bass - treble) * 80));
  const peakBand = bass > mid && bass > treble ? 'BASS' : mid > treble ? 'MID' : 'TREBLE';
  document.getElementById('amplitude-value').textContent = amp.toFixed(3);
  document.getElementById('energy-value').textContent = energy.toFixed(1) + ' J';
  document.getElementById('stability-value').textContent = stability.toFixed(0) + '%';
  document.getElementById('peak-value').textContent = peakBand;

  document.querySelectorAll('.data-value').forEach(el => el.classList.remove('no-input'));

  if (src) {
    if (!audioDataArray || audioDataArray.length !== src.frequencyBinCount) audioDataArray = new Uint8Array(src.frequencyBinCount);
    src.getByteTimeDomainData(audioDataArray);
    const normalized = new Float32Array(audioDataArray.length);
    for (let i = 0; i < audioDataArray.length; i++) {
      normalized[i] = (audioDataArray[i] - 128) / 128;
    }
    updateWaveform(normalized);
  } else if (speechActive) {
    // Soft animated pulse derived from speech activity (no analyser available).
    const w = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      w[i] = Math.sin(i / 64 * Math.PI * 6 + n / 1000 * 8) * (0.25 + S.phonemeIntensity * 1.4) + (Math.random() * 0.08 - 0.04);
    }
    updateWaveform(w);
  } else {
    // Subtle idle breathing line.
    const w = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      w[i] = Math.sin(i / 64 * Math.PI * 2 + n / 1000 * 1.2) * (0.06 + 0.03 * Math.sin(n / 1000 * 0.8));
    }
    updateWaveform(w);
  }
  drawWaveform();
}

let _lastDataUpdate = 0;
function dataPanelLoop(now) {
  requestAnimationFrame(dataPanelLoop);
  if (now - _lastDataUpdate < 50) return;
  _lastDataUpdate = now;
  updateDataPanelMetrics();
}
requestAnimationFrame(dataPanelLoop);

// ============================================================
// Export Chat
// ============================================================
function exportChat() {
  const rows = document.querySelectorAll('.chat-msg');
  if (!rows.length) { showStatus('No chat to export', 'warn'); return; }
  let md = '# AI Face v6 Chat Export\n\n';
  rows.forEach(row => {
    const role = row.classList.contains('user') ? '**You**' : '**AI**';
    const text = row.textContent.trim();
    md += role + ': ' + text + '\n\n';
  });
  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'aiface-chat-' + new Date().toISOString().slice(0,10) + '.md';
  a.click();
  URL.revokeObjectURL(url);
  showStatus('Chat exported as Markdown', 'ok');
  dbg('Chat exported', 'ok');
}

// ============================================================
// Keyboard Shortcuts Overlay (? key)
// ============================================================
function showShortcutsOverlay() {
  let overlay = document.getElementById('shortcutsOverlay');
  if (overlay) { overlay.remove(); return; }
  overlay = document.createElement('div');
  overlay.id = 'shortcutsOverlay';
  overlay.style.cssText = 'position:fixed;inset:0;z-index:300;background:rgba(0,0,0,0.8);display:flex;align-items:center;justify-content:center;backdrop-filter:blur(8px);';
  overlay.innerHTML = `
    <div style="background:var(--panel-bg);border:1px solid var(--panel-border);border-radius:var(--radius);padding:28px 32px;max-width:360px;width:90vw;font-family:'TheGoodMonolith',monospace;color:var(--text-primary);font-size:13px;">
      <h3 style="margin:0 0 16px;font-size:15px;color:var(--active);font-family:'Orbitron',monospace;letter-spacing:0.1em;">KEYBOARD SHORTCUTS</h3>
      <div style="display:grid;grid-template-columns:auto 1fr;gap:6px 16px;">
        <kbd style="color:var(--accent-tertiary);">Ctrl+Shift+D</kbd><span>Toggle debug log</span>
        <kbd style="color:var(--accent-tertiary);">Ctrl+Shift+S</kbd><span>Open settings</span>
        <kbd style="color:var(--accent-tertiary);">Ctrl+Shift+E</kbd><span>Export chat</span>
        <kbd style="color:var(--accent-tertiary);">Ctrl+M</kbd><span>Toggle microphone</span>
        <kbd style="color:var(--accent-tertiary);">Ctrl+Esc</kbd><span>Close settings</span>
        <kbd style="color:var(--accent-tertiary);">?</kbd><span>Show this overlay</span>
      </div>
      <p style="margin:16px 0 0;font-size:11px;color:var(--text-secondary);text-align:center;">Press ? or click outside to close</p>
    </div>`;
  overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
}

document.addEventListener('keydown', (e) => {
  if (e.key === '?' && !e.ctrlKey && !e.shiftKey && !e.altKey && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
    showShortcutsOverlay();
  }
});

// ============================================================
// Load-error banner (CDN / module failures → visible message + Retry)
// ============================================================
function showLoadError(msg) {
  const banner = document.getElementById('loadErrorBanner');
  const msgEl = document.getElementById('loadErrorMsg');
  if (!banner || !msgEl) return;
  msgEl.textContent = (msg || 'Something failed to load.').slice(0, 300);
  banner.hidden = false;
  const loadMsg = document.getElementById('loadMsg');
  if (loadMsg) loadMsg.style.display = 'none';
}

document.getElementById('loadErrorRetry').addEventListener('click', () => location.reload());
document.getElementById('loadErrorDismiss').addEventListener('click', () => {
  document.getElementById('loadErrorBanner').hidden = true;
});

window.addEventListener('error', (e) => {
  const src = (e.filename || '') + (e.lineno != null ? ':' + e.lineno : '');
  const mightBeNetwork = /unpkg\.com|jsdelivr\.net|esm\.run|\.js|Script error|Failed to fetch/i.test((e.message || '') + ' ' + src);
  if (mightBeNetwork) showLoadError('Failed to load a resource (' + (e.message || 'unknown error') + '). Check connection and Retry.');
});

window.addEventListener('unhandledrejection', (e) => {
  const msg = (e.reason && (e.reason.message || String(e.reason))) || 'unknown error';
  if (/fetch|network|load|import|cdn|unreachable|failed/i.test(msg)) {
    showLoadError('A network/resource load failed: ' + msg.slice(0, 200) + '. Check connection and Retry.');
  }
});

dbg('LLM Bridge ready', 'ok');

initScene().catch((err) => {
  dbg('Scene init failed: ' + err.message, 'err');
  const msg = document.getElementById('loadMsg');
  if (msg) { msg.style.display = 'flex'; msg.textContent = 'Failed to start -- open Console (F12)'; }
  showLoadError('Could not start the 3D engine: ' + err.message + '. Check connection and Retry.');
}).finally(() => { try { window.__APP_STATIC_READY = 1; } catch (e) {} });
