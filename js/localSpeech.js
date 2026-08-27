// ============================================================
// AI Face v6 — On-device Speech Engines
// STT:  Whisper (transformers.js, multilingual) — audio never leaves the device
// TTS:  Kokoro (English, high quality) / Meta MMS (Arabic, basic)
// A tiny analyser node is exposed so the app's audio wave + bands keep working.
// ============================================================

export const LocalSpeech = {
  sttReady: false,
  sttLoading: false,
  ttsReady: false,
  ttsLoading: false,
  analyser: null,
};

import { modelProgress } from './progress.js';
import { attachMasterGain, routeOutput, getMasterVolume, getOutputDevice } from './masterBus.js';

// transformers.js progress events → HUD progress overlay.
function wireTfProgress(env, prefix) {
  if (!env) return;
  const handler = (p) => {
    if (!p) return;
    if (p.status === 'done' || p.status === 'ready' || p.status === 'complete') { modelProgress(null); return; }
    let pct = null;
    if (typeof p.progress === 'number') pct = p.progress;            // 0-100
    else if (typeof p.percent === 'number') pct = p.percent * 100;   // 0-1
    else if (p.loaded != null && p.total) pct = (p.loaded / p.total) * 100;
    if (pct != null) modelProgress(pct, prefix + ' ' + (p.file || p.name || 'model'));
  };
  env.progress = handler;
}

let asrPipeline = null;
let mmsPipeline = null;
let kokoro = null;
let whisperModel = 'Xenova/whisper-tiny';

// Switches the on-device Whisper repo id. If a pipeline is already loaded it is
// dropped so the next STT session reloads with the new (larger/smaller) model.
export function setWhisperModel(repoId) {
  if (typeof repoId !== 'string' || !repoId) return;
  whisperModel = repoId;
  if (asrPipeline) {
    asrPipeline = null;
    LocalSpeech.sttReady = false;
  }
}

let mediaRecorder = null;
let recChunks = [];
let recStream = null;
let recLang = 'en';
let recCallbacks = null;

let audioCtx = null;
let currentSource = null;
let masterGainNode = null;
let visemeRaf = 0;
let playStart = 0;
let playText = '';
let lastVisemeIdx = -1;
let onWordCb = null;
let onEndCb = null;
let playStopped = false;

// ---- STT: Whisper ----
async function loadAsr() {
  if (asrPipeline) return true;
  if (LocalSpeech.sttLoading) {
    while (LocalSpeech.sttLoading) await new Promise((r) => setTimeout(r, 250));
    return !!asrPipeline;
  }
  LocalSpeech.sttLoading = true;
  try {
    const tf = await import('@huggingface/transformers');
    wireTfProgress(tf.env, 'Whisper');
    asrPipeline = await tf.pipeline('automatic-speech-recognition', whisperModel, {
      dtype: { encoder_model: 'fp32', decoder_model_merged: 'q8' },
    });
    LocalSpeech.sttReady = true;
    return true;
  } catch (e) {
    console.warn('On-device Whisper failed to load:', e);
    return false;
  } finally {
    LocalSpeech.sttLoading = false;
    modelProgress(null);
  }
}

const BCP47_TO_WHISPER = { 'en-US': 'en', 'en-GB': 'en', 'ar-SA': 'ar', 'fr-FR': 'fr', 'de-DE': 'de', 'es-ES': 'es', 'ja-JP': 'ja' };
export function toWhisperLang(code) {
  return BCP47_TO_WHISPER[code] || 'en';
}

export function startLocalSTT(lang, callbacks) {
  return new Promise(async (resolve) => {
    const ok = await loadAsr();
    if (!ok) { if (callbacks.onError) callbacks.onError('Could not load on-device Whisper. Use the Browser microphone engine instead.'); resolve(false); return; }
    try {
      recStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    } catch (e) {
      if (callbacks.onError) callbacks.onError('Microphone permission denied.');
      resolve(false);
      return;
    }
    recLang = toWhisperLang(lang);
    recCallbacks = callbacks;
    recChunks = [];
    const prefs = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
    const mime = prefs.find((m) => MediaRecorder.isTypeSupported(m)) || '';
    try {
      mediaRecorder = new MediaRecorder(recStream, mime ? { mimeType: mime } : undefined);
    } catch (e) {
      mediaRecorder = new MediaRecorder(recStream);
    }
    mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size) recChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      recStream.getTracks().forEach((t) => t.stop());
      recStream = null;
      try {
        const blob = new Blob(recChunks, { type: mime || 'audio/webm' });
        const f32 = await blobToF32(blob);
        const text = await transcribeF32(f32);
        if (recCallbacks && recCallbacks.onResult) recCallbacks.onResult(text);
      } catch (e) {
        console.warn('Local transcription failed:', e);
        if (recCallbacks && recCallbacks.onError) recCallbacks.onError('Local transcription failed.');
      } finally {
        const cb = recCallbacks;
        recCallbacks = null;
        if (cb && cb.onEnd) cb.onEnd();
      }
    };
    mediaRecorder.start(250);
    resolve(true);
  });
}

export function stopLocalSTT() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    try { mediaRecorder.stop(); } catch (e) {}
  }
}

async function blobToF32(blob) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
  ctx.close();
  const ch = buf.getChannelData(0);
  const out = new Float32Array(ch.length);
  out.set(ch);
  return { data: out, sampleRate: buf.sampleRate };
}

async function to16k(data, sampleRate) {
  if (sampleRate === 16000) return data;
  const AC = window.AudioContext || window.webkitAudioContext;
  const oac = new OfflineAudioContext(1, Math.ceil(data.length * 16000 / sampleRate), 16000);
  const buf = oac.createBuffer(1, data.length, sampleRate);
  buf.copyToChannel(data, 0);
  const src = oac.createBufferSource();
  src.buffer = buf;
  src.connect(oac.destination);
  src.start();
  const rendered = await oac.startRendering();
  return rendered.getChannelData(0);
}

async function transcribeF32(input) {
  const audio16 = await to16k(input.data, input.sampleRate);
  const out = await asrPipeline(audio16, { language: recLang, chunk_length_s: 30 });
  return (out.text || '').trim();
}

// ---- TTS engines ----
async function loadKokoro() {
  if (kokoro) return true;
  try {
    modelProgress(-1, 'Preparing Kokoro voice');
    const { KokoroTTS } = await import('kokoro-js');
    kokoro = new KokoroTTS();
    await kokoro.ready;
    modelProgress(null);
    return true;
  } catch (e) {
    console.warn('Kokoro failed to load:', e);
    modelProgress(null);
    return false;
  }
}

async function loadMMS() {
  if (mmsPipeline) return true;
  const tf = await import('@huggingface/transformers');
  wireTfProgress(tf.env, 'MMS');
  mmsPipeline = await tf.pipeline('text-to-speech', 'Xenova/mms-tts-ara');
  LocalSpeech.ttsReady = true;
  modelProgress(null);
  return true;
}

// Returns { audio: Float32Array, sampling_rate: number }. Throws when language unsupported.
export async function generateLocalAudio(text, lang) {
  if (!lang || lang.startsWith('en')) {
    if (!await loadKokoro()) throw new Error('Kokoro unavailable');
    return kokoro.generate(text, { voiceId: 'af_heart' });
  }
  if (lang.startsWith('ar')) {
    await loadMMS();
    const out = await mmsPipeline(text);
    return { audio: out.audio, sampling_rate: out.sampling_rate };
  }
  throw new Error('Local voice not available for this language');
}

// ---- Playback + analyser ----
export function setLocalCallbacks(cb) {
  onWordCb = (cb && cb.onWord) || null;
  onEndCb = (cb && cb.onEnd) || null;
}

// Live updates from the settings UI: retunes the current playback gain and
// re-routes the context when the user picks a different output device.
export function applyMasterSettings() {
  if (masterGainNode && masterGainNode.gain) masterGainNode.gain.value = getMasterVolume();
  if (audioCtx) routeOutput(audioCtx);
}

export function playLocalAudio(text, res) {
  stopLocalAudio();
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const buf = audioCtx.createBuffer(1, res.audio.length, res.sampling_rate);
  buf.copyToChannel(res.audio, 0);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  LocalSpeech.analyser = analyser;
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  masterGainNode = attachMasterGain(audioCtx, audioCtx.destination);
  src.connect(analyser);
  analyser.connect(masterGainNode);
  currentSource = src;
  playStart = audioCtx.currentTime;
  playText = text;
  lastVisemeIdx = -1;
  routeOutput(audioCtx);

  src.onended = () => {
    cancelAnimationFrame(visemeRaf);
    visemeRaf = 0;
    currentSource = null;
    LocalSpeech.analyser = null;
    if (onEndCb) onEndCb();
  };

  const startNow = () => {
    if (playStopped) return;
    playStopped = false;
    try { src.start(); } catch (e) {}
    driveVisemes();
  };

  // Chrome autoplay policy starts a lazily-created AudioContext suspended when
  // it's created outside a user gesture; resume explicitly or nothing plays.
  if (audioCtx.state === 'suspended') {
    audioCtx.resume().then(startNow).catch(startNow);
  } else {
    startNow();
  }
}

function driveVisemes() {
  visemeRaf = requestAnimationFrame(function tick() {
    visemeRaf = requestAnimationFrame(tick);
    if (!currentSource || !audioCtx) return;
    if (audioCtx.state === 'suspended') { try { audioCtx.resume(); } catch (e) {} }
    const elapsed = audioCtx.currentTime - playStart;
    const dur = (currentSource.buffer && currentSource.buffer.duration) || 0;
    if (dur > 0 && playText && onWordCb) {
      const totalChars = playText.length;
      const idx = Math.max(0, Math.min(totalChars - 1, Math.floor((elapsed / dur) * totalChars)));
      if (idx !== lastVisemeIdx) {
        lastVisemeIdx = idx;
        onWordCb(idx);
      }
    }
  });
}

export function stopLocalAudio() {
  cancelAnimationFrame(visemeRaf);
  visemeRaf = 0;
  playStopped = true;
  if (currentSource) { try { currentSource.stop(); } catch (e) {} currentSource = null; }
  LocalSpeech.analyser = null;
}