// ============================================================
// AI Face v6 — Live Emotion Mirror
// MediaPipe FaceLandmarker running fully on-device (WASM/GPU).
// The avatar can copy your expressions live, or react to your
// detected emotion while keeping the normal idle motion.
// ============================================================

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { detectEmotion } from './pure.js';

const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export const Mirror = {
  ready: false,
  active: false,
  hasFace: false,
  weights: null,      // last blendshape scores, e.g. { eyeBlinkLeft: 0.9, jawOpen: 0.3 }
  emotion: 'neutral', // happy | upset | surprised | curious | neutral
  lastDetect: 0,
};

let landmarker = null;
let video = null;
let stream = null;
let rafId = 0;
let lastRun = 0;

function setPanelPill(state) {
  const el = document.getElementById('mirrorPill');
  if (!el) return;
  el.textContent = state === 'on' ? 'LIVE' : (state === 'off' ? 'OFF' : 'ERR');
  el.className = 'mirror-pill' + (state === 'on' ? ' on' : (state === 'err' ? ' err' : ''));
}

async function createLandmarker(delegate) {
  const vision = await FilesetResolver.forVisionTasks(CDN + '/wasm');
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    outputFaceBlendshapes: true,
    runningMode: 'VIDEO',
    numFaces: 1,
  });
}

async function ensureEngine() {
  if (landmarker) return true;
  try {
    landmarker = await createLandmarker('GPU');
  } catch (e) {
    console.warn('FaceLandmarker GPU init failed, falling back to CPU:', e);
    try {
      landmarker = await createLandmarker('CPU');
    } catch (e2) {
      console.warn('FaceLandmarker init failed:', e2);
      return false;
    }
  }
  Mirror.ready = true;
  return true;
}

export async function startMirror() {
  if (Mirror.active) return true;
  if (!await ensureEngine()) { setPanelPill('err'); return false; }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    });
  } catch (e) {
    console.warn('Camera denied:', e);
    setPanelPill('err');
    return false;
  }

  if (!video) {
    video = document.createElement('video');
    video.muted = true;
    video.setAttribute('playsinline', '');
    video.setAttribute('autoplay', '');
    const panel = document.getElementById('mirrorPanel');
    (panel || document.body).appendChild(video);
  }
  video.srcObject = stream;
  try { await video.play(); } catch (e) {}

  const panel = document.getElementById('mirrorPanel');
  if (panel) panel.style.display = 'block';
  setPanelPill('on');

  Mirror.active = true;
  Mirror.hasFace = false;
  Mirror.weights = null;
  Mirror.emotion = 'neutral';
  lastRun = 0;
  if (!rafId) rafId = requestAnimationFrame(tick);
  return true;
}

export function stopMirror() {
  Mirror.active = false;
  cancelAnimationFrame(rafId);
  rafId = 0;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (video) { video.srcObject = null; }
  Mirror.hasFace = false;
  Mirror.weights = null;
  Mirror.emotion = 'neutral';
  const panel = document.getElementById('mirrorPanel');
  if (panel) panel.style.display = 'none';
  setPanelPill('off');
}

function tick(now) {
  if (!Mirror.active) { rafId = 0; return; }
  rafId = requestAnimationFrame(tick);
  if (now - lastRun < 33 || !video || video.readyState < 2 || !landmarker) return; // ~30fps
  lastRun = now;
  try {
    const res = landmarker.detectForVideo(video, now);
    Mirror.hasFace = !!(res.faceBlendshapes && res.faceBlendshapes.length);
    if (Mirror.hasFace) {
      const w = {};
      for (const c of res.faceBlendshapes[0].categories) {
        if (c.categoryName && c.categoryName !== '_neutral') w[c.categoryName] = c.score;
      }
      Mirror.weights = w;
      Mirror.lastDetect = now;
      Mirror.emotion = detectEmotion(w);
    } else {
      Mirror.weights = null;
    }
  } catch (e) { /* transient frame errors are harmless */ }
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden && Mirror.active) stopMirror();
});