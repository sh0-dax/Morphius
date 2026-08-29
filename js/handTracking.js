// ============================================================
// AI Face — Hand Landmarker for Projection Gesture Control
// MediaPipe HandLandmarker (WASM/GPU) for pinch-zoom + rotate
// ============================================================

import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { lerpWeight } from './pure.js';

const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarker = null;
let handVideo = null;
let videoStream = null;
let rafId = 0;
let lastRun = 0;
const DETECT_INTERVAL_MS = 66; // ~15 FPS

// EMA state for landmark smoothing
const emaState = {
  landmarks: null, // 21 x 3 array
  pinchDist: null,
  handCenter: null, // {x, y} normalized
};

// Callbacks
let onGesture = null; // (delta) => { pinchDelta, rotateDeltaX, rotateDeltaY }
let onStatusChange = null; // (state) => { active, hasHand, pinchStrength }

const THUMB_TIP = 4;
const INDEX_TIP = 8;
const WRIST = 0;
const MIDDLE_MCP = 9;

function normalizeLandmarks(landmarks, videoWidth, videoHeight) {
  return landmarks.map((lm) => ({
    x: lm.x / videoWidth,
    y: lm.y / videoHeight,
    z: lm.z / Math.max(videoWidth, videoHeight),
  }));
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function centroid(landmarks) {
  let x = 0, y = 0, z = 0;
  for (const lm of landmarks) { x += lm.x; y += lm.y; z += lm.z; }
  const n = landmarks.length;
  return { x: x / n, y: y / n, z: z / n };
}

function emaUpdate(target, source, rate = 0.15) {
  if (!target) return { ...source };
  const out = {};
  for (const k of Object.keys(source)) {
    out[k] = lerpWeight(target[k], source[k], rate);
  }
  return out;
}

async function createLandmarker(delegate) {
  const vision = await FilesetResolver.forVisionTasks(CDN + '/wasm');
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands: 1, // v1: single hand only
  });
}

async function ensureEngine() {
  if (landmarker) return true;
  try {
    landmarker = await createLandmarker('GPU');
  } catch (e) {
    console.warn('HandLandmarker GPU init failed, falling back to CPU:', e);
    try {
      landmarker = await createLandmarker('CPU');
    } catch (e2) {
      console.warn('HandLandmarker init failed:', e2);
      return false;
    }
  }
  return true;
}

function detectFrame() {
  if (!landmarker || !handVideo || handVideo.readyState < 2) {
    rafId = requestAnimationFrame(detectFrame);
    return;
  }

  const now = performance.now();
  if (now - lastRun < DETECT_INTERVAL_MS) {
    rafId = requestAnimationFrame(detectFrame);
    return;
  }
  lastRun = now;

  try {
    const results = landmarker.detectForVideo(handVideo, now);
    if (results.landmarks && results.landmarks.length > 0) {
      const lms = results.landmarks[0]; // single hand, 21 landmarks
      const vw = handVideo.videoWidth;
      const vh = handVideo.videoHeight;
      const norm = normalizeLandmarks(lms, vw, vh);

      // EMA smoothing on raw landmarks
      emaState.landmarks = emaUpdate(emaState.landmarks, 
        Object.fromEntries(norm.map((lm, i) => [i, lm])), 0.15);

      // Compute pinch distance (thumb tip 4 to index tip 8)
      const thumb = emaState.landmarks[THUMB_TIP];
      const index = emaState.landmarks[INDEX_TIP];
      const wrist = emaState.landmarks[WRIST];
      const middleMcp = emaState.landmarks[MIDDLE_MCP];

      if (thumb && index && wrist && middleMcp) {
        const pinchDist = distance(thumb, index);
        const refDist = distance(wrist, middleMcp) || 1;
        const normalizedPinch = pinchDist / refDist;

        // EMA on pinch distance
        emaState.pinchDist = emaState.pinchDist !== null
          ? lerpWeight(emaState.pinchDist, normalizedPinch, 0.2)
          : normalizedPinch;

        // Hand center for rotate (centroid of all landmarks)
        const center = centroid(Object.values(emaState.landmarks));
        emaState.handCenter = emaUpdate(emaState.handCenter, center, 0.15);

        // Compute deltas for gestures
        let pinchDelta = 0;
        let rotateDeltaX = 0;
        let rotateDeltaY = 0;

        // Pinch: relative change in normalized pinch distance
        // Dead zone: ignore changes < 0.02 (2% of reference)
        if (onGesture) {
          // We'll track previous pinch in the callback closure or here
          pinchDelta = 0; // will be computed in callback with previous value
        }

        if (onGesture && emaState.prevCenter) {
          rotateDeltaX = emaState.handCenter.x - emaState.prevCenter.x;
          rotateDeltaY = emaState.handCenter.y - emaState.prevCenter.y;
          // Dead zone for rotate: ignore < 0.005 normalized
          const ROT_DEADZONE = 0.005;
          if (Math.abs(rotateDeltaX) < ROT_DEADZONE) rotateDeltaX = 0;
          if (Math.abs(rotateDeltaY) < ROT_DEADZONE) rotateDeltaY = 0;
        }
        emaState.prevCenter = { ...emaState.handCenter };

        // Pinch strength for HUD (0..1)
        const pinchStrength = Math.max(0, Math.min(1, (0.3 - emaState.pinchDist) * 3)); // approx

        if (onStatusChange) {
          onStatusChange({ active: true, hasHand: true, pinchStrength });
        }

        if (onGesture) {
          onGesture({ pinchDist: emaState.pinchDist, pinchDelta, rotateDeltaX, rotateDeltaY });
        }
      }
    } else {
      // No hand detected
      if (onStatusChange) onStatusChange({ active: true, hasHand: false, pinchStrength: 0 });
      emaState.prevCenter = null;
    }
  } catch (e) {
    console.warn('HandLandmarker detect error:', e);
  }

  rafId = requestAnimationFrame(detectFrame);
}

export async function startHandTracking(videoEl, callbacks = {}) {
  if (!videoEl) throw new Error('video element required');
  if (!await ensureEngine()) throw new Error('HandLandmarker init failed');

  handVideo = videoEl;
  onGesture = callbacks.onGesture || null;
  onStatusChange = callbacks.onStatusChange || null;

  // Reset EMA state
  emaState.landmarks = null;
  emaState.pinchDist = null;
  emaState.handCenter = null;
  emaState.prevCenter = null;

  if (handVideo.readyState >= 2) {
    lastRun = 0;
    rafId = requestAnimationFrame(detectFrame);
  } else {
    handVideo.onloadeddata = () => {
      lastRun = 0;
      rafId = requestAnimationFrame(detectFrame);
    };
  }

  return true;
}

export function stopHandTracking() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = 0;
  }
  onGesture = null;
  onStatusChange = null;
  // Reset EMA
  emaState.landmarks = null;
  emaState.pinchDist = null;
  emaState.handCenter = null;
  emaState.prevCenter = null;
}

export function getLandmarker() {
  return landmarker;
}

export function isReady() {
  return !!landmarker;
}