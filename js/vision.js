// ============================================================
// Morphius vision module — real-time object detection
// ------------------------------------------------------------
// Adds real-time object detection to the face/state engine.
//
// Strategy:
//   - If WebGPU is available -> YOLO26n one-to-one via
//     onnxruntime-web (fast, accurate, end-to-end NMS-free)
//   - Otherwise               -> COCO-SSD via TensorFlow.js/WebGL
//                                (broad compatibility)
//
// Public API mirrors the style of mirror.js / localSpeech.js:
//   await Vision.init(videoElement)
//   Vision.start(onDetections)
//   Vision.stop()
//   Vision.getBackend()        -> 'yolo-webgpu' | 'coco-ssd-webgl'
//   Vision.describeScene()     -> stub (VLM not yet wired)
//
// onDetections receives: [{ class, score, bbox: [x, y, w, h] }, ...]
// in normalized [0..1] video-space coordinates, so callers don't need to
// know the source video's pixel size.
//
// NOTE: this module is currently STANDALONE — it is not imported by
// app.js or referenced from index.html. Drive it from the console for
// verification. UI wiring is a later, separate stage.
// ============================================================

import { parseYoloOneToOne } from './pure.js';

let YOLO_MODEL_URL = './models/yolo26n_int8.onnx'; // see scripts/export_model.py
const YOLO_INPUT_SIZE = 640;
const YOLO_SCORE_THRESHOLD = 0.45;
const COCO_SCORE_THRESHOLD = 0.5;

// COCO class list (80 classes) — index-aligned with YOLO/COCO-SSD outputs.
const COCO_CLASSES = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
  'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
  'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
  'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
  'remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
  'toothbrush'
];

let backend = null;          // 'yolo-webgpu' | 'coco-ssd-webgl'
let session = null;          // onnxruntime session (YOLO path)
let cocoModel = null;        // coco-ssd model instance (fallback path)
let videoEl = null;
let rafId = null;
let running = false;
let detectCallback = null;
let offscreen = null;
let offCtx = null;
let maxFps = 0;              // 0 = unlimited
let lastRun = 0;
let dimsLogged = false;      // log output.dims exactly once, then stay quiet

/** Feature-detect WebGPU the same way the rest of the app does for WebLLM. */
async function hasWebGPU() {
  if (!('gpu' in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

/** Lazily load a script (zero-build — matches the project's no-bundler approach). */
function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

async function initYolo() {
  await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
  // eslint-disable-next-line no-undef
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  // eslint-disable-next-line no-undef
  session = await ort.InferenceSession.create(YOLO_MODEL_URL, {
    executionProviders: ['webgpu', 'wasm'],
    graphOptimizationLevel: 'all',
  });
  backend = 'yolo-webgpu';
}

async function initCocoSsd() {
  await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js');
  await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd/dist/coco-ssd.min.js');
  // eslint-disable-next-line no-undef
  await tf.setBackend('webgl');
  // eslint-disable-next-line no-undef
  cocoModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
  backend = 'coco-ssd-webgl';
}

/**
 * Initialize the detector against a given <video> element.
 * Chooses YOLO+WebGPU when available, falls back to COCO-SSD+WebGL otherwise.
 *
 * @param {HTMLVideoElement} video
 * @param {object} [opts]
 * @param {'yolo'|'coco-ssd'} [opts.forceBackend]  force a specific backend
 * @param {string} [opts.modelUrl]                 override the default YOLO model URL
 * @param {number} [opts.maxFPS]                   throttle inference to ~N fps (0 = unlimited)
 * @returns {Promise<string>} the active backend
 */
async function init(video, { forceBackend, modelUrl, maxFPS } = {}) {
  videoEl = video;
  if (modelUrl) YOLO_MODEL_URL = modelUrl;
  maxFps = maxFPS && maxFPS > 0 ? maxFPS : 0;
  offscreen = document.createElement('canvas');
  offCtx = offscreen.getContext('2d', { willReadFrequently: true });

  const wantYolo = forceBackend ? forceBackend === 'yolo' : await hasWebGPU();

  try {
    if (wantYolo) {
      await initYolo();
    } else {
      await initCocoSsd();
    }
  } catch (err) {
    // Any failure in the preferred path falls back to the other one.
    console.warn(`[vision] preferred backend failed (${err.message}), falling back`, err);
    if (wantYolo) {
      await initCocoSsd();
    } else {
      await initYolo();
    }
  }

  if (dimsLogged) dimsLogged = false; // new session → re-log once on next run
  return backend;
}

/**
 * Letterbox-resize a video frame into a square input tensor for YOLO.
 * @returns {{float32: Float32Array, scale: number, dx: number, dy: number}}
 */
function preprocessForYolo() {
  offscreen.width = YOLO_INPUT_SIZE;
  offscreen.height = YOLO_INPUT_SIZE;
  offCtx.fillStyle = '#727272';
  offCtx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);

  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  const scale = Math.min(YOLO_INPUT_SIZE / vw, YOLO_INPUT_SIZE / vh);
  const nw = vw * scale;
  const nh = vh * scale;
  const dx = (YOLO_INPUT_SIZE - nw) / 2;
  const dy = (YOLO_INPUT_SIZE - nh) / 2;
  offCtx.drawImage(videoEl, 0, 0, vw, vh, dx, dy, nw, nh);

  const imgData = offCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE).data;
  const float32 = new Float32Array(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
  const plane = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;

  for (let i = 0; i < plane; i++) {
    float32[i] = imgData[i * 4] / 255;              // channel 0: R
    float32[plane + i] = imgData[i * 4 + 1] / 255;  // channel 1: G
    float32[2 * plane + i] = imgData[i * 4 + 2] / 255; // channel 2: B
  }

  return { float32, scale, dx, dy };
}

/**
 * Run one YOLO inference pass and return normalized detections.
 *
 * YOLO26n is exported with the DEFAULT "one-to-one" head
 * (see https://docs.ultralytics.com/models/yolo26/), so the ONNX output is
 * shaped [1, 300, 6] — each row = [x1, y1, x2, y2, score, class] in absolute
 * letterboxed pixels — already end-to-end NMS-free. We therefore DON'T run
 * NMS here; parseYoloOneToOne() (in pure.js) undoes the letterbox and
 * normalizes to [0..1] video space.
 */
async function detectYolo() {
  const { float32, scale, dx, dy } = preprocessForYolo();
  // eslint-disable-next-line no-undef
  const tensor = new ort.Tensor('float32', float32, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
  const feeds = { [session.inputNames[0]]: tensor };
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];

  // Log the ACTUAL output shape once per session — never assume the docs.
  // Expected (one-to-one): [1, 300, 6]. Legacy one-to-many: [1, 84, 8400].
  if (!dimsLogged) {
    console.log('[vision] YOLO output shape:', output.dims);
    dimsLogged = true;
  }

  return parseYoloOneToOne(output.data, output.dims, {
    scale, dx, dy,
    videoW: videoEl.videoWidth,
    videoH: videoEl.videoHeight,
    scoreThreshold: YOLO_SCORE_THRESHOLD,
  }).map((d) => ({ ...d, class: COCO_CLASSES[d.class] ?? `class_${d.class}` }));
}

/** Run one COCO-SSD inference pass and return normalized detections. */
async function detectCocoSsd() {
  const preds = await cocoModel.detect(videoEl);
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  return preds
    .filter((p) => p.score >= COCO_SCORE_THRESHOLD)
    .map((p) => ({
      class: p.class,
      score: p.score,
      bbox: [p.bbox[0] / vw, p.bbox[1] / vh, p.bbox[2] / vw, p.bbox[3] / vh],
    }));
}

async function loop(now) {
  if (!running) return;
  rafId = requestAnimationFrame(loop);

  // Optional throttling via performance.now() delta (not just rAF cadence).
  if (maxFps > 0 && now - lastRun < 1000 / maxFps) return;
  // Skip inference while the video isn't ready to avoid garbage frames.
  if (!videoEl || videoEl.paused || videoEl.readyState < 2) return;
  lastRun = now;

  try {
    const detections = backend === 'yolo-webgpu' ? await detectYolo() : await detectCocoSsd();
    detectCallback?.(detections, backend);
  } catch (err) {
    console.error('[vision] inference error', err);
  }
}

/** Start the detection loop. Callback fires once per frame with results. */
function start(onDetections) {
  if (!backend) throw new Error('vision: call Vision.init() before start()');
  detectCallback = onDetections;
  running = true;
  lastRun = 0;
  if (!rafId) rafId = requestAnimationFrame(loop);
}

function stop() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
}

function getBackend() {
  return backend;
}

// Isolated placeholder so future code can call Vision.describeScene() without
// changing the API signature when a real VLM backend is added. Never invoked
// by the detection loop, so it has no performance cost when unused.
async function describeScene() {
  console.warn('[vision] describeScene() not yet wired to a VLM backend');
  return { available: false, description: null };
}

export const Vision = { init, start, stop, getBackend, describeScene };
export default Vision;
