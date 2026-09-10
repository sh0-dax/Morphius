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
//   Vision.getInfo()           -> { backend, gpuAccelerated, effectiveFps, configuredFps }
//   Vision.getStats()          -> { backend, running, runCount, lastRun, ... }
//   Vision.describeScene()     -> stub (VLM not yet wired)
//
// onDetections receives: [{ class, score, bbox: [x, y, w, h] }, ...]
// in normalized [0..1] video-space coordinates, so callers don't need to
// know the source video's pixel size.
//
// NOTE: wired into the UI via js/app.js — an opt-in Settings option
// ("Real-time object detection") starts/stops this module with the device
// camera. Detections drive a status pill, a live label, and an optional
// avatar "engaged" reaction when a person is present. Can also be driven
// manually from the console (Vision.init/start/stop) for verification.
// ============================================================

import { parseYoloOneToOne, nonMaxSuppressionPerClass } from './pure.js';
import { clampVisionFps, DEFAULT_VISION_FPS, classifySceneMotion, decideDetectionGate } from './visionLogic.js';

let YOLO_MODEL_URL = './models/yolo26n_int8.onnx'; // see scripts/export_model.py
const YOLO_INPUT_SIZE = 640;
const YOLO_SCORE_THRESHOLD = 0.45;
const COCO_SCORE_THRESHOLD = 0.5;

// Self-hosted onnxruntime-web (no CDN): the UMD build plus the wasm backends
// we vendor under ./libs/ort/. Keeps YOLO fully CDN-independent and offline.
const ORT_LIB_URL = './libs/ort/ort.min.js';
const ORT_WASM_PATH = './libs/ort/wasm/';

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
let timerId = null;
let running = false;
let detectCallback = null;
let offscreen = null;
let offCtx = null;
let maxFps = DEFAULT_VISION_FPS; // detection frames/sec (decoupled from rAF)
let lastRun = 0;
let runCount = 0;        // inferences actually started (diagnostics/smoke)
let dimsLogged = false;      // log output.dims exactly once, then stay quiet

// ---- Idle-floor scene gate ----
// A tiny 16x12 luma probe (not the 640x640 readback) tells us whether the
// scene is static; visionLogic.decideDetectionGate() turns "cold + static"
// into a lower detection rate so we don't burn GPU/CPU re-analyzing frames
// that never change (see gate logic there for the state machine).
const GATE_W = 16;
const GATE_H = 12;
const GATE_FLOOR_FPS = 2;      // detection rate once cold + static
const GATE_BURST_MS = 3000;    // keep full rate briefly after the last person
const GATE_STABLE_MS = 3000;   // how long a scene must sit still before flooring
let gateCanvas = null;         // 16x12 offscreen for the luma probe
let gateCtx = null;
let gateLumaPrev = null;       // Uint8Array(GATE_W*GATE_H) last probe
let gateState = { burstUntil: -Infinity, stableSince: -1 };
let lastPersonPresent = false;

// ---- Backend acceleration ----
// 'yolo-webgpu' can actually mean WebGPU OR a silent CPU (wasm) inference if
// WebGPU is absent. CPU inference is genuinely heavy, so we track whether the
// active backend computes on the GPU and clamp its rate accordingly.
const VISION_CPU_FPS_CAP = 3;  // max detection FPS for a wasm/CPU inference path
let gpuAccelerated = true;

// Worker-based preprocessing keeps the heavy RGBA->CHW permutation off the
// main thread so the Three.js render loop never janks (see visionWorker.js).
let worker = null;           // Worker | null (lazily created in init)
let workerPending = null;    // { resolve, reject } for the in-flight request
let workerAvailable = typeof Worker !== 'undefined';
// Preallocated output buffer reused across frames (worker returns a transferred
// Float32Array; inline path reuses this to avoid per-frame allocation).
let preallocFloat32 = null;  // Float32Array | null

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
  await loadScript(ORT_LIB_URL);
  // eslint-disable-next-line no-undef
  ort.env.wasm.wasmPaths = ORT_WASM_PATH;
  // eslint-disable-next-line no-undef
  session = await ort.InferenceSession.create(YOLO_MODEL_URL, {
    executionProviders: ['webgpu', 'wasm'],
    graphOptimizationLevel: 'all',
  });
  backend = 'yolo-webgpu';
}

// Lazily spin up the preprocessing worker. A failed worker never blocks the
// pipeline: preprocessing simply falls back to the inline path below.
function ensurePreprocessWorker() {
  if (!workerAvailable || worker) return !!worker;
  try {
    worker = new Worker(new URL('./visionWorker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (e) => {
      const d = e.data || {};
      if (d.type === 'preprocess-done' && workerPending) {
        const { resolve } = workerPending;
        workerPending = null;
        resolve(new Float32Array(d.float32));
      }
    };
    worker.onerror = (err) => {
      if (workerPending) {
        const { reject } = workerPending;
        workerPending = null;
        reject(err);
      }
    };
    return true;
  } catch (err) {
    console.warn('[vision] worker unavailable, preprocessing inline:', err);
    workerAvailable = false;
    worker = null;
    return false;
  }
}

function terminatePreprocessWorker() {
  if (worker) {
    try { worker.terminate(); } catch (e) {}
    worker = null;
  }
  workerPending = null;
}

// Asynchronous RGBA->CHW permutation via the worker, with a reusable
// preallocated output Float32Array on the main thread. Returns a Promise of
// a Float32Array of length 3*width*height (R,G,B planes in [0..1]).
function workerPreprocess(rgba, width, height) {
  const size = 3 * width * height;
  if (!preallocFloat32 || preallocFloat32.length !== size) {
    preallocFloat32 = new Float32Array(size);
  }
  return new Promise((resolve, reject) => {
    if (!worker) { reject(new Error('no worker')); return; }
    const transfer = rgba;
    workerPending = { resolve, reject };
    worker.postMessage(
      { type: 'preprocess', size, width, height, rgba },
      [transfer]
    );
  });
}

// Inline fallback: same work as the worker but on the calling thread. Used
// when Worker is unavailable. Reuses preallocFloat32 (resized as needed).
function inlinePreprocess(rgba, width, height) {
  const src = new Uint8ClampedArray(rgba);
  const plane = width * height;
  const size = 3 * plane;
  if (!preallocFloat32 || preallocFloat32.length !== size) {
    preallocFloat32 = new Float32Array(size);
  }
  for (let i = 0; i < plane; i++) {
    const j = i * 4;
    preallocFloat32[i] = src[j] / 255;
    preallocFloat32[plane + i] = src[j + 1] / 255;
    preallocFloat32[2 * plane + i] = src[j + 2] / 255;
  }
  return preallocFloat32;
}
async function initCocoSsd() {
  await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js');
  await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd/dist/coco-ssd.min.js');
  // eslint-disable-next-line no-undef
  await tf.setBackend('webgl');
  // eslint-disable-next-line no-undef
  await tf.ready(); // resolve once the WebGL backend is fully initialized to avoid a race with the model load
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
 * @param {number} [opts.maxFPS]                   detection frames/sec (clamped 1-30; default 8; Infinity = unlimited)
 * @returns {Promise<string>} the active backend
 */
async function init(video, { forceBackend, modelUrl, maxFPS } = {}) {
  videoEl = video;
  if (modelUrl) YOLO_MODEL_URL = modelUrl;
  // Clamp + pin the default detection rate. 0/undefined -> DEFAULT_VISION_FPS;
  // pass Infinity to explicitly opt out of throttling.
  maxFps = maxFPS === Infinity ? Infinity : clampVisionFps(maxFPS);
  offscreen = document.createElement('canvas');
  offCtx = offscreen.getContext('2d', { willReadFrequently: true });

  const gpu = await hasWebGPU();
  const wantYolo = forceBackend ? forceBackend === 'yolo' : gpu;

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

  // 'yolo-webgpu' is a GPU backend ONLY when WebGPU was actually available;
  // otherwise onnxruntime silently ran the argument on CPU (wasm), which is
  // genuinely heavy and gets rate-capped via effectiveMaxFps().
  gpuAccelerated = backend === 'yolo-webgpu' ? gpu : true;

  if (dimsLogged) dimsLogged = false; // new session → re-log once on next run
  if (backend === 'yolo-webgpu') {
    // Best effort: start the preprocessing worker so inference frames don't
    // block the render loop. On failure we transparently fall back to inline.
    if (!worker && workerAvailable) try { ensurePreprocessWorker(); } catch (err) {
      console.warn('[vision] worker init failed, preprocessing inline:', err);
      worker = null;
    }
  }
  return backend;
}

// Clamp the detection rate to the CPU path cap when the active backend computes
// on CPU (wasm YOLO). GPU backends run at the user's configured rate.
function effectiveMaxFps() {
  if (gpuAccelerated) return maxFps;
  if (maxFps === Infinity) return Infinity; // explicit opt-out of throttling
  return Math.min(maxFps, VISION_CPU_FPS_CAP);
}

/**
 * Letterbox-resize a video frame into a square input tensor for YOLO.
 * The heavy RGBA->CHW conversion runs in the preprocessing worker (off the
 * main thread); falls back to inline when a Worker is unavailable.
 * @returns {Promise<{float32: Float32Array, scale: number, dx: number, dy: number}>}
 */
async function preprocessForYolo() {
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

  // getImageData returns a fresh RGBA buffer each call; we transfer it to the
  // worker (buffer ownership moves off the main thread), so no per-frame
  // allocation lingers on the JS heap.
  const rgba = offCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE).data;

  let float32;
  if (worker) {
    try {
      // Returns a transferred Float32Array (length 3*W*H).
      float32 = await workerPreprocess(rgba.buffer, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
    } catch (err) {
      // Worker hiccup — degrade to inline for this frame only.
      float32 = inlinePreprocess(rgba.buffer, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
    }
  } else {
    float32 = inlinePreprocess(rgba.buffer, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  }

  return { float32, scale, dx, dy };
}

/**
 * Run one YOLO inference pass and return normalized detections.
 *
 * YOLO26n is exported with the DEFAULT "one-to-one" head
 * (see https://docs.ultralytics.com/models/yolo26/), so the ONNX output is
 * shaped [1, 300, 6] — each row = [x1, y1, x2, y2, score, class] in absolute
 * letterboxed pixels — already end-to-end NMS-free. parseYoloOneToOne() (in
 * pure.js) undoes the letterbox and normalizes to [0..1] video space, then a
 * per-class NMS safety net drops any duplicate/overlapping boxes (normally a
 * no-op for the one-to-one head, but guarantees no dupes if it ever sees a
 * one-to-many body).
 */
async function detectYolo() {
  const { float32, scale, dx, dy } = await preprocessForYolo();
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

  const dets = parseYoloOneToOne(output.data, output.dims, {
    scale, dx, dy,
    videoW: videoEl.videoWidth,
    videoH: videoEl.videoHeight,
    scoreThreshold: YOLO_SCORE_THRESHOLD,
  });

  // Class-aware NMS as a safety net: the one-to-one head is NMS-free by design
  // (so this is normally a no-op), but per-class suppression guarantees we never
  // emit duplicate/overlapping boxes for the same object if the model ever drops
  // a one-to-many export. numeric `class` is still intact here, before mapping.
  const nmsDets = nonMaxSuppressionPerClass(dets, 0.5);

  return nmsDets.map((d) => ({ ...d, class: COCO_CLASSES[d.class] ?? `class_${d.class}` }));
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

/**
 * Cheap 16x12 luma probe + motion flag for the idle-floor gate. Much cheaper
 * than the 640x640 preprocess (a ~200-byte readback instead of a 1.6MB one), so
 * we can afford to sample it on every loop wake. Returns true when the scene
 * visibly changed since the previous probe.
 */
function gateFrameMotion() {
  if (!videoEl || videoEl.readyState < 2) return false;
  if (!gateCanvas) {
    gateCanvas = document.createElement('canvas');
    gateCanvas.width = GATE_W;
    gateCanvas.height = GATE_H;
    gateCtx = gateCanvas.getContext('2d', { willReadFrequently: true });
  }
  let curr;
  try {
    gateCtx.drawImage(videoEl, 0, 0, GATE_W, GATE_H);
    const data = gateCtx.getImageData(0, 0, GATE_W, GATE_H).data;
    curr = new Uint8Array(GATE_W * GATE_H);
    for (let i = 0; i < GATE_W * GATE_H; i++) {
      curr[i] = ((data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3) | 0;
    }
  } catch (e) {
    // Never trust a failed probe: treat it as motion so the detector stays hot
    // rather than flooring on a broken read.
    return true;
  }
  const motion = gateLumaPrev ? classifySceneMotion(gateLumaPrev, curr) : false;
  gateLumaPrev = curr;
  return motion;
}

async function loop(now) {
  if (!running) return;

  // Skip inference while the video isn't ready to avoid garbage frames.
  if (!videoEl || videoEl.paused || videoEl.readyState < 2) { scheduleNext(0); return; }

  const motion = maxFps !== Infinity && gateFrameMotion();
  const base = effectiveMaxFps();

  const gate = maxFps === Infinity
    ? { shouldRun: true, nextDelayMs: 0 }
    : decideDetectionGate({
        now,
        lastRun,
        personPresent: lastPersonPresent,
        motion,
        state: gateState,
        baseFps: base,
        floorFps: Math.min(GATE_FLOOR_FPS, base),
        burstMs: GATE_BURST_MS,
        stableMs: GATE_STABLE_MS,
      });
  gateState = gate.state || gateState;

  if (gate.shouldRun) {
    lastRun = now;
    runCount++;
    try {
      const detections = backend === 'yolo-webgpu' ? await detectYolo() : await detectCocoSsd();
      lastPersonPresent = detections.some((d) => d.class === 'person');
      detectCallback?.(detections, backend);
    } catch (err) {
      console.error('[vision] inference error', err);
    }
  }
  scheduleNext(gate.nextDelayMs === undefined ? 0 : gate.nextDelayMs);
}

// Detection is decoupled from requestAnimationFrame so it never competes with
// the Three.js render loop for frame budget on the main thread. We self-
// schedule with a fixed interval and only wake when a frame is due.
function scheduleNext(delayMs) {
  if (!running) return;
  timerId = setTimeout(() => loop(performance.now()), delayMs);
}

/** Start the detection loop. Callback fires approx. maxFPS times/sec. */
function start(onDetections) {
  if (!backend) throw new Error('vision: call Vision.init() before start()');
  detectCallback = onDetections;
  running = true;
  lastRun = 0;
  lastPersonPresent = false;
  gateLumaPrev = null;
  gateState = { burstUntil: -Infinity, stableSince: -1 };
  if (!timerId) timerId = setTimeout(() => loop(performance.now()), 0);
}

function stop() {
  running = false;
  if (timerId) { clearTimeout(timerId); timerId = null; }
}

// Full teardown: stop the loop and reclaim the preprocessing worker + buffers.
// Kept separate from stop() so pause/resume (which only calls stop/start) keeps
// the already-created worker alive for the next resume.
function dispose() {
  stop();
  terminatePreprocessWorker();
  session = null;
  cocoModel = null;
  backend = null;
  preallocFloat32 = null;
  dimsLogged = false;
  gpuAccelerated = true;
  lastPersonPresent = false;
  gateLumaPrev = null;
  gateState = { burstUntil: -Infinity, stableSince: -1 };
  runCount = 0;
}

function getBackend() {
  return backend;
}

// Runtime diagnostics: alive counters + the active gate state, useful for
// console debugging and for smoke tests to assert the throttle cadence.
function getStats() {
  return {
    backend,
    running,
    runCount,
    lastRun,
    effectiveFps: effectiveMaxFps(),
    configuredFps: maxFps,
    gpuAccelerated,
    gate: { ...gateState },
  };
}

// Runtime load info so the UI can surface throttle/cap decisions (and so the
// console API stays honest about what's actually computing at full speed).
function getInfo() {
  return {
    backend,
    gpuAccelerated,
    effectiveFps: effectiveMaxFps(),
    configuredFps: maxFps,
  };
}

// Isolated placeholder so future code can call Vision.describeScene() without
// changing the API signature when a real VLM backend is added. Never invoked
// by the detection loop, so it has no performance cost when unused.
async function describeScene() {
  console.warn('[vision] describeScene() not yet wired to a VLM backend');
  return { available: false, description: null };
}

// Stop the detection loop when the tab is hidden to avoid burning CPU/GPU on
// an invisible page (matches the mirror module's behaviour), and resume it
// automatically when the tab comes back if it was running before.
let wasRunningBeforeHide = false;
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    wasRunningBeforeHide = running;
    stop();
  } else if (wasRunningBeforeHide && backend && !running) {
    wasRunningBeforeHide = false;
    running = true;
    lastRun = 0;
    if (!timerId) timerId = setTimeout(() => loop(performance.now()), 0);
  }
});

export const Vision = { init, start, stop, dispose, getBackend, getInfo, getStats, describeScene };
export default Vision;

// Surface for console/smoke driving (promised in the header): the app imports
// and wires Vision internally, but exposing it lets embedders/debuggers inspect
// runtime stats (Vision.getStats()) or drive detection manually.
if (typeof window !== 'undefined') window.Vision = Vision;
