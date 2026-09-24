// ============================================================
// Morphius vision worker — YOLO image preprocessing off-thread
// ------------------------------------------------------------
// The heaviest per-frame vision cost is producing a 640x640 RGBA
// frame and converting it into the CHW (channel-height-width)
// Float32 tensor planes YOLO expects (a ~1.2M-element loop). The
// RGBA readback alone used to force a synchronous GPU pipeline
// flush on the main thread every detection frame, janking the
// Three.js render loop — so this worker owns BOTH steps:
//
// Protocol (in):
//   { type:'preprocess', size, width, height,
//     rgba: <Transferable ArrayBuffer>, buffer?: <Transferable ArrayBuffer> }
//     rgba is a Uint8ClampedArray buffer (R,G,B,A x width x height) —
//     legacy main-thread canvas path.
//   { type:'preprocess-bmp', size, width, height, bitmap: <ImageBitmap>,
//     vw, vh, dx, dy, nw, nh, buffer?: <Transferable ArrayBuffer> }
//     preferred path: the main thread only snapshots the video frame
//     (createImageBitmap, async). The letterbox draw + getImageData
//     + CHW conversion all happen here, off the main thread.
//
// Protocol (out):
//   { type:'preprocess-done', float32: <Transferable ArrayBuffer>,
//     width, height, size }   — Float32Array of length 3*width*height
//                               laid out as R,G,B planes in [0..1]
//   { type:'preprocess-error', message }
//
// `buffer` is the main thread handing back the PREVIOUS frame's
// output (ping-pong) so steady-state preprocessing allocates no new
// garbage on either thread (~1.2MB/frame otherwise).
//
// The worker is deliberately dependency-free (no ort/tf imports), so
// it stays tiny and never loads on-screen script bundles.
// ============================================================

let offscreen = null; // OffscreenCanvas for the bitmap path (cached)
let offCtx = null;

// Reuse the caller-provided output buffer when the size matches; only
// allocate when there is nothing to recycle (first frame / size change).
function acquireOutput(size, buffer) {
  if (buffer && buffer.byteLength === size * 4) return new Float32Array(buffer);
  return new Float32Array(size);
}

// Single pass over each plane; branch-free plane writes are the
// fastest real-world layout for a src.Tensor([1,3,W,H]).
function convertRgbaToChw(src, plane, out) {
  for (let i = 0; i < plane; i++) {
    const j = i * 4;
    out[i] = src[j] / 255;                // R
    out[plane + i] = src[j + 1] / 255;    // G
    out[2 * plane + i] = src[j + 2] / 255; // B
  }
  return out;
}

function ensureSurface(width, height) {
  if (!offscreen || offscreen.width !== width || offscreen.height !== height) {
    offscreen = new OffscreenCanvas(width, height);
    offCtx = offscreen.getContext('2d', { willReadFrequently: true });
  }
  return offCtx;
}

self.onmessage = (e) => {
  const msg = e.data || {};
  try {
    const { size, width, height, buffer } = msg;
    let out;
    if (msg.type === 'preprocess') {
      const src = new Uint8ClampedArray(msg.rgba);
      out = acquireOutput(size, buffer);
      convertRgbaToChw(src, width * height, out);
    } else if (msg.type === 'preprocess-bmp') {
      const { bitmap, vw, vh, dx, dy, nw, nh } = msg;
      const ctx = ensureSurface(width, height);
      ctx.fillStyle = '#727272';
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(bitmap, 0, 0, vw, vh, dx, dy, nw, nh);
      try { bitmap.close(); } catch (err) { /* already closed */ }
      const rgba = ctx.getImageData(0, 0, width, height).data;
      out = acquireOutput(size, buffer);
      convertRgbaToChw(rgba, width * height, out);
    } else {
      return; // unknown message — ignore
    }
    self.postMessage(
      { type: 'preprocess-done', float32: out.buffer, width, height, size },
      [out.buffer]
    );
  } catch (err) {
    // Never leave the transferred bitmap dangling on the error path.
    if (msg.type === 'preprocess-bmp' && msg.bitmap) {
      try { msg.bitmap.close(); } catch (closeErr) { /* neutered */ }
    }
    self.postMessage({ type: 'preprocess-error', message: (err && err.message) || String(err) });
  }
};
