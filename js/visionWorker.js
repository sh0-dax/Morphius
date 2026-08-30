// ============================================================
// Morphius vision worker — YOLO image preprocessing off-thread
// ------------------------------------------------------------
// The heaviest per-frame vision cost is converting a 640x640 RGBA
// ImageData buffer into the CHW (channel-height-width) Float32 tensor
// planes YOLO expects (a ~1.2M-element loop). Doing that on the main
// thread janks the Three.js render loop, so this worker owns it.
//
// Protocol:
//   in:  { type:'preprocess', size, width, height, rgba: <Transferable ArrayBuffer> }
//        rgba is a Uint8ClampedArray buffer (R,G,B,A x width x height).
//   out: { type:'preprocess-done', float32: <Transferable ArrayBuffer>,
//          width, height }
//        float32 is a Float32Array of length 3*width*height laid out
//        as R-plane, G-plane, B-plane (each normalized to [0..1]).
//
// The worker is deliberately dependency-free (no ort/tf imports), so
// it stays tiny and never loads on-screen script bundles.
// ============================================================

self.onmessage = (e) => {
  const { type, size, width, height, rgba } = e.data || {};
  if (type !== 'preprocess') return;

  const src = new Uint8ClampedArray(rgba);
  const plane = width * height;
  const float32 = new Float32Array(3 * plane);

  // Single pass over each plane; branch-free plane writes are the
  // fastest real-world layout for a src.Tensor([1,3,W,H]).
  for (let i = 0; i < plane; i++) {
    const j = i * 4;
    float32[i] = src[j] / 255;            // R
    float32[plane + i] = src[j + 1] / 255; // G
    float32[2 * plane + i] = src[j + 2] / 255; // B
  }

  self.postMessage(
    { type: 'preprocess-done', float32: float32.buffer, width, height, size },
    [float32.buffer]
  );
};
